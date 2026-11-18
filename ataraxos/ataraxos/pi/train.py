"""The perfect-info trainer (2026-11-18): ataraxos's recipe (ataraxos/rl/
train.py, rby4's flags without Expected SARSA, the user's) on the perfect-
info env (src/pi_env.hpp) and net (ataraxos/pi/net.py), for oak's IS-MCTS.

    ~/oak/.venv/bin/python -m ataraxos.pi.train --run runs/x [--key value ...] TEAM_FILE ...

What changes from ataraxos/rl/train.py:
  - one env row per battle step, one net forward per row: the value (P1's
    W/D/L) and both players' policies (oak's flat 315, the legal rows
    gathered). A row where neither side has two choices never reaches the
    net (the env plays it).
  - the data: each player's trajectory is the rows where it has two or more
    choices, as ataraxos's per-player decisions, so returns_and_advantages
    (TD(lambda) W/D/L returns, GAE advantages, gamma 1, terminal rewards) is
    reused as is, on the value from that player's side (P2's: P1's W/D/L
    reversed). A row both players choose in is in both trajectories.
  - no history, so minibatches are shuffled kept decisions, not whole
    trajectories, and there's no causal cache or CUDA-graphed actor.
  - evals against "random", "maxdamage" or checkpoints (ataraxos/pi/agents.py).
Unchanged: clipped PPO, the reverse KL to the data policy, the annealed
reverse KL to the attack_switch magnet, the advantage filter, one epoch, the
EMA, W/D/L value by cross-entropy, the decision-counted LR / temperature
schedules, Exp3 over the team pool, the turn cap as a draw.

The run directory gets config.json, log.jsonl (one line per iteration),
ckpt{it}.pt and teams{it}.npz, as ataraxos's. `--resume latest` carries on
from the newest checkpoint (weights, EMA, optimizer, iteration, Exp3, RNGs;
the games in progress are dropped).
"""

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ataraxos._ext import ataraxos_pi
from ataraxos.pi.agents import NetAgent, make_agent, match
from ataraxos.pi.net import CHOICES, legal_logits, make_net
from ataraxos.rl import metrics, teams
from ataraxos.rl.buffer import returns_and_advantages
from ataraxos.rl.train import (code_version, cpu_seconds, latest_checkpoint, magnet, peak_gb, power_schedule,
                               rss_peak_gb, start_gpu_log)


@dataclass
class Config:
    run: str = "runs/pi_default"
    teams: list = field(default_factory=list)
    size: str = "pi"
    # env
    battles: int = 8192
    threads: int = 8
    seed: int = 1
    max_turns: int = 300
    at_cap: str = "draw"  # "draw" or "drop" (no bootstrap: the env sends no row at the cap)
    # Exp3 (ataraxos's)
    team_gamma: float = 0.1
    team_gamma_floor: float = 0.01
    team_gamma_tau: float = 3e6
    team_gamma_decay: float = 0.5
    team_eta: float = 0.0
    iters: int = 10**9
    # rby4's (rby3's: 8192 games, minibatch 12288)
    games_per_iter: int = 8192
    minibatch: int = 12288
    microbatch: int = 0  # 0: one pass per minibatch
    clip_range: float = 0.2
    td_lambda: float = 0.6
    gae_lambda: float = 0.5
    vf_coef: float = 1.0
    policy_coef: float = 1.0
    kl_coef: float = 0.1
    temperature_coef: float = 0.05
    temperature_decay: float = 0.3
    temperature_ceil: float = 0.1  # rby3's (rby4's 0.0098 continued rby3's schedule)
    temperature_floor: float = 0.005
    lr_coef: float = 0.003
    lr_decay: float = 0.6
    lr_ceil: float = 3e-4  # rby3's (rby4's 1.15e-4 continued rby3's schedule)
    lr_floor: float = 5e-6
    schedule_decisions: float = 5e6
    weight_decay: float = 0.0
    max_grad_norm: float = 0.267
    adv_filt_rate: float = 0.5
    adv_filt_thresh: float = 0.01
    ema_decay: float = 0.98
    magnet: str = "attack_switch"
    anchors: list = field(default_factory=lambda: ["random", "maxdamage"])
    eval_every: int = 25
    eval_battles: int = 512
    save_every: int = 25
    resume: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    gpu_log: float = 2.0


ROW_COLUMNS = ("mon", "hp", "active", "policy", "n", "choices")


@dataclass
class Data:
    """Finished trajectories of decisions (row, player), each contiguous:
    trajectory g is decisions start[g] : start[g] + length[g]. The fields
    returns_and_advantages and metrics.data_stats read are the deciding
    player's: n, choices, logp, pick, value (its side's W/D/L logits)."""
    row: np.ndarray       # [N] into the row arrays
    player: np.ndarray    # [G]
    side: np.ndarray      # [N] the deciding player
    n: np.ndarray         # [N]
    choices: np.ndarray   # [N, CHOICES]
    logp: np.ndarray      # [N, CHOICES]
    pick: np.ndarray      # [N]
    value: np.ndarray     # [N, 3] the deciding player's (loss, draw, win)
    start: np.ndarray     # [G]
    length: np.ndarray    # [G]
    reward: np.ndarray    # [G] the player's
    rows: dict            # ROW_COLUMNS, each [R, ...]
    truncated: np.ndarray = None


class Buffer:
    """ataraxos's Buffer over rows: the env's rows are stored once, and each
    (battle, player)'s open trajectory is its rows with two or more
    choices."""

    def __init__(self, at_cap="draw"):
        if at_cap not in ("draw", "drop"):
            raise ValueError(at_cap)
        self.at_cap = at_cap
        self.parts = []
        self.size = 0
        self.open = {}  # (battle, player) -> (game, [row ids])
        self.done = []  # (row ids, reward, player)

    def end(self, obs):
        for b, g, r, c in zip(obs["end_battle"].tolist(), obs["end_game"].tolist(), obs["end_result"].tolist(),
                              obs["end_capped"].tolist()):
            for p, sign in ((0, 1), (1, -1)):
                t = self.open.pop((b, p), None)
                if t is not None and t[0] == g and t[1] and not (c and self.at_cap == "drop"):
                    self.done.append((t[1], sign * r, p))

    def add(self, obs, picks, logp, value):
        R = len(obs["n"])
        if R == 0:
            return
        part = {k: obs[k] for k in ROW_COLUMNS}
        part.update(pick=picks, logp=logp, value=value)
        self.parts.append(part)
        n = obs["n"]
        for p in (0, 1):
            for i in np.flatnonzero(n[:, p] >= 2).tolist():
                key = (int(obs["battle"][i]), p)
                g = int(obs["game"][i])
                t = self.open.get(key)
                if t is None or t[0] != g:
                    t = (g, [])
                    self.open[key] = t
                t[1].append(self.size + i)
        self.size += R

    def games(self):
        return len(self.done) // 2

    def take(self) -> Data:
        cat = {k: np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]}
        traj = [(t, r, p) for t, r, p in self.done]
        rows = np.array([i for t, _, _ in traj for i in t], np.int64)
        length = np.array([len(t) for t, _, _ in traj], np.int64)
        player = np.array([p for _, _, p in traj], np.int64)
        reward = np.array([r for _, r, _ in traj], np.int64)
        side = np.repeat(player, length)
        # the rows the data needs, renumbered
        used, inv = np.unique(rows, return_inverse=True)
        value = cat["value"][rows]
        value = np.where(side[:, None] == 1, value[:, ::-1], value)
        data = Data(row=inv, player=player, side=side, n=cat["n"][rows, side], choices=cat["choices"][rows, side],
                    logp=cat["logp"][rows, side], pick=cat["pick"][rows, side], value=np.ascontiguousarray(value),
                    start=np.r_[0, np.cumsum(length)[:-1]].astype(np.int64), length=length, reward=reward,
                    rows={k: cat[k][used] for k in ROW_COLUMNS}, truncated=np.zeros(len(length), bool))
        keep = np.unique(np.array([i for _, t in self.open.values() for i in t], np.int64))
        if len(keep):
            self.parts = [{k: v[keep] for k, v in cat.items()}]
            new = {old: j for j, old in enumerate(keep.tolist())}
            self.open = {key: (game, [new[i] for i in t]) for key, (game, t) in self.open.items()}
        else:
            self.parts = []
        self.size = len(keep)
        self.done = []
        return data


class Actor:
    """One forward per row: picks [R, 2] (0 for a side with one choice),
    the data policy's log-probs [R, 2, CHOICES] and the value logits [R, 3]."""

    def __init__(self, net, device, amp):
        self.net, self.dev, self.amp = net, torch.device(device), amp

    @torch.no_grad()
    def act(self, obs):
        d = self.dev
        mon, hp, active = (torch.from_numpy(obs[k]).to(d) for k in ("mon", "hp", "active"))
        with torch.autocast(d.type, torch.bfloat16, enabled=self.amp and d.type == "cuda"):
            value, policy = self.net(mon.long(), hp, active)
        x, _ = legal_logits(policy, torch.from_numpy(obs["policy"]).to(d), torch.from_numpy(obs["n"]).to(d))
        logp = torch.log_softmax(x, -1)
        pick = torch.multinomial(logp.exp().view(-1, CHOICES), 1).view(-1, 2)
        return (pick.to(torch.uint8).cpu().numpy(), logp.cpu().numpy().astype(np.float32),
                value.float().cpu().numpy())


class Learner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.net = make_net(cfg.size).to(self.dev)
        self.ema = make_net(cfg.size).to(self.dev).eval()
        self.ema.load_state_dict(self.net.state_dict())
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.optim = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr_ceil, weight_decay=cfg.weight_decay)
        self.it = 0
        self.decisions = 0

    def schedule_step(self):
        c = self.cfg
        return self.decisions / c.schedule_decisions if c.schedule_decisions > 0 else self.it

    def train(self, data: Data, rng):
        cfg, dev = self.cfg, self.dev
        t0 = time.perf_counter()
        ret, adv = returns_and_advantages(data, cfg.td_lambda, cfg.gae_lambda)
        threshold = max(float(adv.abs().quantile(cfg.adv_filt_rate)) if len(adv) < 2**24 else
                        float(adv.abs()[torch.randperm(len(adv))[:2**24]].quantile(cfg.adv_filt_rate)),
                        cfg.adv_filt_thresh)
        keep = (adv.abs() >= threshold).numpy()
        step = self.schedule_step()
        lr = power_schedule(cfg.lr_coef, step, cfg.lr_decay, cfg.lr_ceil, cfg.lr_floor)
        temperature = power_schedule(cfg.temperature_coef, step, cfg.temperature_decay, cfg.temperature_ceil,
                                     cfg.temperature_floor)
        for g in self.optim.param_groups:
            g["lr"] = lr
        stats = dict(lr=lr, temperature=temperature, schedule_step=step, adv_threshold=threshold,
                     decisions=len(adv), kept=int(keep.sum()))
        stats.update(metrics.data_stats(data, ret, adv))
        # everything on the device once
        T = lambda a, dt=None: torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=dt)
        rows = {k: T(v) for k, v in data.rows.items()}
        dd = dict(row=T(data.row), side=T(data.side), n=T(data.n), choices=T(data.choices),
                  logp=T(data.logp, torch.float), pick=T(data.pick, torch.long), ret=ret.to(dev), adv=adv.to(dev))
        params = [p.detach() for p in self.net.parameters()]
        before = [p.clone() for p in params]
        sums, steps = {}, 0
        self.net.train()
        t1 = time.perf_counter()
        ids = rng.permutation(np.flatnonzero(keep))
        for s in range(0, len(ids), cfg.minibatch):
            group = ids[s:s + cfg.minibatch]
            micros = [group] if cfg.microbatch <= 0 else [group[k:k + cfg.microbatch]
                                                          for k in range(0, len(group), cfg.microbatch)]
            self.optim.zero_grad(set_to_none=True)
            for m in micros:
                loss, out = self.loss(rows, dd, torch.from_numpy(m).to(dev), temperature)
                (loss * (len(m) / len(group))).backward()
                for k, v in out.items():
                    sums[k] = (torch.maximum(sums[k], v) if k in sums and k.endswith("_max") else
                               sums.get(k, 0.0) + v * len(m) / len(group))
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
            self.optim.step()
            for k, v in (("grad_norm", grad_norm), ("grad_clipped", (grad_norm > cfg.max_grad_norm).float())):
                sums[k] = sums.get(k, 0.0) + v
            sums["grad_norm_max"] = torch.maximum(sums["grad_norm_max"], grad_norm) if "grad_norm_max" in sums \
                else grad_norm
            steps += 1
        self.it += 1
        self.decisions += len(data.n)
        with torch.no_grad():
            ema = list(self.ema.parameters())
            torch._foreach_mul_(ema, cfg.ema_decay)
            torch._foreach_add_(ema, params, alpha=1 - cfg.ema_decay)
            for e, p in zip(self.ema.buffers(), self.net.buffers()):
                e.copy_(p)
            delta = torch.stack(torch._foreach_norm(torch._foreach_sub(params, before))).double().pow(2).sum()
            norm = torch.stack(torch._foreach_norm(before)).double().pow(2).sum()
        t2 = time.perf_counter()
        keys = list(sums)
        *vals, delta, norm = torch.stack([torch.as_tensor(sums[k], device=dev).double() for k in keys]
                                         + [delta, norm]).tolist()
        stats.update({k: v if k.endswith("_max") else v / max(1, steps) for k, v in zip(keys, vals)},
                     grad_steps=steps, param_norm=math.sqrt(norm), update_ratio=math.sqrt(delta / max(norm, 1e-30)),
                     train_prep_s=t1 - t0, train_issue_s=t2 - t1, train_wait_s=time.perf_counter() - t2)
        return stats

    def loss(self, rows, dd, idx, temperature):
        cfg, dev = self.cfg, self.dev
        r, side = dd["row"][idx], dd["side"][idx]
        with torch.autocast(dev.type, torch.bfloat16, enabled=cfg.amp and dev.type == "cuda"):
            value, policy = self.net(rows["mon"][r].long(), rows["hp"][r], rows["active"][r])
        K = len(idx)
        k = torch.arange(K, device=dev)
        value = value.float()
        value = torch.where(side[:, None] == 1, value.flip(-1), value)  # the deciding player's W/D/L
        n = dd["n"][idx]
        legal = torch.arange(CHOICES, device=dev)[None] < n[:, None]
        logits = policy[k, side].gather(-1, rows["policy"][r, side].long()).float().masked_fill(~legal, -1e9)
        logp = torch.log_softmax(logits, -1)
        p = logp.exp() * legal
        old = dd["logp"][idx].clamp(min=-1e9).masked_fill(~legal, -1e9)
        a, A, R = dd["pick"][idx][:, None], dd["adv"][idx], dd["ret"][idx]
        ratio = torch.exp(logp.gather(1, a)[:, 0] - old.gather(1, a)[:, 0])
        clipped = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
        policy_loss = -torch.min(A * ratio, A * clipped).mean()
        kl_data = (p * (logp - old)).sum(-1).mean()
        kl_magnet = (p * (logp - magnet(dd["choices"][idx], legal, cfg.magnet))).sum(-1).mean()
        value_loss = -(R * torch.log_softmax(value, -1)).sum(-1).mean()
        loss = cfg.policy_coef * policy_loss + temperature * kl_magnet + cfg.vf_coef * value_loss + cfg.kl_coef * kl_data
        ent = -(p * logp).sum(-1)
        y = logits.detach().masked_fill(~legal, 0)
        y = (y - y.sum(-1, keepdim=True) / n[:, None].float()).masked_fill(~legal, 0)
        return loss, {k: v.detach() for k, v in dict(
            loss=loss, policy_loss=policy_loss, value_loss=value_loss, kl_data=kl_data, kl_magnet=kl_magnet,
            entropy=ent.mean(), entropy_norm_learner=(ent / torch.log(n.float())).mean(),
            clip_frac=((ratio - 1).abs() > cfg.clip_range).float().mean(), ratio_mean=ratio.mean(),
            ratio_max=ratio.max(), logit_abs_max=y.abs().max()).items()}

    def save(self, path, exp3=None, rng=None, team_counts=None, pairs=None):
        ck = dict(net=self.net.state_dict(), ema=self.ema.state_dict(), optim=self.optim.state_dict(), it=self.it,
                  decisions=self.decisions, config=dataclasses.asdict(self.cfg),
                  exp3=exp3.state() if exp3 is not None else None,
                  rng=rng.bit_generator.state if rng is not None else None, torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  team_counts=team_counts, pairs=pairs.state() if pairs is not None else None)
        torch.save(ck, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load(self, ck):
        if ck["config"]["size"] != self.cfg.size:
            raise ValueError(f"checkpoint is size {ck['config']['size']!r}, not {self.cfg.size!r}")
        self.net.load_state_dict(ck["net"])
        self.ema.load_state_dict(ck["ema"])
        self.optim.load_state_dict(ck["optim"])
        self.it, self.decisions = ck["it"], ck["decisions"]


def parse(argv) -> Config:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        if f.name == "teams":
            ap.add_argument("teams", nargs="*")
        elif isinstance(default, list):
            ap.add_argument(f"--{f.name}", type=lambda s: [x for x in s.split(",") if x], default=default)
        elif isinstance(default, bool):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=default)
        else:
            ap.add_argument(f"--{f.name}", type=type(default), default=default)
    return Config(**vars(ap.parse_args(argv)))


def main(argv=None):
    cfg = parse(argv)
    os.makedirs(cfg.run, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    def env_factory(battles, seed=cfg.seed):
        return ataraxos_pi.SelfPlay(battles, seed, teams=cfg.teams, max_turns=cfg.max_turns, threads=cfg.threads)

    learner = Learner(cfg)
    print(f"{sum(p.numel() for p in learner.net.parameters()) / 1e3:.1f}k params on {cfg.device}", flush=True)
    ck = None
    if cfg.resume:
        path = latest_checkpoint(cfg.run) if cfg.resume == "latest" else cfg.resume
        ck = torch.load(path, map_location=learner.dev, weights_only=False)
        learner.load(ck)
        if ck["rng"] is not None:
            rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        print(f"resumed from {path} at iteration {learner.it}", flush=True)
    with open(os.path.join(cfg.run, f"config_resume{learner.it}.json" if ck else "config.json"), "w") as f:
        json.dump({**dataclasses.asdict(cfg), "code": code_version()}, f, indent=1)
    env_seed = cfg.seed if ck is None else int(np.random.SeedSequence([cfg.seed, learner.it]).generate_state(1)[0])
    env = env_factory(cfg.battles, env_seed)
    k = len(env.team_probs)
    exp3 = teams.Exp3(k, cfg.team_gamma, cfg.team_eta, cfg.team_gamma_floor, cfg.team_gamma_tau,
                      cfg.team_gamma_decay) if cfg.teams else None
    counts_before = np.zeros(k, np.int64)
    pairs = teams.Pairs() if exp3 is not None else None
    if ck is not None and exp3 is not None:
        exp3.load(ck["exp3"])
        counts_before = np.asarray(ck["team_counts"], np.int64)
        if ck.get("pairs") is not None:
            pairs.load(ck["pairs"])
    if exp3 is not None:
        env.set_team_weights(exp3.probs())

    def ended(obs):
        if pairs is not None:
            pairs.add(obs["end_team"], obs["end_result"], obs["end_capped"])
        done = (obs["end_capped"] == 0) | (cfg.at_cap == "draw")
        if exp3 is not None and done.any():
            exp3.update(obs["end_team"][done], obs["end_prob"][done], obs["end_result"][done])
            env.set_team_weights(exp3.probs())

    actor = Actor(learner.net, cfg.device, cfg.amp)
    buffer = Buffer(cfg.at_cap)
    obs = env.reset()
    ended(obs)
    log = open(os.path.join(cfg.run, "log.jsonl"), "a")
    if ck is not None:
        log.write(json.dumps({"resume": learner.it}) + "\n")
    anchors = {a: make_agent(a, cfg.device, cfg.seed, cfg.amp) for a in cfg.anchors if "{it}" not in a}
    start_gpu_log(cfg)
    env_before = env.stats

    while learner.it < cfg.iters:
        t0, cpu0 = time.perf_counter(), cpu_seconds()
        peak_gb(learner.dev)
        rows = decisions = 0
        net_s = env_s = 0.0
        learner.net.eval()
        while buffer.games() < cfg.games_per_iter:
            ta = time.perf_counter()
            picks, logp, value = actor.act(obs)
            tb = time.perf_counter()
            buffer.add(obs, picks, logp, value)
            rows += len(picks)
            decisions += int((obs["n"] >= 2).sum())
            tc = time.perf_counter()
            obs = env.step(picks)
            env_s += time.perf_counter() - tc
            net_s += tb - ta
            buffer.end(obs)
            ended(obs)
        t1 = time.perf_counter()
        act_gb = peak_gb(learner.dev)
        data = buffer.take()
        stats = learner.train(data, rng)
        t2 = time.perf_counter()
        train_gb = peak_gb(learner.dev)
        env_now = env.stats
        d_env = {key: env_now[key] - env_before[key] for key in env_now}
        env_before = env_now
        r = data.reward
        stats.update(it=learner.it, games=len(data.length) // 2, act_rows=rows, act_decisions=decisions,
                     rows_per_s=rows / (t1 - t0), act_s=t1 - t0, train_s=t2 - t1, act_net_s=net_s, act_env_s=env_s,
                     act_other_s=t1 - t0 - net_s - env_s, mean_length=float(data.length.mean()),
                     ties=float((r == 0).mean()), capped_rate=d_env["capped"] / max(1, d_env["games"]),
                     forced_frac=d_env["forced"] / max(1, d_env["forced"] + rows))
        if learner.dev.type == "cuda":
            stats.update(act_gb=act_gb, train_gb=train_gb)
        team_counts = counts_before + np.asarray(env.team_counts, np.int64) if exp3 is not None else None
        if exp3 is not None:
            stats.update(teams.summary(exp3, team_counts))
        t3 = time.perf_counter()
        if learner.it % cfg.eval_every == 0:
            opps = dict(anchors)
            for a in cfg.anchors:
                if "{it}" in a and os.path.exists(a.format(it=learner.it)):
                    opps[a] = make_agent(a.format(it=learner.it), cfg.device, cfg.seed, cfg.amp)
            for prefix, net in (("eval", learner.ema), ("eval_net", learner.net)):
                net.eval()
                ours = NetAgent(net, cfg.device, cfg.amp)
                for name, opp in opps.items():
                    w, t, l = match(lambda b: env_factory(b, cfg.seed + 10**6 + learner.it), ours, opp,
                                    cfg.eval_battles)
                    score = (w + t / 2) / max(1, w + t + l)
                    key = f"{prefix}/{os.path.basename(name)}"
                    stats[key] = score
                    stats[key + "_wtl"] = [w, t, l]
                    stats[key + "_ci95"] = list(metrics.wilson(score, w + t + l))
        t4 = time.perf_counter()
        if learner.it % cfg.save_every == 0:
            learner.save(os.path.join(cfg.run, f"ckpt{learner.it}.pt"), exp3, rng, team_counts, pairs)
            if exp3 is not None:
                np.savez(os.path.join(cfg.run, f"teams{learner.it}.npz"), log_w=exp3.log_w, p=exp3.probs(),
                         q_avg=exp3.q_avg(), counts=team_counts, names=np.array(env.team_names), **pairs.arrays())
        t5 = time.perf_counter()
        stats.update(eval_s=t4 - t3, save_s=t5 - t4, iter_s=t5 - t0, time=time.time(),
                     games_per_s=stats["games"] / (t5 - t0), cpu_cores=(cpu_seconds() - cpu0) / (t5 - t0),
                     rss_peak_gb=rss_peak_gb())
        log.write(json.dumps(stats) + "\n")
        log.flush()
        keys = ("it", "games", "games_per_s", "rows_per_s", "act_s", "train_s", "iter_s", "act_gb", "train_gb",
                "loss", "value_loss", "value_ev", "entropy", "entropy_norm", "kl_data", "clip_frac", "grad_clipped",
                "kept", "mean_length", "ties", "teams/gamma", "teams/p_eff", "teams/nash_eff")
        print(" ".join(f"{k}={stats[k]:.4g}" if isinstance(stats.get(k), float) else f"{k}={stats.get(k)}"
                       for k in keys if k in stats) + "".join(f" {k}={v:.3f}" for k, v in stats.items()
                                                              if k.startswith("eval") and isinstance(v, float)),
              flush=True)


if __name__ == "__main__":
    main()
