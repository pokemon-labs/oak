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
pi4 (2026-11-24, docs/pi4.md): `--size pi4` (ataraxos/pi4/net.py) trains on
rby4's env as rows (ataraxos/pi4/env.py: Showdown's choices, perfect-info
state from both sides, both requests), with `--expected_sarsa` (rby4's Q
head: each player's W/D/L per choice, targets the next decision's Q under the
data policy, ataraxos/rl/buffer.py's sarsa_targets; `--sarsa_adv` of its
advantage mixed into GAE's), and evals in rby4's env against random,
maxdamage, rby4, pi1-3 or pi4 checkpoints (ataraxos/pi4/agents.py). Needs
--buffer numpy; no search in training.
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
import concurrent.futures
import contextlib
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
from ataraxos.pi.fast import AsyncActor, Envs, NumpyBuffer, pi_logits, sample_q
from ataraxos.pi.net import CHOICES, make_net
from ataraxos.pi.search import TrainSearch
from ataraxos.pi4 import net as pi4net
from ataraxos.rl import metrics, teams
from ataraxos.rl.buffer import returns_and_advantages, sarsa_targets
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
    torch_threads: int = 4  # torch's CPU threads (0: torch's default, the host's CPUs: on a box with a CPU quota
                            # under its core count that oversubscribes, ~300x slower CPU copies on a Vast 3060 box)
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
    # speed options (ataraxos/pi/fast.py, docs/pi_speed.md); each default is the original code path
    buffer: str = "python"  # "numpy": trajectories found in take, no Python loop per row
    actor: str = "eager"    # "async": pinned buffers, non-blocking copies; "graphs": async + one CUDA graph
    actor_compile: bool = False  # torch.compile the actor's forward (with --actor async|graphs; in the graph)
    pingpong: bool = False  # two half-size envs, one steps while the other's forward runs (actor async|graphs)
    fused_adam: bool = False  # AdamW(fused=True) on CUDA
    tf32: bool = False      # TF32 matmuls and convolutions
    gae: str = "cpu"        # "gpu": returns, advantages and the filter's quantile on the device
    compile_train: str = ""  # torch.compile the loss (forward + backward): default, reduce-overhead, max-autotune
    overlap: bool = False   # learn on iteration t in a thread while collecting t+1 with t's weights (one stale)
    eval_async: bool = False  # evals in a thread on a snapshot of the nets; results as their own log line
    eval_threads: int = 0   # the eval env's threads (0: --threads)
    # search in training (ataraxos/pi/search.py; off at search_p 0, the original path); needs --buffer numpy
    search_p: float = 0.0   # the share of rows (either side 2+ choices) searched, at the start
    search_p_end: float = -1.0  # annealed linearly to this over search_anneal decisions (-1: search_p throughout)
    search_anneal: float = 1e9
    search_budget: int = 64  # depth-1 rollouts a root (search_gain.py: depth 1 budget 64)
    search_beta: float = 0.03  # MMD's lambda, the prior as the magnet (alpha 0; search_gain.py: 0.02-0.05)
    search_floor: float = 0.25  # the rollouts' share of uniform (P1, P2) pairs
    search_shrink: float = 1.0  # cell means' pseudo-counts toward the row / column means
    search_iters: int = 2000
    search_tol: float = 1e-4
    search_coef: float = 1.0  # cross-entropy to the posteriors (searched decisions; out of the PPO loss)
    search_value_mix: float = 0.5  # searched decisions' value target: (1 - mix) TD(lambda) + mix the search's W/D/L
    # search speed options (ataraxos/pi/search.py), each default the original path
    search_leaf: str = "eager"  # the leaves' forward: "async" pinned slots, non-blocking; "graphs" + CUDA graphs
    search_compile: bool = False  # torch.compile the leaves' forward (dynamic shapes)
    search_value_only: bool = False  # the value head alone for the leaves (PiNet.value)
    search_direct: bool = False  # rollouts write the leaves into the pinned slot (search_leaf async|graphs)
    search_async: bool = False  # one env's leaf forward runs while the other steps (needs --pingpong)
    take_gather: bool = False  # NumpyBuffer.take gathers the needed rows from the parts (no full concatenation)
    stats_every: int = 1    # metrics.data_stats (the value's calibration, EV, ...) every N iterations (1: every one)
    search_plan: str = "numpy"  # "cpp": the rollouts' cells drawn in C++
    search_solve: str = "numpy"  # "cpp": cell means, MMD and the searched W/D/L in C++ per root
    # Expected SARSA (rby4's, ataraxos/rl/train.py; pi4's nets have the Q heads)
    expected_sarsa: bool = False
    q_coef: float = 1.0
    sarsa_adv: float = 0.0  # rby4's 0: Q trains, GAE's advantage alone


ROW_COLUMNS = ("mon", "hp", "active", "policy", "n", "choices")
PI4_COLUMNS = ("state", "request", "choices", "n")


def pi4_logits(net, x, with_q=False):
    """pi4's forward on a dict of inputs: (logits [R, 2, CHOICES], -1e9 past
    n; value logits [R, 3]; Q logits [R, 2, CHOICES, 3] with `with_q` else
    None)."""
    out = net(x["state"], x["request"], x["choices"], x["n"], with_q=with_q)
    return out[1], out[0], (out[2] if with_q else None)


class Model:
    """The net family a size names: pi's (ataraxos/pi/net.py, its env
    ataraxos_pi.SelfPlay) or pi4's (ataraxos/pi4/, rby4's env as rows)."""

    def __init__(self, size, q=False):
        self.pi4 = pi4net.is_pi4(size)
        self.size = size
        self.q = q
        if q and not self.pi4:
            raise ValueError("--expected_sarsa needs a pi4 size")
        self.columns = PI4_COLUMNS if self.pi4 else ROW_COLUMNS
        self.inputs = PI4_COLUMNS if self.pi4 else ("mon", "hp", "active", "policy", "n")

    def make_net(self):
        return pi4net.make_net(self.size) if self.pi4 else make_net(self.size)

    def logits(self, net, x, with_q=False):
        return pi4_logits(net, x, with_q) if self.pi4 else pi_logits(net, x)

    def actor_logits(self, net, x):
        return self.logits(net, x, self.q)


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
    q_exp: np.ndarray = None     # [N, 3] with --expected_sarsa: the actor's Q (the player's W/D/L) under the data policy
    q_pick: np.ndarray = None    # [N, 3] and at its pick
    searched: np.ndarray = None  # [N] bool, with --search_p (ataraxos/pi/search.py)
    post: np.ndarray = None      # [N, CHOICES] the search's posterior (probabilities)
    swdl: np.ndarray = None      # [N, 3] the W/D/L under the posteriors, the deciding player's


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
    the data policy's log-probs [R, 2, CHOICES], the value logits [R, 3],
    and with the model's Q the Q columns (q_exp, q_pick [R, 2, 3]) else
    None."""

    def __init__(self, net, device, amp, model=None):
        self.net, self.dev, self.amp = net, torch.device(device), amp
        self.model = model or Model("pi")

    @torch.no_grad()
    def act(self, obs):
        d = self.dev
        x = {k: torch.from_numpy(obs[k]).to(d) for k in self.model.inputs}
        with torch.autocast(d.type, torch.bfloat16, enabled=self.amp and d.type == "cuda"):
            logits, value, q = self.model.actor_logits(self.net, x)
        logp = torch.log_softmax(logits.float(), -1)
        pick = torch.multinomial(logp.exp().view(-1, CHOICES), 1).view(-1, 2)
        extra = None
        if q is not None:
            extra = {k: v.cpu().numpy() for k, v in zip(("q_exp", "q_pick"), sample_q(logp, pick, q))}
        return (pick.to(torch.uint8).cpu().numpy(), logp.cpu().numpy().astype(np.float32),
                value.float().cpu().numpy(), extra)


class Learner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.model = Model(cfg.size, cfg.expected_sarsa)
        self.net = self.model.make_net().to(self.dev)
        self.ema = self.model.make_net().to(self.dev).eval()
        self.ema.load_state_dict(self.net.state_dict())
        for p in self.ema.parameters():
            p.requires_grad_(False)
        fused = cfg.fused_adam and self.dev.type == "cuda"
        self.optim = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr_ceil, weight_decay=cfg.weight_decay,
                                       **({"fused": True} if fused else {}))
        self.it = 0
        self.decisions = 0
        self.loss_fn = self.loss if not cfg.compile_train else torch.compile(
            self.loss, mode=None if cfg.compile_train == "default" else cfg.compile_train)

    def schedule_step(self):
        c = self.cfg
        return self.decisions / c.schedule_decisions if c.schedule_decisions > 0 else self.it

    def train(self, data: Data, rng):
        cfg, dev = self.cfg, self.dev
        t0 = time.perf_counter()
        gd = dev if cfg.gae == "gpu" else torch.device("cpu")
        ret, adv = returns_and_advantages(data, cfg.td_lambda, cfg.gae_lambda, gd)
        q_target, q_stats = None, {}
        if cfg.expected_sarsa:
            q_target, q_adv = sarsa_targets(data)
            q_adv = q_adv.to(adv.device)
            q_stats = dict(adv_gae_std=float(adv.std()), adv_q_std=float(q_adv.std()),
                           adv_q_gae_corr=float(torch.corrcoef(torch.stack([adv.float(), q_adv]))[0, 1]))
            if cfg.sarsa_adv:
                adv = (1 - cfg.sarsa_adv) * adv + cfg.sarsa_adv * q_adv
        threshold = max(float(adv.abs().quantile(cfg.adv_filt_rate)) if len(adv) < 2**24 else
                        float(adv.abs()[torch.randperm(len(adv), device=gd)[:2**24]].quantile(cfg.adv_filt_rate)),
                        cfg.adv_filt_thresh)
        keep = (adv.abs() >= threshold).cpu().numpy()
        searched = data.searched
        if searched is not None:
            keep |= searched  # their cross-entropy targets don't depend on the advantage
            if cfg.search_value_mix > 0:
                m = cfg.search_value_mix
                s = torch.as_tensor(searched, device=ret.device)[:, None]
                ret = torch.where(s, (1 - m) * ret + m * torch.as_tensor(data.swdl, device=ret.device), ret)
        step = self.schedule_step()
        lr = power_schedule(cfg.lr_coef, step, cfg.lr_decay, cfg.lr_ceil, cfg.lr_floor)
        temperature = power_schedule(cfg.temperature_coef, step, cfg.temperature_decay, cfg.temperature_ceil,
                                     cfg.temperature_floor)
        for g in self.optim.param_groups:
            g["lr"] = lr
        stats = dict(lr=lr, temperature=temperature, schedule_step=step, adv_threshold=threshold,
                     decisions=len(adv), kept=int(keep.sum()), **q_stats)
        if searched is not None:
            stats.update(searched_frac=float(searched.mean()))
        ta = time.perf_counter()
        if cfg.stats_every <= 1 or self.it % cfg.stats_every == 0:
            stats.update(metrics.data_stats(data, ret.cpu(), adv.cpu()))
        tb = time.perf_counter()
        # everything on the device once
        T = lambda a, dt=None: torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=dt)
        rows = {k: T(v) for k, v in data.rows.items()}
        dd = dict(row=T(data.row), side=T(data.side), n=T(data.n), choices=T(data.choices),
                  logp=T(data.logp, torch.float), pick=T(data.pick, torch.long), ret=ret.to(dev), adv=adv.to(dev))
        if searched is not None:
            dd.update(searched=T(searched), post=T(data.post, torch.float))
        if q_target is not None:
            dd.update(q_target=q_target.to(dev))
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
                loss, out = self.loss_fn(rows, dd, torch.from_numpy(m).to(dev), temperature)
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
                     train_prep_s=t1 - t0, train_gae_s=ta - t0, train_stats_s=tb - ta, train_upload_s=t1 - tb,
                     train_issue_s=t2 - t1, train_wait_s=time.perf_counter() - t2)
        return stats

    def loss(self, rows, dd, idx, temperature):
        cfg, dev = self.cfg, self.dev
        r, side = dd["row"][idx], dd["side"][idx]
        K = len(idx)
        k = torch.arange(K, device=dev)
        q = None
        with torch.autocast(dev.type, torch.bfloat16, enabled=cfg.amp and dev.type == "cuda"):
            if self.model.pi4:
                both, value, q = pi4_logits(self.net, {c: rows[c][r] for c in PI4_COLUMNS}, "q_target" in dd)
                side_logits = both[k, side]
            else:
                value, policy = self.net(rows["mon"][r].long(), rows["hp"][r], rows["active"][r])
                side_logits = policy[k, side].gather(-1, rows["policy"][r, side].long())
        value = value.float()
        value = torch.where(side[:, None] == 1, value.flip(-1), value)  # the deciding player's W/D/L
        n = dd["n"][idx]
        legal = torch.arange(CHOICES, device=dev)[None] < n[:, None]
        logits = side_logits.float().masked_fill(~legal, -1e9)
        logp = torch.log_softmax(logits, -1)
        p = logp.exp() * legal
        old = dd["logp"][idx].clamp(min=-1e9).masked_fill(~legal, -1e9)
        a, A, R = dd["pick"][idx][:, None], dd["adv"][idx], dd["ret"][idx]
        ratio = torch.exp(logp.gather(1, a)[:, 0] - old.gather(1, a)[:, 0])
        clipped = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
        searched = dd["searched"][idx] if "searched" in dd else None
        pg = torch.min(A * ratio, A * clipped)
        policy_loss = -(pg if searched is None else pg * (~searched).float()).mean()
        kl_data = (p * (logp - old)).sum(-1).mean()
        kl_magnet = (p * (logp - magnet(dd["choices"][idx], legal, cfg.magnet))).sum(-1).mean()
        value_loss = -(R * torch.log_softmax(value, -1)).sum(-1).mean()
        loss = cfg.policy_coef * policy_loss + temperature * kl_magnet + cfg.vf_coef * value_loss + cfg.kl_coef * kl_data
        extra = {}
        if q is not None:  # Expected SARSA: Q at the pick against the next decision's sum
            qa = q[k, side].float().gather(1, a[..., None].expand(-1, -1, 3))[:, 0]
            extra["q_loss"] = -(dd["q_target"][idx] * torch.log_softmax(qa, -1)).sum(-1).mean()
            loss = loss + cfg.q_coef * extra["q_loss"]
        if searched is not None:  # cross-entropy to the posterior, over the searched decisions, as a batch mean
            ce = -(dd["post"][idx] * logp.masked_fill(~legal, 0)).sum(-1)
            search_ce = (ce * searched.float()).mean()
            loss = loss + cfg.search_coef * search_ce
            extra = dict(search_ce=search_ce, search_ce_per=(ce * searched.float()).sum() / searched.sum().clamp(min=1))
        ent = -(p * logp).sum(-1)
        y = logits.detach().masked_fill(~legal, 0)
        y = (y - y.sum(-1, keepdim=True) / n[:, None].float()).masked_fill(~legal, 0)
        return loss, {k: v.detach() for k, v in dict(
            loss=loss, policy_loss=policy_loss, value_loss=value_loss, kl_data=kl_data, kl_magnet=kl_magnet,
            entropy=ent.mean(), entropy_norm_learner=(ent / torch.log(n.float())).mean(),
            clip_frac=((ratio - 1).abs() > cfg.clip_range).float().mean(), ratio_mean=ratio.mean(),
            ratio_max=ratio.max(), logit_abs_max=y.abs().max(), **extra).items()}

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
    if cfg.buffer not in ("python", "numpy"):
        raise ValueError(f"--buffer {cfg.buffer!r}: python or numpy")
    if cfg.actor not in ("eager", "async", "graphs"):
        raise ValueError(f"--actor {cfg.actor!r}: eager, async or graphs")
    if cfg.search_p > 0 and cfg.buffer != "numpy":
        raise ValueError("--search_p needs --buffer numpy")
    if cfg.search_leaf not in ("eager", "async", "graphs"):
        raise ValueError(f"--search_leaf {cfg.search_leaf!r}: eager, async or graphs")
    if cfg.search_direct and cfg.search_leaf == "eager":
        raise ValueError("--search_direct needs --search_leaf async or graphs")
    if cfg.search_async and not cfg.pingpong:
        raise ValueError("--search_async needs --pingpong")
    if cfg.actor_compile and cfg.actor == "eager":
        raise ValueError("--actor_compile needs --actor async or graphs")
    if cfg.pingpong and cfg.actor == "eager":
        raise ValueError("--pingpong needs --actor async or graphs")
    if cfg.actor == "graphs" and not cfg.device.startswith("cuda"):
        raise ValueError("--actor graphs needs CUDA")
    model = Model(cfg.size, cfg.expected_sarsa)
    if model.pi4 and (cfg.buffer != "numpy" or cfg.search_p > 0):
        raise ValueError("pi4 needs --buffer numpy and no search")
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
    if cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(cfg.run, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    def env_factory(battles, seed=cfg.seed):
        if model.pi4:
            from ataraxos.pi4.env import Pi4Env
            return Pi4Env(battles, seed, teams=cfg.teams, max_turns=cfg.max_turns, threads=cfg.threads)
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
    sizes = [cfg.battles // 2, cfg.battles - cfg.battles // 2] if cfg.pingpong else [cfg.battles]
    envs = [env_factory(s, env_seed + 7919 * i) for i, s in enumerate(sizes)]
    env = Envs(envs, sizes) if cfg.pingpong else envs[0]
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

    # --overlap: the actor plays with its own copy of the net, refreshed after each learner step, while the
    # learner runs in a thread on its own CUDA stream (the env releases the GIL while it steps)
    actor_net = learner.net
    if cfg.overlap:
        actor_net = model.make_net().to(learner.dev).eval()
        actor_net.load_state_dict(learner.net.state_dict())
        for p in actor_net.parameters():
            p.requires_grad_(False)
    if cfg.actor == "eager":
        actor = Actor(actor_net, cfg.device, cfg.amp, model)
    else:
        actor = AsyncActor(actor_net, cfg.device, cfg.amp, capacity=max(sizes), slots=len(sizes),
                           graphs=cfg.actor == "graphs", compile=cfg.actor_compile, inputs=model.inputs,
                           logits=model.actor_logits, q=model.q)
    buffer = NumpyBuffer(cfg.at_cap, cfg.take_gather, model.columns) if cfg.buffer == "numpy" else Buffer(cfg.at_cap)
    searcher = None
    if cfg.search_p > 0:
        searcher = TrainSearch(actor_net, cfg.device, cfg.amp, budget=cfg.search_budget, beta=cfg.search_beta,
                               floor=cfg.search_floor, shrink=cfg.search_shrink, iters=cfg.search_iters,
                               tol=cfg.search_tol, threads=cfg.threads, seed=cfg.seed + 7 * learner.it,
                               compile=cfg.search_compile, leaf=cfg.search_leaf, value_only=cfg.search_value_only,
                               direct=cfg.search_direct, slots=len(envs), plan=cfg.search_plan,
                               solve=cfg.search_solve)
    offsets = env.offsets if cfg.pingpong else [0]

    def search_p():
        end = cfg.search_p if cfg.search_p_end < 0 else cfg.search_p_end
        return cfg.search_p + (end - cfg.search_p) * min(1.0, learner.decisions / max(cfg.search_anneal, 1.0))

    def merge(*cols):
        """The buffer's extra columns from the actor (Q) and the search, either None."""
        out = {}
        for c in cols:
            if c:
                out.update(c)
        return out or None

    def search_stats(st, sums):
        for k, v in st.items():
            sums[k] = sums.get(k, 0.0) + v
        sums["search_steps"] = sums.get("search_steps", 0) + (st["search_roots"] > 0)

    def search(e, i, o, picks, logp, sums):
        """Search in training on obs o of env e (index i): overwrites picks, returns the buffer's columns."""
        if searcher is None:
            return None
        cols, st = searcher.run(e, o, picks, logp, sums.setdefault("search_p", search_p()), int(offsets[i]), i)
        search_stats(st, sums)
        return cols
    obs_list = [env.shift(i, e.reset()) if cfg.pingpong else e.reset() for i, e in enumerate(envs)]
    for o in obs_list:
        ended(o)
    state = dict(obs=obs_list[0], env_before=env.stats)
    # async: each env's next forward is queued before the loop and after each step, and collected at its turn
    handles = [actor.launch(o, i) for i, o in enumerate(obs_list)] if cfg.actor != "eager" else None
    log = open(os.path.join(cfg.run, "log.jsonl"), "a")
    if ck is not None:
        log.write(json.dumps({"resume": learner.it}) + "\n")
    if model.pi4:  # evals in rby4's env (ataraxos/pi4/agents.py)
        from ataraxos.pi4 import agents as pi4agents
        agent_of, match_fn = pi4agents.make_agent, pi4agents.match
        ours_of = lambda net: pi4agents.Pi4Agent(net, cfg.device, cfg.amp)
        eval_env = lambda it: pi4agents.eval_env(cfg.seed + 10**6 + it, cfg.teams, cfg.max_turns,
                                                 cfg.eval_threads or cfg.threads)
    else:
        agent_of, match_fn = make_agent, match
        ours_of = lambda net: NetAgent(net, cfg.device, cfg.amp)
        eval_env = lambda it: (lambda b: ataraxos_pi.SelfPlay(b, cfg.seed + 10**6 + it, teams=cfg.teams,
                                                              max_turns=cfg.max_turns,
                                                              threads=cfg.eval_threads or cfg.threads))
    anchors = {a: agent_of(a, cfg.device, cfg.seed, cfg.amp) for a in cfg.anchors if "{it}" not in a}
    start_gpu_log(cfg)
    cuda = learner.dev.type == "cuda"

    def collect():
        """Plays until the buffer holds games_per_iter games; returns the act timings."""
        t0 = time.perf_counter()
        peak_gb(learner.dev)
        rows = decisions = 0
        net_s = env_s = 0.0
        ss = {}  # search stats summed over steps
        if not cfg.overlap:
            learner.net.eval()
        obs = state["obs"]
        while handles is None and buffer.games() < cfg.games_per_iter:
            ta = time.perf_counter()
            picks, logp, value, qx = actor.act(obs)
            tb = time.perf_counter()
            buffer.add(obs, picks, logp, value, merge(qx, search(envs[0], 0, obs, picks, logp, ss)))
            rows += len(picks)
            decisions += int((obs["n"] >= 2).sum())
            tc = time.perf_counter()
            obs = env.step(picks)
            env_s += time.perf_counter() - tc
            net_s += tb - ta
            buffer.end(obs)
            ended(obs)
        state["obs"] = obs
        # async / pingpong: env i's forward runs on the GPU while the other env steps; act_net_s is the
        # time spent waiting for a forward. Forwards queued at an iteration's end ran before the
        # learner's kernels (one stream; with --overlap, before the actor net's refresh, on the actor's
        # stream), so their log-probs are the policy that picked.
        # --search_async: env i's rows are searched up to the leaves' forward (begin), then the other env's
        # pending search is finished and that env stepped, so a leaf forward runs while the CPU steps
        acc = dict(rows=0, decisions=0, net_s=0.0, env_s=0.0)

        def finish_step(i, picks, logp, value, tok):
            cols, st = searcher.finish(tok)
            search_stats(st, ss)
            o = obs_list[i]
            buffer.add(o, picks, logp, value, cols)
            acc["rows"] += len(picks)
            acc["decisions"] += int((o["n"] >= 2).sum())
            tc = time.perf_counter()
            o = envs[i].step(picks)
            acc["env_s"] += time.perf_counter() - tc
            o = env.shift(i, o)
            obs_list[i] = o
            buffer.end(o)
            ended(o)
            handles[i] = actor.launch(o, i)

        pend, i = None, 0
        while handles is not None and searcher is not None and cfg.search_async:
            cur = None
            if buffer.games() < cfg.games_per_iter:
                ta = time.perf_counter()
                picks, logp, value, _ = actor.collect(handles[i])
                acc["net_s"] += time.perf_counter() - ta
                tok = searcher.begin(envs[i], obs_list[i], picks, logp, ss.setdefault("search_p", search_p()),
                                     int(offsets[i]), i)
                cur = (i, picks, logp, value, tok)
            if pend is not None:
                finish_step(*pend)
            if cur is None:
                break
            pend, i = cur, (i + 1) % len(envs)
        rows += acc["rows"]
        decisions += acc["decisions"]
        net_s += acc["net_s"]
        env_s += acc["env_s"]
        while handles is not None and not (searcher is not None and cfg.search_async) and \
                buffer.games() < cfg.games_per_iter:
            for i, e in enumerate(envs):
                ta = time.perf_counter()
                picks, logp, value, qx = actor.collect(handles[i])
                tb = time.perf_counter()
                o = obs_list[i]
                buffer.add(o, picks, logp, value, merge(qx, search(e, i, o, picks, logp, ss)))
                rows += len(picks)
                decisions += int((o["n"] >= 2).sum())
                tc = time.perf_counter()
                o = e.step(picks)
                env_s += time.perf_counter() - tc
                net_s += tb - ta
                o = env.shift(i, o) if cfg.pingpong else o
                obs_list[i] = o
                buffer.end(o)
                ended(o)
                handles[i] = actor.launch(o, i)
        t1 = time.perf_counter()
        n_steps = max(1, ss.pop("search_steps", 1))
        s_summary = {k: v if k in ("search_roots", "search_leaves", "search_p") or k.endswith("_s") else v / n_steps
                        for k, v in ss.items()}
        return dict(**s_summary, act_rows=rows, act_decisions=decisions, rows_per_s=rows / (t1 - t0), act_s=t1 - t0,
                    act_net_s=net_s, act_env_s=env_s, act_other_s=t1 - t0 - net_s - env_s,
                    **({"act_gb": peak_gb(learner.dev)} if cuda else {}))

    def run_evals(it, nets):
        """`nets`: (prefix, net) pairs. Returns the eval stats."""
        out = {}
        opps = dict(anchors)
        for a in cfg.anchors:
            if "{it}" in a and os.path.exists(a.format(it=it)):
                opps[a] = agent_of(a.format(it=it), cfg.device, cfg.seed, cfg.amp)
        for prefix, net in nets:
            net.eval()
            ours = ours_of(net)
            for name, opp in opps.items():
                w, t, l = match_fn(eval_env(it), ours, opp, cfg.eval_battles)
                score = (w + t / 2) / max(1, w + t + l)
                key = f"{prefix}/{os.path.basename(name)}"
                out[key] = score
                out[key + "_wtl"] = [w, t, l]
                out[key + "_ci95"] = list(metrics.wilson(score, w + t + l))
        return out

    def print_evals(stats):
        return "".join(f" {k}={v:.3f}" for k, v in stats.items() if k.startswith("eval") and isinstance(v, float))

    # --eval_async: one eval in flight, on copies of the nets, on its own stream
    eval_pool = concurrent.futures.ThreadPoolExecutor(1) if cfg.eval_async else None
    eval_job = [None]

    def eval_done(wait):
        job = eval_job[0]
        if job is None or not (wait or job[1].done()):
            return
        it, fut, t = job
        out = fut.result()
        eval_job[0] = None
        log.write(json.dumps(dict(eval_it=it, eval_s=time.perf_counter() - t, time=time.time(), **out)) + "\n")
        log.flush()
        print(f"eval_it={it}" + print_evals(out), flush=True)

    def eval_start(it):
        eval_done(wait=True)
        nets = []
        for prefix, net in (("eval", learner.ema), ("eval_net", learner.net)):
            c = model.make_net().to(learner.dev).eval()
            c.load_state_dict(net.state_dict())
            nets.append((prefix, c))
        stream = torch.cuda.Stream() if cuda else None
        if cuda:
            stream.wait_stream(torch.cuda.current_stream())

        def job():
            with torch.cuda.stream(stream) if cuda else contextlib.nullcontext():
                out = run_evals(it, nets)
            if cuda:
                stream.synchronize()
            return out

        eval_job[0] = (it, eval_pool.submit(job), time.perf_counter())

    def finish(stats, data, act, t_start, cpu0):
        """The iteration's log line, evals and checkpoint, after its learner step."""
        env_now = env.stats
        d_env = {key: env_now[key] - state["env_before"][key] for key in env_now}
        state["env_before"] = env_now
        r = data.reward
        stats.update(act)
        stats.update(it=learner.it, games=len(data.length) // 2, mean_length=float(data.length.mean()),
                     ties=float((r == 0).mean()), capped_rate=d_env["capped"] / max(1, d_env["games"]),
                     forced_frac=d_env["forced"] / max(1, d_env["forced"] + act["act_rows"]))
        team_counts = counts_before + np.asarray(env.team_counts, np.int64) if exp3 is not None else None
        if exp3 is not None:
            stats.update(teams.summary(exp3, team_counts))
        t3 = time.perf_counter()
        if eval_pool is not None:
            eval_done(wait=False)
        if learner.it % cfg.eval_every == 0:
            if eval_pool is not None:
                eval_start(learner.it)
            else:
                stats.update(run_evals(learner.it, (("eval", learner.ema), ("eval_net", learner.net))))
        t4 = time.perf_counter()
        if learner.it % cfg.save_every == 0:
            learner.save(os.path.join(cfg.run, f"ckpt{learner.it}.pt"), exp3, rng, team_counts, pairs)
            if exp3 is not None:
                np.savez(os.path.join(cfg.run, f"teams{learner.it}.npz"), log_w=exp3.log_w, p=exp3.probs(),
                         q_avg=exp3.q_avg(), counts=team_counts, names=np.array(env.team_names), **pairs.arrays())
        t5 = time.perf_counter()
        stats.update(eval_s=t4 - t3, save_s=t5 - t4, iter_s=t5 - t_start, time=time.time(),
                     games_per_s=stats["games"] / (t5 - t_start), cpu_cores=(cpu_seconds() - cpu0) / (t5 - t_start),
                     rss_peak_gb=rss_peak_gb())
        log.write(json.dumps(stats) + "\n")
        log.flush()
        keys = ("it", "games", "games_per_s", "rows_per_s", "act_s", "train_s", "iter_s", "act_gb", "train_gb",
                "loss", "value_loss", "value_ev", "entropy", "entropy_norm", "kl_data", "clip_frac", "grad_clipped",
                "kept", "mean_length", "ties", "teams/gamma", "teams/p_eff", "teams/nash_eff", "search_p",
                "search_roots", "search_kl", "search_ce_per", "searched_frac")
        print(" ".join(f"{k}={stats[k]:.4g}" if isinstance(stats.get(k), float) else f"{k}={stats.get(k)}"
                       for k in keys if k in stats) + print_evals(stats), flush=True)

    if not cfg.overlap:
        while learner.it < cfg.iters:
            t0, cpu0 = time.perf_counter(), cpu_seconds()
            act = collect()
            t1 = time.perf_counter()
            data = buffer.take()
            t_take = time.perf_counter()
            stats = learner.train(data, rng)
            stats.update(train_s=time.perf_counter() - t1, take_s=t_take - t1,
                         **({"train_gb": peak_gb(learner.dev)} if cuda else {}))
            finish(stats, data, act, t0, cpu0)
    else:
        # iteration t's learner step runs while t+1 is collected with the weights from before it, so the
        # data is one learner step stale (PPO's ratio uses the stored log-probs). A logged iteration's
        # iter_s / games_per_s is the loop's period: collecting the next iteration, waiting for this
        # one's learner step (overlap_wait_s), its evals and checkpoint.
        pool = concurrent.futures.ThreadPoolExecutor(1)
        learn_stream = torch.cuda.Stream() if cuda else None
        params = [p.detach() for p in learner.net.parameters()] + list(learner.net.buffers())
        actor_params = [p.detach() for p in actor_net.parameters()] + list(actor_net.buffers())

        def train_job(data):
            if cuda:
                learn_stream.wait_stream(torch.cuda.current_stream())  # the data's and the weights' last copies
            t = time.perf_counter()
            with torch.cuda.stream(learn_stream) if cuda else contextlib.nullcontext():
                stats = learner.train(data, rng)
            if cuda:
                learn_stream.synchronize()
            stats.update(train_s=time.perf_counter() - t)
            return stats

        submitted = learner.it
        pending = None
        while submitted < cfg.iters or pending is not None:
            t0, cpu0 = time.perf_counter(), cpu_seconds()
            nxt = None
            if submitted < cfg.iters:
                act = collect()
                tk = time.perf_counter()
                nxt = (buffer.take(), dict(act, take_s=time.perf_counter() - tk))
            if pending is not None:
                fut, data, act_prev = pending
                tw = time.perf_counter()
                stats = fut.result()
                stats.update(overlap_wait_s=time.perf_counter() - tw)
                if cuda:
                    torch.cuda.current_stream().wait_stream(learn_stream)
                with torch.no_grad():
                    torch._foreach_copy_(actor_params, params)
                finish(stats, data, act_prev, t0, cpu0)
                pending = None
            if nxt is not None:
                pending = (pool.submit(train_job, nxt[0]), nxt[0], nxt[1])
                submitted += 1
        pool.shutdown()
    if eval_pool is not None:
        eval_done(wait=True)
        eval_pool.shutdown()

if __name__ == "__main__":
    main()
