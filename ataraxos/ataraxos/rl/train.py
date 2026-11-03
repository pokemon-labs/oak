"""Ataraxos self-play training over ataraxos_env: birch's birch/rl/train.py for Gen 1.

    ~/oak/.venv/bin/python -m ataraxos.rl.train --run runs/x [--key value ...] TEAM_FILE ...

The recipe is ~/stratego's pyengine/core/rl.py (docs/ataraxos-context.md):
clipped PPO, a reverse KL to the data policy, an annealed reverse KL to a
magnet, terminal rewards, separate lambdas for returns and advantages, the
advantage filter, one epoch, an EMA of the weights. The differences:
  - the data's (ataraxos/rl/buffer.py): an iteration plays until `games_per_iter`
    games end, and a minibatch is whole trajectories, since a decision's
    input is its player's game so far;
  - the setup network is Exp3 over the team pool (ataraxos/rl/teams.py; the user's,
    2026-10-23), its draws counted as the Nash estimate;
  - a game ended by the turn cap (`max_turns`, 300 turns) goes by
    `at_cap` (the user's, 2026-10-26): "draw" (the default) is the tie the
    env reports, in the trajectories and Exp3; "drop" (the rule from
    2026-10-25 to 10-26, at 200) keeps no trajectory and no Exp3 round;
    "bootstrap" keeps the trajectories, each ending on its last decision's
    own value, and gives no Exp3 round;
  - the LR and magnet schedules count trained decisions, one step per
    `schedule_decisions` (5M, about one of the paper's iterations), where
    stratego's code counts iterations (Claude's, 2026-10-25), so a change of
    `games_per_iter` doesn't change how fast they anneal; 0 counts
    iterations as stratego does;
  - evaluation plays both the EMA and the current weights against each
    anchor (ataraxos/eval/agents.py: "random", "maxdamage" or a checkpoint),
    the anchor on the target's teams in the exploiter run;
  - with `target` (the exploiter run, 2026-10-30), the learner plays a
    frozen checkpoint instead of itself, P1 in the even battles and P2 in
    the odd ones, and trains on its own side only. Each side draws from its
    own pool through the env's two team distributions: the learner's is
    Exp3 over the team files, one round a game; the target's is fixed, its
    softmax over `target_teams` from `target_team_npz`'s log_w (uniform
    without one) mixed with `target_team_gamma` of uniform. `init` starts
    from a checkpoint's weights, EMA and optimizer, at iteration 0 with no
    trained decisions (the schedules start over; the user's, 2026-10-30). teams{it}.npz is the learner's
    Exp3, its pairs (learner team, target team) from the learner's side,
    and the target's draw counts;
  - the net is the small causal net (ataraxos/nets/causal_net.py, 2026-10-26; `size`
    cs, cs2 or cs_tiny): the env adds the bookkeeping state, the actor
    caches each layer's keys and values (ataraxos/nets/causal_actor.py, CUDA-graphed
    with `graphs`), and the value is W/D/L logits by cross-entropy against
    TD(lambda) over the distribution. The HistoryNet (run D) was removed
    2026-10-30.

Every Config field is a flag (`--gae_lambda 0.5`); a list takes several
comma-separated values (`--anchors random,runs/a/ckpt100.pt`), so the
team files can follow as positional arguments. An anchor with `{it}` in it
(`--anchors /root/anchors/rby1_{it}.pt`) is the checkpoint at the eval's
iteration, skipped where that file doesn't exist; its stats keep the `{it}`. The run directory gets
config.json (the flags, and under "code" the oak commit),
log.jsonl (one line per iteration), ckpt{it}.pt and, with a
pool, teams{it}.npz (Exp3's log-weights, p and q_avg, the draw counts, the names,
and every ordered team pair's wins, ties, losses and capped ties so far from
P1's side, as pair_a, pair_b, pair_wdlc; `ataraxos/eval/pairs.py` prints them).
On CUDA, gpu.csv samples nvidia-smi (`--gpu_log` seconds); log.jsonl's
`time` lines it up with the phases. The diagnostics are ataraxos/rl/metrics.py's.

`--resume latest` (the run's newest checkpoint) or `--resume PATH` carries on
from a checkpoint: weights, EMA, optimizer, iteration, Exp3, the draw counts
and the RNGs. The games in progress at the save are dropped (the user's,
2026-10-23), and the env restarts from a seed drawn from (seed, iteration).
The flags are this invocation's, so anchors or eval settings can change; the
net's size can't. Each resume writes config_resume{it}.json, and log.jsonl
gets a {"resume": it} line, after which a resumed iteration's line replaces
any earlier one with the same `it`.
"""

import argparse
import atexit
import dataclasses
import json
import math
import os
import re
import resource
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ataraxos._ext import ataraxos_env  # noqa: E402
from ataraxos.nets.causal_actor import make_actor  # noqa: E402
from ataraxos.nets.causal_net import FEATURE_PARAMS, CausalPolicy, DeviceData, learner_forward, make_net  # noqa: E402
from ataraxos.eval.agents import NetAgent, load_policy, make_agent, match, subset  # noqa: E402
from ataraxos.rl.buffer import Buffer, minibatches, returns_and_advantages, split  # noqa: E402
from ataraxos.nets.policy import CHOICES, SIZES  # noqa: E402
from ataraxos.rl import metrics  # noqa: E402
from ataraxos.rl import teams  # noqa: E402

@dataclass
class Config:
    run: str = "runs/default"
    teams: list = field(default_factory=list)  # team files; none: truly random teams
    size: str = "cs"
    # env
    battles: int = 1024
    threads: int = 8
    seed: int = 1
    # the turn cap, in turns, and what a capped game counts as: "draw",
    # "drop" or "bootstrap" (the user's, 2026-10-26: 300 and a draw; 200 and
    # dropped before)
    max_turns: int = 300
    at_cap: str = "draw"
    # Exp3 over the team pool (ataraxos/rl/teams.py): uniform exploration gamma_t =
    # max(floor, gamma (1 + rounds / tau)^-decay), rounds two per game, about
    # 0.04 at 16M rounds (a 24 h rental; the user's, 2026-10-25), tau 0 fixed;
    # the step size (0: gamma_t / pool size)
    team_gamma: float = 0.1
    team_gamma_floor: float = 0.01
    team_gamma_tau: float = 3e6
    team_gamma_decay: float = 0.5
    team_eta: float = 0.0
    # the exploiter run (see the top): the frozen opponent, a checkpoint or a
    # policy file (config and weights), and which of its weights; its pool,
    # its log-weights over that pool (a teams{it}.npz, or a checkpoint's Exp3)
    # and the uniform mixed in
    target: str = ""
    target_weights: str = "ema"
    target_teams: list = field(default_factory=list)
    target_team_npz: str = ""
    target_team_gamma: float = 0.05
    # start from this checkpoint (not a resume: iteration 0, Exp3 fresh)
    init: str = ""
    # iteration
    iters: int = 10**9
    # the user's, 2026-10-25: ~6k kept decisions a step, as the paper's 16
    # GPUs; ~50 steps an iteration (est.)
    games_per_iter: int = 16384
    minibatch: int = 6144  # kept decisions per gradient step
    # kept decisions per forward/backward pass, the gradients accumulated over a
    # minibatch's passes (the user's go-ahead, 2026-10-24); 0: one pass
    microbatch: int = 0
    # Ataraxos (stratego's RLConfig defaults)
    clip_range: float = 0.2
    td_lambda: float = 0.8
    gae_lambda: float = 0.5
    vf_coef: float = 1.0
    policy_coef: float = 1.0
    # "ppo": the clipped surrogate above; "neurd": sampled NeuRD (neurd_loss)
    # under PPO's clip, a logit threshold and a cap on 1 / mu (the user's,
    # 2026-10-31: unclipped, cs2c's logits ran to ~18,000 in 19 iterations;
    # clipped, to ~1,100 in 8; clipped with beta 2 and no cap, to ~800 in 7)
    policy_loss: str = "ppo"
    neurd_beta: float = 2.0
    neurd_mu_cap: float = 100.0
    kl_coef: float = 0.1
    temperature_coef: float = 0.05
    temperature_decay: float = 0.3
    temperature_ceil: float = 0.1
    temperature_floor: float = 0.005  # stratego's 0.001; Claude's, 2026-10-25
    lr_coef: float = 0.5
    lr_decay: float = 1.1
    lr_ceil: float = 1e-4
    lr_floor: float = 5e-6
    # trained decisions per schedule step (0: one per iteration, as stratego)
    schedule_decisions: float = 5e6
    weight_decay: float = 0.0
    max_grad_norm: float = 0.267
    adv_filt_rate: float = 0.75
    adv_filt_thresh: float = 0.01
    # per iteration: stratego's 0.999 over 42k iterations; 0.99 averages the
    # last ~100 of a run of hundreds (the user's, 2026-10-25)
    ema_decay: float = 0.99
    # "attack_switch": half on moves, half on switches, uniform within (the
    # default, 2026-10-23); "uniform": uniform over the legal choices
    magnet: str = "attack_switch"
    # evaluation (ataraxos/eval/agents.py): "random", "maxdamage" or checkpoint paths;
    # every ~15 min and saves every ~30 min at ~125 s an iteration (est.)
    anchors: list = field(default_factory=lambda: ["random", "maxdamage"])
    eval_every: int = 7
    eval_battles: int = 512
    save_every: int = 15
    resume: str = ""  # "latest" or a checkpoint path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile: bool = False  # torch.compile the net's summarize, history and decision block (compile_net)
    amp: bool = True  # bf16 autocast on CUDA, actor and learner
    # the causal net's actor as CUDA graphs over fixed shapes (CUDA only;
    # 130k against 78k decisions/s eager at 8192 battles, 2026-10-26)
    graphs: bool = True
    # seconds between nvidia-smi samples into the run's gpu.csv (utilization,
    # memory, power, clocks), to read against log.jsonl's phase times; 0: off
    gpu_log: float = 2.0


def power_schedule(coef, step, decay, ceil, floor):
    return min(max(coef / (step + 1) ** decay, floor), ceil)


def magnet(choices: torch.Tensor, legal: torch.Tensor, kind: str) -> torch.Tensor:
    """The magnet's log-probs [B, CHOICES] (-inf off the legal ones)."""
    legal_f = legal.float()
    if kind == "uniform":
        p = legal_f / legal_f.sum(-1, keepdim=True)
    elif kind == "attack_switch":
        move = ((choices & 3) == 1) & legal
        swap = legal & ~move
        nm, ns = move.sum(-1, keepdim=True), swap.sum(-1, keepdim=True)
        groups = (nm > 0).float() + (ns > 0).float()
        p = (move.float() / nm.clamp(min=1) + swap.float() / ns.clamp(min=1)) / groups
    else:
        raise ValueError(kind)
    return torch.log(p).masked_fill(~legal, -1e9)


def neurd_loss(logits, legal, n, a, old, A, ratio, clip_range, beta, mu_cap):
    """Sampled NeuRD's policy loss under PPO's clip and R-NaD's threshold: each
    legal logit is pushed by f = A min(1 / mu(a), mu_cap) (e_a - legal / n) (mu
    the data policy, `old` its log-probs; the pick a), over the batch's mean, except
    where PPO's min takes the clipped branch (ratio > 1 + clip_range with
    A > 0, < 1 - clip_range with A < 0; the whole row) and where the push
    would take a logit already beta past the legal mean further out. Logits
    are [B, CHOICES], -inf off the legal ones."""
    x = logits.float().masked_fill(~legal, 0)
    y = x - x.sum(-1, keepdim=True) / n[:, None].float()
    keep = ~(((A > 0) & (ratio > 1 + clip_range)) | ((A < 0) & (ratio < 1 - clip_range)))
    w = A * torch.exp(-old.gather(1, a)[:, 0]).clamp(max=mu_cap) * keep
    f = w[:, None] * (torch.nn.functional.one_hot(a[:, 0], legal.shape[1]) - legal.float() / n[:, None].float())
    f = f * (((f > 0) & (y < beta)) | ((f < 0) & (y > -beta)))
    return -(f.detach() * x).sum(-1).mean()


def compile_net(net: CausalPolicy) -> CausalPolicy:
    """Compiles `net`'s summarize, run_core and decide in place, so the
    actor, the learner and the evals all run them compiled, and the
    state_dict (and so every checkpoint) is unchanged. Shapes are dynamic:
    the batch, the lines and the history vary every call."""
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    # torch 2.11's coalescing analysis asserts on summarize's backward with
    # dynamic shapes (tiling_utils.get_pw_red_splits); a tiling heuristic only
    torch._inductor.config.triton.coalesce_tiling_analysis = False
    net.run_core = torch.compile(net.run_core, dynamic=True)
    net.decide = torch.compile(net.decide, dynamic=True)
    net.summarize = torch.compile(net.summarize, dynamic=True)
    return net


class Learner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.policy_loss not in ("ppo", "neurd"):
            raise ValueError(cfg.policy_loss)
        self.dev = torch.device(cfg.device)
        self.net = make_net(SIZES[cfg.size]).to(self.dev)
        self.ema = make_net(SIZES[cfg.size]).to(self.dev).eval()
        self.ema.load_state_dict(self.net.state_dict())
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.optim = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr_ceil, weight_decay=cfg.weight_decay)
        if cfg.compile:
            compile_net(self.net)
            compile_net(self.ema)
        self.dd = None  # the DeviceData for the data being trained on
        self.it = 0
        self.decisions = 0  # trained, for the schedules

    def schedule_step(self):
        cfg = self.cfg
        return self.decisions / cfg.schedule_decisions if cfg.schedule_decisions > 0 else self.it

    def device_data(self, data, ret=None, adv=None):
        """The DeviceData for (data, ret, adv), built once."""
        dd = self.dd
        if dd is None or dd.src is not data or dd.ret_src is not ret or dd.adv_src is not adv:
            self.dd = dd = DeviceData(data, self.dev, ret, adv)
        return dd

    def train(self, data, rng):
        cfg, dev = self.cfg, self.dev
        stats = {}
        t0 = time.perf_counter()
        ret, adv = returns_and_advantages(data, cfg.td_lambda, cfg.gae_lambda)
        threshold = max(float(adv.abs().quantile(cfg.adv_filt_rate)) if len(adv) < 2**24 else
                        float(adv.abs()[torch.randperm(len(adv))[:2**24]].quantile(cfg.adv_filt_rate)),
                        cfg.adv_filt_thresh)
        keep = (adv.abs() >= threshold).numpy()
        step = self.schedule_step()
        lr = power_schedule(cfg.lr_coef, step, cfg.lr_decay, cfg.lr_ceil, cfg.lr_floor)
        temperature = power_schedule(cfg.temperature_coef, step, cfg.temperature_decay,
                                     cfg.temperature_ceil, cfg.temperature_floor)
        for g in self.optim.param_groups:
            g["lr"] = lr
        stats.update(lr=lr, temperature=temperature, schedule_step=step, adv_threshold=threshold,
                     decisions=len(adv), kept=int(keep.sum()))
        stats.update(metrics.data_stats(data, ret, adv))
        self.device_data(data, ret, adv)
        params = [p.detach() for p in self.net.parameters()]
        before = [p.clone() for p in params]
        sums = {}  # tensors on the device, read back once at the end
        steps = 0
        self.net.train()
        t1 = time.perf_counter()
        for group in minibatches(data, keep, cfg.minibatch, rng):
            out = self.step(data, split(data, group, keep, cfg.microbatch), keep, ret, adv, temperature)
            for k, v in out.items():
                sums[k] = torch.maximum(sums[k], v) if k in sums and k.endswith("_max") else sums.get(k, 0.0) + v
            steps += 1
        self.it += 1
        self.decisions += len(data.n)
        with torch.no_grad():
            ema = [e for e in self.ema.parameters()]
            torch._foreach_mul_(ema, cfg.ema_decay)
            torch._foreach_add_(ema, params, alpha=1 - cfg.ema_decay)
            # the iteration's step against the weights' size
            delta = torch.stack(torch._foreach_norm(torch._foreach_sub(params, before))).double().pow(2).sum()
            norm = torch.stack(torch._foreach_norm(before)).double().pow(2).sum()
        t2 = time.perf_counter()
        keys = list(sums)
        *vals, delta, norm = torch.stack([torch.as_tensor(sums[k], device=dev).double() for k in keys]
                                         + [delta, norm]).tolist()
        self.dd = None
        t3 = time.perf_counter()
        stats.update({k: v if k.endswith("_max") else v / max(1, steps) for k, v in zip(keys, vals)},
                     grad_steps=steps, param_norm=math.sqrt(norm), update_ratio=math.sqrt(delta / max(norm, 1e-30)),
                     # the step's CPU time before the loop, issuing it, and waiting for the GPU after it
                     train_prep_s=t1 - t0, train_issue_s=t2 - t1, train_wait_s=t3 - t2)
        return stats

    def forward(self, data, group, keep):
        """The net on trajectories `group` (causal_net.learner_forward).
        Returns (ids [K] of the kept decisions in `data`, logits [K,
        CHOICES], value logits [K, 3], choices [K, CHOICES], n [K]), the
        tensors on the device."""
        ids, _, logits, value, choices, n = learner_forward(self.net, self.device_data(data), group,
                                                            keep, self.dev, self.cfg.amp)
        return ids, logits, value, choices, n

    def step(self, data, micros, keep, ret, adv, temperature):
        """One optimizer step over a minibatch given as micro-batches
        (`split`)."""
        out = self.gradients(data, micros, keep, ret, adv, temperature)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.optim.step()
        # grad_norm is before clipping; grad_clipped averages to the share of steps clipped.
        # Tensors, on the device: train reads them back once.
        return dict(out, grad_norm=grad_norm, grad_norm_max=grad_norm,
                    grad_clipped=(grad_norm > self.cfg.max_grad_norm).float())

    def gradients(self, data, micros, keep, ret, adv, temperature):
        """The minibatch's gradients in .grad, accumulated over its
        micro-batches: each pass's losses are means over its kept decisions,
        weighted by its share of them, so the sum is the mean over the
        minibatch, as one pass over it would give. Returns the stats, means
        over the minibatch's kept decisions."""
        self.optim.zero_grad(set_to_none=True)
        counts = [int(keep[np.concatenate([np.arange(s, s + l) for s, l in
                                           zip(data.start[g], data.length[g])])].sum()) for g in micros]
        total = sum(counts)
        sums = {}
        for g, k in zip(micros, counts):
            loss, out = self.loss(data, g, keep, ret, adv, temperature)
            (loss * (k / total)).backward()
            for key, v in out.items():
                sums[key] = (torch.maximum(sums[key], v) if key in sums and key.endswith("_max") else
                             v if key.endswith("_max") else sums.get(key, 0.0) + v * k / total)
        return sums

    def loss(self, data, group, keep, ret, adv, temperature):
        cfg, dev = self.cfg, self.dev
        dd = self.device_data(data, ret, adv)
        _, idt, logits, value, choices, n = learner_forward(self.net, dd, group, keep, dev, cfg.amp)
        old, a, A, R = dd.logp[idt], dd.pick[idt][:, None], dd.adv[idt], dd.ret[idt]
        legal = torch.arange(CHOICES, device=dev)[None] < n[:, None]
        logp = torch.log_softmax(logits.float().masked_fill(~legal, -1e9), -1)
        p = logp.exp() * legal
        old = old.clamp(min=-1e9).masked_fill(~legal, -1e9)

        ratio = torch.exp(logp.gather(1, a)[:, 0] - old.gather(1, a)[:, 0])
        clipped = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
        policy_loss = (-torch.min(A * ratio, A * clipped).mean() if cfg.policy_loss == "ppo" else
                       neurd_loss(logits, legal, n, a, old, A, ratio, cfg.clip_range, cfg.neurd_beta,
                                  cfg.neurd_mu_cap))
        kl_data = (p * (logp - old)).sum(-1).mean()
        kl_magnet = (p * (logp - magnet(choices, legal, cfg.magnet))).sum(-1).mean()
        value_loss = -(R * torch.log_softmax(value.float(), -1)).sum(-1).mean()  # against the returns' W/D/L
        loss = (cfg.policy_coef * policy_loss + temperature * kl_magnet + cfg.vf_coef * value_loss
                + cfg.kl_coef * kl_data)
        ent = -(p * logp).sum(-1)
        entropy = ent.mean()
        # the largest legal logit's distance from the legal mean (a diagnostic)
        y = logits.detach().float().masked_fill(~legal, 0)
        y = (y - y.sum(-1, keepdim=True) / n[:, None].float()).masked_fill(~legal, 0)
        # NeuRD's importance weight 1 / mu(a), before the cap (diagnostics)
        inv_mu = torch.exp(-old.gather(1, a)[:, 0])
        # detached tensors, not floats: a .item() per stat was a sync per pass
        return loss, {k: v.detach() for k, v in dict(
            loss=loss, policy_loss=policy_loss, value_loss=value_loss, kl_data=kl_data, kl_magnet=kl_magnet,
            entropy=entropy, entropy_norm_learner=(ent / torch.log(n.float())).mean(),
            clip_frac=((ratio - 1).abs() > cfg.clip_range).float().mean(),
            ratio_mean=ratio.mean(), ratio_max=ratio.max(), logit_abs_max=y.abs().max(),
            inv_mu_max=inv_mu.max(), inv_mu_p99=torch.quantile(inv_mu, 0.99),
            inv_mu_capped=(inv_mu > cfg.neurd_mu_cap).float().mean()).items()}

    def save(self, path, exp3=None, rng=None, team_counts=None, pairs=None):
        """Written to a temporary file and renamed, so a sync never copies
        half a checkpoint."""
        ck = dict(net=self.net.state_dict(), ema=self.ema.state_dict(), optim=self.optim.state_dict(),
                  it=self.it, decisions=self.decisions, config=dataclasses.asdict(self.cfg),
                  exp3=exp3.state() if exp3 is not None else None,
                  rng=rng.bit_generator.state if rng is not None else None,
                  torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  team_counts=team_counts,
                  pairs=pairs.state() if pairs is not None else None)
        torch.save(ck, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load(self, ck):
        if ck["config"]["size"] != self.cfg.size:
            raise ValueError(f"checkpoint is size {ck['config']['size']!r}, not {self.cfg.size!r}")
        self.net.load_state_dict(ck["net"])
        self.ema.load_state_dict(ck["ema"])
        self.optim.load_state_dict(ck["optim"])
        self.it = ck["it"]
        self.decisions = ck.get("decisions", 0)

    def init_from(self, ck):
        """`init`: a checkpoint's weights, EMA and optimizer, at iteration 0
        with no trained decisions, so the schedules start at step 0. A policy
        file (config and EMA only) gives the EMA to both and a fresh optimizer.
        A cs2 checkpoint can start a cs3 run: cs3's feature layers keep their
        zero init (so the net computes what the checkpoint did, but for the
        history window) and the optimizer starts fresh."""
        size = ck["config"]["size"]
        if size != self.cfg.size and (size, self.cfg.size) != ("cs2", "cs3"):
            raise ValueError(f"checkpoint is size {size!r}, not {self.cfg.size!r}")
        for net, w in ((self.net, ck.get("net", ck["ema"])), (self.ema, ck["ema"])):
            missing, unexpected = net.load_state_dict(w, strict=size == self.cfg.size)
            if unexpected or any(not k.startswith(FEATURE_PARAMS) for k in missing):
                raise ValueError(f"cs2 -> cs3: missing {missing}, unexpected {unexpected}")
        if "optim" in ck and size == self.cfg.size:
            self.optim.load_state_dict(ck["optim"])
        self.decisions = 0


def latest_checkpoint(run):
    its = [int(m.group(1)) for f in os.listdir(run) if (m := re.fullmatch(r"ckpt(\d+)\.pt", f))]
    if not its:
        raise FileNotFoundError(f"no ckpt*.pt in {run}")
    return os.path.join(run, f"ckpt{max(its)}.pt")


def code_version():
    """The oak commit this process runs (ataraxos is oak's ataraxos/, libpkmn
    oak's submodule): the checkout's COMMIT file (box/push.sh writes it; the
    box has no .git), else git, with "-dirty" for uncommitted changes; None
    if neither says."""
    oak = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    out = {}
    for name, d in (("oak", oak),):
        try:
            with open(os.path.join(d, "COMMIT")) as f:
                out[name] = f.read().strip()
            continue
        except OSError:
            pass
        try:
            git = lambda *a: subprocess.run(["git", "-C", d, *a], capture_output=True, text=True,
                                            check=True).stdout.strip()
            out[name] = git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")
        except (OSError, subprocess.CalledProcessError):
            out[name] = None
    return out


def peak_gb(dev):
    """The device's peak allocation since the last call (CUDA only)."""
    if dev.type != "cuda":
        return None
    gb = torch.cuda.max_memory_allocated(dev) / 2**30
    torch.cuda.reset_peak_memory_stats(dev)
    return gb


def cpu_seconds():
    """This process's user + system CPU time, all threads (the env's included)."""
    t = os.times()
    return t.user + t.system


def rss_peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB on Linux


def start_gpu_log(cfg):
    """nvidia-smi sampling into the run's gpu.csv every cfg.gpu_log seconds,
    appended across resumes; None on CPU, without nvidia-smi, or off. It's
    stopped at a normal exit or Ctrl-C; a killed trainer leaves it running."""
    if cfg.gpu_log <= 0 or not str(cfg.device).startswith("cuda") or shutil.which("nvidia-smi") is None:
        return None
    out = open(os.path.join(cfg.run, "gpu.csv"), "a")
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,"
         "power.limit,clocks.sm,temperature.gpu", "--format=csv,nounits", f"--loop-ms={int(cfg.gpu_log * 1000)}"],
        stdout=out, stderr=subprocess.DEVNULL)
    atexit.register(proc.terminate)
    return proc


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

    state = SIZES[cfg.size].state  # the net reads the bookkeeping state

    def env_factory(battles, seed=cfg.seed, teams=cfg.teams):
        return ataraxos_env.SelfPlay(battles, seed, teams=teams, max_turns=cfg.max_turns, threads=cfg.threads,
                                  state=state)

    exploit = bool(cfg.target)
    if exploit and not (cfg.teams and cfg.target_teams):
        raise ValueError("--target needs the learner's team files and --target_teams")

    learner = Learner(cfg)
    print(f"{sum(p.numel() for p in learner.net.parameters()) / 1e6:.2f}M params on {cfg.device}")
    ck = None
    if cfg.resume:
        path = latest_checkpoint(cfg.run) if cfg.resume == "latest" else cfg.resume
        ck = torch.load(path, map_location=learner.dev, weights_only=False)
        learner.load(ck)
        if ck["rng"] is not None:
            rng.bit_generator.state = ck["rng"]
        # map_location moves these to the device; the setters want CPU tensors
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        print(f"resumed from {path} at iteration {learner.it}")
    elif cfg.init:
        learner.init_from(torch.load(cfg.init, map_location=learner.dev, weights_only=False))
        print(f"initialized from {cfg.init} ({learner.decisions} trained decisions)")
    name = f"config_resume{learner.it}.json" if ck is not None else "config.json"
    with open(os.path.join(cfg.run, name), "w") as f:
        json.dump({**dataclasses.asdict(cfg), "code": code_version()}, f, indent=1)
    # a fresh run keeps the seed; a resumed one mustn't replay the first games
    env_seed = cfg.seed if ck is None else int(np.random.SeedSequence([cfg.seed, learner.it]).generate_state(1)[0])
    # the exploiter run's pool: the target's teams, then the learner's
    if exploit:
        target_names = list(env_factory(1, teams=cfg.target_teams).team_names)
        kt = len(target_names)
        env = env_factory(cfg.battles, env_seed, cfg.target_teams + cfg.teams)
        env.set_alternate(True)  # role 0, the learner's, is P1 in the even battles
    else:
        kt = 0
        env = env_factory(cfg.battles, env_seed)
    k = len(env.team_probs) - kt  # the learner's pool
    exp3 = teams.Exp3(k, cfg.team_gamma, cfg.team_eta, cfg.team_gamma_floor,
                      cfg.team_gamma_tau, cfg.team_gamma_decay) if cfg.teams else None
    if exploit:
        target_exp3 = teams.Exp3(kt, cfg.target_team_gamma)  # never updated: fixed
        if cfg.target_team_npz.endswith(".pt"):  # a checkpoint's Exp3: only the pool size can be checked
            log_w = torch.load(cfg.target_team_npz, map_location="cpu", weights_only=False)["exp3"]["log_w"]
            if len(log_w) != kt:
                raise ValueError(f"{cfg.target_team_npz}'s Exp3 has {len(log_w)} teams, --target_teams {kt}")
            target_exp3.log_w = np.asarray(log_w, np.float64).copy()
        elif cfg.target_team_npz:
            z = np.load(cfg.target_team_npz)
            if list(z["names"]) != target_names:
                raise ValueError(f"{cfg.target_team_npz}'s names don't match --target_teams")
            target_exp3.log_w = np.asarray(z["log_w"], np.float64).copy()
        env.set_team_weights(np.r_[target_exp3.probs(), np.zeros(k)], role=1)
        target = load_policy(cfg.target, cfg.device, cfg.target_weights)
        target_actor = make_actor(target, cfg.battles, cfg.device, amp=cfg.amp, graphs=cfg.graphs)
        print(f"exploiting {cfg.target} ({cfg.target_weights}): learner pool {k}, target pool {kt}, "
              f"target p_eff {np.exp(teams.entropy(target_exp3.probs())):.1f}")

    def eval_env(battles):
        """An eval's env: in the exploiter run, the learner's side draws from
        its Exp3 and the anchor's from the target's teams (the user's,
        2026-10-30), `match`'s P1 in the even battles being role 0."""
        if not exploit:
            return env_factory(battles, cfg.seed + 10**6 + learner.it)
        e = env_factory(battles, cfg.seed + 10**6 + learner.it, cfg.target_teams + cfg.teams)
        e.set_alternate(True)
        e.set_team_weights(np.r_[np.zeros(kt), exp3.probs()], role=0)
        e.set_team_weights(np.r_[target_exp3.probs(), np.zeros(k)], role=1)
        return e

    def set_weights():
        if exploit:
            env.set_team_weights(np.r_[np.zeros(kt), exp3.probs()], role=0)
        else:
            env.set_team_weights(exp3.probs())

    # the env counts its own draws from zero; earlier runs' draws are added here
    counts_before = np.zeros(k, np.int64)
    if ck is not None and exp3 is not None:
        if ck["exp3"] is None or len(ck["exp3"]["log_w"]) != exp3.k:
            raise ValueError("the checkpoint's team pool doesn't match the team files")
        exp3.load(ck["exp3"])
        counts_before = np.asarray(ck["team_counts"], np.int64)
    pairs = teams.Pairs() if exp3 is not None else None
    if ck is not None and pairs is not None and ck.get("pairs") is not None:
        pairs.load(ck["pairs"])
    if exp3 is not None:
        set_weights()
    actor = make_actor(learner.net, cfg.battles, cfg.device, amp=cfg.amp, graphs=cfg.graphs)
    buffer = Buffer(learner.net.c.max_lines, cfg.at_cap, sides=1 if exploit else 2)

    def ended(obs):
        done = (obs["end_capped"] == 0) | (cfg.at_cap == "draw")  # a capped game counts only as a draw
        if exploit:
            # the learner's side: its seat, team (in its pool), draw probability and result
            seat = (obs["end_battle"] & 1).astype(np.int64)
            e = np.arange(len(seat))
            mine, theirs = obs["end_team"][e, seat] - kt, obs["end_team"][e, 1 - seat]
            result = np.where(seat == 0, obs["end_result"], -obs["end_result"])
            pairs.add(np.stack([mine, theirs], 1), result, obs["end_capped"])
            if done.any():
                exp3.update_rounds(mine[done], obs["end_prob"][e, seat][done], (result[done] + 1) / 2)
                set_weights()
            return
        if pairs is not None:
            pairs.add(obs["end_team"], obs["end_result"], obs["end_capped"])
        if exp3 is not None and done.any():
            exp3.update(obs["end_team"][done], obs["end_prob"][done], obs["end_result"][done])
            set_weights()

    obs = env.reset()
    ended(obs)  # a game with no decision to send can end in reset
    log = open(os.path.join(cfg.run, "log.jsonl"), "a")
    if ck is not None:
        log.write(json.dumps({"resume": learner.it}) + "\n")
        log.flush()
    # the fixed anchors once; the "{it}" ones at each eval
    anchors = {a: make_agent(a, cfg.device, cfg.seed, cfg.amp) for a in cfg.anchors if "{it}" not in a}
    start_gpu_log(cfg)
    env_before = env.stats

    while learner.it < cfg.iters:
        t0 = time.perf_counter()
        cpu0 = cpu_seconds()
        peak_gb(learner.dev)
        decisions = 0
        net_s = env_s = 0.0
        learner.net.eval()
        while buffer.games() < cfg.games_per_iter:
            ta = time.perf_counter()
            if exploit:
                mine = (obs["battle"] & 1) == obs["player"]
                ours, theirs = subset(obs, np.flatnonzero(mine)), subset(obs, np.flatnonzero(~mine))
                picks = np.zeros(len(obs["n"]), np.uint8)
                picks[~mine] = target_actor.act(theirs)[0]
                picks[mine], logp, value = actor.act(ours)
                tb = time.perf_counter()
                buffer.add(ours, picks[mine], logp, value)
                decisions += int(mine.sum())
            else:
                picks, logp, value = actor.act(obs)  # picks come back to the CPU, so the GPU work is done
                tb = time.perf_counter()
                buffer.add(obs, picks, logp, value)
                decisions += len(picks)
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
        r = data.reward[::1]
        env_now = env.stats
        d_env = {k: env_now[k] - env_before[k] for k in env_now}
        env_before = env_now
        stats.update(it=learner.it, games=len(data.length) // buffer.sides, act_decisions=decisions,
                     decisions_per_s=decisions / (t1 - t0), act_s=t1 - t0, train_s=t2 - t1,
                     act_net_s=net_s, act_env_s=env_s, act_other_s=t1 - t0 - net_s - env_s,
                     learner_decisions_per_s=len(data.n) / (t2 - t1), learner_kept_per_s=stats["kept"] / (t2 - t1),
                     mean_length=float(data.length.mean()), ties=float((r == 0).mean()),
                     capped_rate=d_env["capped"] / max(1, d_env["games"]),
                     error_rate=d_env["errors"] / max(1, decisions),
                     forced_frac=d_env["forced"] / max(1, d_env["forced"] + decisions),
                     **{f"env_{k}": v for k, v in env_now.items()})
        if learner.dev.type == "cuda":
            stats.update(act_gb=act_gb, train_gb=train_gb)
        if exploit:
            # the learner's games against the target this iteration
            w, t, l = int((r == 1).sum()), int((r == 0).sum()), int((r == -1).sum())
            stats.update({"exploit/score": (w + t / 2) / max(1, len(r)), "exploit/wtl": [w, t, l]})
        team_counts = counts_before + env.team_counts[kt:] if exp3 is not None else None
        if exp3 is not None:
            stats.update(teams.summary(exp3, team_counts))
        t3 = time.perf_counter()
        if learner.it % cfg.eval_every == 0:
            opps = dict(anchors)
            for a in cfg.anchors:
                if "{it}" in a and os.path.exists(a.format(it=learner.it)):
                    opps[a] = make_agent(a.format(it=learner.it), cfg.device, cfg.seed, cfg.amp)
            # the EMA as eval/, the current weights as eval_net/, on the same battles
            for prefix, net in (("eval", learner.ema), ("eval_net", learner.net)):
                net.eval()
                ours = NetAgent(net, cfg.device, cfg.amp)
                for name, opp in opps.items():
                    w, t, l = match(eval_env, ours, opp, cfg.eval_battles)
                    score = (w + t / 2) / max(1, w + t + l)
                    stats[f"{prefix}/{os.path.basename(name)}"] = score
                    stats[f"{prefix}/{os.path.basename(name)}_wtl"] = [w, t, l]
                    stats[f"{prefix}/{os.path.basename(name)}_ci95"] = list(metrics.wilson(score, w + t + l))
        t4 = time.perf_counter()
        if learner.it % cfg.save_every == 0:
            learner.save(os.path.join(cfg.run, f"ckpt{learner.it}.pt"), exp3, rng, team_counts, pairs)
            if exp3 is not None:
                extra = dict(target_counts=env.team_counts[:kt], target_names=np.array(target_names)) if exploit else {}
                np.savez(os.path.join(cfg.run, f"teams{learner.it}.npz"), log_w=exp3.log_w, p=exp3.probs(), q_avg=exp3.q_avg(),
                         counts=team_counts, names=np.array(env.team_names[kt:]), **pairs.arrays(), **extra)
        t5 = time.perf_counter()
        stats.update(eval_s=t4 - t3, save_s=t5 - t4, iter_s=t5 - t0, time=time.time(),
                     games_per_s=stats["games"] / (t5 - t0), decisions_per_s_total=decisions / (t5 - t0),
                     cpu_cores=(cpu_seconds() - cpu0) / (t5 - t0), rss_peak_gb=rss_peak_gb())
        if learner.dev.type == "cuda":
            stats["eval_gb"] = peak_gb(learner.dev)
        log.write(json.dumps(stats) + "\n")
        log.flush()
        keys = ("it", "games", "games_per_s", "decisions_per_s", "act_s", "train_s", "iter_s", "act_gb", "train_gb", "loss",
                "value_loss", "value_ev", "entropy", "entropy_norm", "kl_data", "clip_frac", "grad_clipped", "kept",
                "mean_length", "ties", "exploit/score", "teams/gamma", "teams/p_eff", "teams/nash_eff", "teams/q_avg_eff")
        print(" ".join(f"{k}={stats[k]:.4g}" if isinstance(stats.get(k), float) else f"{k}={stats.get(k)}"
                       for k in keys if k in stats) + "".join(f" {k}={v:.3f}" for k, v in stats.items()
                                               if k.startswith("eval") and isinstance(v, float)), flush=True)


if __name__ == "__main__":
    main()
