"""Which input a checkpoint leans on (2026-11-02, the user's ask): the
history (the causal layers' output, one token into the decision block) or
the bookkeeping state (the 14 state tokens).

Plays `--games` self-play games with the checkpoint (EMA, sampled picks),
then reruns every decision with one input broken and compares with the
intact net:

  - h_shuf: each decision gets another decision's history output;
  - h_win1, h_win4: the causal layers see only this decision's update (the
    last turn's lines), or the last 4;
  - state_shuf: each decision gets another decision's state tokens;
  - foe_shuf: only the foe's 6 Pokémon rows are another decision's;
  - both_shuf: h_shuf and state_shuf.

Per variant, over decisions with 2+ choices: KL(intact || variant), how
often the top choice stays, the mean |change| in the value's
P(win) - P(loss), and the value's cross-entropy against the game's
result (intact's on its own line; "const" is the result's base rate).

    python -m ataraxos.eval.ablate CKPT TEAM_FILE [--games 2048] [--threads 8]
"""

import argparse
import dataclasses

import numpy as np
import torch
import torch.nn.functional as F

from ataraxos._ext import ataraxos_env  # noqa: E402
from ataraxos.eval.agents import load_policy  # noqa: E402
from ataraxos.nets.causal_actor import make_actor  # noqa: E402
from ataraxos.nets.causal_net import ACTIVE_FIELDS, BASE_STATE_FIELDS, MON_FIELDS, LINE_EDGES, line_edge  # noqa: E402
from ataraxos.nets.policy import buckets  # noqa: E402
from ataraxos.rl.buffer import Buffer  # noqa: E402

VARIANTS = ("h_shuf", "h_win1", "h_win4", "state_shuf", "foe_shuf", "both_shuf")


def play(net, teams, games, battles, seed, threads):
    env = ataraxos_env.SelfPlay(battles, seed, teams=teams, threads=threads, state=True)
    actor = make_actor(net, battles, torch.device("cpu"), amp=False)
    buf = Buffer(net.c.max_lines, at_cap="drop")
    obs = env.reset()
    while buf.games() < games:
        buf.end(obs)
        picks, logp, value = actor.act(obs)
        buf.add(obs, picks, logp, value)
        obs = env.step(picks)
    buf.end(obs)
    return buf.take()


def summaries(net, data):
    """Every decision's update summary [N, d]."""
    N = len(data.n)
    off = data.offsets
    lengths = np.minimum(off[1:] - off[:-1], net.c.max_lines)
    tokens = torch.from_numpy(data.tokens)
    request = torch.from_numpy(data.request)
    out = torch.empty(N, net.c.d)
    for rows in buckets(lengths, list(LINE_EDGES), 256):
        L = line_edge(int(lengths[rows].max()))
        j = np.arange(L)
        first = off[rows + 1] - lengths[rows]
        idx = np.minimum(first[:, None] + j, len(data.tokens) - 1)
        x = tokens[torch.from_numpy(idx)] * torch.from_numpy(j < lengths[rows, None])[..., None]
        out[torch.from_numpy(rows)] = net.summarize(x, request[torch.from_numpy(rows)])
    return out


def history(net, data, s, window):
    """The causal layers' output at every decision [N, d], each trajectory
    run alone, attending to the `window` decisions up to its own."""
    old = net.c
    net.c = dataclasses.replace(old, window=window)
    h = torch.empty_like(s)
    try:
        for a in range(0, len(data.length), 64):
            g = np.arange(a, min(a + 64, len(data.length)))
            start, length = data.start[g], data.length[g]
            x = torch.zeros(len(g), int(length.max()), s.shape[1])
            for r, (b, n) in enumerate(zip(start, length)):
                x[r, :n] = s[b:b + n]
            y = net.run_core(x)
            for r, (b, n) in enumerate(zip(start, length)):
                h[b:b + n] = y[r, :n]
    finally:
        net.c = old
    return h


def decide(net, data, h, state):
    logits, value = [], []
    req, ch, n = (torch.from_numpy(a) for a in (data.request, data.choices, data.n))
    for a in range(0, len(h), 4096):
        sl = slice(a, a + 4096)
        lg, v = net.decide(h[sl], req[sl], ch[sl], n[sl], state[sl])
        logits.append(lg)
        value.append(v)
    return torch.cat(logits), torch.cat(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("teams", nargs="+")
    ap.add_argument("--games", type=int, default=2048)
    ap.add_argument("--battles", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--weights", default="ema")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = load_policy(args.ckpt, "cpu", args.weights)
    with torch.no_grad():
        data = play(net, args.teams, args.games, args.battles, args.seed, args.threads)
        N = len(data.n)
        print(f"{len(data.length)} trajectories, {N} decisions, window {net.c.window}", flush=True)
        s = summaries(net, data)
        h = history(net, data, s, net.c.window)
        state = torch.from_numpy(data.state)
        base_lg, base_v = decide(net, data, h, state)

        traj = np.repeat(np.arange(len(data.length)), data.length)
        result = torch.from_numpy(data.reward[traj] + 1)  # (loss, draw, win) class
        live = torch.from_numpy(data.n > 1)
        perm = torch.from_numpy(rng.permutation(N))
        foe = 1 + 2 * ACTIVE_FIELDS + 6 * MON_FIELDS  # the foe's rows start

        def report(name, lg, v):
            p = base_lg.log_softmax(-1)
            q = lg.log_softmax(-1)
            kl = torch.where(p.isfinite(), p.exp() * (p - q), torch.zeros_like(p)).sum(-1)
            top = (base_lg.argmax(-1) == lg.argmax(-1)).float()
            pv, bv = v.softmax(-1), base_v.softmax(-1)
            dv = ((pv[:, 2] - pv[:, 0]) - (bv[:, 2] - bv[:, 0])).abs()
            ce = F.cross_entropy(v, result)
            print(f"{name:11s} KL {kl[live].mean():.4f}  top kept {top[live].mean():.3f}  "
                  f"|dV| {dv.mean():.4f}  value CE {ce:.4f}", flush=True)

        rate = torch.bincount(result, minlength=3).float() / N
        print(f"const       value CE {-(rate[result].log()).mean():.4f}")
        report("intact", base_lg, base_v)
        for name in VARIANTS:
            hv, sv = h, state
            if name in ("h_shuf", "both_shuf"):
                hv = h[perm]
            if name in ("h_win1", "h_win4"):
                hv = history(net, data, s, 1 if name == "h_win1" else 4)
            if name in ("state_shuf", "both_shuf"):
                sv = state[perm]
            if name == "foe_shuf":
                sv = state.clone()
                sv[:, foe:BASE_STATE_FIELDS] = state[perm, foe:BASE_STATE_FIELDS]
            report(name, *decide(net, data, hv, sv))


if __name__ == "__main__":
    main()
