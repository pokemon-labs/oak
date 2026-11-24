"""A perfect-info net (ataraxos/pi/) against a Showdown-view checkpoint
(2026-11-23, the user's: "idc that it's not fair"). One battle feeds both:
the Showdown-view side gets its usual tokens and request (ataraxos_env), the
pi side the battle itself, chance durations included (`SelfPlay.pi_obs`).
Policy only, both sampling their raw policy (temperature 1, no cutoff).

    python -m ataraxos.eval.pi_h2h PI.pt B TEAMFILES... [--battles 2048] [--swap]

B is a Showdown-view checkpoint, "maxdamage" or "random" (the last two
check the pi side's path against its training evals). Prints the pi net's
wins, ties, losses, score and Wilson 95% interval.
"""

import argparse

import numpy as np
import torch

from ataraxos._ext import ataraxos_env
from ataraxos.eval.agents import make_agent, subset
from ataraxos.pi.agents import NetAgent as PiAgent, load_net
from ataraxos.rl import metrics

PI_KEYS = ("mon", "hp", "active", "policy", "n")


def match(env, pi, b, battles, pi_seat, max_steps=100_000):
    """pi plays seat pi_seat(battle) (0 P1, 1 P2). Returns pi's (w, t, l)."""
    b.start(battles)
    result = np.full(battles, -2, dtype=np.int64)
    obs = env.reset()
    for step in range(max_steps + 1):
        for bt, g, r in zip(obs["end_battle"], obs["end_game"], obs["end_result"]):
            if g == 0 and result[bt] == -2:
                result[bt] = r if pi_seat[bt] == 0 else -r
        if (result != -2).all() or step == max_steps:
            break
        mine = pi_seat[obs["battle"]] == obs["player"]
        picks = np.zeros(len(obs["n"]), np.uint8)
        rows = np.flatnonzero(mine)
        if len(rows):
            po = env.pi_obs()
            sub = {k: np.ascontiguousarray(po[k][rows]) for k in PI_KEYS}
            picks[rows] = pi.act(sub, obs["player"][rows].astype(np.int64))
        rows = np.flatnonzero(~mine)
        if len(rows):
            picks[rows] = b.act(subset(obs, rows))
        obs = env.step(picks)
    done = result[result != -2]
    return int((done == 1).sum()), int((done == 0).sum()), int((done == -1).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pi")
    p.add_argument("b")
    p.add_argument("teams", nargs="*")
    p.add_argument("--battles", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--weights", default="ema", choices=["ema", "net"], help="the pi net's")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max_turns", type=int, default=300)
    p.add_argument("--mem_gb", type=float, default=3.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--swap", action="store_true",
                   help="also play each chunk with the seats swapped (same seed: same teams and seats)")
    args = p.parse_args()
    if args.device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 2**30 / total))
    pi = PiAgent(load_net(args.pi, args.device, args.weights), args.device)
    b = make_agent(args.b, args.device, args.seed)
    state = bool(getattr(getattr(b, "net", None), "c", None) and getattr(b.net.c, "state", False))
    W = T = L = 0
    for i in range(0, args.battles, args.chunk):
        n = min(args.chunk, args.battles - i)
        seats = np.arange(n) % 2
        for flip in ((0, 1) if args.swap else (0,)):
            env = ataraxos_env.SelfPlay(n, args.seed + i, teams=args.teams, max_turns=args.max_turns,
                                        threads=args.threads, state=state)
            w, t, l = match(env, pi, b, n, seats ^ flip)
            W, T, L = W + w, T + t, L + l
        g = W + T + L
        print(f"{g} games: {W}-{T}-{L} score {(W + T / 2) / g:.3f}", flush=True)
    g = W + T + L
    s = (W + T / 2) / g
    lo, hi = metrics.wilson(s, g)
    print(f"{args.pi} ({args.weights}, perfect info) vs {args.b}: {W}-{T}-{L}, score {s:.3f}, 95% [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
