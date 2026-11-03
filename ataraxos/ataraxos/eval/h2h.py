"""Checkpoint against checkpoint (2026-10-26): `match` (ataraxos/eval/agents.py) over
the team files, in chunks of battles to keep GPU memory small, with a cap
on this process's share so it can't take memory from a training run on the
same GPU.

    python -m ataraxos.eval.h2h A.pt B.pt TEAMFILES... [--battles 2048] [--weights ema] [--weights_b net] [--swap]

Prints A's wins, ties, losses, score and Wilson 95% interval.
"""

import argparse

import torch

from ataraxos._ext import ataraxos_env  # noqa: E402
from ataraxos.rl import metrics  # noqa: E402
from ataraxos.eval.agents import NetAgent, load_policy, match  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("teams", nargs="*")
    p.add_argument("--battles", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--weights", default="ema", choices=["ema", "net"])
    p.add_argument("--weights_b", default="", choices=["", "ema", "net"], help="B's, if not --weights")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max_turns", type=int, default=300)
    p.add_argument("--mem_gb", type=float, default=3.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--swap", action="store_true",
                   help="also play each chunk's battles with the agents swapped (same teams and seats, less variance)")
    args = p.parse_args()
    if args.device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 2**30 / total))
    a = NetAgent(load_policy(args.a, args.device, args.weights), args.device)
    b = NetAgent(load_policy(args.b, args.device, args.weights_b or args.weights), args.device)
    W = T = L = 0
    for i in range(0, args.battles, args.chunk):
        n = min(args.chunk, args.battles - i)
        seed = args.seed + i
        state = any(getattr(x.net.c, "state", False) for x in (a, b))  # a causal net reads the bookkeeping state
        env = lambda k: ataraxos_env.SelfPlay(k, seed, teams=args.teams, max_turns=args.max_turns,
                                              threads=args.threads, state=state)
        w, t, l = match(env, a, b, n)
        if args.swap:  # the same seed draws the same first teams at reset: B plays A's teams and seats
            l2, t2, w2 = match(env, b, a, n)
            w, t, l = w + w2, t + t2, l + l2
        W, T, L = W + w, T + t, L + l
        g = W + T + L
        print(f"{g} games: {W}-{T}-{L} score {(W + T / 2) / g:.3f}", flush=True)
    g = W + T + L
    s = (W + T / 2) / g
    lo, hi = metrics.wilson(s, g)
    print(f"{args.a} vs {args.b} ({args.weights} vs {args.weights_b or args.weights}): {W}-{T}-{L}, score {s:.3f}, 95% [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
