"""A perfect-info run's newest checkpoint against its earlier ones, to see
whether it is still improving (2026-11-23, the user's: "check for
plateauing"). Policy only, both sides sampling their raw policy, ema
weights; pi.agents.match (seats alternate by battle, first game counts).

    python -m ataraxos.eval.pi_ladder RUN_DIR TEAMFILES... [--new ckpt2625.pt]
        [--old 2500,2250,2000,1500,1000,500] [--battles 2048]

Also plays each old checkpoint against the next one (--chain), which shows
non-transitive cycling that newest-vs-old alone can hide. An --old entry
that isn't a number is a checkpoint path, e.g. another run's newest
(--old runs/pi2/ckpt2675.pt, 2026-11-24: pi3 vs pi2); --chain skips those.
"""

import argparse
import os
import re

import torch

from ataraxos.pi.agents import NetAgent, ataraxos_pi, load_net, match
from ataraxos.rl import metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("teams", nargs="+")
    p.add_argument("--new", default="")
    p.add_argument("--old", default="2500,2250,2000,1500,1000,500")
    p.add_argument("--chain", action="store_true")
    p.add_argument("--battles", type=int, default=2048)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max_turns", type=int, default=300)
    p.add_argument("--mem_gb", type=float, default=3.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 2**30 / total))
    its = sorted(int(m.group(1)) for f in os.listdir(args.run) if (m := re.fullmatch(r"ckpt(\d+)\.pt", f)))
    new = int(re.search(r"\d+", args.new).group()) if args.new else its[-1]
    old = [int(x) if x.isdigit() else x for x in args.old.split(",")]
    path = lambda c: f"{args.run}/ckpt{c}.pt" if isinstance(c, int) else c
    name = lambda c: f"ckpt{c}" if isinstance(c, int) else c
    agent = lambda c: NetAgent(load_net(path(c), args.device), args.device)
    env = lambda b: ataraxos_pi.SelfPlay(b, args.seed, teams=args.teams, max_turns=args.max_turns,
                                         threads=args.threads)

    def play(a, b):
        w, t, l = match(env, agent(a), agent(b), args.battles)
        g = w + t + l
        s = (w + t / 2) / g
        lo, hi = metrics.wilson(s, g)
        print(f"{name(a)} vs {name(b)}: {w}-{t}-{l}, score {s:.3f}, 95% [{lo:.3f}, {hi:.3f}]", flush=True)

    print(f"== newest ckpt{new} vs older")
    for o in old:
        play(new, o)
    if args.chain:
        chain = sorted({o for o in old if isinstance(o, int)} | {new})
        print("== each vs the previous")
        for a, b in zip(chain[1:], chain):
            play(a, b)


if __name__ == "__main__":
    main()
