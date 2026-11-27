"""rby4 -> oak battle net distillation, round 0's data (2026-11-26, the
user's; docs/distill.md, with rby4 as the teacher instead of pi4).

rby4 (a Showdown-view checkpoint, its EMA weights) plays itself in its own
env (state=True), sampling its raw policy; the env records every game as
oak's CompressedFrames (src/battle_data.hpp's `oak`), each libpkmn update
labelled with rby4's policies mapped onto libpkmn's legal choices and its
value (W + D/2, P1's side; both players' views averaged in `nash_value`, the
first one in `empirical_value`). Out: DIR/N.battle.data, oak's format, for
`python -m oak.scripts.battle --data-dir DIR`.

    python -m ataraxos.distill.gen_oak CKPT TEAMFILES... --out DIR --games 200000
"""

import argparse
import os
import time

import numpy as np
import torch

from ataraxos._ext import ataraxos_env
from ataraxos.eval.agents import load_policy
from ataraxos.nets.causal_actor import make_actor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("teams", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=200_000)
    p.add_argument("--battles", type=int, default=4096)
    p.add_argument("--per_file", type=int, default=4096, help="games per .battle.data file")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max_turns", type=int, default=300)
    p.add_argument("--graphs", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    net = load_policy(args.ckpt, dev)
    actor = make_actor(net, args.battles, dev, amp=dev.type == "cuda", graphs=bool(args.graphs))
    env = ataraxos_env.SelfPlay(args.battles, args.seed, teams=list(args.teams), max_turns=args.max_turns,
                                threads=args.threads, state=True)
    env.set_oak(True)
    obs = env.reset()

    done = len([f for f in os.listdir(args.out) if f.endswith(".battle.data")])
    buf, in_buf, games, t0 = [], 0, 0, time.time()
    while games < args.games:
        picks, logp, value = actor.act(obs)
        probs = logp.exp().cpu().numpy().astype(np.float32)
        w = torch.softmax(value, -1).cpu().numpy()  # (loss, draw, win), the decider's
        own = w[:, 2] + 0.5 * w[:, 1]
        p1 = np.where(obs["player"] == 0, own, 1.0 - own).astype(np.float32)
        env.set_oak_targets(probs, p1)
        obs = env.step(picks)
        data, n = env.take_oak()
        if n:
            buf.append(data)
            in_buf += n
            games += n
        if in_buf >= args.per_file or (games >= args.games and in_buf):
            path = os.path.join(args.out, f"{done}.battle.data")
            with open(path + ".tmp", "wb") as f:
                f.write(b"".join(buf))
            os.replace(path + ".tmp", path)
            done += 1
            dt = time.time() - t0
            print(f"{path}: {in_buf} games, total {games} ({games / dt:.1f} games/s, {dt:.0f}s) "
                  f"capped {env.stats['capped']}", flush=True)
            buf, in_buf = [], 0


if __name__ == "__main__":
    main()
