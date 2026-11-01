"""Search against the plain net (2026-11-01): one checkpoint plays both
sides, one side searching (ataraxos/search.py), on a team file split in
half: the history sampler's pool is the first half, the games' teams come
from the second, so every foe is outside the pool (check_history_sampler's
`half`). The searcher is P1 in the even battles and P2 in the odd ones; each
battle's first game counts (agents.match's rules).

    python -m ataraxos.eval.search_h2h CKPT [--battles 32] [--reps 2]
        [--depth 10] [--budget 128] [--worlds 0] [--teams FILE]
"""

import argparse
import os
import tempfile
import time

import numpy as np
import torch

from ataraxos._ext import ataraxos_env
from ataraxos.eval.agents import load_policy, subset
from ataraxos.nets.causal_actor import CausalActor
from ataraxos.search import Search, SearchConfig


def split_teams(path, out_dir):
    """`path`'s teams as two files, the first half and the second."""
    text = open(path).read()
    blocks = ["=== " + b for b in text.split("=== ")[1:]]
    half = len(blocks) // 2
    out = []
    for name, part in (("pool.txt", blocks[:half]), ("games.txt", blocks[half:])):
        p = os.path.join(out_dir, name)
        with open(p, "w") as f:
            f.write("".join(part))
        out.append(p)
    return out


def fmt(times):
    return {k: round(v, 1) for k, v in times.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--teams", default="teams/selfplay/oak1500.txt")
    ap.add_argument("--battles", type=int, default=32)
    ap.add_argument("--worlds", type=int, default=0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.002)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_turns", type=int, default=300)
    ap.add_argument("--max_searches", type=int, default=0, help="stop after this many (0: play out)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    net = load_policy(args.ckpt, dev)
    tmp = tempfile.mkdtemp(prefix="search_h2h_")
    pool, games = split_teams(args.teams, tmp)
    cfg = SearchConfig(worlds=args.worlds, reps=args.reps, depth=args.depth, budget=args.budget,
                       alpha=args.alpha, beta=args.beta)
    search = Search(net, [pool], dev, cfg, threads=args.threads, seed=args.seed)
    B = args.battles
    env = ataraxos_env.SelfPlay(B, seed=args.seed, teams=[games], max_turns=args.max_turns,
                                shards=min(B, 16), threads=args.threads, state=net.c.state)
    env.set_keep_after(True)
    ours, theirs = CausalActor(net, B, dev), CausalActor(net, B, dev)

    result = np.full(B, -2, np.int64)  # the searcher's: 1 win, 0 tie, -1 loss
    obs = env.reset()
    t_start, searched, moved = time.time(), 0, 0
    while True:
        for bt, g, r in zip(obs["end_battle"], obs["end_game"], obs["end_result"]):
            if g == 0:
                result[bt] = r if bt % 2 == 0 else -r
        if (result != -2).all() or (args.max_searches and searched >= args.max_searches):
            break
        mine = (obs["battle"] % 2) == obs["player"]
        picks = np.zeros(len(obs["n"]), np.uint8)
        rows = np.flatnonzero(~mine)
        if len(rows):
            picks[rows] = theirs.act(subset(obs, rows))[0]
        rows = np.flatnonzero(mine)
        if len(rows):
            o = subset(obs, rows)
            p, logp, _ = ours.act(o)
            live = np.flatnonzero(o["game"] == 0)  # only the counted games search
            if len(live):
                s = subset(o, live)
                sp, _, slp = search.run(env, ours, s, logp[live])
                moved += int((sp != torch.argmax(logp[live], -1).cpu().numpy()).sum())
                p[live] = sp
                searched += len(live)
            picks[rows] = p
        obs = env.step(picks)
        done = int((result != -2).sum())
        if searched and searched % 200 < len(rows):
            el = time.time() - t_start
            print(f"{el:7.0f}s  {searched} searches ({searched / el:.2f}/s), {done}/{B} games, "
                  f"stats {search.stats}, times {fmt(search.times)}", flush=True)
    w, t, l = int((result == 1).sum()), int((result == 0).sum()), int((result == -1).sum())
    el = time.time() - t_start
    print(f"search vs net: {w} W {t} T {l} L of {B}, score {(w + t / 2) / B:.3f}; {searched} searches in {el:.0f}s, "
          f"{moved} not the net's argmax; stats {search.stats}, times {fmt(search.times)}")


if __name__ == "__main__":
    main()
