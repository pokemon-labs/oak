"""Checks ataraxos/search.py's caches on self-play (2026-11-01): one
checkpoint on both sides, the searcher P1 in the even battles. At every
search: our copied cache rerun on the root summary must give the root's
log-probs (max abs, bf16 noise only); with the foe's true sets (`truth`),
the foe's first rollout decision against the real foe actor's policy at
the same moment (KL, argmax agreement; not exact: the rebuilt history
estimates the foe's own HP from our percent, and its state's view of us).
The same with pool guesses, for scale.

    python tests/py/check_search.py CKPT [SEARCHES] [BATTLES]
"""

import os
import sys
import tempfile

import numpy as np
import torch

from ataraxos._ext import ataraxos_env
from ataraxos.eval.agents import ACTIVE_MOVE, F_SPECIES, HEADER, SIDE, SIDE0, load_policy, subset
from ataraxos.eval.search_h2h import split_teams
from ataraxos.nets.causal_actor import CausalActor
from ataraxos.search import Search, SearchConfig


def labels(request, choices, n):
    """Each choice by meaning (the numbering differs: the world's foe side is
    by appearance): a move by its ID, a switch by species."""
    sides = request[SIDE0:].reshape(6, SIDE)
    out = []
    for c in choices[:n]:
        kind, slot = int(c) & 3, int(c) >> 2
        if kind == 1:
            out.append(("move", int(request[HEADER + (slot - 1) * ACTIVE_MOVE]) if slot else -1))
        else:
            out.append(("switch", int(sides[slot - 1][F_SPECIES])))
    return out


def compare(real_lp, real_req, real_ch, real_n, lq, req, ch, n):
    """KL(real || rollout) over the real choices by label, argmax agreement,
    and whether the label sets differ."""
    a, b = labels(real_req, real_ch, real_n), labels(req, ch, n)
    if sorted(a) != sorted(b):
        return None
    at = {l: j for j, l in enumerate(b)}
    q = torch.stack([lq[at[l]] for l in a])
    p = real_lp[:real_n]
    kl = float((p.exp() * (p - q)).sum())
    return kl, bool(p.argmax() == q.argmax())


def main():
    ckpt = sys.argv[1]
    searches = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    net = load_policy(ckpt, dev)
    pool, games = split_teams("teams/selfplay/oak1500.txt", tempfile.mkdtemp(prefix="check_search_"))
    search = Search(net, [pool], dev, SearchConfig(budget=32, reps=1, depth=3), threads=4)
    env = ataraxos_env.SelfPlay(B, seed=3, teams=[games], max_turns=300, shards=min(B, 16), threads=4,
                                state=net.c.state)
    env.set_keep_after(True)
    ours, theirs = CausalActor(net, B, dev), CausalActor(net, B, dev)
    stats = {True: ([], [], []), False: ([], [], [])}
    worst = 0.0
    obs = env.reset()
    done = 0
    while done < searches:
        mine = (obs["battle"] % 2) == obs["player"]
        picks = np.zeros(len(obs["n"]), np.uint8)
        rows_t = np.flatnonzero(~mine)
        foe = {}
        if len(rows_t):
            p, lp, _ = theirs.act(subset(obs, rows_t))
            picks[rows_t] = p
            for j, r in enumerate(rows_t):
                foe[int(obs["battle"][r])] = (lp[j], obs["request"][r], obs["choices"][r], int(obs["n"][r]))
        rows = np.flatnonzero(mine)
        if len(rows):
            o = subset(obs, rows)
            p, logp, _ = ours.act(o)
            truth = done % 2 == 0
            check = {}
            p, _, _ = search.run(env, ours, o, logp, truth=truth, check=check)
            worst = max(worst, check["ours"])
            for i, lq, req, ch, n in check["foe"]:
                b = int(o["battle"][i])
                if b not in foe:
                    stats[truth][2].append("rollout foe decides, real doesn't")
                    continue
                c = compare(*foe[b], lq, req, ch, int(n))
                if c is None:
                    stats[truth][2].append("choices differ")
                    continue
                stats[truth][0].append(c[0])
                stats[truth][1].append(c[1])
            picks[rows] = p
            done += len(rows)
        obs = env.step(picks)
    print(f"{done} searches; ours: max |logp - root| {worst:.4f}")
    for truth in (True, False):
        kl, agree, skipped = stats[truth]
        kl = np.array(kl)
        print(f"foe ({'true sets' if truth else 'pool guesses'}): {len(kl)} roots, KL mean {kl.mean():.4f} "
              f"median {np.median(kl):.4f} p90 {np.quantile(kl, 0.9):.4f} max {kl.max():.4f}, "
              f"argmax agrees {np.mean(agree):.3f}; skipped {dict(zip(*np.unique(skipped, return_counts=True)))}")


if __name__ == "__main__":
    main()
