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
from ataraxos.eval.agents import load_policy, subset
from ataraxos.eval.search_h2h import split_teams
from ataraxos.nets.causal_actor import CausalActor
from ataraxos.search import Search, SearchConfig


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
    stats = {True: ([], []), False: ([], [])}
    worst = 0.0
    obs = env.reset()
    done = 0
    while done < searches:
        mine = (obs["battle"] % 2) == obs["player"]
        picks = np.zeros(len(obs["n"]), np.uint8)
        rows_t = np.flatnonzero(~mine)
        foe_lp = {}
        if len(rows_t):
            p, lp, _ = theirs.act(subset(obs, rows_t))
            picks[rows_t] = p
            for j, b in enumerate(obs["battle"][rows_t]):
                foe_lp[int(b)] = lp[j]
        rows = np.flatnonzero(mine)
        if len(rows):
            o = subset(obs, rows)
            p, logp, _ = ours.act(o)
            nan = torch.full_like(logp[0], float("nan"))
            fl = torch.stack([foe_lp.get(int(b), nan) for b in o["battle"]])
            truth = done % 2 == 0
            check = {"foe_logp": fl}
            p, _, _ = search.run(env, ours, o, logp, truth=truth, check=check)
            worst = max(worst, check["ours"])
            stats[truth][0].extend(check["foe"][0])
            stats[truth][1].extend(check["foe"][1])
            picks[rows] = p
            done += len(rows)
        obs = env.step(picks)
    print(f"{done} searches; ours: max |logp - root| {worst:.4f}")
    for truth in (True, False):
        kl, agree = stats[truth]
        kl = np.array(kl)
        print(f"foe ({'true sets' if truth else 'pool guesses'}): {len(kl)} roots, KL mean {kl.mean():.4f} "
              f"median {np.median(kl):.4f} p90 {np.quantile(kl, 0.9):.4f} max {kl.max():.4f}, "
              f"argmax agrees {np.mean(agree):.3f}")


if __name__ == "__main__":
    main()
