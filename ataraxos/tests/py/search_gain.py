"""Cheap 1-round joint search settings scored by policy gain against an
independent deep reference.

Per root (rby self-play, 1 type a side, the foe's true sets):
  cheap runs: (depth, budget, seeds) in CHEAP; a reference run at depth 8,
  budget 256, floor 1 (uniform cells, so every action has a Q), its own seed.
Per (cheap run, lambda): the posterior by MMD on the cheap Q, then
  gain = sum_i (post_i - prior_i) q_ref_i, q_ref = Q_ref against the foe's
  prior (each side with 2+ choices, from its side), in win-minus-loss points
  (x100 in the table); noise = KL between the two seeds' posteriors.

    python tests/py/search_gain.py CKPT ROOTS [BATTLES]
"""

import sys
import tempfile
import time

import numpy as np
import torch

from ataraxos._ext import ataraxos_env
from ataraxos.eval.agents import load_policy
from ataraxos.eval.search_h2h import split_teams
from ataraxos.nets.causal_actor import CausalActor
from ataraxos.search import Search, SearchConfig

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from search_soft import ent, kl, norm, roots_of, solve, sub  # noqa: E402

CHEAP = [(1, 64, 2), (1, 128, 2), (4, 128, 1)]
REF = (8, 256)
LAMS = [(0.002, 0.02), (0.005, 0.05), (0.01, 0.1), (0.02, 0.2), (0.05, 0.5), (0.0, 0.02), (0.0, 0.05), (0.0, 0.1), (0.0, 0.2)]


def main():
    ckpt, R = sys.argv[1], int(sys.argv[2])
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    torch.set_num_threads(12)
    torch.manual_seed(1)
    dev = torch.device("cpu")
    net = load_policy(ckpt, dev)
    pool, games = split_teams("teams/selfplay/oak1500.txt", tempfile.mkdtemp(prefix="soft2_"))
    base = dict(joint=True, joint_mode="per_type", types_a=1, types_b=1, rounds=1, shrink=1.0, act_rows=2048)
    searches = {(d, b): Search(net, [pool], dev, SearchConfig(joint_budget=b, depth=d, floor=0.25, **base),
                               threads=8, amp=False, seed=d * 1000 + b) for d, b, _ in CHEAP}
    ref = Search(net, [pool], dev, SearchConfig(joint_budget=REF[1], depth=REF[0], floor=1.0, **base),
                 threads=8, amp=False, seed=99)
    env = ataraxos_env.SelfPlay(B, seed=11, teams=[games], max_turns=300, shards=min(B, 16), threads=8,
                                state=net.c.state)
    env.set_keep_after(True)
    actor = CausalActor(net, B, dev, amp=False)
    rng = np.random.default_rng(5)
    runs, refs = {}, []
    obs = env.reset()
    done, tic = 0, time.perf_counter()
    while done < R:
        picks, logp, _ = actor.act(obs)
        picks = picks.cpu().numpy().astype(np.uint8) if torch.is_tensor(picks) else picks
        rows = np.flatnonzero((obs["n"] >= 2) & (rng.random(len(obs["n"])) < 0.15))[: min(2, R - done)]
        if len(rows):
            o, lp = sub(obs, rows), logp[rows]
            for d, b, reps in CHEAP:
                for rep in range(reps):
                    check = {}
                    searches[(d, b)].run(env, actor, o, lp, seed=7919 * done + 31 * rep + d * 1000 + b, truth=True,
                                         check=check)
                    runs.setdefault((d, b, rep), []).extend(roots_of(check))
            check = {}
            ref.run(env, actor, o, lp, seed=104729 * (done + 1), truth=True, check=check)
            refs.extend(roots_of(check))
            done += len(rows)
            print(f"{done}/{R} roots, {time.perf_counter() - tic:.0f} s", flush=True)
        obs = env.step(picks)

    print(f"\n{ckpt}: {done} roots; reference depth {REF[0]} budget {REF[1]} (uniform cells); 1 round")
    print(f"{'depth':>5} {'budget':>6} {'fwd/root':>8} | {'alpha':>6} {'beta':>5} {'lam':>5} | {'gain x100':>9} "
          f"{'+-se':>5} {'KL(post|prior)':>14} {'noise KL':>8} {'argmax chg':>10} {'H prior':>7} {'H post':>6}")
    for d, b, reps in CHEAP:
        for alpha, beta in LAMS:
            gains, klp, noise, chg, hp, hq = [], [], [], [], [], []
            for j, (r, q) in enumerate(zip(runs[(d, b, 0)], refs)):
                x = solve(r, alpha, beta)
                y = solve(runs[(d, b, 1)][j], alpha, beta) if reps > 1 else None
                Qr = q[0]
                for side in (0, 1):
                    n1, n2 = r[5], r[6]
                    n = (n1, n2)[side]
                    if n < 2:
                        continue
                    prior = norm(r[1 + side], n)
                    post = norm(x[side], n)
                    if side == 0:
                        qref = Qr[:n1, :n2] @ np.exp(norm(r[2], n2))[:n2]
                    else:
                        qref = -(np.exp(norm(r[1], n1))[:n1] @ Qr[:n1, :n2])
                    gains.append(float((np.exp(post[:n]) - np.exp(prior[:n])) @ qref))
                    klp.append(kl(post, prior, n))
                    chg.append(np.argmax(post[:n]) != np.argmax(prior[:n]))
                    hp.append(ent(prior, n))
                    hq.append(ent(post, n))
                    if y is not None:
                        noise.append(kl(post, norm(y[side], n), n))
            g = np.array(gains) * 100
            print(f"{d:5d} {b:6d} {d * b:8d} | {alpha:6.3f} {beta:5.2f} {alpha + beta:5.3f} | {g.mean():9.2f} "
                  f"{g.std() / np.sqrt(len(g)):5.2f} {np.mean(klp):14.3f} "
                  f"{(np.mean(noise) if noise else float('nan')):8.3f} {np.mean(chg):10.3f} {np.mean(hp):7.3f} "
                  f"{np.mean(hq):6.3f}")


if __name__ == "__main__":
    main()
