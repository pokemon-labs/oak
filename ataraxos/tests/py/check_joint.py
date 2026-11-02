"""Checks the joint search (ataraxos/search.py, `joint`; 2026-11-02), kept
quick:

- MMD on random Bayesian games (3x3 types, 2-9 choices a side): iterations
  and time to the tolerance per step size, and the fixed point's residual
  (each side's policy against softmax(q / lam + magnet));
- with the foe fixed (one choice) MMD gives search.update's closed form;
- per_type == per_pair with one type a side (Search._solve both ways);
- with CKPT: self-play, the searcher joint (the foe's types its true sets
  in every other search): each foe type's root log-probs against the real
  foe actor's at the same moment (KL by choice labels, as check_search),
  `differs`, MMD's iterations, and the time per search.

    python tests/py/check_joint.py [CKPT] [SEARCHES] [BATTLES] [BUDGET]
"""

import sys
import tempfile
import time

import numpy as np
import torch

from ataraxos.rl.train import magnet
from ataraxos.search import Search, SearchConfig, mmd, update


def random_game(rng, a=3, b=3, C=9):
    n1, n2 = rng.integers(2, C + 1, a), rng.integers(2, C + 1, b)
    g1 = np.arange(C)[None] < n1[:, None]
    g2 = np.arange(C)[None] < n2[:, None]
    Q = rng.uniform(-1, 1, (1, a, b, C, C)) * (g1[:, None, :, None] & g2[None, :, None, :])
    m1 = np.log(rng.dirichlet(np.ones(C), a))
    m2 = np.log(rng.dirichlet(np.ones(C), b))
    return Q, g1, g2, m1, m2


def residual(l, q, lam, m, g):
    x = np.where(g, q / lam + m, -1e9)
    x = x - np.logaddexp.reduce(x, -1, keepdims=True)
    return float(np.abs(np.exp(l) - np.exp(x)).max())


def check_mmd(c):
    rng = np.random.default_rng(1)
    lam = c.alpha + c.beta
    games = [random_game(rng) for _ in range(20)]
    for scale in (1.0, 0.5, 0.2):
        its, res, secs = [], [], []
        for Q, g1, g2, m1, m2 in games:
            a, b = g1.shape[0], g2.shape[0]
            tic = time.perf_counter()
            l1, l2, q1, q2, it = mmd(Q, np.full(a, 1 / a), np.full(b, 1 / b), m1[None], m2[None], lam, lam,
                                     g1[None], g2[None], eta=scale / lam, iters=20000, tol=1e-6)
            secs.append(time.perf_counter() - tic)
            its.append(it)
            res.append(max(residual(l1[0], q1[0], lam, m1, g1), residual(l2[0], q2[0], lam, m2, g2)))
        print(f"mmd eta {scale}/lam: iterations median {int(np.median(its))} max {max(its)} "
              f"(20000 = not converged), {np.mean(secs) * 1e3:.1f} ms a solve, residual max {max(res):.2e}")


def check_closed_form(c):
    rng = np.random.default_rng(2)
    lam = c.alpha + c.beta
    worst = 0.0
    for _ in range(50):
        n = int(rng.integers(2, 10))
        q = rng.uniform(-1, 1, n)
        logp = torch.log_softmax(torch.randn(9), -1)
        logp[n:] = -1e9
        logp = torch.log_softmax(logp, -1)
        choices = torch.tensor([1 | (k + 1) << 2 for k in range(4)] + [2 | (k + 2) << 2 for k in range(5)],
                               dtype=torch.uint8)
        ref = update(q, logp, choices, n, c)
        legal = torch.arange(9) < n
        rho = magnet(choices[None].long(), legal[None], c.magnet)[0]
        m1 = ((c.alpha * rho + c.beta * logp) / lam).double().numpy()
        Q = np.zeros((1, 1, 1, 9, 9))
        Q[0, 0, 0, :n, 0] = q
        g2 = np.arange(9)[None] < 1
        l1, _, _, _, _ = mmd(Q, np.ones(1), np.ones(1), m1[None, None], np.zeros((1, 1, 9)), lam, lam,
                             legal.numpy()[None, None], g2[None], iters=20000, tol=1e-9)
        worst = max(worst, float(np.abs(np.exp(l1[0, 0]) - ref.exp().double().numpy()).max()))
    print(f"foe fixed vs the closed form: max |p - p_closed| {worst:.2e}")


def check_modes(c):
    rng = np.random.default_rng(3)
    lam = c.alpha + c.beta
    worst = 0.0
    for _ in range(20):
        Q, g1, g2, m1, m2 = random_game(rng, 1, 1)
        out = []
        for mode in ("per_type", "per_pair"):
            s = Search.__new__(Search)
            s.cfg = SearchConfig(joint=True, joint_mode=mode, mmd_iters=20000, mmd_tol=1e-9)
            x = dict(A=1, B=1, m1=m1, m2=m2, g1=g1, g2=g2, total=Q[0] * 3, count=np.full(Q[0].shape, 3.0),
                     l1=None, l2=None)
            s.cfg.shrink = 0.0
            s._solve(x, lam, lam)
            out.append(np.exp(s._marginal(x, "l1", "m1", "g1")[0]))
        worst = max(worst, float(np.abs(out[0] - out[1]).max()))
    print(f"per_type vs per_pair, one type a side: max |p diff| {worst:.2e}")



def check_selfplay(ckpt, searches, B, budget):
    from ataraxos._ext import ataraxos_env
    from ataraxos.eval.agents import load_policy, subset
    from ataraxos.eval.search_h2h import split_teams
    from ataraxos.nets.causal_actor import CausalActor
    from check_search import compare

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    net = load_policy(ckpt, dev)
    pool, games = split_teams("teams/selfplay/oak1500.txt", tempfile.mkdtemp(prefix="check_joint_"))
    cfg = SearchConfig(joint=True, joint_budget=budget, depth=10)
    search = Search(net, [pool], dev, cfg, threads=4)
    env = ataraxos_env.SelfPlay(B, seed=3, teams=[games], max_turns=300, shards=min(B, 16), threads=4,
                                state=net.c.state)
    env.set_keep_after(True)
    ours, theirs = CausalActor(net, B, dev), CausalActor(net, B, dev)
    stats = {True: ([], [], []), False: ([], [], [])}
    iters, secs = [], []
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
            truth = (done // len(rows)) % 2 == 0
            check = {}
            tic = time.perf_counter()
            p, _, _ = search.run(env, ours, o, logp, truth=truth, check=check)
            secs.append((time.perf_counter() - tic) / len(rows))
            iters += check["iters"]
            lay = check["layout"]
            for i in range(len(rows)):
                b = int(o["battle"][i])
                first, A, Bt = int(lay["first"][i]), int(lay["a"][i]), int(lay["b"][i])
                for y in range(first + A, first + A + Bt):
                    if not lay["decides"][y]:
                        continue
                    if b not in foe:
                        stats[truth][2].append("type decides, real doesn't")
                        continue
                    lq = torch.from_numpy(check["types"][y]).float().to(foe[b][0].device)
                    cmp = compare(*foe[b], lq, lay["dec_request"][y], lay["choices"][y], int(lay["n"][y]))
                    if cmp is None:
                        stats[truth][2].append("choices differ")
                        continue
                    stats[truth][0].append(cmp[0])
                    stats[truth][1].append(cmp[1])
            picks[rows] = p
            done += len(rows)
        obs = env.step(picks)
    st = search.stats
    print(f"{done} searches, budget {budget}: {np.mean(secs):.2f} s a search; differs {st['differs']} of "
          f"{st['rollouts']} rollouts; MMD iterations median {int(np.median(iters))} max {max(iters)} "
          f"(cap {cfg.mmd_iters}); times {', '.join(f'{k} {v:.1f}s' for k, v in search.times.items())}")
    for truth in (True, False):
        kl, agree, skipped = stats[truth]
        if not kl:
            continue
        kl = np.array(kl)
        print(f"foe types ({'true sets' if truth else 'pool guesses'}): {len(kl)}, KL mean {kl.mean():.4f} "
              f"median {np.median(kl):.4f} p90 {np.quantile(kl, 0.9):.4f}, argmax agrees {np.mean(agree):.3f}; "
              f"skipped {dict(zip(*np.unique(skipped, return_counts=True)))}")


def main():
    c = SearchConfig(joint=True)
    check_mmd(c)
    check_closed_form(c)
    check_modes(c)
    if len(sys.argv) > 1:
        check_selfplay(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40,
                       int(sys.argv[3]) if len(sys.argv) > 3 else 4, int(sys.argv[4]) if len(sys.argv) > 4 else 512)


if __name__ == "__main__":
    main()
