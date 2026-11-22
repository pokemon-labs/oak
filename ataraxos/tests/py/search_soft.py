"""How far cheap 1-round joint MMD search moves rby4's policy, per (alpha, beta).

Self-play with rby4 (EMA), both players on the net. At a random share of rows
where the deciding player has 2+ choices, the joint search runs with 1 type a
side and the foe's true sets (truth=True): near perfect info. Each root is
searched at each cheap budget twice (two seeds) and once at a reference budget.
With 1 round, the rollouts are sampled from the magnet (+ floor) and don't
depend on lambda, so each root's cell means are re-solved by MMD for every
(alpha, beta) here.

    python tests/py/search_soft.py CKPT ROOTS [BATTLES] [CHEAP,BUDGETS] [REF] [OUT.npz]
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
from ataraxos.rl.train import magnet
from ataraxos.search import Search, SearchConfig, mmd

CONFIGS = [(0.002, 0.02), (0.01, 0.1), (0.02, 0.2), (0.05, 0.5), (0.1, 1.0), (0.05, 0.1)]


def sub(obs, rows):
    off = obs["offsets"]
    lengths = off[rows + 1] - off[rows]
    idx = np.repeat(off[rows] - np.cumsum(np.r_[0, lengths[:-1]]), lengths) + np.arange(lengths.sum())
    out = {k: obs[k][rows] for k in ("battle", "game", "player", "request", "choices", "n", "state") if k in obs}
    out["tokens"] = obs["tokens"][idx]
    out["offsets"] = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    return out


def roots_of(check):
    """Per root: (Q [C, C], our lp, foe lp, our choices, foe choices, our n, foe n)."""
    lay, out = check["layout"], []
    for i, Q in enumerate(check["Q"]):
        y = int(lay["first"][i])
        out.append((Q[0, 0], check["types"][y], check["types"][y + 1], lay["choices"][y], lay["choices"][y + 1],
                    int(lay["n"][y]), int(lay["n"][y + 1])))
    return out


def solve(root, alpha, beta, kind="attack_switch"):
    Q, lp1, lp2, ch1, ch2, n1, n2 = root
    C = Q.shape[-1]
    lam = alpha + beta
    g1, g2 = np.arange(C) < n1, np.arange(C) < n2
    rho = magnet(torch.from_numpy(np.stack([ch1, ch2])).long(), torch.from_numpy(np.stack([g1, g2])), kind).double().numpy()
    m1 = (alpha * rho[0] + beta * lp1) / lam
    m2 = (alpha * rho[1] + beta * lp2) / lam
    l1, l2, q1, q2, it = mmd(Q[None, None, None], np.ones(1), np.ones(1), m1[None, None], m2[None, None], lam, lam,
                             g1[None, None], g2[None, None], eta=1.0, iters=4000, tol=1e-7, extra=True)
    return l1[0, 0], l2[0, 0], q1[0, 0], it


def norm(lp, n):
    lp = np.where(np.arange(len(lp)) < n, lp, -np.inf)
    return lp - np.logaddexp.reduce(lp[:n])


def kl(lp, lq, n):
    p = np.exp(lp[:n])
    return float((p * (lp[:n] - lq[:n])).sum())


def ent(lp, n):
    p = np.exp(lp[:n])
    return float(-(p * lp[:n]).sum())


def main():
    ckpt, R = sys.argv[1], int(sys.argv[2])
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    cheap = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "64,128").split(",")]
    ref = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    out_path = sys.argv[6] if len(sys.argv) > 6 else None
    torch.set_num_threads(12)
    torch.manual_seed(1)
    dev = torch.device("cpu")
    net = load_policy(ckpt, dev)
    pool, games = split_teams("teams/selfplay/oak1500.txt", tempfile.mkdtemp(prefix="soft_search_"))
    base = dict(joint=True, joint_mode="per_type", types_a=1, types_b=1, rounds=1, depth=4, floor=0.25,
                shrink=1.0, act_rows=2048)
    searches = {b: Search(net, [pool], dev, SearchConfig(joint_budget=b, **base), threads=8, amp=False, seed=b)
                for b in sorted(set(cheap + [ref]))}
    env = ataraxos_env.SelfPlay(B, seed=7, teams=[games], max_turns=300, shards=min(B, 16), threads=8,
                                state=net.c.state)
    env.set_keep_after(True)
    actor = CausalActor(net, B, dev, amp=False)
    rng = np.random.default_rng(5)
    runs = {}  # (budget, rep) -> list of roots
    obs = env.reset()
    done, tic = 0, time.perf_counter()
    while done < R:
        picks, logp, _ = actor.act(obs)
        picks = picks.cpu().numpy().astype(np.uint8) if torch.is_tensor(picks) else picks
        rows = np.flatnonzero((obs["n"] >= 2) & (rng.random(len(obs["n"])) < 0.15))[: R - done]
        if len(rows):
            o, lp = sub(obs, rows), logp[rows]
            for b in sorted(set(cheap + [ref])):
                for rep in (range(2) if b in cheap else range(1)):
                    check = {}
                    searches[b].run(env, actor, o, lp, seed=1000 * done + 17 * rep + b, truth=True, check=check)
                    runs.setdefault((b, rep), []).extend(roots_of(check))
            done += len(rows)
            print(f"{done}/{R} roots, {time.perf_counter() - tic:.0f} s", flush=True)
        obs = env.step(picks)

    print(f"\n{done} roots (the deciding player 2+ choices; foe 1 choice in "
          f"{sum(r[6] < 2 for r in runs[(ref, 0)])}); budgets {cheap} x2 seeds, ref {ref}; 1 round, depth 4")
    # Q noise: q1 under the prior foe policy, two seeds
    for b in cheap:
        d = []
        for ra, rb in zip(runs[(b, 0)], runs[(b, 1)]):
            n1, n2 = ra[5], ra[6]
            p2 = np.exp(norm(ra[2], n2))[:n2]
            qa, qb = ra[0][:n1, :n2] @ p2, rb[0][:n1, :n2] @ p2
            d.append((qa - qb) / np.sqrt(2))
        d = np.concatenate(d)
        print(f"budget {b}: per-action q noise (std, two seeds) {d.std():.3f}")
    hdr = f"{'alpha':>6} {'beta':>5} {'lam':>5} | {'budget':>6} {'KL(post|prior)':>14} {'H prior':>7} {'H post':>6} " \
          f"{'argmax chg':>10} {'KL seedA|B':>10} {'KL(ref|post)':>12} {'KL(ref|prior)':>13}"
    print(hdr)
    summary = {}
    for alpha, beta in CONFIGS:
        refs = [solve(r, alpha, beta) for r in runs[(ref, 0)]]
        for b in cheap + [ref]:
            sa = [solve(r, alpha, beta) for r in runs[(b, 0)]]
            sb = [solve(r, alpha, beta) for r in runs[(b, 1)]] if b in cheap else sa
            m = {k: [] for k in ("klp", "hp", "hq", "chg", "ab", "kref", "kprior")}
            for r, x, y, z in zip(runs[(b, 0)], sa, sb, refs):
                for side in (0, 1):
                    n = r[5 + side]
                    if n < 2:
                        continue
                    prior = norm(r[1 + side], n)
                    post, post2, rp = norm(x[side], n), norm(y[side], n), norm(z[side], n)
                    m["klp"].append(kl(post, prior, n))
                    m["hp"].append(ent(prior, n))
                    m["hq"].append(ent(post, n))
                    m["chg"].append(np.argmax(post[:n]) != np.argmax(prior[:n]))
                    m["ab"].append(kl(post, post2, n))
                    m["kref"].append(kl(rp, post, n))
                    m["kprior"].append(kl(rp, prior, n))
            s = {k: float(np.mean(v)) for k, v in m.items()}
            summary[(alpha, beta, b)] = s
            print(f"{alpha:6.3f} {beta:5.2f} {alpha + beta:5.3f} | {b:6d} {s['klp']:14.3f} {s['hp']:7.3f} {s['hq']:6.3f} "
                  f"{s['chg']:10.3f} {(s['ab'] if b in cheap else float('nan')):10.3f} {s['kref']:12.3f} {s['kprior']:13.3f}")
    if out_path:
        np.savez(out_path, summary=np.array([(k, v) for k, v in summary.items()], dtype=object))


if __name__ == "__main__":
    main()
