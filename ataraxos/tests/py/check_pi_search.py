"""Checks search in training's pieces (ataraxos/pi/search.py): the C++ MMD
(ataraxos_pi.mmd) against search_np.mmd at a tight tolerance on random
games; with Q = 0 the posterior is the prior; SelfPlay.rollouts is
deterministic per seed, a new seed changes leaves, and the env's battles are
untouched; TrainSearch.run on real rows (a random net) gives normalized
legal posteriors and legal picks. CPU, seconds.

    python tests/py/check_pi_search.py [TEAM_FILE]
"""

import sys

import numpy as np
import torch

from ataraxos._ext import ataraxos_pi
from ataraxos.pi.net import CHOICES, legal_logits, make_net
from ataraxos.pi.search import LeafEval, TrainSearch
from ataraxos.search_np import cell_means, mmd

C = CHOICES


def random_games(rng, G):
    Q = rng.uniform(-1, 1, (G, C, C)).astype(np.float32)
    n1, n2 = rng.integers(1, C + 1, G).astype(np.uint8), rng.integers(1, C + 1, G).astype(np.uint8)
    m1 = np.log(rng.dirichlet(np.ones(C), G)).astype(np.float32)
    m2 = np.log(rng.dirichlet(np.ones(C), G)).astype(np.float32)
    legal1, legal2 = np.arange(C)[None] < n1[:, None], np.arange(C)[None] < n2[:, None]
    Q = np.where(legal1[:, :, None] & legal2[:, None, :], Q, 0).astype(np.float32)
    return Q, m1, m2, n1, n2, legal1, legal2


def main():
    team = sys.argv[1] if len(sys.argv) > 1 else "teams/selfplay/oak150.txt"
    rng = np.random.default_rng(0)
    ok = True
    Q, m1, m2, n1, n2, legal1, legal2 = random_games(rng, 200)
    for lam in (0.03, 0.2):
        l1, l2, its = ataraxos_pi.mmd(Q, m1, m2, n1, n2, lam, 1.0, 20000, 1e-7, 4)
        r1, r2, *_ = mmd(Q[:, None, None].astype(np.float64), np.ones(1), np.ones(1),
                         m1[:, None].astype(np.float64), m2[:, None].astype(np.float64), lam, lam,
                         legal1[:, None], legal2[:, None], iters=20000, tol=1e-7)
        e = max(np.abs((np.exp(l1) - np.exp(r1[:, 0])) * legal1).max(),
                np.abs((np.exp(l2) - np.exp(r2[:, 0])) * legal2).max())
        print(f"mmd lam {lam}: max |cpp - numpy| {e:.1e} (iters max {its.max()})")
        ok &= e < 1e-4
    l1, l2, _ = ataraxos_pi.mmd(np.zeros_like(Q), m1, m2, n1, n2, 0.03, 1.0, 2000, 1e-6, 4)
    pr1 = np.where(legal1, np.exp(m1), 0)
    e = np.abs(np.exp(l1) * legal1 - pr1 / pr1.sum(-1, keepdims=True)).max()
    print(f"mmd Q = 0: max |post - prior| {e:.1e}")
    ok &= e < 1e-5

    env = ataraxos_pi.SelfPlay(64, 3, teams=[team], max_turns=300, threads=2)
    obs = env.reset()
    for _ in range(5):
        n = np.maximum(obs["n"].astype(np.int64), 1)
        obs = env.step((rng.random(n.shape) * n).astype(np.uint8))
    before = {k: v.copy() for k, v in obs.items()}
    K = 32
    b = np.repeat(obs["battle"], K)
    n = np.repeat(np.maximum(obs["n"], 1), K, 0)
    p = (rng.random(n.shape) * n).astype(np.uint8)
    seeds = rng.integers(0, 2**62, len(b), dtype=np.int64).astype(np.uint64)
    a, a2 = env.rollouts(b, p, seeds), env.rollouts(b, p, seeds)
    c = env.rollouts(b, p, seeds + np.uint64(1))
    same = all(np.array_equal(a[k], a2[k]) for k in a)
    differ = float((np.abs(a["hp"] - c["hp"]).sum((1, 2)) > 0).mean())
    print(f"rollouts: same seed same leaves {same}, a new seed moves HP in {differ:.0%} of leaves, "
          f"done {a['done'].mean():.1%}")
    ok &= same and differ > 0.05
    # the env is unchanged: the same picks give the same next rows as a fresh replay would
    picks = np.zeros((len(obs["n"]), 2), np.uint8)
    nxt = env.step(picks)
    ok &= len(nxt["n"]) == len(before["n"])

    net = make_net("pi_tiny").eval()
    ts = TrainSearch(net, "cpu", False, budget=16, threads=2, seed=1)
    obs = nxt
    with torch.no_grad():
        _, policy = net(torch.from_numpy(obs["mon"]).long(), torch.from_numpy(obs["hp"]),
                        torch.from_numpy(obs["active"]))
        x, _ = legal_logits(policy, torch.from_numpy(obs["policy"]), torch.from_numpy(obs["n"]))
        logp = torch.log_softmax(x, -1).numpy()
    picks = np.zeros((len(obs["n"]), 2), np.uint8)
    cols, st = ts.run(env, obs, picks, logp, 0.5)
    s = cols["searched"].astype(bool)
    nn = np.maximum(obs["n"].astype(np.int64), 1)
    legal = np.arange(C)[None, None] < nn[:, :, None]
    post = cols["post"][s]
    good = (np.allclose(post.sum(-1), 1, atol=1e-5) and not (post * ~legal[s]).any()
            and (picks < nn).all() and np.allclose(cols["swdl"][s].sum(-1), 1, atol=1e-4))
    print(f"TrainSearch: {s.sum()} of {len(s)} rows searched, posteriors legal and normalized {good}, "
          f"KL to the prior {st['search_kl']:.3f}")
    ok &= good and s.any()
    # --search_solve cpp against numpy, on the same rollouts (real leaves, a random net's values)
    for solve in ("numpy", "cpp"):
        t = TrainSearch(net, "cpu", False, budget=32, threads=2, seed=5, solve=solve)
        pk = np.zeros((len(obs["n"]), 2), np.uint8)
        tok = t.begin(env, obs, pk, logp, 0.5)
        if solve == "numpy":
            ref_tok = tok
            ref = t.finish(tok)[0]
        else:
            tok.update({k: ref_tok[k] for k in ("roots", "cells", "done", "result", "handle", "nr", "legal",
                                                 "prior", "logp")})
            got = t.finish(tok)[0]
    e = max(np.abs(ref["post"] - got["post"]).max(), np.abs(ref["swdl"] - got["swdl"]).max())
    print(f"solve cpp vs numpy: max diff {e:.1e}")
    ok &= e < 1e-4 and np.array_equal(ref["searched"], got["searched"])

    # the same with random leaf values (so Q, the cell means and MMD move), at the array level
    S, B = 300, 64
    nr = np.stack([rng.integers(1, C + 1, S), rng.integers(1, C + 1, S)], 1)
    cells = np.stack([(rng.random((S, B)) * nr[:, None, 0]).astype(np.int64),
                      (rng.random((S, B)) * nr[:, None, 1]).astype(np.int64)], -1)
    wdl = rng.dirichlet(np.ones(3), S * B).astype(np.float32)
    legal = np.arange(C)[None, None] < nr[:, :, None]
    m = np.where(legal, np.log(rng.dirichlet(np.ones(C), (S, 2))), -1e9).astype(np.float32)
    flat = (np.repeat(np.arange(S), B) * C + cells[..., 0].ravel()) * C + cells[..., 1].ravel()
    count = np.bincount(flat, minlength=S * C * C).reshape(S, 1, C, C).astype(np.float64)
    cl = (legal[:, 0, :, None] & legal[:, 1, None, :])[:, None]
    Q = cell_means(np.bincount(flat, (wdl[:, 2] - wdl[:, 0]).astype(np.float64), S * C * C).reshape(S, 1, C, C),
                   count, cl, 1.0)[:, 0]
    W = np.stack([cell_means(np.bincount(flat, wdl[:, k].astype(np.float64), S * C * C).reshape(S, 1, C, C),
                             count, cl, 1.0)[:, 0] for k in range(3)], -1)
    n1, n2 = nr[:, 0].astype(np.uint8), nr[:, 1].astype(np.uint8)
    l1, l2, _ = ataraxos_pi.mmd(Q.astype(np.float32), np.ascontiguousarray(m[:, 0]), np.ascontiguousarray(m[:, 1]),
                                n1, n2, 0.03, 1.0, 2000, 1e-4, 2)
    p1, p2 = np.exp(l1) * legal[:, 0], np.exp(l2) * legal[:, 1]
    sw = np.einsum("sijk,si,sj->sk", W, p1 / p1.sum(-1, keepdims=True), p2 / p2.sum(-1, keepdims=True))
    c1, c2, csw, _ = ataraxos_pi.solve(cells.astype(np.uint8), wdl, n1, n2, np.ascontiguousarray(m[:, 0]),
                                       np.ascontiguousarray(m[:, 1]), 1.0, 0.03, 1.0, 2000, 1e-4, 2)
    e = max(np.abs((np.exp(c1) - np.exp(l1)) * legal[:, 0]).max(), np.abs((np.exp(c2) - np.exp(l2)) * legal[:, 1]).max(),
            np.abs(csw - sw).max())
    kl = float((p1 * (l1 - np.where(legal[:, 0], m[:, 0], 0))).sum(-1).mean())
    print(f"solve cpp vs numpy, random leaves: max diff {e:.1e} (posteriors moved: KL to the prior {kl:.3f})")
    ok &= e < 1e-4 and kl > 0.01

    # --search_plan cpp: the cells' frequencies against (1 - floor) prior x prior + floor uniform
    S, B, floor = 4, 200000, 0.25
    pr = rng.dirichlet(np.ones(C), (S, 2)).astype(np.float32)
    na, nb = np.array([9, 3, 1, 5], np.uint8), np.array([2, 9, 4, 1], np.uint8)
    for side, nn_ in ((0, na), (1, nb)):
        pr[:, side] *= np.arange(C)[None] < nn_[:, None]
        pr[:, side] /= pr[:, side].sum(-1, keepdims=True)
    cells = ataraxos_pi.plan(pr, na, nb, B, floor, 123, 2)
    worst = 0.0
    for r in range(S):
        f = np.zeros((C, C))
        np.add.at(f, (cells[r, :, 0], cells[r, :, 1]), 1)
        f /= B
        uni = np.outer(np.arange(C) < na[r], np.arange(C) < nb[r]) / (int(na[r]) * int(nb[r]))
        want = (1 - floor) * np.outer(pr[r, 0], pr[r, 1]) + floor * uni
        worst = max(worst, np.abs(f - want).max())
    print(f"plan cpp: max |freq - expected| {worst:.4f} over {B} draws a root")
    ok &= worst < 0.005

    # --search_direct: rollouts_into gives rollouts' leaves
    K = len(b)
    out = dict(mon=np.full((K + 3, 2, 6, ataraxos_pi.MON_FIELDS), 7, np.int16),
               hp=np.full((K + 3, 2, 6), 7, np.float32), active=np.full((K + 3, 2, ataraxos_pi.ACTIVE_DIM), 7, np.float32),
               done=np.zeros(K, np.uint8), result=np.zeros(K, np.int8))
    obs3 = env.step(np.zeros((len(nxt["n"]), 2), np.uint8))
    b = np.repeat(obs3["battle"], 8)
    n = np.repeat(np.maximum(obs3["n"], 1), 8, 0)
    p = (rng.random(n.shape) * n).astype(np.uint8)
    seeds = rng.integers(0, 2**62, len(b), dtype=np.int64).astype(np.uint64)
    K = len(b)
    out = {k: v[:K] if k in ("done", "result") else v for k, v in out.items()}
    out["done"], out["result"] = np.zeros(K, np.uint8), np.zeros(K, np.int8)
    a = env.rollouts(b, p, seeds)
    env.rollouts_into(b, p, seeds, out["mon"], out["hp"], out["active"], out["done"], out["result"])
    same = all(np.array_equal(a[k], out[k][:K]) for k in ("mon", "hp", "active", "done", "result"))
    print(f"rollouts_into vs rollouts: equal {same}")
    ok &= same

    # --search_value_only: PiNet.value against forward's value
    with torch.no_grad():
        x = [torch.from_numpy(obs3[k]) for k in ("mon", "hp", "active")]
        e = (net.value(x[0].long(), x[1], x[2]) - net(x[0].long(), x[1], x[2])[0]).abs().max().item()
    print(f"value_only vs forward: max diff {e:.1e}")
    ok &= e < 1e-6

    # --search_leaf async|graphs (+ compile, value_only) against eager, on the GPU
    if torch.cuda.is_available():
        gnet = make_net("pi").cuda().eval()
        leaves = env.rollouts(b, p, seeds)
        ref = LeafEval(gnet, "cuda", True, "eager").collect((0, K, leaves), leaves["done"], leaves["result"])
        for mode, comp, vo in (("async", False, False), ("graphs", False, False), ("graphs", True, True)):
            le = LeafEval(gnet, "cuda", True, mode, comp, vo)
            got = None
            for _ in range(2):  # the second call replays the captured graph
                got = le.collect(le.launch(0, K, leaves), leaves["done"], leaves["result"])
            e = np.abs(got - ref).max()
            print(f"leaf {mode} compile {comp} value_only {vo} vs eager: max diff {e:.1e}")
            ok &= e < 2e-2  # bf16; compile reorders
    print("ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
