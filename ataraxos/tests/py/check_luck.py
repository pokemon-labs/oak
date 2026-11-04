"""Where a turn's luck comes from (2026-11-03): self-play with one
checkpoint, and at our decisions the whole one-turn matrix. Each cell
(our choice i, the foe's choice j) gets K rollouts of one turn (the joint
search's rollouts: the foe's true sets, fresh dice and hidden counters per
rollout), valued at our next decision (P(win) - P(loss); a game that ends,
its outcome). With pi our root policy and sigma the foe's (its root policy
on its true sets), the one-step TD error's variance splits into

    ours  sum_i pi alpha_i^2          alpha_i = (Q sigma)_i - V
    blunder  sum_j sigma beta_j^2     beta_j = (pi^T Q)_j - V
    read  sum pi sigma gamma^2        gamma = Q - V - alpha - beta
    dice  sum pi sigma s^2            s^2 the cell's variance over its K

(read has the cell means' noise, sum pi sigma s^2 / K, taken out). The
opponent-averaged advantage removes blunder + read; a function of their
pick alone at most blunder. CPU by default.

    python tests/py/check_luck.py CKPT [ROOTS] [BATTLES] [K] [EVERY]
"""

import sys
import tempfile
import time

import numpy as np
import torch

from ataraxos._ext import ataraxos_env
from ataraxos.eval.agents import load_policy, subset
from ataraxos.eval.search_h2h import split_teams
from ataraxos.nets.causal_actor import CausalActor
from ataraxos.search import Search, SearchConfig, gather_rows


@torch.no_grad()
def matrices(search, real, obs, logp, K, seed):
    """The roots' one-turn matrices: per root (pi, sigma, cell sums, sums of
    squares, counts), our side's values."""
    r, dev = search.r, search.device
    D = len(obs["n"])
    keys = obs["battle"].astype(np.int64) * 2 + obs["player"]
    r.clear()
    for i in range(D):
        r.add_joint_selfplay(search._env, int(obs["battle"][i]), int(obs["player"][i]), 1, 1, 1, seed + i, True)
    lay = r.joint_layout()
    Y = len(lay["n"])
    first = lay["first"].astype(np.int64)
    n = lay["n"].astype(np.int64)
    cells = []
    for i in range(D):
        na, nb = n[first[i]], n[first[i] + 1]
        for ci in range(na):
            for cj in range(nb):
                cells.append((i, 0, 0, ci, cj, K))
    N = len(cells) * K
    actor = search._actor(N + (Y + 1) // 2)
    tkeys = 2 * N + np.arange(Y, dtype=np.int64)
    search.prefill(actor, tkeys, lay)
    real_y = np.flatnonzero(lay["real"])
    src = torch.from_numpy(keys[lay["root"][real_y].astype(np.int64)]).to(real.device)
    k = torch.from_numpy(tkeys[real_y]).to(dev)
    actor.k[:, k] = real.k[:, src].to(dev, actor.k.dtype)
    actor.v[:, k] = real.v[:, src].to(dev, actor.v.dtype)
    actor.t[k] = real.t[src].to(dev)
    actor.game[k] = 0
    # the foe's root policy (its true sets), as search_joint
    C = logp.shape[1]
    lp_t = np.full((Y, C), -np.inf)
    lp_t[:, 0] = 0.0
    dec = np.flatnonzero((lay["decides"] > 0) & (lay["real"] == 0))
    if len(dec):
        sub = {"tokens": lay["dec_tokens"], "offsets": lay["dec_offsets"], "request": lay["dec_request"],
               "choices": lay["choices"], "n": lay["n"]}
        if "dec_state" in lay:
            sub["state"] = lay["dec_state"]
        sub = gather_rows(sub, dec)
        sub["battle"] = (N + dec // 2).astype(np.uint32)
        sub["player"] = (dec % 2).astype(np.uint8)
        sub["game"] = np.zeros(len(dec), np.uint32)
        _, lp, _ = actor.act(sub)
        lp_t[dec] = lp.float().cpu().numpy()
    lp_t[real_y] = logp[lay["root"][real_y].astype(np.int64)].float().cpu().numpy()
    r.clear_rollouts()
    r.schedule(np.array(cells, np.int64).reshape(-1, 6))
    vals = r.joint_values()
    root_of = vals["root"].astype(np.int64)
    us = np.asarray(obs["player"], np.int64)[root_of]
    idx = np.arange(len(root_of), dtype=np.int64)
    ka = torch.from_numpy(tkeys[first[root_of]]).to(dev)
    kb = torch.from_numpy(tkeys[first[root_of] + 1]).to(dev)
    for dst, sk in ((torch.from_numpy(idx * 2 + us).to(dev), ka), (torch.from_numpy(idx * 2 + 1 - us).to(dev), kb)):
        actor.k[:, dst] = actor.k[:, sk]
        actor.v[:, dst] = actor.v[:, sk]
        actor.t[dst] = actor.t[sk]
        actor.game[dst] = 0
    search._rollout_loop(actor, r, seed + 7)
    vals = r.joint_values()
    out = []
    for i in range(D):
        na, nb = n[first[i]], n[first[i] + 1]
        m = vals["root"] == i
        at = (vals["action"][m], vals["foe_action"][m])
        s1, s2, c = np.zeros((na, nb)), np.zeros((na, nb)), np.zeros((na, nb))
        np.add.at(s1, at, vals["value"][m])
        np.add.at(s2, at, vals["value"][m] ** 2)
        np.add.at(c, at, 1)
        pi = np.exp(lp_t[first[i], :na] - np.logaddexp.reduce(lp_t[first[i], :na]))
        sg = np.exp(lp_t[first[i] + 1, :nb] - np.logaddexp.reduce(lp_t[first[i] + 1, :nb]))
        out.append((pi, sg, s1, s2, c, int(vals["differs"][m].sum())))
    return out


def split(pi, sg, s1, s2, c):
    Q = s1 / c
    s2c = (s2 - c * Q ** 2) / np.maximum(c - 1, 1)  # the cell's variance
    V = pi @ Q @ sg
    al = Q @ sg - V
    be = pi @ Q - V
    ga = Q - V - al[:, None] - be[None, :]
    w = pi[:, None] * sg[None, :]
    dice = float((w * s2c).sum())
    noise = float((w * s2c / c).sum())
    return dict(ours=float(pi @ al ** 2), blunder=float(sg @ be ** 2),
                read=max(0.0, float((w * ga ** 2).sum()) - noise), dice=dice)


def main():
    ckpt = sys.argv[1]
    roots = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    every = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    net = load_policy(ckpt, dev)
    pool, games = split_teams("teams/selfplay/oak1500.txt", tempfile.mkdtemp(prefix="check_luck_"))
    search = Search(net, [pool], dev, SearchConfig(joint=True), threads=4, amp=dev.type == "cuda")
    env = ataraxos_env.SelfPlay(B, seed=5, teams=[games], max_turns=300, shards=min(B, 16), threads=4,
                                state=net.c.state)
    env.set_keep_after(True)
    search._env = env
    actor = CausalActor(net, B, dev, amp=dev.type == "cuda")
    rows_out = []
    differs = 0
    tic = time.perf_counter()
    obs = env.reset()
    while len(rows_out) < roots:
        picks, logp, _ = actor.act(obs)
        picks = picks.cpu().numpy().astype(np.uint8) if torch.is_tensor(picks) else np.asarray(picks, np.uint8)
        rows = np.flatnonzero((obs["n"] >= 2) & (rng.random(len(obs["n"])) < 1 / every))
        if len(rows):
            o = subset(obs, rows)
            for pi, sg, s1, s2, c, d in matrices(search, actor, o, logp[rows], K, int(rng.integers(1 << 40))):
                differs += d
                rows_out.append((len(sg) >= 2, split(pi, sg, s1, s2, c)))
            print(f"{len(rows_out)} roots, {time.perf_counter() - tic:.0f}s", flush=True)
        obs = env.step(picks)

    def report(name, sel):
        if not sel:
            return
        tot = {k: np.mean([x[k] for x in sel]) for k in sel[0]}
        all_ = sum(tot.values())
        share = " ".join(f"{k} {v / all_:.1%}" for k, v in tot.items())
        opp = np.array([(x["blunder"] + x["read"]) / max(1e-12, sum(x.values())) for x in sel])
        print(f"{name}: {len(sel)} roots, one-step TD var {all_:.4f} (sd {np.sqrt(all_):.3f}); {share}; "
              f"foe share per root median {np.median(opp):.1%}, >50% in {np.mean(opp > 0.5):.1%}")

    print(f"K {K}, differs {differs}")
    report("all", [x for _, x in rows_out])
    report("foe has a choice", [x for f, x in rows_out if f])


if __name__ == "__main__":
    main()
