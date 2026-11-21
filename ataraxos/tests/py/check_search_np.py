"""Checks the torch-free search (ataraxos/arena/search.py) against the torch
one (ataraxos/search.py) on the same roots (2026-11-21). Needs torch.

    python tests/py/check_search_np.py CKPT [--battles 16] [--warmup 12] [--seed 1] \
        [--budget 128] [--depth 4] [--teams teams/selfplay/oak1500.txt] [--threads 4]

Both real actors play the same self-play games for `--warmup` steps (torch's
picks), then both searches run on that step's player-0 roots with the same
seed, vanilla and joint per_type (3x3 types, 4 rounds: applemagpie's, at a
smaller budget). Picks inside the rollouts are greedy on both sides (the
searches' only draws besides the env's own, which the seed fixes), so the
rollouts are the same unless a near-tie's argmax flips on float rounding.
Prints, per mode, the roots whose search policy (legal log-probs) and q
match within 1e-3, the max differences, and both sides' time."""

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ataraxos._ext import ataraxos_env  # noqa: E402
from ataraxos.arena import export  # noqa: E402
from ataraxos.arena import search as np_search  # noqa: E402
from ataraxos.arena.actor import Actor, load  # noqa: E402
from ataraxos.eval.agents import load_policy, subset  # noqa: E402
from ataraxos.nets import causal_actor  # noqa: E402
from ataraxos.nets.causal_actor import make_actor  # noqa: E402
from ataraxos.search import Search, SearchConfig  # noqa: E402

TOL = 1e-3


def greedy_actors():
    """Rollout picks greedy on both sides."""
    act_t = causal_actor.CausalActor.act
    causal_actor.CausalActor.act = lambda self, obs, greedy=False: act_t(self, obs, greedy=True)
    act_c = Actor.act
    Actor.act = lambda self, obs, greedy=False: act_c(self, obs, greedy=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--battles", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--teams", default=os.path.join(os.path.dirname(__file__), "..", "..", "teams", "selfplay",
                                                    "oak1500.txt"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    path = os.path.join(tempfile.mkdtemp(), "net.bin")
    export.main([a.ckpt, path])
    net_t, net_c = load_policy(a.ckpt, "cpu"), load(path, a.threads)
    B = a.battles
    real_t = make_actor(net_t, B, "cpu", amp=False)
    real_c = Actor(net_c, B)
    env = ataraxos_env.SelfPlay(B, a.seed, teams=[a.teams], state=True, threads=a.threads)
    env.set_keep_after(True)
    obs = env.reset()
    for _ in range(a.warmup):
        picks, _, _ = real_t.act(obs)
        real_c.act(obs)
        obs = env.step(picks)
    _, lp_t, _ = real_t.act(obs)
    _, lp_c, _ = real_c.act(obs)
    rows = np.flatnonzero((obs["player"] == 0) & (obs["n"] >= 2))
    assert len(rows), "no roots"
    roots = subset(obs, rows)
    lp_t, lp_c = lp_t[torch.from_numpy(rows)], lp_c[rows]
    greedy_actors()
    modes = {
        "vanilla": SearchConfig(budget=a.budget, reps=2, depth=a.depth),
        "joint": SearchConfig(joint=True, joint_mode="per_type", types_a=3, types_b=3, rounds=4,
                              joint_budget=a.budget, depth=a.depth),
    }
    for name, cfg in modes.items():
        st = Search(net_t, [a.teams], "cpu", cfg, threads=a.threads, amp=False)
        sc = np_search.Search(net_c, [a.teams], cfg, threads=a.threads)
        tic = time.perf_counter()
        _, q_t, l_t = st.run(env, real_t, roots, lp_t, seed=a.seed)
        t_t = time.perf_counter() - tic
        tic = time.perf_counter()
        _, q_c, l_c = sc.run(env, real_c, roots, lp_c, seed=a.seed)
        t_c = time.perf_counter() - tic
        l_t = l_t.numpy()
        same, worst_l, worst_q = 0, 0.0, 0.0
        for i in range(len(rows)):
            n = int(roots["n"][i])
            dl = float(np.abs(l_t[i, :n] - l_c[i, :n]).max())
            dq = float(np.abs(np.asarray(q_t[i], np.float64) - np.asarray(q_c[i], np.float64)).max())
            worst_l, worst_q = max(worst_l, dl), max(worst_q, dq)
            same += dl <= TOL and dq <= TOL
        print(f"{name}: {same}/{len(rows)} roots match within {TOL:g}; max |d logp| {worst_l:.2e}, "
              f"|d q| {worst_q:.2e}; torch {t_t:.1f} s, numpy {t_c:.1f} s; rollouts {st.stats['rollouts']} / "
              f"{sc.stats['rollouts']}, steps {st.stats['steps']} / {sc.stats['steps']}")


if __name__ == "__main__":
    main()
