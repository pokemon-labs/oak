"""Checks the torch-free net (src/net.hpp, ataraxos/arena/) against torch's
(2026-11-21). Needs torch, ataraxos_env and ataraxos_net built; CI runs it
on the real checkpoint (.github/workflows/ataraxos-arena.yml).

    python tests/py/check_net.py CKPT [--battles 64] [--steps 200] [--seed 1] \
        [--teams teams/selfplay/oak1500.txt] [--threads 4] [--tol 2e-3]

1. export: ataraxos/arena/export.py's torch-free reader gives every EMA
   tensor exactly as torch.load does, and the config the net was built
   with.
2. self-play: the torch CausalActor (CPU, float32) and arena.actor.Actor
   act on the same decisions (torch's picks are played), from a reset
   through `--steps` steps (many games, so new-game cache restarts and
   positions past the window are covered): legal log-probs and value
   logits within `--tol` at every step, the same legal set, and the same
   argmax wherever torch's top two differ by more than `--tol`.
3. prefill: decide=False writes the same cache as decide=True.
4. the caches: every slot's keys and values within `--tol` at the end.
Prints the max differences and both sides' decisions per second."""

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
from ataraxos.arena.actor import Actor, load  # noqa: E402
from ataraxos.eval.agents import load_policy  # noqa: E402
from ataraxos.nets.causal_actor import make_actor  # noqa: E402
from ataraxos.nets.causal_net import net_config  # noqa: E402


def check_export(ckpt, path):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ours = export.load(ckpt)
    for k, v in ck["ema"].items():
        a = ours["ema"][k]
        assert a.shape == tuple(v.shape), (k, a.shape, v.shape)
        assert np.array_equal(a, v.float().numpy()), k
    assert set(ours["ema"]) == set(ck["ema"])
    export.main([ckpt, path])
    c = net_config(ck["config"])
    got = load(path).config
    for k in ("d", "heads", "window", "ff", "field_d", "line_ff", "max_lines", "max_pos", "depth", "update_depth",
              "causal_depth", "state", "features", "q_head"):
        assert got[k] == getattr(c, k), (k, got[k], getattr(c, k))
    print(f"export: {len(ck['ema'])} tensors equal, config {got}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--battles", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--teams", default=os.path.join(os.path.dirname(__file__), "..", "..", "teams", "selfplay",
                                                    "oak1500.txt"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tol", type=float, default=2e-3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    path = os.path.join(tempfile.mkdtemp(), "net.bin")
    check_export(a.ckpt, path)

    net_t = load_policy(a.ckpt, "cpu")
    net_c = load(path, a.threads)
    B = a.battles
    ref = make_actor(net_t, B, "cpu", amp=False)
    ours = Actor(net_c, B, np.random.default_rng(a.seed))
    env = ataraxos_env.SelfPlay(B, a.seed, teams=[a.teams], state=net_t.c.state, threads=a.threads)
    obs = env.reset()
    worst = {"logp": 0.0, "value": 0.0}
    decisions, t_ref, t_ours, flips, games = 0, 0.0, 0.0, 0, 0
    for step in range(a.steps):
        D = len(obs["n"])
        if D == 0:
            obs = env.step(np.zeros(0, np.uint8))
            continue
        tic = time.perf_counter()
        picks, lp_t, v_t = ref.act(obs)
        t_ref += time.perf_counter() - tic
        lp_t, v_t = lp_t.numpy(), v_t.numpy()
        if step == 3:  # prefill: the same cache writes without the decision block
            k0, v0, t0, g0 = ours.k.copy(), ours.v.copy(), ours.t.copy(), ours.game.copy()
            key, t = ours.positions(obs)
            ours.run(obs, key, t, decide=False)
            kp, vp = ours.k.copy(), ours.v.copy()
            ours.k[...], ours.v[...], ours.t[...], ours.game[...] = k0, v0, t0, g0
        tic = time.perf_counter()
        _, lp_c, v_c = ours.act(obs)
        t_ours += time.perf_counter() - tic
        if step == 3:
            assert np.array_equal(kp, ours.k) and np.array_equal(vp, ours.v), "prefill's cache differs"
        legal = np.arange(lp_t.shape[1])[None] < obs["n"][:, None].astype(np.int64)
        assert (np.isfinite(lp_c) == legal).all(), "legal sets differ"
        d_lp = np.abs(np.where(legal, lp_t, 0) - np.where(legal, lp_c, 0)).max()
        d_v = np.abs(v_t - v_c).max()
        worst["logp"] = max(worst["logp"], float(d_lp))
        worst["value"] = max(worst["value"], float(d_v))
        assert d_lp <= a.tol and d_v <= a.tol, f"step {step}: logp {d_lp:.2e} value {d_v:.2e}"
        top = np.sort(np.where(legal, lp_t, -np.inf), -1)
        clear = (top[:, -1] - top[:, -2] > a.tol) | (obs["n"] < 2)
        flips += int(((lp_t.argmax(-1) != lp_c.argmax(-1)) & clear).sum())
        decisions += D
        obs = env.step(picks)
        games += len(obs["end_battle"])
    assert flips == 0, f"{flips} argmax flips"
    used = ref.t.numpy()[: 2 * B] > 0
    dk = np.abs(ref.k.float().numpy()[:, : 2 * B][:, used] - ours.k[:, used]).max()
    dv = np.abs(ref.v.float().numpy()[:, : 2 * B][:, used] - ours.v[:, used]).max()
    assert dk <= a.tol and dv <= a.tol, f"caches: k {dk:.2e} v {dv:.2e}"
    print(f"self-play: {decisions} decisions over {a.steps} steps, {games} games ended; max |d logp| "
          f"{worst['logp']:.2e}, |d value| {worst['value']:.2e}, cache k {dk:.2e} v {dv:.2e}; "
          f"torch {decisions / t_ref:.0f}/s, C++ {decisions / t_ours:.0f}/s ({a.threads} threads)")
    print("ok")


if __name__ == "__main__":
    main()
