"""Checks the GPU forward (src/net_metal.mm, set_gpu) against the CPU one
(src/net.hpp) on self-play decisions, and times both (2026-11-28). No torch:
macOS CI runs it on the real net.bin (.github/workflows/ataraxos-arena.yml).

    python tests/py/check_net_gpu.py NET.BIN [--battles 256] [--steps 150] [--seed 1] \
        [--teams teams/selfplay/oak1500.txt] [--threads 3] [--tol 2e-3]

1. arena.actor.check_gpu's random batches (every index, the cache ring, a
   prefill).
2. self-play: a CPU actor and a GPU actor on the same decisions (the CPU's
   picks are played): legal log-probs and value logits within --tol every
   step, the same legal sets, the caches within --tol at the end.
Exits 2 (skipped) when there is no usable Metal device (GitHub's macOS VMs
show a paravirtual one), 1 on a kernel compile error."""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ataraxos._ext import ataraxos_env  # noqa: E402
from ataraxos.arena.actor import Actor, check_gpu, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--battles", type=int, default=256)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--teams", default=os.path.join(os.path.dirname(__file__), "..", "..", "teams", "selfplay",
                                                    "oak1500.txt"))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--tol", type=float, default=2e-3)
    a = ap.parse_args()
    cpu, gpu = load(a.net, a.threads), load(a.net, a.threads)
    why = gpu.set_gpu(True)
    if why.startswith("Metal compile"):  # our kernels' bug, not the runner's
        print(f"FAILED: {why}")
        sys.exit(1)
    if why:  # no device, or one without what the kernels need (CI's VMs)
        print(f"skipped: {why}")
        sys.exit(2)
    tic = time.perf_counter()
    ok, how = check_gpu(gpu)
    print(f"random batches: {'ok' if ok else 'FAILED'}, {how} ({time.perf_counter() - tic:.1f} s)")
    assert ok, how
    B = a.battles
    ours_c = Actor(cpu, B, np.random.default_rng(a.seed))
    ours_g = Actor(gpu, B, np.random.default_rng(a.seed))
    env = ataraxos_env.SelfPlay(B, a.seed, teams=[a.teams], state=bool(cpu.config["state"]), threads=a.threads)
    obs = env.reset()
    worst, decisions, t_c, t_g = {"logp": 0.0, "value": 0.0}, 0, 0.0, 0.0
    for step in range(a.steps):
        D = len(obs["n"])
        if D == 0:
            obs = env.step(np.zeros(0, np.uint8))
            continue
        tic = time.perf_counter()
        picks, lp_c, v_c = ours_c.act(obs)
        t_c += time.perf_counter() - tic
        tic = time.perf_counter()
        _, lp_g, v_g = ours_g.act(obs)
        t_g += time.perf_counter() - tic
        legal = np.isfinite(lp_c)
        assert (np.isfinite(lp_g) == legal).all(), f"step {step}: legal sets differ"
        d_lp = float(np.abs(np.where(legal, lp_c - lp_g, 0)).max())
        d_v = float(np.abs(v_c - v_g).max())
        worst["logp"], worst["value"] = max(worst["logp"], d_lp), max(worst["value"], d_v)
        assert d_lp <= a.tol and d_v <= a.tol, f"step {step}: logp {d_lp:.2e} value {d_v:.2e}"
        decisions += D
        obs = env.step(picks)
    dk = float(np.abs(ours_c.k - ours_g.k).max())
    dv = float(np.abs(ours_c.v - ours_g.v).max())
    assert dk <= a.tol and dv <= a.tol, f"caches: k {dk:.2e} v {dv:.2e}"
    tg = gpu.timing
    print(f"self-play: {decisions} decisions over {a.steps} steps; max |d logp| {worst['logp']:.2e}, "
          f"|d value| {worst['value']:.2e}, cache k {dk:.2e} v {dv:.2e}; CPU {decisions / t_c:.0f}/s, "
          f"GPU {decisions / t_g:.0f}/s (GPU busy {tg['gpu_seconds']:.2f} of {tg['seconds']:.2f} s in the net)")
    print("ok")


if __name__ == "__main__":
    main()
