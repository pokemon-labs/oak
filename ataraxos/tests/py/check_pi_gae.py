"""Checks returns_and_advantages on the GPU (the pi trainer's --gae gpu)
against the CPU: random self-play on the perfect-info env into NumpyBuffer,
fake values, and both devices' returns and advantages compared, then the
advantage filter's threshold. Needs CUDA; seconds.

    python tests/py/check_pi_gae.py [BATTLES] [STEPS] [TEAM_FILE]
"""

import sys

import numpy as np
import torch

from ataraxos._ext import ataraxos_pi
from ataraxos.pi.fast import NumpyBuffer
from ataraxos.pi.net import CHOICES
from ataraxos.rl.buffer import returns_and_advantages


def main():
    battles = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    team = sys.argv[3:4] or ["teams/selfplay/oak150.txt"]
    if not torch.cuda.is_available():
        sys.exit("needs CUDA")
    rng = np.random.default_rng(1)
    env = ataraxos_pi.SelfPlay(battles, 3, teams=team, max_turns=100, threads=2)
    buf = NumpyBuffer("draw")
    obs = env.reset()
    buf.end(obs)
    for _ in range(steps):
        n = np.maximum(obs["n"].astype(np.int64), 1)
        picks = (rng.random(n.shape) * n).astype(np.uint8)
        buf.add(obs, picks, rng.standard_normal((len(picks), 2, CHOICES)).astype(np.float32),
                rng.standard_normal((len(picks), 3)).astype(np.float32))
        obs = env.step(picks)
        buf.end(obs)
    data = buf.take()
    print(f"{len(data.length)} trajectories, {len(data.n)} decisions, longest {data.length.max()}")
    worst = 0.0
    for td, gae in ((0.6, 0.5), (1.0, 1.0), (0.0, 0.0)):
        r0, a0 = returns_and_advantages(data, td, gae)
        r1, a1 = returns_and_advantages(data, td, gae, "cuda")
        e = max((r0 - r1.cpu()).abs().max().item(), (a0 - a1.cpu()).abs().max().item())
        q0, q1 = a0.abs().quantile(0.5).item(), a1.abs().quantile(0.5).item()
        print(f"td {td} gae {gae}: max |cpu - gpu| {e:.2e}, threshold {q0:.6f} vs {q1:.6f}")
        worst = max(worst, e, abs(q0 - q1))
    print("ok" if worst < 1e-5 else "MISMATCH")
    sys.exit(0 if worst < 1e-5 else 1)


if __name__ == "__main__":
    main()
