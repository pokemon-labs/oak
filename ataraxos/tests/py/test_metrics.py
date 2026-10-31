"""Checks ataraxos/rl/metrics.py on hand-built data (2026-10-25):

    ~/oak/.venv/bin/python tests/py/test_metrics.py

- explained variance: 1 for a perfect predictor, 0 for the mean, negative
  for worse;
- calibration: a calibrated predictor's error is near 0, a shifted one's
  is the shift;
- data_stats: normalized entropy 1 for uniform play whatever the number of
  legal choices; the switch rate and the switch-only share counted by
  Choice type; P1's score read by player, not by position;
- wilson: covers the score, narrows with n, matches a known value.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ataraxos.rl.buffer import Data  # noqa: E402
from ataraxos.rl import metrics  # noqa: E402

CHOICES = 9


def move(slot):
    return (slot << 2) | 1


def swap(slot):
    return (slot << 2) | 2


def make_data():
    """Two games, four trajectories: P2's first in game 0 (stored order
    mustn't matter). Decisions: uniform policies over 2, 3 or 4 legal
    choices; some switch-only."""
    rows = [  # (choices, pick)
        ([move(1), move(2), swap(2)], 2),           # can move, switches
        ([swap(2), swap(3)], 0),                    # switch only
        ([move(1), move(2), move(3), move(4)], 1),  # can move, attacks
        ([move(1), swap(2)], 0),                    # can move, attacks
        ([swap(2), swap(3), swap(4)], 1),           # switch only
        ([move(1), swap(2), swap(3)], 1),           # can move, switches
    ]
    N = len(rows)
    choices = np.zeros((N, CHOICES), np.uint8)
    logp = np.full((N, CHOICES), -np.inf, np.float32)
    n = np.zeros(N, np.uint8)
    pick = np.zeros(N, np.uint8)
    for i, (c, k) in enumerate(rows):
        choices[i, :len(c)] = c
        logp[i, :len(c)] = -np.log(len(c))
        n[i], pick[i] = len(c), k
    return Data(tokens=np.zeros((0, 1), np.int16), offsets=np.zeros(N + 1, np.int64),
                request=np.zeros((N, 1), np.int16), choices=choices, n=n, pick=pick, logp=logp,
                value=np.zeros((N, 3), np.float32), start=np.array([0, 2, 3, 5]), length=np.array([2, 1, 2, 1]),
                reward=np.array([-1, 1, 0, 0]), player=np.array([1, 0, 0, 1]))


def main():
    rng = np.random.default_rng(0)
    t = rng.normal(size=1000)
    assert abs(metrics.explained_variance(t, t) - 1) < 1e-12
    assert abs(metrics.explained_variance(np.full_like(t, t.mean()), t)) < 1e-12
    assert metrics.explained_variance(-t, t) < 0

    p = rng.random(200_000)
    z = (rng.random(len(p)) < p).astype(float)
    ece, table = metrics.calibration(p, z)
    assert ece < 0.01 and len(table) == 10 and sum(r[2] for r in table) == len(p)
    ece, _ = metrics.calibration(np.clip(p + 0.05, 0, 1), z)
    assert 0.03 < ece < 0.06

    d = make_data()
    s = metrics.data_stats(d, np.zeros((len(d.n), 3)), np.zeros(len(d.n)))
    assert abs(s["entropy_norm"] - 1) < 1e-6, s["entropy_norm"]
    assert abs(s["entropy_norm_move"] - 1) < 1e-6 and abs(s["entropy_norm_switch_only"] - 1) < 1e-6
    assert s["switch_only_frac"] == 2 / 6
    assert s["switch_rate"] == 2 / 4
    assert s["p1_score"] == (1 + 0.5) / 2  # P1's rewards: 1 and 0
    assert s["value_mean"] == 0.5 and abs(s["value_brier"] - 0.25 * 3 / 6) < 1e-12  # z: 0,0,1,½,½,½
    assert s["length_p50"] == 1.5

    lo, hi = metrics.wilson(0.6, 512)
    assert lo < 0.6 < hi and abs(lo - 0.5568) < 1e-3 and abs(hi - 0.6417) < 1e-3
    lo2, hi2 = metrics.wilson(0.6, 4096)
    assert hi2 - lo2 < hi - lo
    print("ok")


if __name__ == "__main__":
    main()
