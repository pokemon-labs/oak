"""Diagnostics for log.jsonl (2026-10-25): what an iteration's data says about
the value net, the policy and the games, computed from what the loop already
has (the actor's log-probs and values, the returns and advantages), so they
cost no forward passes.

- value: explained variance against the returns (PPO's usual one; partly
  circular, since the returns are built from the same values) and against
  the game's outcome (a tie 1/2), the Brier score against the outcome beside
  a constant predictor's (the outcome's variance), and the calibration error
  over ten bins of the predicted P(win);
- policy: the data policy's entropy over log(legal choices), so 2-choice and
  9-choice decisions read on one scale, split into decisions with a move
  legal and decisions with only switches (faint and Baton Pass
  replacements); how often it switches when it could attack;
- games: trajectory length percentiles (decisions per player), P1's score
  (a side bias check), the advantages' spread;
- eval: a Wilson interval for a score over n games.
"""

import math

import numpy as np

MOVE, SWITCH = 1, 2  # libpkmn's choice type, bits 0-1


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    var = float(np.var(target))
    return float("nan") if var == 0 else 1.0 - float(np.var(target - pred)) / var


def calibration(p: np.ndarray, z: np.ndarray, bins: int = 10):
    """Expected calibration error (count-weighted |mean p - mean z| over
    equal-width bins) and the bins as [mean p, mean z, count]."""
    idx = np.minimum((p * bins).astype(np.int64), bins - 1)
    count = np.bincount(idx, minlength=bins)
    sp = np.bincount(idx, p, minlength=bins)
    sz = np.bincount(idx, z, minlength=bins)
    used = count > 0
    mp, mz = sp[used] / count[used], sz[used] / count[used]
    ece = float((count[used] * np.abs(mp - mz)).sum() / max(1, len(p)))
    table = [[round(float(a), 4), round(float(b), 4), int(c)] for a, b, c in zip(mp, mz, count[used])]
    return ece, table


def data_stats(data, ret, adv) -> dict:
    """`data` a buffer.Data, `ret` and `adv` its returns and advantages
    (tensors or arrays; W/D/L returns [N, 3], advantages [N]). The value is
    read as P(win) with a tie worth 1/2."""
    ret, adv = np.asarray(ret, np.float64), np.asarray(adv, np.float64)
    N = len(data.n)
    row = np.repeat(np.arange(len(data.length)), data.length)
    v = data.value.astype(np.float64)  # (loss, draw, win) logits
    q = np.exp(v - v.max(-1, keepdims=True))
    q /= q.sum(-1, keepdims=True)
    p = q[:, 2] + q[:, 1] / 2
    ret = ret[:, 2] + ret[:, 1] / 2
    z = (data.reward[row].astype(np.float64) + 1) / 2
    ece, table = calibration(p, z)

    n = data.n.astype(np.int64)
    legal = np.arange(data.choices.shape[1])[None] < n[:, None]
    logp = np.where(legal, data.logp.astype(np.float64), 0.0)
    ent = -(np.exp(logp) * logp * legal).sum(-1) / np.log(n)
    kind = data.choices & 3
    can_move = ((kind == MOVE) & legal).any(-1)
    picked = kind[np.arange(N), data.pick.astype(np.int64)]

    out = {
        "value_ev": explained_variance(p, ret),
        "value_ev_outcome": explained_variance(p, z),
        "value_brier": float(np.mean((p - z) ** 2)),
        "value_brier_base": float(np.var(z)),
        "value_ece": ece,
        "value_calibration": table,
        "value_mean": float(p.mean()),
        "adv_mean": float(adv.mean()),
        "adv_std": float(adv.std()),
        "entropy_norm": float(ent.mean()),
        "entropy_norm_move": float(ent[can_move].mean()) if can_move.any() else float("nan"),
        "entropy_norm_switch_only": float(ent[~can_move].mean()) if (~can_move).any() else float("nan"),
        "switch_only_frac": float((~can_move).mean()),
        "switch_rate": float((picked[can_move] == SWITCH).mean()) if can_move.any() else float("nan"),
        "legal_mean": float(n.mean()),
    }
    for q in (10, 50, 90):
        out[f"length_p{q}"] = float(np.percentile(data.length, q))
    p1 = data.player == 0
    if p1.any():
        out["p1_score"] = float((data.reward[p1] + 1).mean() / 2)
    return out


def wilson(score: float, n: int, z: float = 1.96):
    """The 95% Wilson interval for a score (a tie 1/2) over n games."""
    if n == 0:
        return float("nan"), float("nan")
    c = (score + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h
