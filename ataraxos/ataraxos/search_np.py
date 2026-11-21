"""The search's numpy pieces (moved out of ataraxos/search.py 2026-11-21,
unchanged, so the pkmn.ai bundle's torch-free search, ataraxos/arena/search.py,
shares them): SearchConfig, MMD, the cells' means and gather_rows.
ataraxos/search.py re-exports them."""

import warnings
from dataclasses import dataclass

import numpy as np

@dataclass
class SearchConfig:
    worlds: int = 0      # history samples per root; 0: from the budget
    reps: int = 2        # rollouts per (world, choice)
    depth: int = 10      # turns before the leaf
    budget: int = 128    # rollouts per root, with worlds 0
    alpha: float = 0.002  # the magnet's coefficient
    beta: float = 0.02    # the net policy's
    magnet: str = "attack_switch"
    # the joint search
    joint: bool = False
    joint_mode: str = "per_type"  # or "per_pair"
    types_a: int = 3       # ours, A0 our real team
    types_b: int = 3       # the foe's
    joint_budget: int = 512  # rollouts per root over all rounds
    rounds: int = 4
    floor: float = 0.25    # the uniform share of each round's cell sampling
    shrink: float = 1.0    # pseudo-counts toward a cell's row and column means
    foe_beta: float = -1.0  # the foe's beta; < 0: ours
    eta: float = 0.0       # MMD's step; 0: 1 (extragradient), else lam / 4
    extra: bool = True     # extragradient MMD (plain MMD cycles at big steps)
    mmd_iters: int = 2000
    mmd_tol: float = 1e-5
    act_rows: int = 1024   # the rollouts' decisions per net call (the activations' memory)
    graphs: bool = False   # the rollouts' actor as one compiled step with CUDA graphs, padded to act_rows rows


def mmd(Q, w1, w2, m1, m2, lam1, lam2, legal1, legal2, l1=None, l2=None, eta=None, iters=2000, tol=1e-4,
        extra=True):
    """MMD on both sides of G Bayesian games to the fixed point: Q [G, a, b,
    N, M] (ours, means), type weights w1 [a], w2 [b], magnets' log-probs m1
    [G, a, N], m2 [G, b, M] (rho^alpha pi^beta over lam), regularization
    lam1, lam2, legal masks like m1, m2. Starts from log-policies l1, l2
    (default the magnets). Returns (l1, l2, q1, q2, iterations): q1 [G, a,
    N] = sum_B w2 sum_j Q p2, q2 [G, b, M] = -sum_A w1 sum_i Q p1. Each step
    l <- (l + eta q + eta lam m) / (1 + eta lam), normalized. `extra`
    (default): extragradient (mirror-prox) steps, q read at a half step:
    plain MMD needs eta <= lam / L^2 to settle in a zero-sum game (it
    cycles at larger steps), extragradient only eta < 1 / L (L ~ max |Q|),
    with the same fixed point."""
    neg = -1e9

    def norm(l, legal):
        l = np.where(legal, l, neg)
        return l - np.logaddexp.reduce(l, -1, keepdims=True)

    m1, m2 = norm(m1, legal1), norm(m2, legal2)
    l1 = m1.copy() if l1 is None else norm(l1, legal1)
    l2 = m2.copy() if l2 is None else norm(l2, legal2)
    eta = (1.0 if extra else min(lam1, lam2) / 4) if not eta else eta
    Qw1 = Q * w1[None, :, None, None, None]
    Qw2 = Q * w2[None, None, :, None, None]

    def step(a1, a2, at1, at2):
        """MMD's step from (a1, a2) with q read at (at1, at2)."""
        q1 = np.einsum("gabij,gbj->gai", Qw2, np.exp(at2))
        q2 = -np.einsum("gabij,gai->gbj", Qw1, np.exp(at1))
        return (norm((a1 + eta * q1 + eta * lam1 * m1) / (1 + eta * lam1), legal1),
                norm((a2 + eta * q2 + eta * lam2 * m2) / (1 + eta * lam2), legal2))

    it = 0
    for it in range(1, iters + 1):
        p1, p2 = np.exp(l1), np.exp(l2)
        h1, h2 = step(l1, l2, l1, l2)
        n1, n2 = step(l1, l2, h1, h2) if extra else (h1, h2)
        d = max(np.abs(np.exp(n1) - p1).max(), np.abs(np.exp(n2) - p2).max())
        l1, l2 = n1, n2
        if d < tol:
            break
    p1, p2 = np.exp(l1), np.exp(l2)
    q1 = np.einsum("gabij,gbj->gai", Qw2, p2)
    q2 = -np.einsum("gabij,gai->gbj", Qw1, p1)
    return l1, l2, q1, q2, it


def cell_means(total, count, legal, shrink):
    """Q [a, b, N, M] from the cells' value sums and counts: an empty cell
    reads the mean of its row's and its column's means in its (A, B) block
    (one of them if the other is empty, else the block's, else 0), and
    every cell is shrunk toward that by `shrink` pseudo-counts."""
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nanmean of two NaNs
        rs, rc = total.sum(3, keepdims=True), count.sum(3, keepdims=True)
        cs, cc = total.sum(2, keepdims=True), count.sum(2, keepdims=True)
        bs, bc = total.sum((2, 3), keepdims=True), count.sum((2, 3), keepdims=True)
        block = np.where(bc > 0, bs / bc, 0.0)
        row = np.where(rc > 0, rs / rc, np.nan)
        col = np.where(cc > 0, cs / cc, np.nan)
        prior = np.nanmean(np.stack(np.broadcast_arrays(row, col)), 0)
        prior = np.where(np.isnan(prior), block, prior)
        q = (total + shrink * prior) / (count + shrink) if shrink > 0 else np.where(count > 0, total / count, prior)
    return np.where(legal, q, 0.0)


def gather_rows(obs, rows, keys=("request", "state", "choices", "n")):
    """Rows `rows` of a decisions dict (tokens by offsets)."""
    off = obs["offsets"]
    parts = [obs["tokens"][off[r]:off[r + 1]] for r in rows]
    lens = np.array([len(x) for x in parts], np.int64)
    out = {k: obs[k][rows] for k in keys if k in obs}
    out["tokens"] = np.concatenate(parts) if len(parts) else obs["tokens"][:0]
    out["offsets"] = np.r_[0, np.cumsum(lens)].astype(np.int64)
    return out
