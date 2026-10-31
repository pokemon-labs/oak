"""Checks the team draws (src/env.hpp) and Exp3 (ataraxos/rl/teams.py):

    ~/oak/.venv/bin/python tests/py/test_teams.py TEAM_FILE ...

- the env's draws follow the weights it's given: each End's probabilities
  are the weights' at its draw, the draw counts are 2 per game started, and
  the drawn teams' frequencies match the weights (a chi-square bound);
- a zero weight is never drawn;
- Exp3 against itself on a symmetric zero-sum matrix game with a known
  Nash (rock-paper-scissors plus dominated teams): its draws' empirical
  distribution gets near the Nash, and the dominated teams lose weight;
  with gamma annealed (eta coupled), the draws and the softmax average get
  nearer, the dominated teams' share falling below the fixed gamma's floor;
- Exp3's state round-trips, the annealed gamma included.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ataraxos._ext import ataraxos_env  # noqa: E402
from ataraxos.rl.teams import Exp3  # noqa: E402


def env_draws(files, battles=512, steps=400, seed=1):
    env = ataraxos_env.SelfPlay(battles, seed, teams=files, max_turns=30, threads=8)
    K = len(env.team_probs)
    assert len(env.team_names) == K and np.allclose(env.team_probs, 1 / K)
    rng = np.random.default_rng(seed)
    w = rng.random(K) ** 3
    w[:K // 10] = 0  # never drawn
    env.set_team_weights(w)
    p = w / w.sum()
    assert np.allclose(env.team_probs, p)
    obs = env.reset()  # draws every battle's first game under w
    seen = np.zeros(K)
    games = 0
    for _ in range(steps):
        obs = env.step((rng.random(len(obs["n"])) * obs["n"]).astype(np.uint8))
        t, q = obs["end_team"], obs["end_prob"]
        assert np.allclose(q, p[t])
        np.add.at(seen, t.ravel(), 1)
        games += len(t)
    counts = env.team_counts
    assert counts.sum() == 2 * (battles + games), (counts.sum(), battles, games)
    assert (counts[:K // 10] == 0).all()
    n = counts.sum()
    live = p > 0
    chi2 = (((counts - n * p)[live]) ** 2 / (n * p[live])).sum()
    dof = live.sum() - 1
    assert chi2 < dof + 6 * np.sqrt(2 * dof), (chi2, dof)
    try:
        env.set_team_weights(np.zeros(K))
        raise AssertionError("no error")
    except ValueError:
        pass
    print(f"env: {K} teams, {n} draws over {games} ended games, chi2 {chi2:.0f} on {dof} dof: ok")


def play(exp3, M, rounds, rng, batch=64):
    """Exp3 against itself on M (P(row beats column)), `batch` games per
    weight update, as the env's steps batch them. Returns the draw counts."""
    K = len(M)
    counts = np.zeros(K)
    for _ in range(rounds // batch):
        p = exp3.probs()
        t = rng.choice(K, size=(batch, 2), p=p)
        np.add.at(counts, t.ravel(), 1)
        win = rng.random(batch) < M[t[:, 0], t[:, 1]]
        tie = rng.random(batch) < 0.02
        result = np.where(tie, 0, np.where(win, 1, -1))
        exp3.update(t, p[t], result)
    return counts


def exp3_nash(rounds=400_000, seed=0):
    # P(row beats column): rock-paper-scissors by margins 0.7 / 0.3 between
    # teams 0-2, and teams 3-5 beaten by everything at 0.8.
    K = 6
    M = np.full((K, K), 0.5)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        M[i, j], M[j, i] = 0.7, 0.3
    for d in range(3, 6):
        for i in range(K):
            if i != d:
                M[i, d], M[d, i] = 0.8, 0.2
    rng = np.random.default_rng(seed)
    exp3 = Exp3(K, gamma=0.05, eta=0.002)
    counts = play(exp3, M, rounds, rng)
    emp = counts / counts.sum()
    # The Nash is uniform on 0-2; exploration keeps gamma / K on each team.
    nash = np.r_[np.full(3, 1 / 3), np.zeros(3)] * (1 - exp3.gamma) + exp3.gamma / K
    gap = np.abs(emp - nash).max()
    exploit = (M @ emp).max() - 0.5  # the best team's edge against the average
    assert gap < 0.05 and exploit < 0.03, (emp, exploit)
    print(f"exp3: empirical {np.round(emp, 3)}, max gap to Nash {gap:.3f}, "
          f"best response edge {exploit:.3f}: ok")

    # annealed: gamma 0.3 -> 0.01, eta coupled
    nash = np.r_[np.full(3, 1 / 3), np.zeros(3)]
    fixed = Exp3(K, gamma=0.3)
    fixed_emp = play(fixed, M, rounds, np.random.default_rng(seed + 1))
    fixed_emp /= fixed_emp.sum()
    ann = Exp3(K, gamma=0.3, gamma_floor=0.01, gamma_tau=rounds / 50, gamma_decay=0.5)
    emp = play(ann, M, rounds, np.random.default_rng(seed + 1))
    emp /= emp.sum()
    assert abs(ann.gamma - max(0.01, 0.3 * (1 + ann.rounds / ann.gamma_tau) ** -0.5)) < 1e-12
    assert ann.gamma < 0.05 and abs(ann.eta - ann.gamma / K) < 1e-15
    q = ann.q_avg()
    dominated_fixed, dominated_ann, dominated_q = fixed_emp[3:].sum(), emp[3:].sum(), q[3:].sum()
    assert dominated_ann < dominated_fixed and dominated_q < dominated_ann, (dominated_fixed, dominated_ann, dominated_q)
    assert dominated_ann < 0.3 / 2 and np.abs(q - nash).max() < 0.05, (emp, q)
    exploit = (M @ emp).max() - 0.5
    assert exploit < 0.03, exploit
    print(f"exp3 annealed: gamma {ann.gamma:.3f}, dominated share: fixed gamma 0.3 {dominated_fixed:.3f}, "
          f"annealed draws {dominated_ann:.3f}, softmax average {dominated_q:.3f}; "
          f"q_avg gap to Nash {np.abs(q - nash).max():.3f}: ok")

    again = Exp3(K, gamma=0.3, gamma_floor=0.01, gamma_tau=rounds / 50, gamma_decay=0.5)
    again.load(ann.state())
    assert again.gamma == ann.gamma and np.array_equal(again.q_sum, ann.q_sum) and np.allclose(again.probs(), ann.probs())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        env_draws(sys.argv[1:])
    exp3_nash()
