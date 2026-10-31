"""Exp3 over the team pool (draft, 2026-10-23): the team-building half of
Ataraxos, amortized into a fixed pool (the user's, 2026-10-23). The setup
network becomes one weight per team.

Both sides of every self-play game draw from one distribution p, mixed with
uniform exploration: p = (1 - gamma) softmax(log_w) + gamma / K. A game is
two Exp3 rounds, one per side: P1's team gets s = (result + 1) / 2 (a tie is
1/2), P2's 1 - s, each as the importance-weighted estimate s / p_i with p_i
the probability the team was drawn with (the env reports it), so feedback
that arrives after p has moved stays unbiased. It's Exp3 against itself, an
adversarial bandit, so the move net changing under it is fine.

The weights persist over the whole run. The draws are counted too (the
env's team_counts): their empirical distribution is the Nash estimate, as a
no-regret learner's average play is (Exp3 against itself in a symmetric
zero-sum game: every team's payoff against the average is at most 1/2 plus
the average regret). The current p can cycle; the counts are what converges.

Annealed exploration (the user's, 2026-10-25). A fixed gamma costs about
(e - 1) gamma of average regret forever (Auer et al.'s bound), so gamma
decays over Exp3 rounds (two per game; kept in the checkpoint):
gamma_t = max(floor, gamma (1 + t / tau)^-decay), tau 0 keeping it fixed.
With eta 0, eta_t = gamma_t / k, Auer's coupling, which bounds one update's
step in log w by 1 (s / p <= k / gamma_t); a fixed eta lets a rare team's
lucky win jump by up to eta k / gamma_t. The draw counts include the
uniform part, so `q_sum` also accumulates the softmax alone, weighted by
rounds: its normalization is a second Nash estimate without the
exploration's floor.
"""

import numpy as np


class Exp3:
    def __init__(self, k: int, gamma: float = 0.1, eta: float = 0.0, gamma_floor: float = 0.0,
                 gamma_tau: float = 0.0, gamma_decay: float = 0.5):
        """`eta` 0 takes gamma_t / k; `gamma_tau` 0 keeps gamma fixed."""
        self.k = k
        self.gamma0, self.gamma_floor, self.gamma_tau, self.gamma_decay = gamma, gamma_floor, gamma_tau, gamma_decay
        self.eta0 = eta
        self.log_w = np.zeros(k)
        self.q_sum = np.zeros(k)
        self.rounds = 0

    @property
    def gamma(self) -> float:
        if self.gamma_tau <= 0:
            return self.gamma0
        return max(self.gamma_floor, self.gamma0 * (1 + self.rounds / self.gamma_tau) ** -self.gamma_decay)

    @property
    def eta(self) -> float:
        return self.eta0 if self.eta0 > 0 else self.gamma / self.k

    def softmax(self) -> np.ndarray:
        q = np.exp(self.log_w - self.log_w.max())
        return q / q.sum()

    def probs(self) -> np.ndarray:
        g = self.gamma
        return (1 - g) * self.softmax() + g / self.k

    def q_avg(self) -> np.ndarray:
        """The softmax's average over rounds: the Nash estimate without exploration."""
        total = self.q_sum.sum()
        return self.q_sum / total if total > 0 else np.full(self.k, 1 / self.k)

    def update(self, team: np.ndarray, prob: np.ndarray, result: np.ndarray):
        """Ended games: team i32 [E, 2], prob f64 [E, 2] (each side's draw
        probability), result [E] P1's (1, 0, -1)."""
        if len(result) == 0:
            return
        s = (result.astype(np.float64) + 1) / 2
        self.update_rounds(team, prob, np.stack([s, 1 - s], 1))

    def update_rounds(self, team: np.ndarray, prob: np.ndarray, reward: np.ndarray):
        """Rounds of any shape: each team drawn, the probability it was drawn
        with and its reward in [0, 1]. The exploiter run's (2026-10-30) has
        one a game, the learner's side only."""
        if team.size == 0:
            return
        self.q_sum += team.size * self.softmax()
        np.add.at(self.log_w, team.ravel(), self.eta * (reward / prob).ravel())
        self.log_w -= self.log_w.max()  # only differences matter
        self.rounds += team.size

    def state(self) -> dict:
        return dict(log_w=self.log_w.copy(), q_sum=self.q_sum.copy(), rounds=self.rounds, gamma=self.gamma,
                    eta=self.eta)

    def load(self, state: dict):
        self.log_w = np.asarray(state["log_w"], dtype=np.float64).copy()
        self.rounds = int(state["rounds"])
        self.q_sum = np.asarray(state.get("q_sum", np.zeros(self.k)), dtype=np.float64).copy()


class Pairs:
    """Every ended game's result per ordered (P1 team, P2 team) pair, from
    P1's side: wins, ties, losses and ties by max_turns (the user's,
    2026-10-26: explicit W/D/L for every pair). Sparse, since a dense
    table for a 4,863-team pool would be ~280 MB."""

    def __init__(self):
        self.counts = {}  # (a, b) -> [win, tie, loss, capped]

    def add(self, team: np.ndarray, result: np.ndarray, capped: np.ndarray):
        for (a, b), r, c in zip(team.tolist(), result.tolist(), capped.tolist()):
            row = self.counts.setdefault((a, b), [0, 0, 0, 0])
            row[3 if c else 1 - r] += 1

    def arrays(self) -> dict:
        keys = np.array(list(self.counts), dtype=np.int32).reshape(-1, 2)
        wdlc = np.array(list(self.counts.values()), dtype=np.int64).reshape(-1, 4)
        return {"pair_a": keys[:, 0], "pair_b": keys[:, 1], "pair_wdlc": wdlc}

    def state(self) -> dict:
        return self.arrays()

    def load(self, state: dict):
        self.counts = {(int(a), int(b)): [int(x) for x in row]
                       for a, b, row in zip(state["pair_a"], state["pair_b"], state["pair_wdlc"])}


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def summary(exp3: Exp3, counts: np.ndarray, top: int = 5) -> dict:
    """Log lines: the effective number of teams (exp entropy) under the
    current p and under the empirical draws, and the empirical top teams."""
    p = exp3.probs()
    total = max(1, int(counts.sum()))
    emp = counts / total
    order = np.argsort(-emp)[:top]
    q = exp3.q_avg()
    q_order = np.argsort(-q)[:top]
    return {"teams/p_eff": float(np.exp(entropy(p))), "teams/p_max": float(p.max()),
            "teams/gamma": exp3.gamma, "teams/eta": exp3.eta, "teams/rounds": exp3.rounds,
            "teams/nash_eff": float(np.exp(entropy(emp))), "teams/draws": total,
            "teams/nash_top": [[int(i), float(emp[i])] for i in order],
            "teams/q_avg_eff": float(np.exp(entropy(q))),
            "teams/q_avg_top": [[int(i), float(q[i])] for i in q_order]}
