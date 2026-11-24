"""pi4's training env (2026-11-24, docs/pi4.md): rby4's env (ataraxos_env's
SelfPlay with `state`, light: no tokens or per-decision state) seen as the
perfect-info trainer's rows (ataraxos/pi/train.py): one row per battle
with a decision, `pi4_obs`'s inputs, both players' choices, and `step`
taking picks [R, 2] (0 where a side doesn't decide)."""

import numpy as np

from ataraxos._ext import ataraxos_env

END_KEYS = ("end_battle", "end_game", "end_result", "end_capped", "end_team", "end_prob")


class Pi4Env:
    def __init__(self, battles, seed=1, teams=(), max_turns=300, threads=8):
        self.env = ataraxos_env.SelfPlay(battles, seed, teams=list(teams), max_turns=max_turns, threads=threads,
                                         state=True)
        self.env.set_light(True)
        self.dec = None

    def _obs(self, o):
        r = self.env.pi4_obs(True)
        self.dec = r.pop("dec")
        for k in END_KEYS:
            r[k] = o[k]
        return r

    def reset(self):
        return self._obs(self.env.reset())

    def step(self, picks):
        picks = np.asarray(picks)
        out = np.zeros(self.env.decisions, np.uint8)
        for s in (0, 1):
            m = self.dec[:, s] >= 0
            out[self.dec[m, s]] = picks[m, s]
        return self._obs(self.env.step(out))

    def set_team_weights(self, w):
        self.env.set_team_weights(w)

    @property
    def team_probs(self):
        return self.env.team_probs

    @property
    def team_counts(self):
        return self.env.team_counts

    @property
    def team_names(self):
        return self.env.team_names

    @property
    def stats(self):
        return self.env.stats

    @property
    def battles(self):
        return self.env.battles
