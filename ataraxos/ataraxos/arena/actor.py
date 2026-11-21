"""CausalActor (ataraxos/nets/causal_actor.py) on the C++ net (src/net.hpp,
module ataraxos_net), numpy only (2026-11-21, for pkmn.ai's bundle).

The same cache semantics: each layer's keys and values per (battle,
player) slot, a ring of the last window - 1 positions (`k`, `v` [layers,
slots, window - 1, d] float32), each slot's next position `t` and its
`game` (a new game restarts the slot at position 0). The search reads and
copies these arrays directly, as it does the torch actor's tensors.

`act` returns (picks u8 [D], log-probs f32 [D, CHOICES] (-inf past n),
value logits f32 [D, 3]), numpy throughout; picks are drawn from `rng`
(numpy's Generator), or the argmax with `greedy`."""

import numpy as np

from ataraxos._ext import ataraxos_net


def log_softmax(x):
    m = x.max(-1, keepdims=True)
    z = np.log(np.exp(x - m).sum(-1, keepdims=True))
    return x - m - z


def sample(logp, rng):
    """One draw per row from exp(logp)."""
    p = np.exp(logp.astype(np.float64))
    c = np.cumsum(p, -1)
    u = rng.random(len(p))[:, None] * c[:, -1:]
    return np.minimum((u >= c).sum(-1), (p > 0).sum(-1) - 1).astype(np.uint8)


class Actor:
    def __init__(self, net, battles: int, rng=None):
        self.net = net
        c = net.config
        self.c = c
        self.R = c["window"] - 1
        self.slots = battles * 2
        self.k = np.zeros((c["causal_depth"], self.slots, self.R, c["d"]), np.float32)
        self.v = np.zeros_like(self.k)
        self.t = np.zeros(self.slots, np.int64)
        self.game = np.full(self.slots, -1, np.int64)
        self.rng = rng if rng is not None else np.random.default_rng()

    def positions(self, obs):
        """The rows' cache keys and positions; marks their games."""
        key = obs["battle"].astype(np.int64) * 2 + obs["player"]
        game = obs["game"].astype(np.int64)
        new = self.game[key] != game
        self.game[key] = game
        return key, np.where(new, 0, self.t[key])

    def run(self, obs, key, t, decide=True):
        """The net on rows `obs` at cache keys `key`, positions `t`; advances t."""
        out = self.net.act(obs["tokens"], obs["offsets"], obs["request"], obs.get("state") if self.c["state"] else None,
                           obs["choices"], obs["n"], key, t, self.k, self.v, decide)
        self.t[key] = t + 1
        return out

    def act(self, obs, greedy=False):
        D = len(obs["n"])
        if D == 0:
            return np.zeros(0, np.uint8), np.zeros((0, ataraxos_net.CHOICES), np.float32), np.zeros((0, 3), np.float32)
        key, t = self.positions(obs)
        logits, value = self.run(obs, key, t)
        logp = log_softmax(logits)
        picks = logp.argmax(-1).astype(np.uint8) if greedy else sample(logp, self.rng)
        return picks, logp, value


def load(path, threads=None):
    """The net from export.py's file, its loops on `threads` threads."""
    if threads:
        ataraxos_net.set_threads(int(threads))
    return ataraxos_net.Net(path)
