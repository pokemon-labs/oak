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


def probe_obs(rng, D=24, lines=40):
    """A random batch with every index in range (check_gpu's): line counts
    past max_lines, masked lines (KIND 0), all choice kinds."""
    lens = rng.integers(0, lines, D)
    T = int(lens.sum())
    hi = [43, 256, 17, 17, 166, 166, 152, 300, 400, 400, 256]
    tokens = np.stack([rng.integers(0, h, T) for h in hi], 1).astype(np.int16)
    request = rng.integers(0, 17, (D, ataraxos_net.REQUEST_FIELDS)).astype(np.int16)
    request[:, 0] = rng.integers(0, 3, D)
    state = rng.integers(0, 17, (D, ataraxos_net.STATE_FIELDS)).astype(np.int16)
    n = rng.integers(1, ataraxos_net.CHOICES + 1, D).astype(np.uint8)
    moves = [(s << 2) | 1 for s in range(5)]
    switches = [(s << 2) | 2 for s in range(1, 7)]
    choices = rng.choice(moves + switches, (D, ataraxos_net.CHOICES)).astype(np.uint8)
    return {"tokens": tokens, "offsets": np.r_[0, np.cumsum(lens)].astype(np.int64), "request": request,
            "state": state, "choices": choices, "n": n}


def check_gpu(net, seed=0, tol=2e-3):
    """The GPU forward against the CPU's on random batches (random caches,
    positions before and past the window, a prefill): (ok, the max
    difference over each output's scale, or why the GPU is off).
    Leaves the net on the GPU only when ok."""
    rng = np.random.default_rng(seed)
    c = net.config
    R, L, d = c["window"] - 1, c["causal_depth"], c["d"]
    worst = 0.0
    for step in range(3):
        obs = probe_obs(rng)
        D = len(obs["n"])
        S = 2 * D + 4
        k = rng.standard_normal((L, S, R, d)).astype(np.float32)
        v = rng.standard_normal((L, S, R, d)).astype(np.float32)
        keys = rng.permutation(S)[:D].astype(np.int64)
        t = rng.integers(0, 3 * R + 2, D).astype(np.int64)
        st = obs["state"] if c["state"] else None
        out = {}
        for gpu in (False, True):
            why = net.set_gpu(gpu)
            if why:
                return False, why
            kk, vv = k.copy(), v.copy()
            res = net.act(obs["tokens"], obs["offsets"], obs["request"], st, obs["choices"], obs["n"], keys, t,
                          kk, vv, step != 2)
            out[gpu] = (res, kk, vv)
        (r0, k0, v0), (r1, k1, v1) = out[False], out[True]
        def rel(a, b):  # max |a - b| over a's scale
            return float(np.abs(a - b).max() / max(1.0, float(np.abs(a).max())))

        diffs = [rel(k0, k1), rel(v0, v1)]
        if r0 is not None:
            fin = np.isfinite(r0[0])
            if not (fin == np.isfinite(r1[0])).all():
                net.set_gpu(False)
                return False, "legal sets differ"
            diffs += [rel(np.where(fin, r0[0], 0), np.where(fin, r1[0], 0)), rel(r0[1], r1[1])]
        worst = max(worst, max(float(x) for x in diffs))
    ok = worst <= tol
    net.set_gpu(ok)
    return ok, worst
