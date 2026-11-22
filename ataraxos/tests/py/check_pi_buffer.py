"""Checks NumpyBuffer (ataraxos/pi/fast.py) against train.Buffer: random
self-play on the perfect-info env, both buffers fed the same rows, ends and
fake log-probs / values, and every `take` compared (the same trajectories,
as sets of (rows, player, reward); the same rows, n, choices, log-probs,
picks and values per trajectory). Also with two envs through Envs (battle
ids offset), as --pingpong feeds the buffer; and NumpyBuffer(gather=True)
(--take_gather) against the default NumpyBuffer, field for field, with
search in training's columns. CPU only, a few seconds.

    python tests/py/check_pi_buffer.py [BATTLES] [TAKES] [STEPS_PER_TAKE] [drop]
"""

import sys

import numpy as np

from ataraxos._ext import ataraxos_pi
from ataraxos.pi.fast import Envs, NumpyBuffer
from ataraxos.pi.net import CHOICES
from ataraxos.pi.train import Buffer


def random_picks(rng, obs):
    n = np.maximum(obs["n"].astype(np.int64), 1)
    return (rng.random(n.shape) * n).astype(np.uint8)


def canon(d):
    """Each trajectory as (its rows' contents, player, reward), sorted."""
    out = []
    for g in range(len(d.length)):
        s, l = int(d.start[g]), int(d.length[g])
        sl = slice(s, s + l)
        rows = d.row[sl]
        key = (int(d.player[g]), int(d.reward[g]),
               d.rows["mon"][rows].tobytes(), d.rows["hp"][rows].tobytes(), d.n[sl].tobytes(),
               d.choices[sl].tobytes(), d.logp[sl].tobytes(), d.pick[sl].tobytes(), d.value[sl].tobytes())
        out.append(key)
    return sorted(out)


def run(battles, takes, steps, at_cap, pingpong):
    rng = np.random.default_rng(1)
    sizes = [battles // 2, battles - battles // 2] if pingpong else [battles]
    envs = [ataraxos_pi.SelfPlay(s, 3 + i, max_turns=60, threads=1) for i, s in enumerate(sizes)]
    env = Envs(envs, sizes)
    a, b, c = Buffer(at_cap), NumpyBuffer(at_cap), NumpyBuffer(at_cap, gather=True)
    obs = [env.shift(i, e.reset()) for i, e in enumerate(envs)]
    for o in obs:
        a.end(o)
        b.end(o)
        c.end(o)
    total = 0
    for t in range(takes):
        for _ in range(steps):
            for i, e in enumerate(envs):
                o = obs[i]
                picks = random_picks(rng, o)
                logp = rng.standard_normal((len(picks), 2, CHOICES)).astype(np.float32)
                value = rng.standard_normal((len(picks), 3)).astype(np.float32)
                a.add(o, picks, logp, value)
                R = len(picks)
                extra = dict(searched=(rng.random(R) < 0.2).astype(np.uint8),
                             post=rng.random((R, 2, CHOICES)).astype(np.float32),
                             swdl=rng.random((R, 3)).astype(np.float32))
                b.add(o, picks, logp, value, extra)
                c.add(o, picks, logp, value, extra)
                o = env.shift(i, e.step(picks))
                obs[i] = o
                a.end(o)
                b.end(o)
                c.end(o)
        ga, gb = a.games(), b.games()
        da, db, dc = a.take(), b.take(), c.take()
        fields = ("row", "player", "side", "n", "choices", "logp", "pick", "value", "start", "length", "reward",
                  "searched", "post", "swdl")
        same = all(np.array_equal(getattr(db, f), getattr(dc, f)) for f in fields) and \
            all(np.array_equal(db.rows[k], dc.rows[k]) for k in db.rows) and b.size == c.size
        if not same:
            print(f"take {t}: gather differs from the default numpy take")
            return False
        ca, cb = canon(da), canon(db)
        ok = ca == cb and a.size == b.size
        total += len(da.length)
        print(f"{'pingpong ' if pingpong else ''}{at_cap} take {t}: {len(da.length)} vs {len(db.length)} trajectories, "
              f"games {ga} vs {gb}, kept rows {a.size} vs {b.size}, gather equal: {'ok' if ok else 'DIFFERENT'}")
        if not ok:
            return False
    return total > 0


def main():
    battles = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    takes = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    caps = [sys.argv[4]] if len(sys.argv) > 4 else ["draw", "drop"]
    ok = all(run(battles, takes, steps, c, p) for c in caps for p in (False, True))
    print("all ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
