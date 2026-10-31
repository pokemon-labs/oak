"""Checks ataraxos_env.SelfPlay (src/env.hpp) under a random policy, and times it.

    ~/oak/.venv/bin/python tests/py/test_env.py [--battles N] [--steps N] [--seed N]
        [--max-turns N] [TEAM_FILE ...]

With no team files the teams are truly random.

- every decision has 2..MAX_CHOICES choices, each a valid Choice;
- offsets cover the tokens, with every token's KIND nonzero;
- a battle has at most one decision per player, of the battle's current game;
- every game a battle starts ends once, and the games' results are in range;
- the output doesn't depend on the thread count;
- a pick past its choices, or the wrong number of picks, raises;
- with --max-turns, the capped games are ties.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ataraxos._ext import ataraxos_env  # noqa: E402


def play(env, steps, seed):
    rng = np.random.default_rng(seed)
    obs = env.reset()
    yield obs
    for _ in range(steps):
        picks = (rng.random(len(obs["n"])) * obs["n"]).astype(np.uint8)
        obs = env.step(picks)
        yield obs


def check(obs, games, ties):
    D = len(obs["n"])
    n, choices = obs["n"], obs["choices"]
    assert ((n >= 2) & (n <= ataraxos_env.MAX_CHOICES)).all()
    kind, slot = choices & 3, choices >> 2
    live = np.arange(ataraxos_env.MAX_CHOICES)[None] < n[:, None]
    assert (kind[live] != 3).all() and (choices[~live] == 0).all()
    assert ((kind[live] != 1) | (slot[live] <= 4)).all()
    assert ((kind[live] != 2) | ((slot[live] >= 2) & (slot[live] <= 6))).all()
    off = obs["offsets"]
    assert off[0] == 0 and off[-1] == len(obs["tokens"]) and (np.diff(off) >= 0).all()
    assert (obs["tokens"][:, 0] != 0).all()
    assert obs["request"].shape == (D, ataraxos_env.REQUEST_FIELDS)
    key = obs["battle"].astype(np.int64) * 2 + obs["player"]
    assert len(np.unique(key)) == D
    for b, g, r in zip(obs["end_battle"], obs["end_game"], obs["end_result"]):
        assert games[b] == g and r in (-1, 0, 1)
        games[b] += 1
        ties[0] += r == 0
    assert (obs["game"] == games[obs["battle"]]).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("teams", nargs="*")
    args = ap.parse_args()
    battles, steps, seed = args.battles, args.steps, args.seed
    kw = dict(teams=args.teams, max_turns=args.max_turns)

    games = np.zeros(battles, dtype=np.int64)
    ties = [0]
    a = ataraxos_env.SelfPlay(battles, seed, threads=1, **kw)
    b = ataraxos_env.SelfPlay(battles, seed, threads=8, **kw)
    for x, y in zip(play(a, steps, seed), play(b, steps, seed)):
        check(x, games, ties)
        for k in x:
            assert np.array_equal(x[k], y[k]), k

    env = ataraxos_env.SelfPlay(4, seed)
    obs = env.reset()
    for bad in (np.full(len(obs["n"]), 9), np.zeros(len(obs["n"]) + 1)):
        try:
            env.step(bad)
            raise AssertionError("no error")
        except ValueError:
            pass

    s = a.stats
    assert s["capped"] <= ties[0] and (args.max_turns or s["capped"] == 0)
    print(f"{battles} battles x {steps} steps, seed {seed}: ok. {s['games']} games "
          f"({ties[0]} ties, {s['capped']} at the turn cap), {s['forced']} one-choice requests "
          f"skipped, {s['errors']} rejected picks")

    for threads in (8, 16):
        env = ataraxos_env.SelfPlay(battles * 4, seed, threads=threads, **kw)
        decisions = 0
        t = time.perf_counter()
        for obs in play(env, steps, seed):
            decisions += len(obs["n"])
        dt = time.perf_counter() - t
        s = env.stats
        print(f"{battles * 4} battles, {threads} threads: {decisions / dt / 1e6:.2f}M decisions/s "
              f"(random policy in numpy), {decisions / max(1, s['games']) / 2:.1f} net decisions "
              f"per side per game, {s['forced'] / (s['forced'] + decisions):.1%} of requests "
              f"had one choice")


if __name__ == "__main__":
    main()
