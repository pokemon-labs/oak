"""Checks ataraxos/eval/agents.py and the Buffer's handling of capped games (2026-10-25; at_cap 2026-10-26).

    ~/oak/.venv/bin/python tests/py/test_agents.py [--battles N] [--seed N] [TEAM_FILE ...]

- MaxDamage's foe, tracked from the log alone, is the other player's active
  species (from that player's request) whenever both choose a move in the
  same step;
- MaxDamage beats RandomAgent: above 0.75 on pool teams, 3σ above 1/2 on
  truly random ones;
- RandomAgent against itself scores about 1/2;
- with a tiny turn cap, the Buffer keeps exactly the games that ended
  without the cap under at_cap "drop", and every game under "draw" and
  "bootstrap", the capped ones truncated under "bootstrap" alone.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ataraxos._ext import ataraxos_env  # noqa: E402
from ataraxos.eval.agents import F_ACTIVE, F_SPECIES, SIDE, SIDE0, MaxDamage, RandomAgent, match, subset  # noqa: E402
from ataraxos.rl.buffer import AT_CAP, Buffer  # noqa: E402

MOVE_REQUEST = 0  # RequestData::Move_


def active_species(request):
    sides = request[SIDE0:].reshape(6, SIDE)
    return int(sides[np.flatnonzero(sides[:, F_ACTIVE])[0], F_SPECIES])


def check_tracking(teams, battles, seed, steps=400):
    env = ataraxos_env.SelfPlay(battles, seed, teams=teams)
    agent = MaxDamage(seed)
    agent.start(battles)
    obs = env.reset()
    compared = 0
    for _ in range(steps):
        picks = agent.act(obs)
        moves = obs["request"][:, 0] == MOVE_REQUEST
        at = {(int(b), int(p)): i for i, (b, p) in enumerate(zip(obs["battle"], obs["player"]))}
        for (b, p), i in at.items():
            j = at.get((b, 1 - p))
            if j is None or not (moves[i] and moves[j]):
                continue
            foe = agent.foe[(b, p)][1]
            assert foe == active_species(obs["request"][j]), (b, p, foe, active_species(obs["request"][j]))
            compared += 1
        obs = env.step(picks)
    assert compared > 1000, compared
    return compared


def check_capped(teams, battles, seed, at_cap, steps=2000):
    """max_turns 30: many games are capped. Under "drop" the Buffer must keep
    exactly the others (both sides' trajectories) and none of the capped;
    under "draw" and "bootstrap" all of them, the capped as ties, marked
    truncated under "bootstrap" alone."""
    env = ataraxos_env.SelfPlay(battles, seed, teams=teams, max_turns=30)
    buf = Buffer(max_lines=64, at_cap=at_cap)
    rng = np.random.default_rng(seed)
    asked = set()  # (battle, game, player) with a decision
    kept = capped = capped_kept = 0
    obs = env.reset()
    for _ in range(steps):
        for b, g, p in zip(obs["battle"].tolist(), obs["game"].tolist(), obs["player"].tolist()):
            asked.add((b, g, p))
        picks = (rng.random(len(obs["n"])) * obs["n"]).astype(np.uint8)
        buf.add(obs, picks, *_dummy(len(picks)))
        obs = env.step(picks)
        for b, g, r, c in zip(obs["end_battle"].tolist(), obs["end_game"].tolist(), obs["end_result"].tolist(),
                              obs["end_capped"].tolist()):
            assert not c or r == 0
            sides = ((b, g, 0) in asked) + ((b, g, 1) in asked)
            if c:
                capped += 1
                capped_kept += sides
            else:
                kept += sides
        buf.end(obs)
    assert capped > 0 and kept > 0, (capped, kept)
    if at_cap != "drop":
        kept += capped_kept
    assert len(buf.done) == kept, (len(buf.done), kept)
    data = buf.take()
    assert len(data.length) == kept
    assert data.truncated.sum() == (capped_kept if at_cap == "bootstrap" else 0)
    assert (data.reward[data.truncated] == 0).all()
    return capped, kept


def _dummy(D):
    import torch
    return torch.zeros(D, ataraxos_env.MAX_CHOICES), torch.zeros(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("teams", nargs="*")
    args = ap.parse_args()

    n = check_tracking(args.teams, args.battles, args.seed)
    print(f"tracking: the foe matched the other side's active species in {n} decisions")

    def factory(b):
        return ataraxos_env.SelfPlay(b, args.seed + 1, teams=args.teams, max_turns=200)

    w, t, l = match(factory, MaxDamage(args.seed), RandomAgent(args.seed), args.battles)
    score = (w + t / 2) / (w + t + l)
    # truly random teams draw each level from 1-100, which decides most games
    # whoever plays: MaxDamage scores about 0.59 there (2026-10-25)
    bar = 0.75 if args.teams else 0.5 + 3 * 0.5 / np.sqrt(w + t + l)
    assert score > bar, (w, t, l, bar)
    print(f"maxdamage vs random: {w}-{t}-{l}, score {score:.3f}")
    w, t, l = match(factory, RandomAgent(args.seed), RandomAgent(args.seed + 1), args.battles)
    score = (w + t / 2) / (w + t + l)
    assert abs(score - 0.5) < 4 * 0.5 / np.sqrt(w + t + l), (w, t, l)
    print(f"random vs random: {w}-{t}-{l}, score {score:.3f}")

    for at_cap in AT_CAP:
        capped, kept = check_capped(args.teams, args.battles, args.seed, at_cap)
        print(f"capped ({at_cap}): {capped} games capped, {kept} trajectories kept")
    print("ok")


if __name__ == "__main__":
    main()
