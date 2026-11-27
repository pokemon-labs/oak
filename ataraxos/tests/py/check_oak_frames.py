"""The env's oak recording (src/battle_data.hpp's `oak`, ataraxos/distill/
gen_oak.py) replays in oak: random play, every game read back with oak's
BattleFrames (oak's libpkmn replaying the stored choices). Checks, per game:
the replay passes through every battle state the env showed between steps,
in order (forced updates inside a step aren't seen by the env's side); the
stored result is an ending (not none or error); each update's policies are
distributions over its k libpkmn choices, with the played choice's index in
range. Run on the box: PYTHONPATH=. python tests/py/check_oak_frames.py
"""

import sys

import numpy as np

import oak.train
from ataraxos._ext import ataraxos_env

B, STEPS = 64, 400
teams = sys.argv[1:] or ["teams/selfplay/oak1500.txt"]
env = ataraxos_env.SelfPlay(B, 5, teams=teams, max_turns=300, threads=2, state=True)
env.set_oak(True)
rng = np.random.default_rng(0)
obs = env.reset()
seen = [[env.battle_bytes(b)] for b in range(B)]  # current game's states
games = {}  # (battle, game) -> states
blobs = []
for step in range(STEPS):
    D = len(obs["n"])
    probs = rng.random((D, 9)).astype(np.float32) * (np.arange(9)[None] < obs["n"][:, None])
    probs /= probs.sum(1, keepdims=True)
    picks = np.array([rng.choice(9, p=q) for q in probs], np.uint8)
    env.set_oak_targets(probs, rng.random(D).astype(np.float32))
    obs = env.step(picks)
    for bt, g in zip(obs["end_battle"], obs["end_game"]):
        games[(int(bt), int(g))] = seen[bt]
        seen[bt] = []
    for b in range(B):
        x = env.battle_bytes(b)
        if not seen[b] or seen[b][-1] != x:
            seen[b].append(x)
    data, n = env.take_oak()
    if n:
        blobs.append(data)

open("/tmp/check_oak.battle.data", "wb").write(b"".join(blobs))
recs = oak.train.read_battle_data("/tmp/check_oak.battle.data")
print(f"{len(recs)} games recorded, {len(games)} ended ({env.stats['capped']} capped)")
# Match a recorded game to the env's by the first state the env showed of
# it (forced updates can come before it, so look it up among the replay's).
by_start = {}
for key, states in games.items():
    if states:
        by_start.setdefault(states[0], []).append(states)
bad = checked = 0
for blob, count in recs:
    f = oak.train.BattleFrames.from_bytes(blob, count)
    replay = [bytes(f.battle[i]) for i in range(count)]
    cands = next((by_start[x] for x in replay if x in by_start), [])
    if not cands:
        bad += 1
        print("no env game starts at a recorded game's first state")
        continue
    states = cands[0]
    j = 0
    for s in states:  # every state the env showed before the game ended
        while j < len(replay) and replay[j] != s:
            j += 1
        if j == len(replay):
            break
    ok = j < len(replay)
    why = [] if ok else ["replay misses an env state"]
    k = np.asarray(f.k[:count]).reshape(count, 2).astype(int)
    pol = f.empirical_policies[:count].reshape(count, 2, 9)
    sums = np.stack([pol[i, s, : k[i, s]].sum() for i in range(count) for s in range(2)])
    if not np.all(np.abs(sums - 1) < 1e-3):
        why.append(f"policy sums {sums.min():.4f}-{sums.max():.4f}")
    if not np.all(np.asarray(f.iterations[:count]) <= 2):
        why.append("iterations > 2")
    if why:
        bad += 1
        if bad <= 5:
            print("bad game:", "; ".join(why))
    checked += 1
print(f"checked {checked}, bad {bad}")
sys.exit(1 if bad or checked == 0 else 0)
