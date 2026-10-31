"""Win/draw/loss per team and per team pair from a run's teams{it}.npz
(2026-10-26).

    python -m ataraxos.eval.pairs runs/x/teams100.npz [--pairs N] [--matrix]

Both seats are combined: a pair's record is from the first team's side, over
the games it played as P1 and as P2. A tie by max_turns counts as a draw
here and is also shown apart (`capped`); training drops those games.
Prints every team's record sorted by score (mirror games left out), then the N pairs with the most
games (all of them by default), and with --matrix the score of row against
column (pools up to about 60 teams fit a terminal).
"""

import argparse

import numpy as np


def records(z):
    """[K, K, 4]-like dict (i, j) -> [win, draw, loss, capped] from i's side,
    both seats combined, i < j and i == j (mirror pairs kept once)."""
    out = {}
    for a, b, (w, t, l, c) in zip(z["pair_a"].tolist(), z["pair_b"].tolist(), z["pair_wdlc"].tolist()):
        if a <= b:
            row = out.setdefault((a, b), [0, 0, 0, 0])
            row[0] += w; row[1] += t + c; row[2] += l; row[3] += c
        else:  # P1 was b: flip to a's side
            row = out.setdefault((b, a), [0, 0, 0, 0])
            row[0] += l; row[1] += t + c; row[2] += w; row[3] += c
    return out


def score(w, d, l):
    n = w + d + l
    return (w + d / 2) / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--pairs", type=int, default=0, help="the pairs with the most games; 0: all")
    ap.add_argument("--matrix", action="store_true")
    args = ap.parse_args()
    z = np.load(args.npz)
    if "pair_a" not in z:
        raise SystemExit(f"{args.npz} has no pair records (written before 2026-10-26)")
    names = [str(n) for n in z["names"]]
    k = len(names)
    rec = records(z)

    team = np.zeros((k, 4), np.int64)  # mirror games left out: they're a win and a loss for the team
    for (i, j), (w, d, l, c) in rec.items():
        if i != j:
            team[i] += [w, d, l, c]
            team[j] += [l, d, w, c]
    games = sum(w + d + l for w, d, l, _ in rec.values())
    print(f"{args.npz}: {k} teams, {games} games, {len(rec)} pairs played")
    print(f"\n{'team':>5} {'score':>6} {'W':>6} {'D':>5} {'L':>6} {'capped':>6}  name")
    for i in sorted(range(k), key=lambda i: -np.nan_to_num(score(*team[i, :3]), nan=-1)):
        w, d, l, c = team[i]
        if w + d + l:
            print(f"{i:>5} {score(w, d, l):6.3f} {w:6d} {d:5d} {l:6d} {c:6d}  {names[i]}")

    print(f"\n{'A':>5} {'B':>5} {'n':>5} {'A score':>7} {'W':>5} {'D':>4} {'L':>5}  (A's side)")
    order = sorted(rec.items(), key=lambda kv: -sum(kv[1][:3]))
    for (i, j), (w, d, l, c) in order[:args.pairs or None]:
        print(f"{i:>5} {j:>5} {w + d + l:5d} {score(w, d, l):7.3f} {w:5d} {d:4d} {l:5d}")

    if args.matrix:
        print("\nrow's score against column ('.' no games)")
        print("     " + "".join(f"{j:>5}" for j in range(k)))
        for i in range(k):
            cells = []
            for j in range(k):
                w, d, l, _ = rec[(i, j)] if i <= j and (i, j) in rec else (
                    [rec[(j, i)][2], rec[(j, i)][1], rec[(j, i)][0], 0] if (j, i) in rec else [0, 0, 0, 0])
                cells.append(f"{score(w, d, l):5.2f}" if w + d + l else "    .")
            print(f"{i:>5}" + "".join(cells))


if __name__ == "__main__":
    main()
