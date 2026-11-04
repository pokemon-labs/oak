"""A ladder pool (`--teams`) of a run's top N teams by Nash draw counts
(2026-11-04, the user's: rby4's top 20 for applemagpie on Showdown): the
counts of a teams{it}.npz, the sets from the run's team file, packed as
Showdown's Teams.pack packs a gen1 export with only species and moves
(codegen/pack_teams.cjs's output, no node needed).

    python codegen/top_teams.py runs/rby4/teams1750.npz teams/selfplay/oak1500.txt 20 > pool.tsv

The draw counts and shares go to stderr.
"""

import re
import sys

import numpy as np


def read_teams(path):
    """Showdown export -> {name: [(species, [moves])]}, names as in team_names."""
    out = {}
    parts = re.split(r"^(=== .* ===)$", open(path).read(), flags=re.M)
    for head, body in zip(parts[1::2], parts[2::2]):
        name = re.sub(r"^=== (\[[^\]]*\] )?", "", head).removesuffix(" ===")
        team = []
        for block in body.strip().split("\n\n"):
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            if not lines:
                continue
            assert all(l.startswith("- ") for l in lines[1:]), (name, lines)
            team.append((lines[0], [l[2:] for l in lines[1:]]))
        out[name] = team
    return out


def pack_name(s):
    return re.sub(r"[^A-Za-z0-9]+", "", s)


def pack(team):
    # name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|happiness
    return "]".join(f"{sp}||||{','.join(pack_name(m) for m in moves)}|||||||" for sp, moves in team)


def main():
    npz, teams_file, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    z = np.load(npz)
    counts, names = z["counts"], [str(x) for x in z["names"]]
    teams = read_teams(teams_file)
    order = np.argsort(-counts, kind="stable")[:n]
    total = counts.sum()
    for i in order:
        print(f"{names[i]}\t{pack(teams[names[i]])}")
        print(f"{names[i]}: {int(counts[i])} draws ({counts[i] / total:.4f})", file=sys.stderr)


if __name__ == "__main__":
    main()
