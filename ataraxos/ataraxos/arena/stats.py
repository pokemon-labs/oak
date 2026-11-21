"""The arena's usage stats (2026-11-21): pkmn.ai gives an era-matched
data.pkmn.cc file at $PKMN_STATS_DIR/<format>.json (/opt/arena/stats on the
ladder, readable inside the sandbox). Not used by the search yet (the
user's: the sampler stays on teams1500); main.py logs what it finds.

The file: {"battles", "metagame", "pokemon": {SPECIES: {"usage": {"raw",
"real", "weighted"}, "lead", "count", "weight", "viability", "moves": {MOVE:
share}, "teammates": {SPECIES: share}, "counters", "spreads": {"Hardy:
HP/Atk/Def/SpA/SpD/Spe" (stat exp): share}, "abilities", "items", ...}}}.
Gen 1's spreads show players lowering a special attacker's Atk stat exp
(Slowbro 21% 0 Atk, Chansey 11%): the sets aren't all maxed."""

import json
import os

FORMAT = "gen1ou"


def path(fmt=FORMAT):
    d = os.environ.get("PKMN_STATS_DIR")
    return os.path.join(d, fmt + ".json") if d else None


def load(fmt=FORMAT):
    """The stats for `fmt`, or None when the arena gave none."""
    p = path(fmt)
    if not p or not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def summary(stats):
    """One line for the log."""
    mons = stats.get("pokemon", {})
    top = sorted(mons, key=lambda s: -mons[s].get("usage", {}).get("weighted", 0))[:6]
    return f"{stats.get('battles', '?')} battles, {len(mons)} species, top {', '.join(top)}"


def usage(stats, weighting="weighted"):
    """Species -> usage share."""
    return {s: x.get("usage", {}).get(weighting, 0.0) for s, x in stats.get("pokemon", {}).items()}


def moves(stats, species):
    """A species' moves -> share of its sets, most used first."""
    m = stats.get("pokemon", {}).get(species, {}).get("moves", {})
    return dict(sorted(m.items(), key=lambda kv: -kv[1]))
