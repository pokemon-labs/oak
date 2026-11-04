"""One league match (2026-11-04): two search configs, both searching, one
checkpoint, on search_h2h's team split (the sampler's pool the first half of
the file, the games' teams the second). A sits in P1 in the even battles and
P2 in the odd ones (`--flip` swaps: the same seed draws the same teams, so a
task and its flip play each team pair both ways); each battle's first game
counts. Every counted game is appended to `--out` as a JSON line when it
ends, so a league's standings move while its matches run.

    python -m ataraxos.eval.search_match CKPT --a SPEC --b SPEC [--battles 16]
        [--seed 1] [--flip] [--out games.jsonl] [--act_rows 8192]

SPEC is NAME=key:value,... (keys: rounds, budget, depth, types, mode, floor,
alpha, beta, foe_beta, shrink) for the joint search, budget its joint_budget
as on the ladder, or NAME=plain for the net with no search.
"""

import argparse
import json
import tempfile
import time

import numpy as np
import torch

from ataraxos._ext import ataraxos_env
from ataraxos.eval.agents import load_policy, subset
from ataraxos.eval.search_h2h import split_teams
from ataraxos.nets.causal_actor import CausalActor
from ataraxos.search import Search, SearchConfig

INTS = {"rounds", "budget", "depth", "types"}
FLOATS = {"floor", "alpha", "beta", "foe_beta", "shrink"}


def parse_spec(spec, act_rows=8192):
    """NAME=k:v,... -> (name, SearchConfig or None for plain)."""
    name, _, kv = spec.partition("=")
    if kv in ("", "plain"):
        return name, None
    d = {}
    for item in kv.split(","):
        k, v = item.split(":")
        if k not in INTS | FLOATS | {"mode"}:
            raise ValueError(f"{spec}: unknown key {k}")
        d[k] = int(v) if k in INTS else float(v) if k in FLOATS else v
    types = d.pop("types", 3)
    budget = d.pop("budget", 2048)
    mode = d.pop("mode", "per_type")
    cfg = SearchConfig(joint=True, joint_mode=mode, types_a=types, types_b=types, budget=budget,
                       joint_budget=budget, rounds=d.pop("rounds", 2), depth=d.pop("depth", 4),
                       act_rows=act_rows, **d)
    return name, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--teams", default="teams/selfplay/oak1500.txt")
    ap.add_argument("--battles", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max_turns", type=int, default=300)
    ap.add_argument("--act_rows", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--task", default="", help="the league's task tag, in each game line (a retry's games replace)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    net = load_policy(args.ckpt, dev)
    pool, games = split_teams(args.teams, tempfile.mkdtemp(prefix="search_match_"))
    sides = []
    for i, spec in enumerate((args.a, args.b)):
        name, cfg = parse_spec(spec, args.act_rows)
        search = Search(net, [pool], dev, cfg, threads=args.threads, seed=args.seed * 2 + i) if cfg else None
        sides.append((name, search))
    B = args.battles
    env = ataraxos_env.SelfPlay(B, seed=args.seed, teams=[games], max_turns=args.max_turns,
                                shards=min(B, 16), threads=args.threads, state=net.c.state)
    env.set_keep_after(True)
    actors = [CausalActor(net, B, dev), CausalActor(net, B, dev)]
    flip = int(args.flip)
    (na, _), (nb, _) = sides

    result = np.full(B, -2, np.int64)  # A's: 1 win, 0 tie, -1 loss
    turns = np.zeros(B, np.int64)
    obs = env.reset()
    t0, searched, search_s = time.time(), [0, 0], [0.0, 0.0]
    out = open(args.out, "a") if args.out else None
    while True:
        for bt, g, r in zip(obs["end_battle"], obs["end_game"], obs["end_result"]):
            if g == 0 and result[bt] == -2:
                a_p1 = (bt + flip) % 2 == 0
                result[bt] = r if a_p1 else -r
                w, t, l = (int((result == v).sum()) for v in (1, 0, -1))
                print(f"{time.time() - t0:7.0f}s  {na} vs {nb} game {w + t + l}/{B} "
                      f"{'WTL'[1 - result[bt]]}: {w}-{t}-{l}, searches {searched}", flush=True)
                if out:
                    out.write(json.dumps({"task": args.task, "a": na, "b": nb, "seed": args.seed, "flip": flip, "battle": int(bt),
                                          "a_p1": bool(a_p1), "res": int(result[bt]), "decisions": int(turns[bt]),
                                          "searches": list(searched), "search_s": [round(x, 2) for x in search_s],
                                          "t": round(time.time() - t0, 1)}) + "\n")
                    out.flush()
        if (result != -2).all():
            break
        a_rows = ((obs["battle"] + flip) % 2) == obs["player"]  # A is P1 (player 0) in even battles
        picks = np.zeros(len(obs["n"]), np.uint8)
        for side, rows in enumerate((np.flatnonzero(a_rows), np.flatnonzero(~a_rows))):
            if not len(rows):
                continue
            o = subset(obs, rows)
            p, logp, _ = actors[side].act(o)
            search = sides[side][1]
            live = np.flatnonzero(o["game"] == 0)  # only the counted games search
            if search is not None and len(live):
                ts = time.time()
                sp, _, _ = search.run(env, actors[side], subset(o, live), logp[live])
                search_s[side] += time.time() - ts
                p[live] = sp
                searched[side] += len(live)
            picks[rows] = p
        turns[np.unique(obs["battle"][obs["game"] == 0])] += 1
        obs = env.step(picks)
    w, t, l = (int((result == v).sum()) for v in (1, 0, -1))
    peak = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0
    print(f"{na} vs {nb}: {w}-{t}-{l} in {time.time() - t0:.0f}s, searches {searched}, peak {peak:.2f} GiB",
          flush=True)


if __name__ == "__main__":
    main()
