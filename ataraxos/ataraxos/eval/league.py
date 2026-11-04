"""A search-config league (2026-11-04): every pair of configs plays
search_match tasks (`--battles` battles, then the same seed flipped, so each
team pair is played from both seats), round by round, with `--procs` matches
at once on the GPU. Games stream into DIR/games.jsonl; DIR/standings.txt (and
stdout) has the Bradley-Terry Elo of each config (anchored to the first,
95% intervals from a bootstrap over games) and the pair matrix, refreshed
each minute and whenever a match ends. A restart skips the tasks in DIR/done.
A match starts only while the GPU has `--min_free_gb` free (one at a time,
`--stagger` s apart, so the last one's memory shows first); a failed one
(an OOM) goes back in the queue, up to `--tries` times; its games are keyed
by (task, battle), so a retry's replace the failed attempt's.

    python -m ataraxos.eval.league CKPT DIR --config SPEC [--config SPEC ...]
        [--rounds 4] [--battles 8] [--procs 8] [--act_rows 4096] [--min_free_gb 4]

    python -m ataraxos.eval.league --standings DIR   # print standings only
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np


def load_games(d):
    p = os.path.join(d, "games.jsonl")
    if not os.path.exists(p):
        return []
    out = {}
    for line in open(p):
        try:
            g = json.loads(line)
        except json.JSONDecodeError:  # a line being written
            continue
        out[(g.get("task", ""), g["seed"], g["flip"], g["a"], g["b"], g["battle"])] = g  # a retry's game replaces
    return list(out.values())


def bt_elo(names, games, iters=200):
    """Bradley-Terry strengths by MM (ties half a win each), as Elo with
    names[0] at 0."""
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    w = np.zeros((k, k))
    for g in games:
        a, b = idx[g["a"]], idx[g["b"]]
        s = (g["res"] + 1) / 2
        w[a, b] += s
        w[b, a] += 1 - s
    n = w + w.T
    p = np.ones(k)
    for _ in range(iters):
        wins = w.sum(1) + 0.5  # a half-game prior, so an unbeaten config stays finite
        denom = (n / (p[:, None] + p[None, :])).sum(1) + 1.0 / (p + 1.0)
        p = wins / denom
        p /= p[0]
    return 400 * np.log10(p)


def standings(d, names=None, boot=200, seed=0):
    games = load_games(d)
    if names is None:
        names = []
        for g in games:
            for n in (g["a"], g["b"]):
                if n not in names:
                    names.append(n)
    games = [g for g in games if g["a"] in names and g["b"] in names]
    lines = [f"{time.strftime('%H:%M:%S')}  {len(games)} games"]
    if not games:
        return "\n".join(lines)
    elo = bt_elo(names, games)
    rng = np.random.default_rng(seed)
    samples = np.array([bt_elo(names, [games[i] for i in rng.integers(len(games), size=len(games))])
                        for _ in range(boot)])
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    score = {n: [0.0, 0] for n in names}
    for g in games:
        s = (g["res"] + 1) / 2
        score[g["a"]][0] += s
        score[g["a"]][1] += 1
        score[g["b"]][0] += 1 - s
        score[g["b"]][1] += 1
    lines.append(f"{'config':<14}{'elo':>6}  {'95%':>13}  {'games':>6}  {'score':>6}")
    for i in np.argsort(-elo):
        n = names[i]
        s, c = score[n]
        lines.append(f"{n:<14}{elo[i]:6.0f}  [{lo[i]:5.0f},{hi[i]:5.0f}]  {c:6d}  {s / max(c, 1):6.3f}")
    # the pair matrix: row's score vs column (games)
    pair = {}
    for g in games:
        s = (g["res"] + 1) / 2
        for x, y, v in ((g["a"], g["b"], s), (g["b"], g["a"], 1 - s)):
            t = pair.setdefault((x, y), [0.0, 0])
            t[0] += v
            t[1] += 1
    order = [names[i] for i in np.argsort(-elo)]
    w = max(10, max(len(n) for n in names) + 1)
    lines.append("")
    lines.append(" " * w + "".join(f"{n[:9]:>10}" for n in order))
    for x in order:
        row = "".join(f"{(pair[(x, y)][0] / pair[(x, y)][1]):5.2f}({pair[(x, y)][1]:3d})"
                      if (x, y) in pair else f"{'-':>10}" for y in order)
        lines.append(f"{x:<{w}}{row}")
    return "\n".join(lines)


def gpu_free_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return float(out.split()[0]) / 1024
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--battles", type=int, default=8)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--act_rows", type=int, default=4096)
    ap.add_argument("--min_free_gb", type=float, default=4.0)
    ap.add_argument("--stagger", type=float, default=15.0)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--standings", default="", help="print DIR's standings and exit")
    args = ap.parse_args()
    if args.standings:
        print(standings(args.standings))
        return

    d = args.dir
    os.makedirs(os.path.join(d, "done"), exist_ok=True)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    names = [s.partition("=")[0] for s in args.config]
    assert len(set(names)) == len(names), "config names must differ"
    spec = dict(zip(names, args.config))
    with open(os.path.join(d, "configs.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "configs": args.config, "battles": args.battles}, f, indent=1)
    tasks = []
    for rd in range(args.rounds):
        pairs = list(itertools.combinations(names, 2))
        np.random.default_rng(rd).shuffle(pairs)
        for a, b in pairs:
            for flip in (0, 1):
                tasks.append((f"r{rd}_{a}_vs_{b}_f{flip}", a, b, 1000 + rd, flip))
    todo = [t for t in tasks if not os.path.exists(os.path.join(d, "done", t[0]))]
    print(f"{len(tasks)} tasks, {len(todo)} to run, {args.procs} at once", flush=True)
    running = {}
    tries = {}
    last, launched = 0.0, 0.0
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    while todo or running:
        if todo and len(running) < args.procs and time.time() - launched > args.stagger \
                and gpu_free_gb() >= args.min_free_gb:
            task = todo.pop(0)
            tag, a, b, seed, flip = task
            tries[tag] = tries.get(tag, 0) + 1
            launched = time.time()
            cmd = [sys.executable, "-u", "-m", "ataraxos.eval.search_match", args.ckpt, "--a", spec[a],
                   "--b", spec[b], "--battles", str(args.battles), "--seed", str(seed),
                   "--out", os.path.join(d, "games.jsonl"), "--act_rows", str(args.act_rows),
                   "--threads", str(args.threads), "--task", tag] + (["--flip"] if flip else [])
            log = open(os.path.join(d, "logs", tag + ".log"), "a")
            running[tag] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log, task)
        ended = False
        for tag, (p, log, task) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            log.close()
            del running[tag]
            ended = True
            if rc == 0:
                open(os.path.join(d, "done", tag), "w").close()
                msg = "done"
            elif tries[tag] < args.tries:
                todo.append(task)
                msg = f"exited {rc}, try {tries[tag]}: requeued (see its log)"
            else:
                msg = f"exited {rc}, try {tries[tag]}: dropped (see its log)"
            print(f"{time.strftime('%H:%M:%S')}  {tag} {msg}", flush=True)
        if ended or time.time() - last > 60:
            text = standings(d, names)
            with open(os.path.join(d, "standings.txt"), "w") as f:
                f.write(text + "\n")
            print(text + f"\n  running {len(running)}, queued {len(todo)}, gpu free {gpu_free_gb():.1f} GB\n",
                  flush=True)
            last = time.time()
        time.sleep(2)
    print("league done", flush=True)


if __name__ == "__main__":
    main()
