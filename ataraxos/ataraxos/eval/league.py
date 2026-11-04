"""A search-config league (2026-11-04): search configs play search_match
matches (`--battles` battles, then the same seed with the seats flipped, so
each team pair is played from both seats), `--procs` at once on the GPU.

The configs live in DIR/configs.txt, re-read every loop, so configs can be
added or retired while it runs: a line NAME=SPEC is active, -NAME=SPEC
retired (its games stay in the ratings; it gets no new matches). Each free
slot goes to the active config with the fewest games (the anchor's count
divided by `--anchor_weight`, so it plays less), against the active config
it has played least; the flipped half of a match runs next.

A config is retired automatically when, with at least `--min_games` games,
under `--cut` of the bootstrap samples rate it at or above the leader (the
top active config by Elo); never the anchor (the first config). Each cut is
logged with its numbers in DIR/league.log.

Games stream into DIR/games.jsonl (keyed by (task, battle): a retried
match's games replace the failed attempt's). DIR/standings.txt has the
Bradley-Terry Elo (anchor 0, 95% bootstrap intervals), each config's
record, P(>= leader), its search seconds per root (batched over the
match's live roots and shared with the other matches on the GPU: a relative
cost, not the ladder's time per move) and the pair records, refreshed each
minute and whenever a match ends. A match starts only while the GPU has
`--min_free_gb` free, `--stagger` s after the last; a failed one (an OOM)
is retried up to `--tries` times.

    python -m ataraxos.eval.league CKPT DIR [--config SPEC ...] [--procs 12]
        [--battles 8] [--act_rows 4096] [--min_free_gb 4] [--anchor_weight 0.5]
        [--min_games 24] [--cut 0.05]

    python -m ataraxos.eval.league --standings DIR   # print standings only

Names: d{depth}r{rounds}b{budget, k}{m}x{n}_{team|pair}: _team the per_type
mode (a policy per own team, marginal over the other side's), _pair per_pair
(a joint policy per team pair), e.g. d8r2b2k3x3_team (`config_name`).
DIR/aliases.json ({"old": "new"}) renames configs in the games as they load,
so matches still running under an old name keep counting.

`--config` only seeds a new DIR/configs.txt; after that, edit the file.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np


def config_name(spec):
    """The full-parameter name of a SPEC's key:value part (search_match's)."""
    kv = spec.partition("=")[2] if "=" in spec else spec
    if kv in ("", "plain"):
        return "plain"
    d = dict(x.split(":") for x in kv.split(","))
    t = d.get("types", "3")
    b = int(d.get("budget", 2048))
    name = f"d{d.get('depth', 4)}r{d.get('rounds', 2)}b{f'{b // 1024}k' if b % 1024 == 0 else f'{b}_'}" \
           f"{d.get('ta', t)}x{d.get('tb', t)}" + {"per_type": "_team", "per_pair": "_pair"}[d.get("mode", "per_type")]
    extra = [f"{k}{v}" for k, v in sorted(d.items())
             if k not in ("depth", "rounds", "budget", "types", "ta", "tb", "mode")]
    return "_".join([name] + extra)


def load_games(d):
    p = os.path.join(d, "games.jsonl")
    if not os.path.exists(p):
        return []
    ap = os.path.join(d, "aliases.json")
    alias = json.load(open(ap)) if os.path.exists(ap) else {}
    out = {}
    for line in open(p):
        try:
            g = json.loads(line)
        except json.JSONDecodeError:  # a line being written
            continue
        g["a"], g["b"] = alias.get(g["a"], g["a"]), alias.get(g["b"], g["b"])
        out[(g.get("task", ""), g["seed"], g["flip"], g["a"], g["b"], g["battle"])] = g  # a retry's game replaces
    return list(out.values())


def read_configs(d):
    """DIR/configs.txt -> [(name, spec, active)] in file order."""
    out = []
    for line in open(os.path.join(d, "configs.txt")):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        active = not line.startswith("-")
        spec = line.lstrip("-")
        out.append((spec.partition("=")[0], spec, active))
    return out


def retire(d, name):
    p = os.path.join(d, "configs.txt")
    lines = open(p).read().split("\n")
    with open(p, "w") as f:
        f.write("\n".join("-" + l if l.strip().partition("=")[0] == name else l for l in lines))


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
        wins = w.sum(1) + 0.5  # a half-game prior against a virtual equal, so an unbeaten config stays finite
        denom = (n / (p[:, None] + p[None, :])).sum(1) + 1.0 / (p + 1.0)
        p = wins / denom
        p /= p[0]
    return 400 * np.log10(p)


def rate(names, games, active, boot=300, seed=0, min_games=24):
    """(elo [k], lo, hi, P(>= the leader) [k], leader index); the leader is
    the top active config by point Elo among those with `min_games` games
    (any active one if none has), so a hot start on a few games can't cut
    the field."""
    elo = bt_elo(names, games)
    rng = np.random.default_rng(seed)
    samples = np.array([bt_elo(names, [games[i] for i in rng.integers(len(games), size=len(games))])
                        for _ in range(boot)])
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    count = {n: 0 for n in names}
    for g in games:
        count[g["a"]] += 1
        count[g["b"]] += 1
    act = [i for i, n in enumerate(names) if n in active]
    act = [i for i in act if count[names[i]] >= min_games] or act
    leader = max(act, key=lambda i: elo[i])
    p_top = (samples >= samples[:, [leader]] - 1e-9).mean(0)
    return elo, lo, hi, p_top, leader


def costs(games):
    """Search seconds per root, per config, from each task's last game line."""
    last = {}
    for g in games:
        if "search_s" in g:
            k = g.get("task", "")
            if k not in last or g["t"] > last[k]["t"]:
                last[k] = g
    sec, n = {}, {}
    for g in last.values():
        for side, name in enumerate((g["a"], g["b"])):
            sec[name] = sec.get(name, 0.0) + g["search_s"][side]
            n[name] = n.get(name, 0) + g["searches"][side]
    return {k: sec[k] / n[k] for k in sec if n[k]}


def standings(d, configs=None, min_games=24):
    games = load_games(d)
    if configs is None:
        configs = read_configs(d) if os.path.exists(os.path.join(d, "configs.txt")) else []
    names = [c[0] for c in configs]
    for g in games:
        for n in (g["a"], g["b"]):
            if n not in names:
                names.append(n)
                configs.append((n, n, False))
    active = {c[0] for c in configs if c[2]}
    games = [g for g in games if g["a"] in names and g["b"] in names]
    head = f"{time.strftime('%H:%M:%S')}  {len(games)} games"
    if not games:
        return head, None
    elo, lo, hi, p_top, leader = rate(names, games, active, min_games=min_games)
    rec = {n: [0, 0, 0] for n in names}
    pair = {}
    for g in games:
        i = {1: 0, 0: 1, -1: 2}[g["res"]]
        rec[g["a"]][i] += 1
        rec[g["b"]][2 - i] += 1
        for x, y, j in ((g["a"], g["b"], i), (g["b"], g["a"], 2 - i)):
            pair.setdefault((x, y), [0, 0, 0])[j] += 1
    cost = costs(games)
    nw = max(10, max(len(n) for n in names) + 1)
    lines = [head, f"  {'config':<{nw}}{'elo':>6}  {'95%':>13}  {'W-T-L':>9}  {'P(top)':>6}  {'s/root':>6}"]
    order = list(np.argsort(-elo))
    for i in order:
        n = names[i]
        w, t, l = rec[n]
        mark = " " if n in active else "x"
        c = f"{cost[n]:6.2f}" if n in cost else f"{'-':>6}"
        lines.append(f"{mark} {n:<{nw}}{elo[i]:6.0f}  [{lo[i]:5.0f},{hi[i]:5.0f}]  {f'{w}-{t}-{l}':>9}  "
                     f"{p_top[i]:6.2f}  {c}")
    lines.append(f"  (x retired; P(top) the bootstrap share at or above the leader, the top with {min_games}+ games;"
                 " s/root batched, shared GPU)")
    lines.append("")
    lines.append("pair records (row's W-L vs column):")
    w = max(10, max(len(n) for n in names) + 2)
    on = [names[i] for i in order]
    cw = max(9, max(len(n) for n in names) + 1)
    lines.append(" " * w + "".join(f"{n:>{cw}}" for n in on))
    for x in on:
        lines.append(f"{x:<{w}}" + "".join(
            f"{f'{pair[(x, y)][0] + pair[(x, y)][1] / 2:g}-{pair[(x, y)][2] + pair[(x, y)][1] / 2:g}':>{cw}}"
            if (x, y) in pair else f"{'.':>{cw}}" for y in on))
    return "\n".join(lines), dict(names=names, games=games, elo=elo, p_top=p_top, leader=leader, active=active)


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
    ap.add_argument("--battles", type=int, default=8)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--act_rows", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min_free_gb", type=float, default=4.0)
    ap.add_argument("--stagger", type=float, default=15.0)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--anchor_weight", type=float, default=0.5)
    ap.add_argument("--min_games", type=int, default=24)
    ap.add_argument("--cut", type=float, default=0.05)
    ap.add_argument("--standings", default="", help="print DIR's standings and exit")
    args = ap.parse_args()
    if args.standings:
        print(standings(args.standings)[0])
        return

    d = args.dir
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    cfile = os.path.join(d, "configs.txt")
    if not os.path.exists(cfile):
        with open(cfile, "w") as f:
            f.write("\n".join(args.config) + "\n")
    logf = open(os.path.join(d, "league.log"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    seed = 1 + max([g["seed"] for g in load_games(d)] + [1999])  # a fresh seed per match pair, past league1's
    running, tries, pending = {}, {}, []
    last, launched = 0.0, 0.0
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log(f"league: {d}, ckpt {args.ckpt}, {args.procs} procs, {args.battles} battles a half")
    while True:
        configs = read_configs(d)
        spec = {n: s for n, s, _ in configs}
        active = [n for n, _, a in configs if a]
        anchor = configs[0][0]
        if (pending or len(active) >= 2) and len(running) < args.procs \
                and time.time() - launched > args.stagger and gpu_free_gb() >= args.min_free_gb:
            # the flipped halves first (their configs may have been retired since: still play them)
            if pending:
                task = pending.pop(0)
            else:
                games = load_games(d)
                n_games, n_pair = {n: 0.0 for n in active}, {}
                for g in games:
                    for x, y in ((g["a"], g["b"]), (g["b"], g["a"])):
                        if x in n_games:
                            n_games[x] += 1
                        n_pair[(x, y)] = n_pair.get((x, y), 0) + 1
                for _, _, a, b, _ in [v[2] for v in running.values()]:  # count running matches as played
                    for x, y in ((a, b), (b, a)):
                        if x in n_games:
                            n_games[x] += 2 * args.battles
                        n_pair[(x, y)] = n_pair.get((x, y), 0) + 2 * args.battles
                load = {n: n_games[n] / (args.anchor_weight if n == anchor else 1.0) + random.random()
                        for n in active}
                a = min(active, key=load.get)
                b = min((n for n in active if n != a),
                        key=lambda n: n_pair.get((a, n), 0) + (0.5 if n == anchor else 0) + random.random() * 0.1)
                task = (f"s{seed}_{a}_vs_{b}_f0", seed, a, b, 0)
                pending.append((f"s{seed}_{a}_vs_{b}_f1", seed, a, b, 1))
                seed += 1
            tag, sd, a, b, flip = task
            tries[tag] = tries.get(tag, 0) + 1
            launched = time.time()
            cmd = [sys.executable, "-u", "-m", "ataraxos.eval.search_match", args.ckpt, "--a", spec[a],
                   "--b", spec[b], "--battles", str(args.battles), "--seed", str(sd),
                   "--out", os.path.join(d, "games.jsonl"), "--act_rows", str(args.act_rows),
                   "--threads", str(args.threads), "--task", tag] + (["--flip"] if flip else [])
            lf = open(os.path.join(d, "logs", tag + ".log"), "a")
            running[tag] = (subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env), lf, task)
            log(f"start {tag} (try {tries[tag]})")
        ended = False
        for tag, (p, lf, task) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            del running[tag]
            ended = True
            if rc == 0:
                log(f"done {tag}")
            elif tries[tag] < args.tries:
                pending.insert(0, task)
                log(f"{tag} exited {rc} (try {tries[tag]}): retrying (see logs/{tag}.log)")
            else:
                log(f"{tag} exited {rc} (try {tries[tag]}): dropped (see logs/{tag}.log)")
        if ended or time.time() - last > 60:
            text, r = standings(d, read_configs(d), args.min_games)
            if r is not None:
                counts = {n: 0 for n in r["names"]}
                for g in r["games"]:
                    counts[g["a"]] += 1
                    counts[g["b"]] += 1
                leader = r["names"][r["leader"]]
                for i, n in enumerate(r["names"]):
                    if n in r["active"] and n != anchor and n != leader and counts[n] >= args.min_games \
                            and r["p_top"][i] < args.cut:
                        retire(d, n)
                        log(f"retired {n}: P(>= {leader}) {r['p_top'][i]:.3f} < {args.cut}, "
                            f"elo {r['elo'][i]:.0f} vs {r['elo'][r['leader']]:.0f}, {counts[n]} games")
            with open(os.path.join(d, "standings.txt"), "w") as f:
                f.write(text + "\n")
            print(text + f"\n  running {len(running)}, flipped halves queued {len(pending)}, "
                         f"gpu free {gpu_free_gb():.1f} GB\n", flush=True)
            last = time.time()
        time.sleep(2)


if __name__ == "__main__":
    main()
