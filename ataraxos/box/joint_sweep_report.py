"""The joint sweep's scores so far (box/joint_sweep.sh): per arm, games,
W-T-L, score with its standard error, and the paired difference to
`default` over the seeds both have.

    python box/joint_sweep_report.py [runs/joint_sweep/results.jsonl]
"""

import json
import sys
from collections import defaultdict

import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/joint_sweep/results.jsonl"
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(dict)  # arm -> seed -> per-game scores
    secs = defaultdict(list)
    for r in rows:
        res = np.array(r["result"])
        by[r["tag"]][r["args"]["seed"]] = (res + 1) / 2
        secs[r["tag"]].append(r["seconds"] / max(1, r["searches"]))
    base = by.get("default", {})
    for arm, seeds in by.items():
        s = np.concatenate(list(seeds.values()))
        line = (f"{arm:12s} {len(s):4d} games  {int((s == 1).sum())}-{int((s == 0.5).sum())}-{int((s == 0).sum())}  "
                f"score {s.mean():.3f} ± {s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0:.3f}  "
                f"{np.mean(secs[arm]):.2f} s/search")
        both = sorted(set(seeds) & set(base))
        if arm != "default" and both:
            d = np.concatenate([seeds[k] - base[k] for k in both])
            line += f"  vs default {d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0:.3f}"
        print(line)


if __name__ == "__main__":
    main()
