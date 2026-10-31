"""Checks ataraxos/rl/train.py's --resume on a CPU, with a tiny net:

    ~/oak/.venv/bin/python tests/py/test_resume.py TEAM_FILE ...

Runs 2 iterations saving each, then resumes from `latest` for 2 more, and
checks: the resumed learner starts from the checkpoint's weights, EMA,
optimizer and RNGs; the iterations carry on (3, 4); Exp3's rounds and the
team draw counts carry on rather than restart; log.jsonl has the resume
marker; a checkpoint from another size is refused. The games in progress
are dropped, so the resumed run isn't compared with an uninterrupted one.
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ataraxos.rl import train  # noqa: E402


def main():
    team_files = sys.argv[1:]
    with tempfile.TemporaryDirectory() as run:
        common = ["--run", run, "--size", "cs_tiny", "--device", "cpu", "--battles", "32", "--threads", "2",
                  "--games_per_iter", "32", "--minibatch", "64", "--eval_every", "2", "--eval_battles", "16",
                  "--save_every", "1", *team_files]
        train.main(["--iters", "2", *common])
        ck2 = torch.load(os.path.join(run, "ckpt2.pt"), weights_only=False)
        assert ck2["it"] == 2

        # what the resumed learner holds before its first step
        cfg = train.parse(["--iters", "2", *common])
        learner = train.Learner(cfg)
        learner.load(ck2)
        for a, b in zip(learner.net.state_dict().values(), ck2["net"].values()):
            assert torch.equal(a, b)
        for a, b in zip(learner.ema.state_dict().values(), ck2["ema"].values()):
            assert torch.equal(a, b)
        assert learner.optim.state_dict()["state"].keys() == ck2["optim"]["state"].keys()

        train.main(["--iters", "4", "--resume", "latest", *common])
        ck4 = torch.load(os.path.join(run, "ckpt4.pt"), weights_only=False)
        ck3 = torch.load(os.path.join(run, "ckpt3.pt"), weights_only=False)
        assert ck3["it"] == 3 and ck4["it"] == 4
        assert not all(torch.equal(a, b) for a, b in zip(ck2["net"].values(), ck3["net"].values()))
        if team_files:
            c2, c4 = np.asarray(ck2["team_counts"]), np.asarray(ck4["team_counts"])
            assert (c4 >= c2).all() and c4.sum() > c2.sum(), "draw counts restarted"
            assert ck4["exp3"]["rounds"] > ck2["exp3"]["rounds"], "Exp3 restarted"
        lines = [json.loads(l) for l in open(os.path.join(run, "log.jsonl"))]
        assert [l.get("it", l.get("resume")) for l in lines] == [1, 2, 2, 3, 4], lines
        assert "resume" in lines[2]
        assert os.path.exists(os.path.join(run, "config_resume2.json"))
        assert not any(f.endswith(".tmp") for f in os.listdir(run))

        try:
            train.main(["--iters", "5", "--resume", "latest", *common[:3], "cs", *common[4:]])
        except ValueError as e:
            assert "size" in str(e)
        else:
            raise AssertionError("a checkpoint of another size was loaded")
    print("ok: resume")


if __name__ == "__main__":
    main()
