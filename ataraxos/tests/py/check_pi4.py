"""Checks pi4's Python side (2026-11-24, docs/pi4.md), on the box (a GPU, a
few minutes):
  1. ataraxos/pi4/env.py's rows: shapes, a decision index for every side
     with 2+ choices and none otherwise, every decision answered once, games
     ending under random play;
  2. the net (ataraxos/pi4/net.py): its size, logits -1e9 past n, Q's shape;
  3. AsyncActor with CUDA graphs (and torch.compile) against the eager
     Actor on the same rows: log-probs, values and Q columns;
  4. NumpyBuffer + Expected SARSA: a take's Q columns and targets, and one
     learner step (finite losses, q_loss there);
  5. a few eval games (ataraxos/pi4/agents.py) against random, maxdamage
     and the checkpoints given (rby4, pi3).

    python tests/py/check_pi4.py TEAM_FILE [CHECKPOINT ...]
"""

import sys
import time

import numpy as np
import torch

from ataraxos.pi import train
from ataraxos.pi.fast import AsyncActor, NumpyBuffer
from ataraxos.pi4 import agents
from ataraxos.pi4.env import Pi4Env
from ataraxos.pi4.net import PI4_FIELDS, make_net
from ataraxos.rl.buffer import sarsa_targets


def random_picks(rng, obs):
    n = np.maximum(obs["n"].astype(np.int64), 1)
    return (rng.random(n.shape) * n).astype(np.uint8)


def main():
    teams = [sys.argv[1]]
    ckpts = sys.argv[2:]
    dev = "cuda"
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # 1. env rows
    env = Pi4Env(256, 1, teams=teams, threads=4)
    obs = env.reset()
    ends = rows = 0
    for _ in range(300):
        R = len(obs["n"])
        assert obs["state"].shape == (R, 2, PI4_FIELDS) and obs["request"].shape[:2] == (R, 2)
        assert obs["choices"].shape == (R, 2, 9) and obs["n"].shape == (R, 2)
        dec = env.dec
        assert ((obs["n"] >= 2) == (dec >= 0)).all(), "a side decides iff it has a decision"
        used = np.sort(dec[dec >= 0])
        D = env.env.decisions
        assert len(used) == D and (used == np.arange(D)).all(), "every decision answered once"
        rows += R
        obs = env.step(random_picks(rng, obs))
        ends += len(obs["end_battle"])
    assert ends > 0
    print(f"1. env: {rows} rows over 300 steps, {ends} games ended", flush=True)

    # 2. the net
    net = make_net("pi4").to(dev).eval()
    params = sum(p.numel() for p in net.parameters())
    x = {k: torch.from_numpy(obs[k]).to(dev) for k in train.PI4_COLUMNS}
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        value, logits, q = net(x["state"], x["request"], x["choices"], x["n"], with_q=True)
    R = len(obs["n"])
    assert value.shape == (R, 3) and logits.shape == (R, 2, 9) and q.shape == (R, 2, 9, 3)
    past = torch.arange(9, device=dev)[None, None] >= x["n"].clamp(min=1)[..., None]
    assert (logits[past] == -1e9).all()
    print(f"2. net: pi4 {params / 1e6:.2f}M params; logits masked past n; Q [R, 2, 9, 3]", flush=True)

    # 3. graphed + compiled actor = eager
    model = train.Model("pi4", q=True)
    eager = train.Actor(net, dev, True, model)
    _, lp_e, v_e, q_e = eager.act(obs)
    t = time.perf_counter()
    fast = AsyncActor(net, dev, True, capacity=env.battles, graphs=True, compile=True, inputs=model.inputs,
                      logits=model.actor_logits, q=True)
    _, lp_g, v_g, q_g = fast.act(obs)
    live = obs["n"][..., None] > np.arange(9)
    err = dict(logp=np.abs(lp_e - lp_g)[live].max(), value=np.abs(v_e - v_g).max(),
               q_exp=np.abs(q_e["q_exp"] - q_g["q_exp"]).max())
    print(f"3. graphed + compiled actor ({time.perf_counter() - t:.0f}s to build) vs eager: "
          + ", ".join(f"{k} {v:.2e}" for k, v in err.items()), flush=True)
    assert err["logp"] < 0.1 and err["value"] < 0.1 and err["q_exp"] < 0.05

    # 4. buffer, SARSA, one learner step (pi4_tiny)
    cfg = train.Config(size="pi4_tiny", expected_sarsa=True, buffer="numpy", minibatch=4096, compile_train="",
                       device=dev)
    learner = train.Learner(cfg)
    actor = train.Actor(learner.net, dev, True, learner.model)
    buf = NumpyBuffer("draw", True, train.PI4_COLUMNS)
    env = Pi4Env(256, 2, teams=teams, threads=4)
    obs = env.reset()
    while buf.games() < 128:
        picks, logp, value, qx = actor.act(obs)
        buf.add(obs, picks, logp, value, qx)
        obs = env.step(picks)
        buf.end(obs)
    data = buf.take()
    target, q_adv = sarsa_targets(data)
    assert data.q_exp.shape == (len(data.n), 3) and target.shape == (len(data.n), 3)
    assert torch.allclose(target.sum(-1), torch.ones(len(target)), atol=1e-4)
    stats = learner.train(data, rng)
    assert np.isfinite(stats["loss"]) and "q_loss" in stats
    print(f"4. buffer + SARSA: {len(data.length)} trajectories, {len(data.n)} decisions; learner step loss "
          f"{stats['loss']:.3f} q_loss {stats['q_loss']:.3f} adv_q_gae_corr {stats['adv_q_gae_corr']:.2f}",
          flush=True)

    # 5. evals
    ours = agents.Pi4Agent(learner.ema, dev)
    env_of = agents.eval_env(7, teams, 300, 4)
    for name in ["random", "maxdamage", *ckpts]:
        opp = agents.make_agent(name, dev, 7)
        t = time.perf_counter()
        w, ti, l = agents.match(env_of, ours, opp, 64)
        print(f"5. untrained pi4_tiny vs {name}: {w}-{ti}-{l} ({time.perf_counter() - t:.0f}s)", flush=True)
    print("check_pi4: ok")


if __name__ == "__main__":
    main()
