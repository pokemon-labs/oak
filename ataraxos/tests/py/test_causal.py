"""Checks the causal net's path (ataraxos/nets/causal_net.py, ataraxos/nets/causal_actor.py) and
the env's bookkeeping state, in float32:

    ~/oak/.venv/bin/python tests/py/test_causal.py [--device cuda] [--games N] [TEAM_FILE ...]

- the state (src/state_tokens.hpp): the foe's active is its slot 1, ours
  is the request's, boosts within -6..6, nothing seen that doesn't exist;
- the learner, on the stored trajectories, gives every decision the
  log-probs and W/D/L values the actor gave it (the key/value ring past its
  window, game resets, the buffer's state column and its compaction across
  takes);
- W/D/L returns and advantages match a plain loop per trajectory, also
  with trajectories truncated (at_cap "bootstrap");
- the gradients accumulated over micro-batches (train.split) match one
  pass's, and the magnets are distributions over the legal choices (both
  from the HistoryNet's tests/py/test_train.py, removed 2026-10-30);
- on CUDA, the graphed actor (fixed shapes, padding rows) gives the eager
  actor's log-probs and values;
- one training iteration runs and its losses are finite.
"""

import argparse
import dataclasses
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ataraxos._ext import ataraxos_env  # noqa: E402
from ataraxos.rl.buffer import Buffer, returns_and_advantages  # noqa: E402
from ataraxos.nets.causal_actor import CausalActor  # noqa: E402
from ataraxos.nets import causal_net as cn  # noqa: E402
from ataraxos.nets.policy import CHOICES, SIZES  # noqa: E402
from ataraxos.rl import train  # noqa: E402

TEAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "teams", "selfplay", "oak1500.txt")
# the env's state (the Gen 1 bookkeeping) is phase 2: off until written
STATE = SIZES["cs"].state


def check_state(teams, seed):
    assert ataraxos_env.STATE_FIELDS == cn.STATE_FIELDS
    env = ataraxos_env.SelfPlay(64, seed, teams=teams, max_turns=200, state=STATE)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    F, A, M = cn.FIELD_FIELDS, cn.ACTIVE_FIELDS, cn.MON_FIELDS
    rows = 0
    for _ in range(300):
        s = obs["state"]
        assert s.shape == (len(obs["n"]), cn.STATE_FIELDS)
        foe, slot1 = s[:, F + A], s[:, F + 2 * A + 2]
        assert ((foe == slot1) | (foe == 0)).all()
        ours, req = s[:, F], obs["request"][:, 7 + 16]  # the request's side slot 0 species
        assert ((ours == req) | (ours == 0)).all()
        boosts = np.concatenate([s[:, F + 3:F + 10], s[:, F + A + 3:F + A + 10]], 1)
        assert ((boosts >= -6) & (boosts <= 6)).all()
        mons = s[:, F + 2 * A:].reshape(-1, 6, M)
        assert (mons[..., 1] <= mons[..., 0]).all()
        rows += len(s)
        obs = env.step((rng.random(len(obs["n"])) * obs["n"]).astype(np.uint8))
    plain = ataraxos_env.SelfPlay(4, seed, teams=teams)
    assert "state" not in plain.reset()
    return rows


def plain_returns(data, td_lambda, gae_lambda):
    out_r, out_a = [], []
    truncated = data.truncated if data.truncated is not None else np.zeros(len(data.length), bool)
    for s, l, r, u in zip(data.start, data.length, data.reward, truncated):
        v = torch.softmax(torch.from_numpy(data.value[s:s + l]).double(), -1).numpy()
        target = np.r_[v[1:], v[-1:] if u else np.eye(3)[r + 1][None]]
        delta = target - v
        e = delta[:, 2] - delta[:, 0]
        ret, adv = np.zeros((l, 3)), np.zeros(l)
        acc, sacc = np.zeros(3), 0.0
        for t in range(l - 1, -1, -1):
            acc = delta[t] + td_lambda * acc
            sacc = e[t] + gae_lambda * sacc
            ret[t], adv[t] = acc + v[t], sacc
        out_r.append(ret)
        out_a.append(adv)
    return np.concatenate(out_r), np.concatenate(out_a)


def accumulation_gap(learner, data, ret, adv, rng):
    """One pass over a group of trajectories against micro-batches of it
    (train.split), a random half of the decisions kept: the largest gap
    between the two gradients, relative to the one pass's largest entry."""
    keep = rng.random(len(data.n)) < 0.5
    group = rng.permutation(len(data.length))[:16]
    micros = train.split(data, group, keep, 5)
    assert len(micros) > 1
    kept = np.add.reduceat(keep.astype(np.int64), data.start)
    assert sorted(np.concatenate(micros).tolist()) == sorted(g for g in group.tolist() if kept[g])
    grads = []
    for parts in ([group], micros):
        learner.gradients(data, parts, keep, ret, adv, 0.05)
        grads.append(torch.cat([p.grad.flatten() for p in learner.net.parameters() if p.grad is not None]))
    learner.optim.zero_grad(set_to_none=True)
    assert grads[0].shape == grads[1].shape
    return ((grads[0] - grads[1]).abs().max() / grads[0].abs().max()).item()


def play(actor, env, games, buffer, obs):
    while buffer.games() < games:
        picks, logp, value = actor.act(obs)
        buffer.add(obs, picks, logp, value)
        obs = env.step(picks)
        buffer.end(obs)
    return obs


def check_learner(cfg, teams, games, seed):
    torch.manual_seed(seed)
    learner = train.Learner(cfg)
    learner.net.eval()
    dev = learner.dev
    env = ataraxos_env.SelfPlay(cfg.battles, seed, teams=teams, max_turns=cfg.max_turns, state=STATE)
    actor = CausalActor(learner.net, cfg.battles, dev, amp=False)
    buffer = Buffer(learner.net.c.max_lines)
    obs = env.reset()
    worst_p = worst_v = worst_g = 0.0
    decisions = 0
    for take in range(2):  # the second take reads trajectories compacted across the first
        obs = play(actor, env, games, buffer, obs)
        data = buffer.take()
        assert data.state is not None and data.state.shape == (len(data.n), cn.STATE_FIELDS)
        assert data.value.shape == (len(data.n), 3)
        assert data.length.max() > learner.net.c.window  # the ring wraps
        keep = np.ones(len(data.n), bool)
        with torch.no_grad():
            ids, logits, value, choices, n = learner.forward(data, np.arange(len(data.length)), keep)
        legal = torch.arange(CHOICES, device=dev)[None] < n[:, None]
        mag = train.magnet(choices, legal, "attack_switch").exp() * legal
        assert torch.allclose(mag.sum(-1), torch.ones(len(ids), device=dev), atol=1e-6)
        logp = torch.log_softmax(logits.float(), -1)
        old = torch.from_numpy(data.logp[ids]).to(dev)
        worst_p = max(worst_p, (logp - old).abs()[legal].max().item())
        worst_v = max(worst_v, (value.float().cpu() - torch.from_numpy(data.value[ids])).abs().max().item())
        decisions += len(ids)
        ret, adv = returns_and_advantages(data, cfg.td_lambda, cfg.gae_lambda)
        pr, pa = plain_returns(data, cfg.td_lambda, cfg.gae_lambda)
        assert np.allclose(ret.numpy(), pr, atol=1e-5) and np.allclose(adv.numpy(), pa, atol=1e-5)
        assert torch.allclose(ret.sum(-1), torch.ones(len(ret)), atol=1e-5)
        saved = data.truncated  # at_cap "bootstrap": the last target is the last value
        data.truncated = np.random.default_rng(len(data.n)).random(len(data.length)) < 0.5
        ret, adv = returns_and_advantages(data, cfg.td_lambda, cfg.gae_lambda)
        pr, pa = plain_returns(data, cfg.td_lambda, cfg.gae_lambda)
        assert np.allclose(ret.numpy(), pr, atol=1e-5) and np.allclose(adv.numpy(), pa, atol=1e-5)
        data.truncated = saved
        ret, adv = returns_and_advantages(data, cfg.td_lambda, cfg.gae_lambda)
        learner.net.train()
        worst_g = max(worst_g, accumulation_gap(learner, data, ret, adv, np.random.default_rng(take)))
        learner.net.eval()
    assert worst_p < 1e-4 and worst_v < 1e-4 and worst_g < 1e-4, (worst_p, worst_v, worst_g)
    learner.net.train()
    stats = learner.train(data, np.random.default_rng(seed))
    assert all(np.isfinite(stats[k]) for k in ("loss", "policy_loss", "value_loss", "kl_data", "kl_magnet"))
    return decisions, worst_p, worst_v, worst_g, stats


def check_neurd(seed):
    """neurd_loss's gradient on the logits against -A min(1 / mu(a), cap)
    (e_a - legal / n) / B, zero where PPO's clip binds or the push takes a logit beta past the
    legal mean further out, finite with -inf off the legal choices, and none
    on them."""
    g = torch.Generator().manual_seed(seed)
    B = 64
    n = torch.randint(1, CHOICES + 1, (B,), generator=g)
    legal = torch.arange(CHOICES)[None] < n[:, None]
    logits = (3 * torch.randn(B, CHOICES, generator=g)).masked_fill(~legal, float("-inf")).requires_grad_()
    old = torch.log_softmax(torch.randn(B, CHOICES, generator=g).masked_fill(~legal, -1e9), -1)
    a = (torch.rand(B, generator=g) * n).long()[:, None]
    A = torch.randn(B, generator=g)
    ratio = torch.rand(B, generator=g) * 2
    loss = train.neurd_loss(logits, legal, n, a, old, A, ratio, 0.2, 2.0, 20.0)
    loss.backward()
    keep = ~(((A > 0) & (ratio > 1.2)) | ((A < 0) & (ratio < 0.8)))
    assert 0 < keep.sum() < B
    inv_mu = 1 / old.gather(1, a)[:, 0].exp()
    assert (inv_mu > 20).any() and (inv_mu < 20).any()
    w = A * inv_mu.clamp(max=20) * keep
    f = w[:, None] * (torch.nn.functional.one_hot(a[:, 0], CHOICES) - legal.float() / n[:, None])
    x = logits.detach().masked_fill(~legal, 0)
    y = x - x.sum(-1, keepdim=True) / n[:, None]
    held = legal & (f != 0) & (((f > 0) & (y >= 2)) | ((f < 0) & (y <= -2)))
    assert held.any() and (legal & (f != 0) & ~held).any()
    want = -f.masked_fill(held, 0) / B
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert (logits.grad[~legal] == 0).all()
    worst = (logits.grad - want).abs().max().item()
    assert worst < 1e-5, worst
    return worst


def check_graphs(teams, seed, battles=96):
    """The graphed actor against the eager one on the same games (float32;
    the eager actor's picks drive both)."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    net = cn.CausalPolicy(SIZES["cs_tiny"]).to(dev).eval()
    eager = CausalActor(net, battles, dev, amp=False)
    graphed = CausalActor(net, battles, dev, amp=False, graphs=True, bucket=64)
    env = ataraxos_env.SelfPlay(battles, seed, teams=teams, max_turns=200, state=STATE)
    obs = env.reset()
    worst = 0.0
    for _ in range(150):
        picks, lp, v = eager.act(obs)
        _, lg, vg = graphed.act(obs)
        live = torch.isfinite(lp)
        worst = max(worst, (lp - lg).abs()[live].max().item(), (v - vg).abs().max().item())
        obs = env.step(picks)
    assert worst < 1e-3, worst
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("teams", nargs="*")
    p.add_argument("--device", default="cpu")
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()
    teams = args.teams or [TEAMS]
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if STATE:
        print(f"state: {check_state(teams, args.seed)} rows, invariants hold")
    worst, params = cn.check()
    print(f"causal_net.check: {worst:.2e}; {params / 1e6:.2f}M params at cs")
    cfg = train.Config(size="cs_tiny", device=args.device, battles=16, threads=4, minibatch=256, microbatch=0,
                       amp=False, compile=False, graphs=False, teams=teams)
    decisions, wp, wv, wg, stats = check_learner(cfg, teams, args.games, args.seed)
    print(f"learner vs actor: {decisions} decisions, max |logp diff| {wp:.2e}, max |value diff| {wv:.2e}; "
          f"accumulated vs one-pass gradients {wg:.2e} (relative); "
          f"W/D/L returns match the plain loop; a step: loss {stats['loss']:.3f}")
    print(f"neurd_loss's gradient vs the formula: {check_neurd(args.seed):.2e}")
    cfg = dataclasses.replace(cfg, policy_loss="neurd")
    _, _, _, _, stats = check_learner(cfg, teams, args.games, args.seed)
    print(f"a neurd step: loss {stats['loss']:.3f}, logit_abs_max {stats['logit_abs_max']:.3f}")
    if args.device == "cuda":
        print(f"graphed vs eager actor: max diff {check_graphs(teams, args.seed):.2e}")
    print("ok")


if __name__ == "__main__":
    main()
