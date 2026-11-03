"""Checks the causal net's path (ataraxos/nets/causal_net.py, ataraxos/nets/causal_actor.py) and
the env's bookkeeping state, in float32:

    ~/oak/.venv/bin/python tests/py/test_causal.py [--device cuda] [--games N] [TEAM_FILE ...]

- the state (src/state_tokens.hpp): each active is its side's active
  Pokémon (ours the request's) unless transformed, boosts within -6..6,
  nothing seen that doesn't exist, an unseen Pokémon and an empty move
  slot all zeros, our team all seen; the matchup features
  (src/features.hpp) in their ranges, an empty move slot's all zeros, and
  the request's moves found among our active's (counted);
- cs3 (features on) loaded from a cs_tiny net's weights gives cs_tiny's
  log-probs and values (its feature layers start at zero), and the learner
  vs actor check at cs3_tiny;
- the learner, on the stored trajectories, gives every decision the
  log-probs and W/D/L values the actor gave it (the key/value ring past its
  window, game resets, the buffer's state column and its compaction across
  takes);
- W/D/L returns and advantages match a plain loop per trajectory, also
  with trajectories truncated (at_cap "bootstrap");
- the gradients accumulated over micro-batches (train.split) match one
  pass's, and the magnets are distributions over the legal choices (both
  from the HistoryNet's tests/py/test_train.py, removed 2026-10-30);
- Expected SARSA (`expected_sarsa`): off, the net has no Q parameters and
  the others' init is the same as with it on; on, the learner gives each
  decision the actor's Q at the pick and summed over the data policy, the
  targets and advantages match a plain loop (truncated too), `init` goes
  both ways between the two, and a step's Q loss is finite;
- on CUDA, the graphed actor (fixed shapes, padding rows) gives the eager
  actor's log-probs and values, and Q;
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
from ataraxos.rl.buffer import Buffer, returns_and_advantages, sarsa_targets  # noqa: E402
from ataraxos.nets.causal_actor import CausalActor  # noqa: E402
from ataraxos.nets import causal_net as cn  # noqa: E402
from ataraxos.nets.policy import CHOICES, SIZES  # noqa: E402
from ataraxos.rl import train  # noqa: E402

TEAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "teams", "selfplay", "oak1500.txt")
STATE = SIZES["cs"].state


def check_state(teams, seed):
    assert ataraxos_env.STATE_FIELDS == cn.STATE_FIELDS
    env = ataraxos_env.SelfPlay(64, seed, teams=teams, max_turns=200, state=STATE)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    A, M = cn.ACTIVE_FIELDS, cn.MON_FIELDS
    transformed = 1 << 15  # the last active flag
    rows = missing = move_rows = 0
    for _ in range(300):
        s = obs["state"]
        assert s.shape == (len(obs["n"]), cn.STATE_FIELDS)
        act = s[:, 1:1 + 2 * A].reshape(-1, 2, A)
        mons = s[:, 1 + 2 * A:cn.BASE_STATE_FIELDS].reshape(-1, 2, 6, M)
        exists, seen, active = mons[..., 0], mons[..., 1], mons[..., 2]
        assert (seen <= exists).all() and (active <= seen).all() and (active.sum(-1) <= 1).all()
        # an unseen Pokémon is all zeros past its flags; an empty slot has no PP
        assert (mons[..., 3:][seen == 0] == 0).all()
        assert (mons[..., 13:17][mons[..., 9:13] == 0] == 0).all()
        assert (act[..., 17:21][act[..., 13:17] == 0] == 0).all()
        # our side is the request's team, all seen
        assert (seen[:, 0] == exists[:, 0]).all()
        # each active is its side's active Pokémon, unless transformed
        for side in range(2):
            species = act[:, side, 0]
            mon = (mons[:, side, :, 3] * active[:, side]).sum(-1)
            ok = (species == mon) | (species == 0) | ((act[:, side, -1].astype(np.uint16) & transformed) > 0)
            assert ok.all(), side
        req = obs["request"][:, 7 + 16]  # the request's side slot 0 species: our active
        ours = (mons[:, 0, :, 3] * active[:, 0]).sum(-1)
        assert ((ours == req) | (ours == 0)).all()
        boosts = act[..., 3:9]
        assert ((boosts >= -6) & (boosts <= 6)).all()
        assert (s[:, 0] >= 0).all() and (s[:, 0] <= 1000).all()
        f = s[:, cn.BASE_STATE_FIELDS:]
        mv = f[:, :8 * cn.MOVE_FEATS].reshape(-1, 2, 4, cn.MOVE_FEATS)
        assert (mv[act[..., 13:17] == 0] == 0).all()
        assert ((mv[..., 0] >= 0) & (mv[..., 0] <= mv[..., 1]) & (mv[..., 1] <= 2000)).all()
        assert ((mv[..., 2] >= 0) & (mv[..., 2] <= 2000)).all()
        assert ((mv[..., 3:5] >= 0) & (mv[..., 3:5] <= 256)).all()
        assert ((mv[..., 5] >= 0) & (mv[..., 5] <= 255) & (mv[..., 6] >= 0) & (mv[..., 6] <= 256)).all()
        assert np.isin(mv[..., 7], [0, 1, 2, 4, 8, 16]).all() and np.isin(mv[..., 8], [-1, 0, 1]).all()
        sw = f[:, 8 * cn.MOVE_FEATS:].reshape(-1, 6, cn.SWITCH_FEATS)
        assert ((sw[..., [0, 2]] >= 0) & (sw[..., [0, 2]] <= 2000)).all()
        assert ((sw[..., [1, 3]] >= 0) & (sw[..., [1, 3]] <= 256)).all() and np.isin(sw[..., 4], [-1, 0, 1]).all()
        assert (sw[mons[:, 0, :, 5] == 0] == 0).all()  # a fainted Pokémon
        rid = obs["request"][:, 7:7 + 16:4].astype(np.int64)
        moves = (obs["request"][:, 0] == 0) & (rid > 0).any(1)  # a move request
        have = (rid[:, :, None] == act[:, 0, None, 13:17]).any(-1) | (rid <= 0)
        missing += int((~have.all(1) & moves).sum())
        move_rows += int(moves.sum())
        rows += len(s)
        obs = env.step((rng.random(len(obs["n"])) * obs["n"]).astype(np.uint8))
    plain = ataraxos_env.SelfPlay(4, seed, teams=teams)
    assert "state" not in plain.reset()
    return rows, missing, move_rows


def check_cs3_init(teams, seed):
    """A cs3-shaped net with cs_tiny's weights (its feature layers at
    zero) against cs_tiny itself, on a few self-play decisions: the largest
    log-prob and value differences."""
    torch.manual_seed(seed)
    old = cn.make_net(SIZES["cs_tiny"]).eval()
    new = cn.make_net(SIZES["cs3_tiny"]).eval()
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(cn.FEATURE_PARAMS) for k in missing), (missing, unexpected)
    env = ataraxos_env.SelfPlay(32, seed, teams=teams, max_turns=200, state=True)
    a, b = CausalActor(old, 32, torch.device("cpu"), amp=False), CausalActor(new, 32, torch.device("cpu"), amp=False)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    wp = wv = 0.0
    with torch.no_grad():
        for _ in range(60):
            _, lp, v = a.act(obs)
            _, lq, w = b.act(obs)
            live = torch.isfinite(lp)
            wp = max(wp, (lp[live] - lq[live]).abs().max().item() if live.any() else 0.0)
            wv = max(wv, (v - w).abs().max().item() if len(v) else 0.0)
            obs = env.step((rng.random(len(obs["n"])) * obs["n"]).astype(np.uint8))
    return wp, wv


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
        assert (data.state is not None and data.state.shape == (len(data.n), cn.STATE_FIELDS)) if STATE else data.state is None
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


def plain_sarsa(data):
    """sarsa_targets as a loop per trajectory."""
    target, adv = np.zeros((len(data.n), 3)), np.zeros(len(data.n))
    for g in range(len(data.length)):
        s, L = data.start[g], data.length[g]
        for t in range(L):
            i = s + t
            if t + 1 < L:
                target[i] = data.q_exp[i + 1]
            elif data.truncated is not None and data.truncated[g]:
                target[i] = data.q_exp[i]
            else:
                target[i] = np.eye(3)[data.reward[g] + 1]
            adv[i] = (data.q_pick[i, 2] - data.q_exp[i, 2]) - (data.q_pick[i, 0] - data.q_exp[i, 0])
    return target, adv


def check_sarsa(cfg, teams, games, seed):
    """`expected_sarsa` on against off (see the top). Returns (decisions,
    max |Q at the pick diff|, max |Q over the policy diff|, the step's stats)."""
    on = dataclasses.replace(cfg, expected_sarsa=True)
    torch.manual_seed(seed)
    plain = train.Learner(cfg)
    torch.manual_seed(seed)
    learner = train.Learner(on)
    a, b = plain.net.state_dict(), {k: v.clone() for k, v in learner.net.state_dict().items()}
    assert not any(k.startswith(cn.Q_PARAMS) for k in a) and any(k.startswith(cn.Q_PARAMS) for k in b)
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    ck = lambda l, c: dict(config=dataclasses.asdict(c), net=l.net.state_dict(), ema=l.ema.state_dict())
    learner.init_from(ck(plain, cfg))  # off -> on: the Q head keeps its init
    assert all(torch.equal(v, b[k]) for k, v in learner.net.state_dict().items())
    plain.init_from(ck(learner, on))  # on -> off: the Q head is dropped
    learner.net.eval()
    dev = learner.dev
    env = ataraxos_env.SelfPlay(cfg.battles, seed, teams=teams, max_turns=cfg.max_turns, state=STATE)
    actor = CausalActor(learner.net, cfg.battles, dev, amp=False, q=True)
    buffer = Buffer(learner.net.c.max_lines)
    obs = env.reset()
    while buffer.games() < games:
        picks, logp, value = actor.act(obs)
        assert actor.last_q.shape == (len(picks), CHOICES, 3)
        buffer.add(obs, picks, logp, value, actor.last_q)
        obs = env.step(picks)
        buffer.end(obs)
    data = buffer.take()
    assert data.q_pick.shape == data.q_exp.shape == (len(data.n), 3)
    keep = np.ones(len(data.n), bool)
    dd = learner.device_data(data)
    with torch.no_grad():
        ids, _, logits, _, _, n, q = cn.learner_forward(learner.net, dd, np.arange(len(data.length)), keep, dev,
                                                       False, with_q=True)
    P = torch.softmax(q.float(), -1).cpu()
    pi = torch.softmax(logits.float(), -1).cpu()
    pick = torch.from_numpy(data.pick[ids].astype(np.int64))
    worst_pick = (P[torch.arange(len(ids)), pick] - torch.from_numpy(data.q_pick[ids])).abs().max().item()
    worst_exp = ((pi[..., None] * P).sum(1) - torch.from_numpy(data.q_exp[ids])).abs().max().item()
    for truncated in (data.truncated, np.random.default_rng(len(data.n)).random(len(data.length)) < 0.5):
        data.truncated = truncated
        target, adv = sarsa_targets(data)
        pt, pa = plain_sarsa(data)
        assert np.allclose(target.numpy(), pt, atol=1e-6) and np.allclose(adv.numpy(), pa, atol=1e-6)
        assert torch.allclose(target.sum(-1), torch.ones(len(target)), atol=1e-5)
    assert worst_pick < 1e-4 and worst_exp < 1e-4, (worst_pick, worst_exp)
    learner.net.train()
    stats = learner.train(data, np.random.default_rng(seed))
    assert all(np.isfinite(stats[k]) for k in ("loss", "q_loss", "value_loss", "adv_q_std", "adv_q_gae_corr"))
    return len(ids), worst_pick, worst_exp, stats


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
    worst = 0.0
    for q in (False, True):
        net = cn.CausalPolicy(dataclasses.replace(SIZES["cs_tiny"], q_head=q)).to(dev).eval()
        eager = CausalActor(net, battles, dev, amp=False, q=q)
        graphed = CausalActor(net, battles, dev, amp=False, graphs=True, bucket=64, q=q)
        env = ataraxos_env.SelfPlay(battles, seed, teams=teams, max_turns=200, state=STATE)
        obs = env.reset()
        for _ in range(150):
            picks, lp, v = eager.act(obs)
            _, lg, vg = graphed.act(obs)
            live = torch.isfinite(lp)
            worst = max(worst, (lp - lg).abs()[live].max().item(), (v - vg).abs().max().item())
            assert (eager.last_q is None) == (graphed.last_q is None) == (not q)
            if q:
                worst = max(worst, (eager.last_q - graphed.last_q).abs().max().item())
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
        rows, missing, move_rows = check_state(teams, args.seed)
        print(f"state: {rows} rows, invariants hold; request moves not among our active's in "
              f"{missing} of {move_rows} move requests")
        wp, wv = check_cs3_init(teams, args.seed)
        print(f"cs3 from cs_tiny's weights vs cs_tiny: max |logp diff| {wp:.2e}, max |value diff| {wv:.2e}")
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
    decisions, wq, we, stats = check_sarsa(dataclasses.replace(cfg, policy_loss="ppo"), teams, args.games, args.seed)
    print(f"expected sarsa: off has no Q and the same init; learner vs actor: {decisions} decisions, "
          f"max |Q pick diff| {wq:.2e}, max |Q over pi diff| {we:.2e}; targets match the plain loop; "
          f"a step: q_loss {stats['q_loss']:.3f}, adv_q_std {stats['adv_q_std']:.4f}, "
          f"corr with GAE {stats['adv_q_gae_corr']:.3f}")
    cfg = dataclasses.replace(cfg, size="cs3_tiny", policy_loss=train.Config.policy_loss)
    decisions, wp, wv, _, stats = check_learner(cfg, teams, args.games, args.seed)
    print(f"cs3_tiny learner vs actor: {decisions} decisions, max |logp diff| {wp:.2e}, "
          f"max |value diff| {wv:.2e}; a step: loss {stats['loss']:.3f}")
    if args.device == "cuda":
        print(f"graphed vs eager actor: max diff {check_graphs(teams, args.seed):.2e}")
    print("ok")


if __name__ == "__main__":
    main()
