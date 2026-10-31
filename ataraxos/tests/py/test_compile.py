"""Compiled against eager (train.compile_net), on the same weights:

    python tests/py/test_compile.py [--ckpt PATH] [--device cuda] [--games N] [TEAM_FILE ...]

Two copies of the net, one eager and one compiled, act on the same
decisions (the eager one picks), then the learner's forward and gradients
run on the stored trajectories with each. Reports the largest log-prob and
value gaps and the gradients' relative gap, beside the same gaps between
eager bf16 and eager float32 for scale, and each copy's time after a
warm-up. Without --ckpt the net is cs_tiny, randomly initialized.
Asserts that compiled float32 matches eager float32 to 1e-4 (CPU, or
CUDA with --fp32); bf16 gaps are reported, not checked.
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ataraxos._ext import ataraxos_env  # noqa: E402
from ataraxos.rl.buffer import Buffer, returns_and_advantages  # noqa: E402
from ataraxos.nets.causal_actor import CausalActor  # noqa: E402
from ataraxos.nets.causal_net import make_net  # noqa: E402
from ataraxos.nets.policy import CHOICES, SIZES  # noqa: E402
from ataraxos.rl import train  # noqa: E402


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def logp_gap(a, b, n):
    legal = torch.arange(CHOICES, device=n.device)[None] < n[:, None]
    return (torch.log_softmax(a.float(), -1) - torch.log_softmax(b.float(), -1))[legal].abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="autocast off on CUDA")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--battles", type=int, default=64)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--mem", type=float, default=0.0, help="CUDA memory fraction cap (0: none)")
    ap.add_argument("teams", nargs="*")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cuda" and args.mem > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem, dev.index or 0)
    torch.manual_seed(0)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        size = ck["config"]["size"]
        state = ck["net"]
    else:
        size, state = "cs_tiny", None
    amp = dev.type == "cuda" and not args.fp32
    cfg = train.Config(size=size, device=args.device, battles=args.battles, teams=args.teams, amp=amp,
                       graphs=False)
    eager = train.Learner(cfg)
    if state is not None:
        eager.net.load_state_dict(state)
    eager.net.eval()
    comp = train.Learner(cfg)
    comp.net.load_state_dict(eager.net.state_dict())
    train.compile_net(comp.net)
    comp.net.eval()
    ref = make_net(SIZES[size]).to(dev).eval()  # float32 eager, for scale
    ref.load_state_dict(eager.net.state_dict())

    env = ataraxos_env.SelfPlay(args.battles, 3, teams=args.teams, max_turns=100, threads=4, state=SIZES[size].state)
    actors = [CausalActor(eager.net, args.battles, dev, amp=amp), CausalActor(comp.net, args.battles, dev, amp=amp),
              CausalActor(ref, args.battles, dev, amp=False)]
    times = [0.0, 0.0, 0.0]
    buffer = Buffer(eager.net.c.max_lines)
    obs = env.reset()
    gap_p = gap_v = ref_p = ref_v = 0.0
    steps = 0
    while buffer.games() < args.games:
        out = []
        for i, a in enumerate(actors):
            sync(dev)
            t = time.perf_counter()
            out.append(a.act(obs))
            sync(dev)
            if steps >= 20:  # after the compiles
                times[i] += time.perf_counter() - t
        (picks, lp_e, v_e), (_, lp_c, v_c), (_, lp_r, v_r) = out
        legal = torch.arange(CHOICES, device=dev)[None] < torch.from_numpy(obs["n"]).to(dev)[:, None]
        if len(picks):
            gap_p = max(gap_p, (lp_e - lp_c)[legal].abs().max().item())
            gap_v = max(gap_v, (v_e - v_c).abs().max().item())
            ref_p = max(ref_p, (lp_e - lp_r)[legal].abs().max().item())
            ref_v = max(ref_v, (v_e - v_r).abs().max().item())
        buffer.add(obs, picks, lp_e, v_e)
        obs = env.step(picks)
        buffer.end(obs)
        steps += 1
    print(f"actor, {steps} steps, {buffer.size} decisions buffered: "
          f"compiled vs eager log-probs {gap_p:.2e}, values {gap_v:.2e}; "
          f"eager vs float32 log-probs {ref_p:.2e}, values {ref_v:.2e}")
    print(f"actor time after 20 steps: eager {times[0]:.2f} s, compiled {times[1]:.2f} s, "
          f"float32 eager {times[2]:.2f} s")

    data = buffer.take()
    keep = np.ones(len(data.n), bool)
    ret, adv = returns_and_advantages(data, cfg.td_lambda, cfg.gae_lambda)
    groups = np.array_split(np.random.default_rng(0).permutation(len(data.length)), args.groups)
    lg_p = lg_v = 0.0
    for g in groups:
        with torch.no_grad():
            _, le, ve, _, n = eager.forward(data, g, keep)
            _, lc, vc, _, _ = comp.forward(data, g, keep)
        lg_p = max(lg_p, logp_gap(le, lc, n))
        lg_v = max(lg_v, (ve.float() - vc.float()).abs().max().item())
    grads, ltimes = [], []
    for learner in (eager, comp):
        learner.net.train()
        for rep in range(3):  # the first passes compile
            sync(dev)
            t = time.perf_counter()
            learner.gradients(data, groups, keep, ret, adv, 0.05)
            sync(dev)
        ltimes.append(time.perf_counter() - t)
        grads.append(torch.cat([p.grad.flatten().float() for p in learner.net.parameters() if p.grad is not None]))
    g_gap = ((grads[0] - grads[1]).abs().max() / grads[0].abs().max()).item()
    cos = torch.nn.functional.cosine_similarity(grads[0], grads[1], 0).item()
    print(f"learner, {len(data.n)} decisions: compiled vs eager log-probs {lg_p:.2e}, values {lg_v:.2e}; "
          f"gradients {g_gap:.2e} (relative max), cosine {cos:.6f}")
    print(f"learner time (one minibatch of {len(groups)} micro-batches): eager {ltimes[0]:.2f} s, "
          f"compiled {ltimes[1]:.2f} s")
    if not amp:
        assert gap_p < 1e-4 and gap_v < 1e-4 and lg_p < 1e-4 and lg_v < 1e-4 and g_gap < 1e-3, "float32 gap"
        print("ok: float32 compiled matches eager")


if __name__ == "__main__":
    main()
