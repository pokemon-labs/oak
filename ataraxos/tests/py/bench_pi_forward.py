"""The perfect-info actor's step (forward, legal logits, sampling) in rows/s
on the GPU: eager vs torch.compile vs AsyncActor's CUDA graph, at several
row counts, bf16 autocast (docs/pi_speed.md, benchmark step 6). Inputs are
real env rows (random self-play), tiled to the row count.

    python tests/py/bench_pi_forward.py [SIZE] [ROWS,...] [SECONDS]
"""

import sys
import time

import numpy as np
import torch

from ataraxos._ext import ataraxos_pi
from ataraxos.pi.fast import INPUTS, AsyncActor
from ataraxos.pi.net import CHOICES, legal_logits, make_net


def rows_from_env(R):
    env = ataraxos_pi.SelfPlay(min(R, 4096), 1, teams=["teams/selfplay/oak150.txt"], max_turns=200, threads=4)
    rng = np.random.default_rng(1)
    obs = env.reset()
    for _ in range(20):  # past the opening
        n = np.maximum(obs["n"].astype(np.int64), 1)
        obs = env.step((rng.random(n.shape) * n).astype(np.uint8))
    reps = -(-R // len(obs["n"]))
    return {k: np.ascontiguousarray(np.concatenate([obs[k]] * reps)[:R]) for k in INPUTS}


def timeit(fn, seconds):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
        n += 1
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    size = sys.argv[1] if len(sys.argv) > 1 else "pi"
    counts = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "4096,16384,65536,131072").split(",")]
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    dev = torch.device("cuda")
    torch.set_num_threads(4)  # the host's CPU count oversubscribes a box's CPU quota (train.py's --torch_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    net = make_net(size).to(dev).eval()
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, {size} "
          f"({sum(p.numel() for p in net.parameters()) / 1e3:.0f}k params)")

    def step(x):
        with torch.autocast("cuda", torch.bfloat16):
            value, policy = net(x["mon"].long(), x["hp"], x["active"])
        logits, _ = legal_logits(policy, x["policy"], x["n"])
        logp = torch.log_softmax(logits, -1)
        u = torch.rand_like(logp).clamp_(1e-10, 1 - 1e-7)
        return (logp - torch.log(-torch.log(u))).argmax(-1), logp, value

    compiled = torch.compile(step, mode="max-autotune-no-cudagraphs", dynamic=False)
    print(f"{'rows':>8} | {'eager':>12} {'compile':>12} {'graph':>12} {'graph+copies':>13} {'cgraph':>12}"
          f" {'cgraph+copies':>14}  (M rows/s; cgraph: the compiled forward in the graph)")
    for R in counts:
        obs = rows_from_env(R)
        x = {k: torch.from_numpy(v).to(dev) for k, v in obs.items()}
        out = {}
        with torch.no_grad():
            out["eager"] = timeit(lambda: step(x), seconds)
            try:
                out["compile"] = timeit(lambda: compiled(x), seconds)
            except Exception as e:  # noqa: BLE001 (a missing compiler shouldn't stop the bench)
                print(f"compile failed: {type(e).__name__}: {e}")
                out["compile"] = float("nan")
        actor = AsyncActor(net, dev, True, capacity=R, slots=1, graphs=True)
        actor.collect(actor.launch(obs))  # captures
        out["graph"] = timeit(lambda: actor.graph.replay(), seconds)
        out["copies"] = timeit(lambda: actor.collect(actor.launch(obs)), seconds)
        cactor = AsyncActor(net, dev, True, capacity=R, slots=1, graphs=True, compile=True)
        cactor.collect(cactor.launch(obs))  # compiles, captures
        out["cgraph"] = timeit(lambda: cactor.graph.replay(), seconds)
        out["ccopies"] = timeit(lambda: cactor.collect(cactor.launch(obs)), seconds)
        print(f"{R:8d} | " + " ".join(f"{R / out[k] / 1e6:12.2f}" for k in ("eager", "compile", "graph"))
              + f" {R / out['copies'] / 1e6:13.2f} {R / out['cgraph'] / 1e6:12.2f}"
              + f" {R / out['ccopies'] / 1e6:14.2f}", flush=True)


if __name__ == "__main__":
    main()
