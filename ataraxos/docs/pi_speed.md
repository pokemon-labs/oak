# Perfect-info training: speed plan

2026-11-22. Goal: the most training games per rented dollar for the
perfect-info net. Nothing here has been measured on a GPU yet.

## Where things stand (main2)

- **Env:** `src/pi_env.hpp`, C++ self-play on oak's libpkmn_chance, one
  row per battle step, OpenMP over shards. 3.2M rows/s random play on 2
  laptop threads.
- **Net:** `ataraxos/pi/net.py`, size `pi`: d 64, 2 heads, 1 block,
  MLP x2, pool, MLP 128, W/D/L value, two policy heads. 203k params (78k
  in the Pokémon / moves / active subnets).
- **Trainer:** `ataraxos/pi/train.py`, ataraxos's PPO (rby4's flags, no
  SARSA). 8192 battles, 8192 games an iteration, minibatch 12288, one
  epoch, eager PyTorch.
- **Not done:** CUDA graphs / `torch.compile` for the actor, any GPU run,
  search during training.

## Plan

1. **Rent a cheap 30xx** (8 GB+) on a host with a CPU like the one we'll
   train on (16+ fast cores). Not the applemagpie box while the bot runs.
2. **Baseline there, before changing anything** (10 min, see Benchmark).
3. **Implement the ideas there**, measuring each against the baseline.
4. **Final hour on the 4090 and/or 5090:** pick the card by games per
   dollar and tune `--battles`, `--threads` and the CUDA-graph row sizes,
   which depend on the card.

## Where the time goes

Today every step waits for the one before it:

```
today      CPU  [env][   ][env][   ][env][   ]
           GPU  [   ][fwd][   ][fwd][   ][fwd][ learner (CPU idle) ]

ideas 1+2  CPU  [env A][env B][env A][env B]
           GPU  [fwd B][fwd A][fwd B][fwd A]       (actor stream)
           GPU  [ learner on the previous iteration ] (own stream)
```

Without search the forward is tiny (~0.3M multiply-adds a row), so
launches, copies and Python dominate. With search the rollouts' forwards
outweigh everything else and the GPU becomes the limit.

## Benchmark

`log.jsonl` already has the split: `act_net_s` (forward), `act_env_s`
(env step), `act_other_s` (buffer, Python), `train_s` with
`train_prep_s` / `train_issue_s` / `train_wait_s`, `games_per_s`,
`cpu_cores`.

1. **Box check.** `nvidia-smi`, `lscpu`, `nproc`, torch and CUDA
   versions. Build libpkmn's chance mode (zig 0.16), then
   `cmake --build build --target ataraxos_pi`. The last box never built
   `ataraxos_pi` (no chance build), so do this first.
2. **Baseline.** `python -m ataraxos.pi.train --run runs/bench0 --iters 10
   --eval_every 1000 --save_every 1000 --threads $(nproc) [teams...]`.
   Note iterations 3-10's timings and `games_per_s`.
3. **Battles:** 16384, 32768, 65536. If `games_per_s` keeps rising,
   launch overhead is the limit.
4. **Threads:** half and all of `nproc` at the best battle count. Shows
   whether the env is CPU-bound.
5. **Minibatch:** 12288, 24576, 49152. A big drop in `train_issue_s` means
   the learner is launch-bound (birch's notes say birch's was).
6. **Forward microbench** (new, ~30 lines): `PiNet` rows/s at 4k, 16k,
   64k, 128k rows; eager, `torch.compile(mode="max-autotune")`, CUDA
   graphs; bf16. The ceiling for the actor and for search rollouts.
7. **One profile** (`torch.profiler` or `nsys`) of an iteration, for the
   GPU's idle gaps during collection.

## Ideas

Gains are guesses until the baseline is in.

| # | Idea | Guessed gain | Effort | Moves |
| --- | --- | --- | --- | --- |
| 1 | Ping-pong env and actor: two halves of the battles; the GPU runs A's forward while the CPU steps B | up to 2x on collection | small (a CUDA stream) | `act_net_s` + `act_env_s` become a max |
| 2 | Collect while learning: train on iteration t while actors collect t+1 with t's weights (one iteration stale; PPO uses the stored log-probs) | up to 2x when act and train are close | medium | `iter_s` toward max(`act_s`, `train_s`) |
| 3 | CUDA graphs or `torch.compile` for the actor, rows padded to a few fixed sizes | 1.5-3x on the forward | small | `act_net_s` |
| 4 | Bigger lockstep batch (32k-65k battles) | fills the GPU | a flag | `games_per_s` |
| 5 | Pinned buffers, fewer syncs: env writes into pinned memory, `non_blocking` copies, only picks back to the CPU, log-probs and values in a GPU ring buffer | 10-30% | small-medium | `act_other_s`, `act_net_s` |
| 6 | Learner overhead: fused AdamW, compile or CUDA-graph the train step, bigger minibatch | up to 2x on `train_s` if launch-bound | small | `train_issue_s` |
| 7 | Buffer and GAE on the GPU | depends on `act_other_s`, `train_prep_s` | medium | `act_other_s`, `train_prep_s` |
| 8 | Evals off the critical path (side thread or process on a saved checkpoint) | ~5% | small | `eval_s` |
| 9 | TF32 / bf16 everywhere | small at this size | trivial | `train_issue_s` |

Order: 4, 3, 6 (flags or a few lines), then 1, then 2 once the numbers
show act and train are close.

### Flags (written 2026-11-22, not yet run on a GPU)

Every idea is a `Config` flag in `ataraxos/pi/train.py`; each default is
the original code path, so a run with no new flags is the baseline and any
one flag can be switched off for an ablation. Code: `ataraxos/pi/fast.py`.

| Flag | Idea | Default |
| --- | --- | --- |
| `--battles N` | 4 | 8192 |
| `--buffer numpy` | 7 (part): trajectories built with numpy in `take`, no Python loop per row in `add` | `python` |
| `--actor async` | 5: pinned buffers, non-blocking copies, `launch` / `collect` | `eager` |
| `--actor graphs` | 3 + 5: the forward and sampling as one CUDA graph (Gumbel-max sampling) | `eager` |
| `--pingpong 1` | 1: two half-size envs, one steps while the other's forward runs (needs `async` or `graphs`) | off |
| `--fused_adam 1` | 6 (part) | off |
| `--tf32 1` | 9 | off |
| `--gae gpu` | 7 (rest): returns, advantages and the filter's quantile on the device | `cpu` |
| `--compile_train MODE` | 6 (rest): `torch.compile` of the loss, forward and backward (`default`, `reduce-overhead`, `max-autotune`); one recompile for the last, smaller minibatch, then dynamic | off |
| `--overlap 1` | 2: the learner in a thread on its own stream, the actor on its own copy of the net refreshed after each learner step (one step stale) | off |
| `--eval_async 1` | 8: evals in a thread on copies of the nets, own stream, at most one in flight; results as their own `eval_it` log line | off |
| `--eval_threads N` | the eval env's threads (0: `--threads`), so async evals don't oversubscribe | 0 |

Checks for when a machine is free: `tests/py/check_pi_buffer.py` (CPU,
seconds: NumpyBuffer gives train.Buffer's trajectories, with and without
two envs), `tests/py/bench_pi_forward.py` (GPU: eager vs compile vs
graph rows/s) and `tests/py/check_pi_gae.py` (GPU: `--gae gpu` against the
CPU).

Laptop smoke run (CPU, 256 battles, 4 iterations, oak150): every new flag
and `--overlap 1 --eval_async 1 --buffer numpy --actor async --pingpong 1`
together run, save and resume-format checkpoints; `--gae gpu` (on the CPU)
and `--compile_train default` give the baseline's losses exactly. With
`--overlap`, a logged iteration's `iter_s` / `games_per_s` is the loop's
period (collecting the next iteration while this one trains), plus
`overlap_wait_s`; its env / team stats include the next collection. The
learner thread shares the GIL with the collection loop (the env releases
it while stepping): whether that costs more than it saves is for the box.

Birch's 3x: no record of which change gave it. Birch's commits show a
GPU-resident learner (`e3a5e85`) and history grouped by length
(`68b3542`); its notes say the learner is "bound by CPU kernel issue". In
ataraxos, `pad_chunks` on the GPU gave 4.6x and CUDA graphs ~1.5x on
rollouts. All of them removed overhead around the net, not inside it.

## 4090 or 5090

Choose the host's CPU first and the GPU second. Without search the net
can't fill either card; with search the rollouts can, and the 5090's
extra SMs and bandwidth should show.

| | RTX 4090 | RTX 5090 |
| --- | --- | --- |
| SMs | 128 | 170 |
| Memory | 24 GB GDDR6X | 32 GB GDDR7 |
| Bandwidth | ~1.0 TB/s | ~1.8 TB/s |
| fp32 | ~83 TFLOPS | ~105 TFLOPS |
| Software | any CUDA 12 torch | CUDA 12.8+, torch 2.7+ (cu128), driver 570+ |

Specs from memory, approximate. Check that the 5090 host's driver and a
cu128 torch work before renting it for a long run (the last box had
driver 550 and needed torch 2.6).

## Search during training

Agreed (2026-11-22): one round of joint MMD, 64-128 rollouts,
applemagpie's update, both sides' posteriors as CE targets; rows where
either player has 2+ choices (forced switch-ins included); the searched
action is played and masked out of the PG loss; the value trains by CE
on the rollouts' empirical W/D/L; p anneals up as the value head learns.

Cost: about p x budget x depth extra forwards and env steps per env row.

| p | Budget | Depth | Extra forwards per env row |
| --- | --- | --- | --- |
| 0.05 | 64 | 4 | ~13x |
| 0.10 | 128 | 4 | ~51x |
| 0.25 | 128 | 4 | ~128x |

Batching: rollouts join the env's lockstep batch as extra battles
(birch's `docs/OVO.md`); a searching game waits for its rollouts while
the others step, so one forward serves both.

Settings test (2026-11-22, rby2 ckpt250 as a weak net, 80 roots, 1 round,
1 type a side with the foe's true sets). Score: the posterior's gain over
the prior, sum (post - prior) q_ref, with q_ref from an independent depth 8
budget 256 run with uniform cells, against the foe's prior; in
win-minus-loss points x100 (+- one standard error):

| Depth, budget (forwards) | alpha 0, beta 0.02 | alpha 0, beta 0.05 | alpha 0, beta 0.2 | 1:10, lambda 0.022 | 1:10, lambda 0.22 |
| --- | --- | --- | --- | --- | --- |
| 1, 64 (64) | 0.97 +- 0.31 | 0.70 +- 0.20 | 0.23 +- 0.06 | 0.93 | 0.17 |
| 1, 128 (128) | 1.04 +- 0.34 | 0.66 +- 0.20 | 0.23 +- 0.07 | 0.98 | 0.16 |
| 4, 128 (512) | 1.47 +- 0.52 | 1.07 +- 0.34 | 0.42 +- 0.10 | 1.43 | 0.34 |

- The gain keeps rising as lambda falls to 0.02 (smaller wasn't tested).
- alpha 0 (the prior alone as the magnet) beats 1:10 at every lambda: the
  attack_switch magnet costs win rate and adds entropy.
- Depth 1 budget 64 gives the most gain per forward (~0.015 / forward vs
  ~0.003 at depth 4); budget 128 adds nothing at depth 1.
- Entropy at alpha 0, lambda 0.02: 0.63 -> 0.53 nats (no collapse).
- Seed noise is large at small lambda (KL 0.25 at budget 64), but CE is
  linear in the target, so it is zero-mean gradient noise; the gain above
  already includes noise's effect on the mean.
- An earlier test on rby4 ckpt1750 scored against a 512 reference at the
  same depth and favoured lambda ~0.2; that criterion was wrong (it counted
  noise as harm and graded against a same-depth run).

Starting point: depth 1, budget 64, 1 round, alpha 0, beta 0.02-0.05.

## Open decisions

- Minimum host: 16 cores or 32?
- One-iteration-stale actor (idea 2) acceptable?
- Starting p, budget, depth and lambda, after the softness run.
- Does oak's C++ forward for IS-MCTS wait until after the first GPU run?
