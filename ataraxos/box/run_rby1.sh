#!/usr/bin/env bash
# The first Gen 1 run rby1 (2026-10-31): birch's cs2b flags on ataraxos,
# self-play over teams/selfplay/oak1500.txt, evals vs random and maxdamage
# every 25 iterations, 512 battles (the user: same as birch). On the box, in
# ~/oak-ataraxos/ataraxos, after box/setup.sh:
#
#   box/run_rby1.sh [--key value ...]   # extra flags go to both runs
#
# Runs a smoke test at the run's battles and microbatch (3 short iterations,
# an eval and a save; out runs/rby1_smoke.out), then, if it exits 0, starts
# the run in the background (out runs/rby1.out) and box/hf_sync.sh if it
# isn't running. PY (default /venv/main/bin/python).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/venv/main/bin/python}
teams=teams/selfplay/oak1500.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# cs2b's command line, but its teams, anchors and resume
flags=(--size cs2 --battles 8192 --threads 12 --seed 1 --max_turns 300 --at_cap draw
       --team_gamma 0.5 --team_gamma_floor 0.01 --team_gamma_tau 0.0 --team_gamma_decay 0.5 --team_eta 1e-05
       --minibatch 6144 --microbatch 1024 --clip_range 0.2 --td_lambda 0.8 --gae_lambda 0.5
       --vf_coef 1.0 --policy_coef 1.0 --kl_coef 0.1
       --temperature_coef 0.05 --temperature_decay 0.3 --temperature_ceil 0.1 --temperature_floor 0.005
       --lr_coef 0.5 --lr_decay 1.1 --lr_ceil 0.0003 --lr_floor 5e-06 --schedule_decisions 5000000.0
       --weight_decay 0.0 --max_grad_norm 0.267 --adv_filt_rate 0.75 --adv_filt_thresh 0.01
       --ema_decay 0.961 --magnet attack_switch --anchors random,maxdamage
       --device cuda --compile 1 --amp 1 --graphs 1 --gpu_log 2.0)

mkdir -p runs
rm -rf runs/rby1_smoke
echo "== smoke (runs/rby1_smoke.out)"
nice -n 5 $PY -m ataraxos.rl.train --run runs/rby1_smoke "${flags[@]}" --games_per_iter 2048 --iters 3 \
  --eval_every 2 --eval_battles 64 --save_every 3 "$@" "$teams" > runs/rby1_smoke.out 2>&1 \
  || { tail -30 runs/rby1_smoke.out; echo "== smoke failed" >&2; exit 1; }
tail -5 runs/rby1_smoke.out
[ -f runs/rby1_smoke/ckpt3.pt ] || { echo "== smoke saved no ckpt3.pt" >&2; exit 1; }

[ ! -e runs/rby1 ] || { echo "runs/rby1 exists; resume it by hand (--resume latest)" >&2; exit 1; }
echo "== rby1 (runs/rby1.out)"
setsid nohup nice -n 5 $PY -m ataraxos.rl.train --run runs/rby1 "${flags[@]}" --games_per_iter 8192 \
  --iters 1000000000 --eval_every 25 --eval_battles 512 --save_every 50 "$@" "$teams" \
  > runs/rby1.out 2>&1 < /dev/null &
echo "train pid $!"
if ! pgrep -x -f "bash box/hf_sync.sh opencover/ataraxos-runs" >/dev/null; then
  setsid nohup bash box/hf_sync.sh opencover/ataraxos-runs > runs/hf_sync.out 2>&1 < /dev/null &
  echo "hf_sync pid $!"
fi
