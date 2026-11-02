#!/usr/bin/env bash
# The joint search's one-at-a-time sweep (2026-11-02, the user's: defaults
# and, for 3 key dims, one arm each with that dim moved far), each arm vs
# the plain net (search_h2h, ckpt1100), plus vanilla at the same budget for
# reference. 8 seeds x 16 battles an arm, the same seeds in every arm
# (paired games); jobs seed-major so partial results cover every arm. 3 jobs
# at once, each capped at 0.09 of the GPU (an OOM stays in the job, not the
# training run beside it). Results: runs/joint_sweep/results.jsonl, logs
# per job. On the box, in ~/oak-ataraxos/ataraxos:
#
#   nohup box/joint_sweep.sh > runs/joint_sweep/sweep.out 2>&1 &
#
# CKPT, SEEDS, BATTLES, JOBS, BUDGET (every arm's, joint and vanilla), DEPTH
# override; OUT the results' directory.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/venv/main/bin/python}
CKPT=${CKPT:-/root/arena/beanseggyum/ckpt1100.pt}
SEEDS=${SEEDS:-8}
BATTLES=${BATTLES:-16}
JOBS=${JOBS:-3}
BUDGET=${BUDGET:-512}
DEPTH=${DEPTH:-10}
OUT=${OUT:-runs/joint_sweep}
mkdir -p "$OUT"

declare -A ARMS=(
  [default]="--joint --joint_budget $BUDGET"
  [per_pair]="--joint --joint_budget $BUDGET --joint_mode per_pair"
  [types1x1]="--joint --joint_budget $BUDGET --types_a 1 --types_b 1"
  [foe_beta0.2]="--joint --joint_budget $BUDGET --foe_beta 0.2"
  [vanilla]="--budget $BUDGET --reps 2"
)
ORDER="default per_pair types1x1 foe_beta0.2 vanilla"

for seed in $(seq 1 "$SEEDS"); do
  for arm in $ORDER; do
    echo "$arm $seed ${ARMS[$arm]}"
  done
done | xargs -P "$JOBS" -L 1 bash -c '
  arm=$0; seed=$1; shift 2
  log='"$OUT"'/$arm.$seed.log
  echo "$(date +%T) start $arm seed $seed"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 10 '"$PY"' -m ataraxos.eval.search_h2h '"$CKPT"' \
    --battles '"$BATTLES"' --depth '"$DEPTH"' --seed $seed --threads 4 --mem_frac 0.09 --out '"$OUT"'/results.jsonl --tag $arm "$@" \
    > $log 2>&1
  echo "$(date +%T) done $arm seed $seed (exit $?): $(tail -1 $log | cut -c1-120)"
'
echo "$(date +%T) sweep done"
