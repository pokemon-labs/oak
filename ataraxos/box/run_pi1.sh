#!/usr/bin/env bash
# pi1 (2026-11-22, the user: "just start training"): the perfect-info net
# (ataraxos/pi/train.py), all the speed flags, search in training on Claude's
# proposed schedule (p 0.05 -> 0.25 over 1e9 decisions, value mix 0.5,
# budget 64, beta 0.03), oak1500, evals vs random and maxdamage every 25
# iterations (async). Restart loop: resumes from the newest checkpoint;
# stops when the trainer exits 0. Out: runs/pi1/train.log, log.jsonl, ckpts.
#
#   box/run_pi1.sh    (from ataraxos/, in tmux)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PY:-/venv/main/bin/python}
mkdir -p runs/pi1
while true; do
  res=""
  ls runs/pi1/ckpt*.pt >/dev/null 2>&1 && res="--resume latest"
  $PY -u -m ataraxos.pi.train --run runs/pi1 $res \
    --threads 12 --torch_threads 4 --battles 8192 \
    --buffer numpy --take_gather 1 --actor graphs --actor_compile 1 --pingpong 1 --overlap 1 \
    --compile_train default --gae gpu --fused_adam 1 --eval_async 1 --eval_threads 2 \
    --search_p 0.05 --search_p_end 0.25 --search_anneal 1e9 --search_budget 64 --search_beta 0.03 \
    --search_solve cpp --search_plan cpp --search_leaf graphs --search_direct 1 --search_value_only 1 \
    --search_compile 1 --search_async 1 \
    teams/selfplay/oak1500.txt >> runs/pi1/train.log 2>&1
  rc=$?
  echo "== exited $rc $(date +%T)" >> runs/pi1/train.log
  [ $rc -eq 0 ] && break
  sleep 30
done
