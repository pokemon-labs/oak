#!/usr/bin/env bash
# pi4 (2026-11-24, docs/pi4.md): the perfect-info teacher on rby4's inputs
# (ataraxos/pi4/, --size pi4, ~13.5M params) in rby4's env as rows, rby4's
# settings with fresh schedules, Expected SARSA (rby4's --q_coef 1
# --sarsa_adv 0), no search, every speed flag. Evals every 25 iterations
# (async) in rby4's env against maxdamage, rby4 ckpt1750 and pi3 ckpt1950.
# Restart loop: resumes from the newest checkpoint; stops when the trainer
# exits 0. Out: runs/$RUN/train.log, log.jsonl, ckpts.
#
#   box/run_pi4.sh    (from ataraxos/, in tmux; RUN defaults to pi4)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PY:-/venv/main/bin/python}
RUN=${RUN:-pi4}
SIZE=${SIZE:-pi4}
MICRO=${MICRO:-1024}
RBY=${RBY:-/root/h2h/runs/rby4/ckpt1750.pt}
PI3=${PI3:-runs/pi3/ckpt1950.pt}
mkdir -p runs/$RUN
while true; do
  res=""
  ls runs/$RUN/ckpt*.pt >/dev/null 2>&1 && res="--resume latest"
  $PY -u -m ataraxos.pi.train --run runs/$RUN --size $SIZE $res \
    --threads 12 --torch_threads 4 --battles 8192 --microbatch $MICRO \
    --team_gamma 0.5 --team_gamma_floor 0.01 --team_gamma_tau 0.0 --team_gamma_decay 0.5 --team_eta 1e-05 \
    --expected_sarsa 1 --q_coef 1 --sarsa_adv 0 \
    --buffer numpy --take_gather 1 --actor graphs --actor_compile 1 --pingpong 1 --overlap 1 \
    --compile_train default --gae gpu --fused_adam 1 --eval_async 1 --eval_threads 2 \
    --anchors maxdamage,$RBY,$PI3 \
    teams/selfplay/oak1500.txt >> runs/$RUN/train.log 2>&1
  rc=$?
  echo "== exited $rc $(date +%T)" >> runs/$RUN/train.log
  [ $rc -eq 0 ] && break
  sleep 30
done
