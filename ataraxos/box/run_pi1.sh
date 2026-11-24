#!/usr/bin/env bash
# pi1 (2026-11-22, the user: "just start training"): the perfect-info net
# (ataraxos/pi/train.py), all the speed flags, search in training on Claude's
# proposed schedule (p 0.05 -> 0.25 over 1e9 decisions, value mix 0.5,
# budget 64, beta 0.03), oak1500, evals vs random and maxdamage every 25
# iterations (async). Restart loop: resumes from the newest checkpoint;
# stops when the trainer exits 0. Out: runs/$RUN/train.log, log.jsonl, ckpts.
#
# pi2 (2026-11-22): pi1 ran on the trainer's Exp3 defaults (gamma 0.1
# decaying, eta gamma/k ~6.7e-5), which collapsed the team pool to ~13
# teams by it 100 while the evals draw uniformly from all 1500. The team
# flags below are rby1-4's (box/run_rby1.sh), so pi2 is fresh with them.
#
# pi3 (2026-11-23): pi2's flags with --size pi3 (1.42M params; pi2 lost
# 0.157 to rby4 at ckpt2575).
#
#   RUN=pi2 box/run_pi1.sh    (from ataraxos/, in tmux; RUN defaults to pi2)
#   RUN=pi3 SIZE=pi3 box/run_pi1.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PY:-/venv/main/bin/python}
RUN=${RUN:-pi2}
SIZE=${SIZE:-pi}
mkdir -p runs/$RUN
while true; do
  res=""
  ls runs/$RUN/ckpt*.pt >/dev/null 2>&1 && res="--resume latest"
  $PY -u -m ataraxos.pi.train --run runs/$RUN --size $SIZE $res \
    --threads 12 --torch_threads 4 --battles 8192 \
    --team_gamma 0.5 --team_gamma_floor 0.01 --team_gamma_tau 0.0 --team_gamma_decay 0.5 --team_eta 1e-05 \
    --buffer numpy --take_gather 1 --actor graphs --actor_compile 1 --pingpong 1 --overlap 1 \
    --compile_train default --gae gpu --fused_adam 1 --eval_async 1 --eval_threads 2 \
    --search_p 0.05 --search_p_end 0.25 --search_anneal 1e9 --search_budget 64 --search_beta 0.03 \
    --search_solve cpp --search_plan cpp --search_leaf graphs --search_direct 1 --search_value_only 1 \
    --search_compile 1 --search_async 1 \
    teams/selfplay/oak1500.txt >> runs/$RUN/train.log 2>&1
  rc=$?
  echo "== exited $rc $(date +%T)" >> runs/$RUN/train.log
  [ $rc -eq 0 ] && break
  sleep 30
done
