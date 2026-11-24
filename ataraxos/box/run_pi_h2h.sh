#!/usr/bin/env bash
# A perfect-info run's newest checkpoint (ema, policy only) against rby4
# ckpt1750 (Showdown view, ataraxos/eval/pi_h2h.py, 2048 battles swapped)
# and against other perfect-info checkpoints (VS=, comma-separated paths,
# ataraxos/eval/pi_ladder.py, 4096 battles), after a check vs maxdamage.
#
# 2026-11-23 (pi2 vs rby4, the user's): paused pi2's trainer with SIGSTOP.
# 2026-11-24 (pi3 vs rby4 and pi2's newest, the user's): a stopped process
# keeps its GPU memory (pi3's trainer held 11.7 of the 3060's 12.3 GB), so
# this waits for the run's next checkpoint (and the iteration after it: the
# teams file is written after the checkpoint), stops the restart loop and the
# trainer, plays, and starts the loop again in tmux $RUN (it resumes from
# that checkpoint), whatever happens. From ataraxos/, in tmux:
#
#   RUN=pi3 SIZE=pi3 VS=runs/pi2/ckpt2675.pt box/run_pi_h2h.sh   Out: ~/h2h/run.log
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PY:-/venv/main/bin/python}
RUN=${RUN:-pi3} SIZE=${SIZE:-pi3} VS=${VS:-}
OUT=~/h2h
mkdir -p $OUT
exec >> $OUT/run.log 2>&1
newest() { ls runs/$RUN | sed -n 's/^ckpt\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1; }
last=$(newest)
echo "== start $(date +%T), $RUN at ckpt$last: waiting for the next checkpoint"
while :; do
  it=$(newest)
  [ "$it" -gt "$last" ] && tail -n 20 runs/$RUN/train.log | grep -q "^it=$((it + 1)) " && break
  sleep 5
done
trainer=$(pgrep -o -f "ataraxos.pi.train --run runs/$RUN ")
loop=$(ps -o ppid= -p "$trainer" | tr -d ' ')
restart() {
  tmux has-session -t $RUN 2>/dev/null && tmux kill-session -t $RUN
  tmux new-session -d -s $RUN -c "$PWD" "RUN=$RUN SIZE=$SIZE box/run_pi1.sh"
  echo "== $RUN restarted $(date +%T)"
}
trap restart EXIT
echo "== stopping $RUN at ckpt$it $(date +%T): loop $loop, trainer $trainer"
echo "== stopped for h2h at ckpt$it $(date +%T)" >> runs/$RUN/train.log
kill "$loop"
kill "$trainer"
timeout 120 tail --pid="$trainer" -f /dev/null || { kill -9 "$trainer"; sleep 5; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
RBY=$OUT/runs/rby4/ckpt1750.pt
[ -f $RBY ] || HF_TOKEN_PATH=~/.cache/huggingface/token /venv/main/bin/hf download opencover/ataraxos-runs runs/rby4/ckpt1750.pt --local-dir $OUT
PI=runs/$RUN/ckpt$it.pt
T=teams/selfplay/oak1500.txt
A="--threads 8 --mem_gb 4"
echo "== check: $PI vs maxdamage (training eval ~0.89)"
$PY -m ataraxos.eval.pi_h2h $PI maxdamage $T --battles 512 $A | tail -1
echo "== h2h: $PI vs rby4, 2048 battles, swapped"
$PY -m ataraxos.eval.pi_h2h $PI $RBY $T --battles 2048 --swap $A
for v in ${VS//,/ }; do
  echo "== h2h: $PI vs $v, 4096 battles"
  $PY -m ataraxos.eval.pi_ladder runs/$RUN $T --new ckpt$it --old $v --battles 4096 $A
done
echo "== done $(date +%T)"
