#!/usr/bin/env bash
# rby4 on the main Showdown gen1ou ladder (2026-11-04, the user's: account
# "apple magpie", rby4's newest ckpt (1750, EMA), its top 20 teams by Nash
# draw counts, the league's best search config d4r4b1k3x3_team: joint
# per_type, 3x3 types, 4 rounds, budget 1024, depth 4, sampler pool
# oak1500.txt, on the GPU; temperature 1 and no min-prob, as in the league).
# Restart loop; extra args go to the bot; stops when the bot exits 0.
#
#   box/ladder_main.sh [--games N ...]
#
# Needs in ~/ladder/applemagpie: login.txt (name, then password; chmod 600;
# never in a repo or a log), ckpt.pt, pool.tsv (codegen/top_teams.py).
# Out: bot.log and games/ there. Control: botctl --port 8780
# (`games 0` to stop after the current game).
set -uo pipefail
d=~/ladder/applemagpie
ROUNDS=${ROUNDS:-4} DEPTH=${DEPTH:-4} BUDGET=${BUDGET:-1024} PY=${PY:-/venv/main/bin/python}
mkdir -p $d/games
cd ~/oak-ataraxos/ataraxos
while true; do
  nice -n 10 $PY -u -m ataraxos.play.showdown --server showdown --login $d/login.txt --ladder \
    --ckpt $d/ckpt.pt --teams $d/pool.tsv \
    --leave-after 5 --idle-timer 20 \
    --games-dir $d/games --threads 4 --control-port 8780 \
    --search-pool teams/selfplay/oak1500.txt --search-budget $BUDGET --search-depth $DEPTH \
    --search-joint per_type --search-types 3 --search-rounds $ROUNDS "$@" >> $d/bot.log 2>&1
  rc=$?; echo "== bot exited $rc $(date +%T)" >> $d/bot.log
  [ $rc -eq 0 ] && break
  sleep 10
done
