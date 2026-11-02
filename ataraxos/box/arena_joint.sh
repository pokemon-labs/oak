#!/usr/bin/env bash
# A joint-search bot on pkmn.bot (2026-11-02, the user's: one per_type, one
# per_pair): rby1 ckpt1100 (EMA), eggbeanswoo1100's 50 teams, the joint
# search (budget 512 over 4 rounds, 3x3 types, depth 10, the sampler pool
# oak1500.txt, on the GPU). Restart loop; extra args go to the bot; stops
# when the bot exits 0. Out: ~/arena/NAME/bot.log, games in ~/arena/NAME/games.
#
#   box/arena_joint.sh NAME per_type|per_pair CONTROL_PORT [--key value ...]
#
# Token: ~/arena/token_NAME (chmod 600; never in a repo or a log).
set -uo pipefail
name=$1 mode=$2 port=$3
shift 3
export ARENA_BOT_NAME="$name"
export ARENA_BOT_TOKEN="$(cat ~/arena/token_$name)"
d=~/arena/$name
mkdir -p $d/games
cd ~/oak-ataraxos/ataraxos
while true; do
  nice -n 10 /venv/main/bin/python -u -m ataraxos.play.showdown --server arena \
    --ckpt ~/arena/eggbeanswoo1100/ckpt1100.pt --teams ~/arena/beanseggyum/pool.tsv \
    --games-dir $d/games --threads 4 --control-port $port \
    --search-pool teams/selfplay/oak1500.txt --search-budget 512 --search-depth 10 \
    --search-joint $mode --search-types 3 --search-rounds 4 "$@" >> $d/bot.log 2>&1
  rc=$?; echo "== bot exited $rc $(date +%T)" >> $d/bot.log
  [ $rc -eq 0 ] && break
  sleep 10
done
