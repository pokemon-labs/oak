#!/usr/bin/env bash
# Uploads the box's checkpoints to a private Hugging Face repo, only those at
# iterations that are multiples of 500 (the user's, 2026-10-28: by checkpoint,
# not by time; every 25 minutes filled the 100 GB private limit). On the box,
# in oak/ataraxos, after `hf auth login`; leave it running (nohup or tmux):
#
#   box/hf_sync.sh opencover/ataraxos-runs
#
# Each new runs/RUN/ckptN.pt with N % 500 == 0 goes up with its teamsN.npz and
# the run's logs (*.jsonl, *.json, *.csv). Checkpoints are written as .tmp and
# renamed, so a ckptN.pt is complete. Pushed ones are listed in
# runs/.hf_pushed. Nothing is deleted, locally or on the Hub (the user's,
# 2026-11-04: local disk is box/disk_guard.sh's job). EVERY=seconds between
# looks for new checkpoints (default 60); ONCE=1 looks once; HF= the hf binary
# (default: hf on PATH, else the venv's); MOD= the iteration multiple
# uploaded (default 500; 25 for the perfect-info net's few-MB checkpoints,
# 2026-11-22).
set -uo pipefail
[ $# -ge 1 ] || { sed -n 2,15p "$0"; exit 2; }
repo=$1
cd "$(dirname "$0")/.."
HF=${HF:-$(command -v hf || echo /venv/main/bin/hf)}
pushed=runs/.hf_pushed
touch "$pushed"
while true; do
  for ck in runs/*/ckpt*.pt; do
    [ -e "$ck" ] || continue
    it=$(basename "$ck" .pt); it=${it#ckpt}
    [[ $it =~ ^[0-9]+$ ]] && (( it % ${MOD:-500} == 0 )) || continue
    grep -qxF "$ck" "$pushed" && continue
    run=$(basename "$(dirname "$ck")")
    if $HF upload "$repo" runs runs --repo-type model \
         --include "$run/ckpt$it.pt" --include "$run/teams$it.npz" --include "$run/*.jsonl" \
         --include "$run/*.json" --include "$run/*.csv" \
         --commit-message "$run ckpt$it" >/dev/null; then
      echo "$ck" >> "$pushed"
      echo "$(date '+%F %T') uploaded $ck"
    else
      rc=$?
      echo "$(date '+%F %T') upload of $ck failed (exit $rc); retrying next look" >&2
    fi
  done
  [ -n "${ONCE:-}" ] && break
  sleep "${EVERY:-60}"
done
