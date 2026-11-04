#!/usr/bin/env bash
# Keeps room on the box's disk for new checkpoints (the user's, 2026-11-04:
# hf_sync leaves local files alone; this deletes only when the disk is
# short). On the box, in oak/ataraxos; leave it running (nohup):
#
#   box/disk_guard.sh
#
# When the runs/ filesystem has under MIN_FREE_GB free (default 15), deletes
# runs/*/ckptN.pt (with its teamsN.npz), oldest first, until FREE_GB free
# (default 25). Never a run's newest KEEP checkpoints (default 4), nor a
# multiple of 500 that isn't in runs/.hf_pushed yet (not on the Hub).
# Nothing else is touched. EVERY=seconds between looks (default 300); ONCE=1
# looks once; DRY=1 only prints what it would delete.
set -uo pipefail
cd "$(dirname "$0")/.."
MIN=${MIN_FREE_GB:-15} TARGET=${FREE_GB:-25} KEEP=${KEEP:-4}
pushed=runs/.hf_pushed
free_gb() { df -P --block-size=1G runs | awk 'NR == 2 { print $4 }'; }
candidates() {  # deletable checkpoints, oldest first
  for d in runs/*/; do
    ls "$d" | sed -n 's/^ckpt\([0-9]\+\)\.pt$/\1/p' | sort -n | head -n -"$KEEP" | while read -r it; do
      ck="${d}ckpt$it.pt"
      (( it % 500 == 0 )) && ! grep -qxF "$ck" "$pushed" 2>/dev/null && continue
      echo "$(stat -c %Y "$ck") $ck"
    done
  done | sort -n | cut -d' ' -f2
}
while true; do
  if (( $(free_gb) < MIN )); then
    for ck in $(candidates); do
      (( $(free_gb) >= TARGET )) && break
      t=$(sed 's|/ckpt\([0-9]*\)\.pt$|/teams\1.npz|' <<< "$ck")
      if [ -n "${DRY:-}" ]; then
        echo "$(date '+%F %T') would delete $ck"
        continue
      fi
      rm -f "$ck" "$t"
      echo "$(date '+%F %T') deleted $ck ($(free_gb) GB free)"
    done
  fi
  [ -n "${ONCE:-}" ] && break
  sleep "${EVERY:-300}"
done
