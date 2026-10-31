#!/usr/bin/env bash
# Copies what ataraxos needs from oak to a rented box and runs box/setup.sh
# there (birch's box/push.sh for oak). From the laptop, anywhere in oak:
#
#   ataraxos/box/push.sh root@HOST PORT    # Vast's "ssh -p PORT root@HOST"
#   ataraxos/box/push.sh local DIR         # a rehearsal: copy into DIR, set up there
#
# ataraxos/ goes as its working tree (tracked and untracked files, not
# ignored ones, so uncommitted work goes too), with oak's C++ headers,
# pybind11 (oak's submodule) and the built libpkmn (dev/libpkmn's ReleaseFast
# "client" build and the pkmn.h it generates, both ignored by git, so the box
# needs no zig). The oak commit goes in oak/COMMIT (with -dirty for
# uncommitted work), for ataraxos/rl/train.py's config.json. Over rsync, so
# the box needs no GitHub access. Nothing on the box is deleted.
# Extra environment for setup.sh (PY, TORCH_INDEX) passes through.
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,14p "$0"; exit 2; }
host=$1 arg=$2
oak=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
commit=$(git -C "$oak" rev-parse HEAD)
[ -z "$(git -C "$oak" status --porcelain -- ataraxos cpp/include)" ] || commit+=-dirty
lib=cpp/libpkmn/ReleaseFast/client/libpkmn-showdown.a
[ -f "$oak/$lib" ] && [ -f "$oak/cpp/include/pkmn.h" ] || { echo "build libpkmn first (dev/libpkmn)" >&2; exit 1; }

if [ "$host" = local ]; then
  root=$(realpath -m "$arg")
  mkdir -p "$root"
  copy() { rsync -a --ignore-missing-args --files-from=- "$oak" "$root/oak/"; }
  run() { bash -c "$1"; }
else
  root='~'
  ssh_cmd="ssh -p $arg -o StrictHostKeyChecking=accept-new"
  copy() { rsync -az --ignore-missing-args -e "$ssh_cmd" --files-from=- "$oak" "$host:oak/"; }
  run() { $ssh_cmd "$host" "$1"; }
fi

run "mkdir -p $root/oak"
{
  git -C "$oak" ls-files -co --exclude-standard -- ataraxos cpp/include
  git -C "$oak/cpp/extern/pybind11" ls-files | sed 's#^#cpp/extern/pybind11/#'
  echo "$lib"
  echo cpp/include/pkmn.h
} | copy
run "echo $commit > $root/oak/COMMIT"
echo "copied oak $commit"

env_vars=""
for v in PY TORCH_INDEX; do
  [ -n "${!v:-}" ] && env_vars+="$v=$(printf %q "${!v}") "
done
run "cd $root/oak/ataraxos && ${env_vars}bash box/setup.sh"
