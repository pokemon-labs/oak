#!/usr/bin/env bash
# Sets up a box for training and checks it (birch's box/setup.sh for oak's
# ataraxos). Run from oak/ataraxos (box/push.sh does): installs what's
# missing, builds the binding with CMake against the copied libpkmn, records
# the GPU, and runs the tests. Writes everything it prints to runs/setup.log
# too.
#
#   PY       the Python to use (default: python3); needs torch, or pip can
#            install it (TORCH_INDEX, e.g. https://download.pytorch.org/whl/cpu
#            for a CPU rehearsal)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
exec > >(tee runs/setup.log) 2>&1

PY=${PY:-python3}
echo "== oak $(cat ../COMMIT 2>/dev/null || echo '(commit unknown)')"

apt_install() {  # root on a rented box; best effort
  if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@"
  else
    return 1
  fi
}

command -v cmake >/dev/null || apt_install cmake || true
command -v g++ >/dev/null || apt_install g++ || true
command -v rsync >/dev/null || apt_install rsync || true

# python: torch, numpy, and the headers
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
if ! $PY -c "import torch" 2>/dev/null; then
  echo "== installing torch"
  $PY -m pip install -q torch ${TORCH_INDEX:+--index-url "$TORCH_INDEX"}
fi
$PY -c "import numpy" 2>/dev/null || $PY -m pip install -q numpy
inc=$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")
[ -f "$inc/Python.h" ] || apt_install python3-dev || true
[ -f "$inc/Python.h" ] || { echo "no Python.h in $inc" >&2; exit 1; }
$PY -c "import torch, numpy, sys; print('==', sys.version.split()[0], 'torch', torch.__version__,
       'numpy', numpy.__version__, 'cuda', torch.cuda.is_available())"

echo "== building ataraxos_env and the checkers with $(g++ --version | head -1)"
cmake -S . -B build -DPython_EXECUTABLE="$(command -v $PY)" -DPYBIND11_FINDPYTHON=ON >/dev/null
make -C build -j"$(nproc)" >/dev/null

device=cpu
if $PY -c "import torch, sys; sys.exit(not torch.cuda.is_available())"; then
  device=cuda
  echo "== GPU"
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,power.max_limit,clocks.max.sm \
    --format=csv
fi
echo "== $(nproc) CPUs, $(free -g | awk '/Mem:/ {print $2}') GB RAM, $(df -h . | awk 'NR==2 {print $4}') disk free"

teams=teams/selfplay/oak1500.txt
echo "== tests"
for t in build/check_*; do [ -x "$t" ] && "$t"; done
$PY ataraxos/nets/policy.py
$PY tests/py/test_env.py --battles 256 "$teams"
$PY tests/py/test_env.py --battles 256
$PY tests/py/test_teams.py "$teams"
$PY tests/py/test_metrics.py
$PY tests/py/test_agents.py "$teams"
$PY tests/py/test_causal.py --device $device "$teams"
$PY tests/py/test_resume.py "$teams"
$PY tests/py/test_exploit.py
echo "== setup ok ($device)"
