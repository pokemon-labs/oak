#!/usr/bin/env bash
# Assembles pkmn.ai's native bundle in DIST (2026-11-21; the rules:
# https://pkmn.ai/rules, an executable ./run at the root, <= 500 MiB
# unpacked and compressed, <= 25,000 entries). DIST must already hold
# python/ (a standalone CPython with numpy installed) and net.bin
# (ataraxos/arena/export.py's); the extensions ataraxos_client, ataraxos_env
# and ataraxos_net must be built for that Python (ataraxos/_ext/). Adds:
#
#   run                 the entry point (ataraxos/arena/main.py)
#   app/ataraxos/       the torch-free modules and the extensions
#   oak1500.txt         the history sampler's pool
#
# then prunes the Python, strips and (macOS) ad-hoc signs the binaries and
# prints the size. OMP_DIR (macOS): libomp's prefix, copied in beside the
# extensions that link it.
#
#   ataraxos/arena/bundle.sh DIST
set -euo pipefail
[ $# -eq 1 ] || { sed -n 2,16p "$0"; exit 2; }
here=$(cd "$(dirname "$0")/../.." && pwd)  # ataraxos/
dist=$1
py=$dist/python/bin/python3
[ -x "$py" ] || { echo "no $py" >&2; exit 1; }
[ -f "$dist/net.bin" ] || { echo "no $dist/net.bin" >&2; exit 1; }
tag=$("$py" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

app=$dist/app/ataraxos
rm -rf "$dist/app"
mkdir -p "$app/_ext" "$app/arena"
cp "$here/ataraxos/__init__.py" "$here/ataraxos/search_np.py" "$app/"
cp "$here/ataraxos/_ext/__init__.py" "$app/_ext/"
for m in ataraxos_client ataraxos_env ataraxos_net; do
  cp "$here/ataraxos/_ext/$m$tag" "$app/_ext/"
done
cp "$here"/ataraxos/arena/{__init__,actor,search,main,export,stats}.py "$app/arena/"
cp "$here/teams/selfplay/oak1500.txt" "$dist/oak1500.txt"

if [ "$(uname)" = Darwin ]; then
  for so in "$app"/_ext/*.so; do
    omp=$(otool -L "$so" | awk '/libomp/ {print $1}')
    [ -n "$omp" ] || continue
    if [ ! -f "$app/_ext/libomp.dylib" ]; then
      cp "${OMP_DIR:?set OMP_DIR to the libomp prefix}/lib/libomp.dylib" "$app/_ext/libomp.dylib"
      chmod u+w "$app/_ext/libomp.dylib"
      install_name_tool -id @loader_path/libomp.dylib "$app/_ext/libomp.dylib"
    fi
    install_name_tool -change "$omp" @loader_path/libomp.dylib "$so"
  done
fi

cat > "$dist/run" <<'EOF'
#!/bin/bash
# pkmn.ai's entry point: the ataraxos bot (ataraxos/arena/main.py).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$DIR/app"
exec "$DIR/python/bin/python3" -s -u -m ataraxos.arena.main --net "$DIR/net.bin" --pool "$DIR/oak1500.txt" "$@"
EOF
chmod +x "$dist/run"

# prune: what the bot never imports
lib=$(ls -d "$dist"/python/lib/python3.* | head -1)
rm -rf "$dist/python/include" "$dist/python/share" "$dist/python/lib/pkgconfig" \
  "$lib"/{test,idlelib,tkinter,turtledemo,ensurepip,pydoc_data,lib2to3} \
  "$dist"/python/lib/libtcl* "$dist"/python/lib/libtk* "$dist"/python/lib/tcl* "$dist"/python/lib/tk* \
  "$lib"/config-* "$lib"/site-packages/{pip,setuptools}* "$lib"/site-packages/numpy/*/tests \
  "$lib"/site-packages/numpy/tests "$lib"/site-packages/numpy/_pyinstaller
find "$dist" -name '*.a' -delete
find "$dist" -name '*.pyi' -delete

# strip (and, on macOS, re-sign: arm64 refuses unsigned code)
find "$dist" -type f \( -name '*.so' -o -name '*.dylib' -o -path '*/bin/python3*' \) | while read -r f; do
  chmod u+w "$f"
  if [ "$(uname)" = Darwin ]; then
    strip -x "$f" 2>/dev/null || true
    codesign --force --sign - "$f" 2>/dev/null || true
  else
    strip --strip-unneeded "$f" 2>/dev/null || true
  fi
done

"$py" - "$dist" <<'EOF'
import os, sys
root = sys.argv[1]
files, entries = 0, 0
for dp, dn, fn in os.walk(root, followlinks=False):
    entries += len(dn) + len(fn)
    files += sum(os.path.getsize(os.path.join(dp, f)) for f in fn if not os.path.islink(os.path.join(dp, f)))
print(f"bundle: {files / 2**20:.1f} MiB unpacked, {entries} entries (limits 500 MiB, 25000)")
assert files <= 500 * 2**20 and entries <= 25000
EOF
