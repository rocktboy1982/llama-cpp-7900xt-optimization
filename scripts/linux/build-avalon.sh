#!/usr/bin/env bash
# Build Avalon (freriadoc/Avalon, a llama.cpp fork for Qwen3.8-Flash-Next with the experts in system RAM) for an RX 7900 XT against ROCm 10.
#
# usage: build-avalon.sh [--check-only]
#   --check-only  clone/checkout only, do not compile
# env: AVALON_DIR (default ~/Avalon), AVALON_REPO, AVALON_COMMIT (default a91372e), ROCM (default /opt/rocm/core-10.0), JOBS (default nproc)
#
# Avalon's own install.sh refuses ROCm 10 (it checks for 7.x) and does not set the HIP compiler, so CMake fails with
# "Failed to find ROCm root directory" on ROCm 10's layout. This script builds the same target with HIPCXX set.
set -euo pipefail
CHECK=0
for a in "$@"; do case "$a" in --check-only) CHECK=1;; -h|--help) sed -n 2,9p "$0"; exit 0;; *) echo "unknown option: $a" >&2; exit 2;; esac; done
ROCM=${ROCM:-/opt/rocm/core-10.0}; COMMIT=${AVALON_COMMIT:-a91372e}; REPO=${AVALON_REPO:-https://github.com/freriadoc/Avalon}
DIR=${AVALON_DIR:-$HOME/Avalon}; JOBS=${JOBS:-$(nproc)}

[ -f "$ROCM/lib/cmake/hip-lang/hip-lang-config.cmake" ] || { echo "ROCm dev files not found under $ROCM (run install-rocm10.sh)" >&2; exit 1; }
if [ ! -d "$DIR/.git" ]; then git clone "$REPO" "$DIR"; fi
cd "$DIR"
git cat-file -e "$COMMIT^{commit}" 2>/dev/null || git fetch origin
git checkout -q "$COMMIT"
echo "Avalon at $(git log -1 --format='%h %s')"
# a checkout made on Windows has CRLF line endings, which breaks every shell script in the tree
if git ls-files --eol | grep -q 'i/lf *w/crlf'; then
  echo "converting a Windows (CRLF) checkout back to LF"
  git ls-files -z | xargs -0 touch && git checkout-index -f -a
fi
[ "$CHECK" = 1 ] && { echo "check-only: not compiling"; exit 0; }

HIPCXX="$ROCM/lib/llvm/bin/clang++" HIP_PATH="$ROCM" ROCM_PATH="$ROCM" \
  cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM/bin/amdclang" -DCMAKE_CXX_COMPILER="$ROCM/bin/amdclang++"
cmake --build build -j"$JOBS" --target llama-server

echo; echo "Built. Devices (ROCm1 is the Vega iGPU on a 5700G; the launcher hides it with HIP_VISIBLE_DEVICES=0):"
LD_LIBRARY_PATH="$ROCM/lib" build/bin/llama-server --list-devices 2>&1 | grep -E 'ROCm' || true
echo "Next: download the model (docs/avalon-guide.md), then install configs/linux/avalon-server and configs/linux/avalon.service"
