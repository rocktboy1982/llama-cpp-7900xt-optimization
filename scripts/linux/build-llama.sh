#!/usr/bin/env bash
# Build llama.cpp for an RX 7900 XT (HIP / gfx1100, optionally Vulkan) against ROCm 10.
#
# usage: build-llama.sh [--turboquant] [--no-vulkan] [--check-only]
#   --turboquant  apply patches/turboquant-on-upstream-2ca15f540.patch (HIP only; adds turbo3/turbo4 KV types)
#   --no-vulkan   HIP only (implied by --turboquant)
#   --check-only  clone/checkout and verify the patch applies, do not compile
# env: LLAMA_DIR (default ~/llama.cpp, or ~/llama.cpp-turboquant with --turboquant), LLAMA_REPO, LLAMA_COMMIT (default 2ca15f540),
#      ROCM (default /opt/rocm/core-10.0), JOBS (default nproc), TARGETS (default "llama-server llama-bench")
set -euo pipefail
TQ=0; VK=1; CHECK=0
for a in "$@"; do case "$a" in --turboquant) TQ=1; VK=0;; --no-vulkan) VK=0;; --check-only) CHECK=1;;
  -h|--help) sed -n 2,9p "$0"; exit 0;; *) echo "unknown option: $a" >&2; exit 2;; esac; done
HERE=$(cd "$(dirname "$0")" && pwd); PATCH="$HERE/../../patches/turboquant-on-upstream-2ca15f540.patch"
ROCM=${ROCM:-/opt/rocm/core-10.0}; COMMIT=${LLAMA_COMMIT:-2ca15f540}; REPO=${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}
DIR=${LLAMA_DIR:-$HOME/llama.cpp}; [ "$TQ" = 1 ] && DIR=${LLAMA_DIR:-$HOME/llama.cpp-turboquant}
JOBS=${JOBS:-$(nproc)}; TARGETS=${TARGETS:-llama-server llama-bench}

[ -f "$ROCM/lib/cmake/hip-lang/hip-lang-config.cmake" ] || { echo "ROCm dev files not found under $ROCM (run install-rocm10.sh)" >&2; exit 1; }
if [ ! -d "$DIR/.git" ]; then git clone "$REPO" "$DIR"; fi
cd "$DIR"
git cat-file -e "$COMMIT^{commit}" 2>/dev/null || git fetch origin
git checkout -q "$COMMIT"
echo "llama.cpp at $(git log -1 --format='%h %s')"
if [ "$TQ" = 1 ]; then
  if git apply --reverse --check "$PATCH" 2>/dev/null; then echo "TurboQuant patch already applied"
  else git apply --check "$PATCH" && echo "TurboQuant patch applies cleanly" && { [ "$CHECK" = 1 ] || git apply "$PATCH"; }; fi
fi
[ "$CHECK" = 1 ] && { echo "check-only: not compiling"; exit 0; }

OPTS=(-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release)
[ "$VK" = 1 ] && OPTS+=(-DGGML_VULKAN=ON)
[ "$TQ" = 1 ] && OPTS+=(-DLLAMA_BUILD_TESTS=ON) && TARGETS="$TARGETS test-turboquant test-backend-ops"
HIPCXX="$ROCM/llvm/bin/clang" HIP_PATH="$ROCM" CMAKE_PREFIX_PATH="$ROCM" cmake -S . -B build "${OPTS[@]}"
# shellcheck disable=SC2086
cmake --build build -j"$JOBS" --target $TARGETS

echo; echo "Built. Devices:"
LD_LIBRARY_PATH="$ROCM/lib" build/bin/llama-server --list-devices 2>&1 | grep -E 'ROCm|Vulkan' || true
echo "Run with: export LD_LIBRARY_PATH=$ROCM/lib; $DIR/build/bin/llama-server ...   (see configs/linux/run-llama.sh)"
