#!/usr/bin/env bash
# Install ROCm 10.0.0 for an RX 7900 XT (gfx1100) on Ubuntu 26.04 from AMD's package repository.
# Installs ONLY the gfx1100 packages plus the compile-time dev packages. Do not install the "amdrocm10.0" or
# "amdrocm-core-dev10.0" meta-packages: they pull libraries for every GPU generation (~11 GB extra).
#
# usage: install-rocm10.sh [--dry-run] [--purge-old] [--with-vulkan]
#   --dry-run      print the commands, change nothing
#   --purge-old    first remove ROCm 7.x ("rocm-*", "amdgpu-install" and related) - destructive, asks first
#   --with-vulkan  also install the Vulkan build/runtime packages (needed for the iGPU or a Vulkan build)
set -euo pipefail
DRY=0; PURGE=0; VULKAN=0
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --purge-old) PURGE=1;; --with-vulkan) VULKAN=1;;
  -h|--help) sed -n 2,11p "$0"; exit 0;; *) echo "unknown option: $a" >&2; exit 2;; esac; done
run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }

. /etc/os-release
[ "${VERSION_ID:-}" = "26.04" ] || echo "WARNING: written for Ubuntu 26.04 (found ${VERSION_ID:-unknown}); the repo path below is ubuntu2604" >&2

PKGS=(amdrocm10.0-gfx1100 amdrocm-core-dev10.0-gfx1100 amdrocm-runtime-dev10.0 amdrocm-llvm-dev10.0 amdrocm-blas-dev10.0 amdrocm-hipblas-common-dev10.0)
VK=(glslc libvulkan-dev vulkan-tools mesa-vulkan-drivers spirv-headers)

if [ "$PURGE" = 1 ]; then
  OLD=$(dpkg -l | awk '/^ii/ && ($2 ~ /^(rocm|amdgpu-install|hip|roc|miopen|migraphx|comgr|hsa)/) {print $2}' | tr '\n' ' ')
  echo "Would remove old ROCm packages: ${OLD:-none}"
  if [ "$DRY" = 0 ] && [ -n "$OLD" ]; then read -r -p "Remove them with apt purge? [y/N] " a; [ "$a" = y ] || exit 1; fi
  [ -z "$OLD" ] || run sudo apt purge -y $OLD
fi

run sudo mkdir -p -m0755 /etc/apt/keyrings
if [ "$DRY" = 1 ]; then echo "+ wget -qO- https://stable.repo.amd.com/rocm/gpg/packages.gpg | gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg"
else wget -qO- https://stable.repo.amd.com/rocm/gpg/packages.gpg | gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg >/dev/null; fi
SRC='X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes'
if [ "$DRY" = 1 ]; then echo "+ write /etc/apt/sources.list.d/amdrocm-stable.sources"; else echo "$SRC" | sudo tee /etc/apt/sources.list.d/amdrocm-stable.sources >/dev/null; fi
run sudo apt update
run sudo apt install -y "${PKGS[@]}"
[ "$VULKAN" = 0 ] || run sudo apt install -y "${VK[@]}"

cat <<MSG

Done. ROCm 10 lives under /opt/rocm/core-10.0 (there is no /opt/rocm symlink tree).
  export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
  /opt/rocm/core-10.0/bin/rocminfo | grep gfx          # expect gfx1100
  /opt/rocm/core-10.0/bin/rocm-smi --showmeminfo vram
The Vega iGPU (gfx90c) is not supported by ROCm 10; use Vulkan for it.
MSG
