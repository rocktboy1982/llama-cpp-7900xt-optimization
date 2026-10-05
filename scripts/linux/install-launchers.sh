#!/usr/bin/env bash
# Install the launcher scripts into $HOME and (optionally) GNOME desktop shortcuts.
#   ~/run-llama.sh        llama.cpp servers: dGPU 27B (:11437, API key), iGPU 4B (:8081), CPU 2B (:8082)
#   ~/start-llama.sh      desktop launcher: asks before stopping TabbyAPI, then runs run-llama.sh
#   ~/start-tabbyapi.sh   desktop launcher: asks before stopping llama.cpp, then runs the ExLlamaV3-ROCm launcher
# usage: install-launchers.sh [--no-desktop]      env: DESKTOP_DIR (default: xdg Desktop folder)
# Existing different files are kept as <name>.bak-<timestamp>. The tabbyapi launcher expects ~/exllamav3-rocm and ~/tabbyAPI
# (see docs/tabbyapi-guide.md); it is installed regardless and reports a clear error if they are missing.
set -euo pipefail
DESK=1; [ "${1:-}" = --no-desktop ] && DESK=0
SRC=$(cd "$(dirname "$0")/../../configs/linux" && pwd)
for f in run-llama.sh start-llama.sh start-tabbyapi.sh; do
  if [ -f "$HOME/$f" ] && ! cmp -s "$SRC/$f" "$HOME/$f"; then cp "$HOME/$f" "$HOME/$f.bak-$(date +%Y%m%d%H%M%S)"; echo "kept a backup of the existing ~/$f"; fi
  install -m 0755 "$SRC/$f" "$HOME/$f"; echo "installed ~/$f"
done
[ "$DESK" = 1 ] || exit 0
D=${DESKTOP_DIR:-$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")}
[ -d "$D" ] || { echo "no desktop folder at $D, skipping shortcuts"; exit 0; }
mk() { f="$D/$1.desktop"; cat > "$f" <<EOT
[Desktop Entry]
Type=Application
Name=$2
Comment=$3
Exec=$HOME/$4
Icon=utilities-terminal
Terminal=true
Categories=Development;
EOT
  chmod +x "$f"; gio set "$f" metadata::trusted true 2>/dev/null || true; echo "created $f"; }
mk start-llama "Start llama.cpp servers" "27B on :11437 (dGPU), 4B on :8081 (iGPU), 2B on :8082 (CPU)" start-llama.sh
mk start-tabbyapi "Start TabbyAPI" "Qwen3.8-27B EXL3 + DFlash2 on :11437 (dGPU)" start-tabbyapi.sh
