#!/usr/bin/env bash
# Desktop launcher: TabbyAPI (ExLlamaV3-ROCm) on :11437 via the fork's run_tabbyapi.sh.
# llama.cpp and TabbyAPI cannot share the 7900 XT, so ask before stopping llama-server.
TP=$(ss -ltnp 2>/dev/null | grep ':11437 ' | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "$TP" ] && tr '\0' ' ' < /proc/$TP/cmdline 2>/dev/null | grep -q 'main.py'; then
    echo "TabbyAPI is already running (PID $TP) on port 11437."
    read -r -p "Press Enter to close..."; exit 0
fi
if pgrep -x llama-server >/dev/null; then
    echo "llama.cpp servers are running and hold the GPU."
    read -r -p "Stop them and start TabbyAPI? [y/N] " a
    [ "$a" = y ] || [ "$a" = Y ] || { echo "Cancelled."; read -r -p "Press Enter to close..."; exit 0; }
    pkill -x llama-server; sleep 4
fi
cd "$HOME/exllamav3-rocm" && source .venv-rocm/bin/activate || exit 1
echo "Starting TabbyAPI (Ctrl+C stops it)..."
rocm/scripts/run_tabbyapi.sh config.yml "$HOME/tabbyAPI"
echo; read -r -p "TabbyAPI stopped. Press Enter to close..."
