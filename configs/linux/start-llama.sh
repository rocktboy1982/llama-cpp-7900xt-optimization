#!/usr/bin/env bash
# Desktop launcher: llama.cpp servers (dGPU 27B on :11437, iGPU 4B on :8081, CPU 2B on :8082).
# TabbyAPI and llama.cpp cannot share the 7900 XT, so ask before stopping TabbyAPI.
TP=$(ss -ltnp 2>/dev/null | grep ':11437 ' | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "$TP" ] && tr '\0' ' ' < /proc/$TP/cmdline 2>/dev/null | grep -q 'main.py'; then
    echo "TabbyAPI is running (PID $TP) and holds the GPU."
    read -r -p "Stop TabbyAPI and start the llama.cpp servers? [y/N] " a
    [ "$a" = y ] || [ "$a" = Y ] || { echo "Cancelled."; read -r -p "Press Enter to close..."; exit 0; }
    kill -TERM "$TP"; for _ in $(seq 1 40); do kill -0 "$TP" 2>/dev/null || break; sleep 1; done
fi
"$HOME/run-llama.sh"
echo; read -r -p "Servers stopped. Press Enter to close..."
