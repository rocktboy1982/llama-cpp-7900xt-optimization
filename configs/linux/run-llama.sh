#!/usr/bin/env bash
# llama.cpp servers: dGPU (7900 XT, ROCm) / iGPU (Vulkan) / CPU.
# Binary: ~/llama.cpp/build (upstream 2ca15f540, ROCm 10 + Vulkan). Tuning results: ~/bench/results*.jsonl
# No `set -e`: one crashing server must not take down the others.
set -uo pipefail

# ── Settings (override via environment) ─────────────────────────────────────
DGPU_HOST="${DGPU_HOST:-0.0.0.0}"         # dGPU model is reachable from the network (iGPU/CPU servers stay local)
DGPU_PORT="${DGPU_PORT:-11437}"
LOCAL_HOST="127.0.0.1"
# API key for the dGPU server: generated once into ~/.llama-api-key. LLAMA_API_KEY overrides it, LLAMA_NO_AUTH=1 disables auth.
KEY_FILE="$HOME/.llama-api-key"
if [ -z "${LLAMA_API_KEY:-}" ] && [ -z "${LLAMA_NO_AUTH:-}" ]; then
    [ -s "$KEY_FILE" ] || { umask 077; openssl rand -hex 24 > "$KEY_FILE"; }
    LLAMA_API_KEY="$(cat "$KEY_FILE")"
fi
DGPU_CTX="${DGPU_CTX:-131072}"            # 131072 + q8_0 peaks ~19 GiB of 20. Max: 262144 with DGPU_KV=q4_0 DGPU_UBATCH=512
DGPU_KV="${DGPU_KV:-q8_0}"                # K and V use the same type (mixed types fall off the fast attention path)
DGPU_UBATCH="${DGPU_UBATCH:-1024}"
DGPU_EFFORT="${DGPU_EFFORT:-medium}"      # reasoning effort default: low | medium | xhigh (requests can override)

echo "Killing any existing llama-server processes..."
pkill -x llama-server 2>/dev/null
sleep 2

LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
MODEL_DIR="$HOME/models"
# ROCm 10 keeps its libraries under core-10.0
export LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DGPU_MODEL="$MODEL_DIR/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"  # has the MTP head (blk.64.nextn.*)
IGPU_MODEL="$MODEL_DIR/Qwen3.8-4B-Distill-Q4_K_M.gguf"   # tool calling 7/7, ~11 t/s on the iGPU (the 2B distill only passes 3/7)
CPU_MODEL="$MODEL_DIR/Qwen3.8-2B-Distill-Q4_K_M.gguf"    # ~22 t/s on the CPU; light chat only, NOT reliable for tool calls

AUTH=()
[ -n "${LLAMA_API_KEY:-}" ] && AUTH=(--api-key "$LLAMA_API_KEY")

PIDS=()
cleanup() {
    echo "Shutting down..."
    [ ${#PIDS[@]} -gt 0 ] && kill "${PIDS[@]}" 2>/dev/null && wait "${PIDS[@]}" 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# ── dGPU — RX 7900 XT via ROCm 10 (gfx1100), Swift 1.5 27B + MTP ────────────
# Measured (131k ctx, q8_0): ~62 t/s new code, ~175 t/s code edits, ~790 t/s prefill; decode falls to ~29 t/s at 100k depth.
# --spec-type: MTP head drafts tokens; ngram-simple drafts runs copied from the prompt (edits: 98 -> 175 t/s vs ngram-mod 12/16).
# Not worth it (tested): HIP graphs, unroll flag, COMPUTE power profile, --spec-draft-p-min, checkpoint/cache-ram tuning, server threads,
# draft length 8, turbo4 KV (-20% decode at depth), AMD-Ecosystem fork, Vulkan.
if [ -f "$DGPU_MODEL" ]; then
    USED=$(/opt/rocm/core-10.0/bin/rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c "import json,sys; print(int(int(json.load(sys.stdin)['card0']['VRAM Total Used Memory (B)'])/2**30))" 2>/dev/null || echo 0)
    if [ "${USED:-0}" -ge 2 ]; then
        echo "!! dGPU already has ${USED} GiB in use (another model server?). Skipping the dGPU server to avoid an OOM."
    else
        echo "Starting dGPU (Swift 1.5 27B, ROCm, ctx $DGPU_CTX, KV $DGPU_KV) on port $DGPU_PORT..."
        "$LLAMA_SERVER" \
            --model "$DGPU_MODEL" --device ROCm0 \
            --host "$DGPU_HOST" --port "$DGPU_PORT" ${AUTH[@]+"${AUTH[@]}"} \
            --n-gpu-layers 999 --ctx-size "$DGPU_CTX" \
            --flash-attn on --cache-type-k "$DGPU_KV" --cache-type-v "$DGPU_KV" \
            --batch-size 2048 --ubatch-size "$DGPU_UBATCH" \
            --spec-type ngram-simple,draft-mtp --spec-draft-n-max 4 \
            --threads 8 --parallel 1 --jinja --reasoning-format deepseek \
            --chat-template-kwargs "{\"reasoning_effort\":\"$DGPU_EFFORT\"}" \
            --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
            2>&1 | sed 's/^/[dGPU 27B] /' &
        PIDS+=($!)
    fi
else
    echo "!! Missing $DGPU_MODEL - skipping dGPU server."
fi

# ── iGPU — Vega 8 via Vulkan (ROCm 10 does not support gfx90c) ──────────────
if [ -f "$IGPU_MODEL" ]; then
    echo "Starting iGPU (Qwen3.8-4B-Distill, Vulkan) on port 8081..."
    "$LLAMA_SERVER" \
        --model "$IGPU_MODEL" --device Vulkan0 \
        --host "$LOCAL_HOST" --port 8081 \
        --n-gpu-layers 999 --ctx-size 32768 \
        --threads 2 --parallel 1 --jinja --reasoning-format deepseek \
        2>&1 | sed 's/^/[iGPU 7B] /' &
    PIDS+=($!)
else
    echo "!! Missing $IGPU_MODEL - skipping iGPU server."
fi

# ── CPU — Ryzen 7 5700G ─────────────────────────────────────────────────────
if [ -f "$CPU_MODEL" ]; then
    echo "Starting CPU (Qwen3.8-2B-Distill) on port 8082..."
    "$LLAMA_SERVER" \
        --model "$CPU_MODEL" --device none \
        --host "$LOCAL_HOST" --port 8082 \
        --n-gpu-layers 0 --ctx-size 32768 \
        --threads 12 --parallel 1 --jinja --reasoning-format deepseek \
        2>&1 | sed 's/^/[CPU 7B] /' &
    PIDS+=($!)
else
    echo "!! Missing $CPU_MODEL - skipping CPU server."
fi

echo ""
echo "=== Running ==="
echo "  dGPU model: http://$DGPU_HOST:$DGPU_PORT  (auth: $([ -n "${LLAMA_API_KEY:-}" ] && echo "API key in $KEY_FILE" || echo OFF))"
echo "  iGPU: 127.0.0.1:8081   CPU: 127.0.0.1:8082 (local only)"
echo "Monitor: watch -n 1 /opt/rocm/core-10.0/bin/rocm-smi   |   Ctrl+C stops all."
[ ${#PIDS[@]} -gt 0 ] && wait
