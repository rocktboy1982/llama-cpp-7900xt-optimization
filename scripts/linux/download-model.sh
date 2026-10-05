#!/usr/bin/env bash
# Download GGUF models into ~/models (resumable, size-checked, skipped if already complete).
#
# usage: download-model.sh [swift-27b | small-4b | small-2b | all]     (default: swift-27b)
#   swift-27b : Swift 1.5 Qwen3.8-27B GSQ-RCO IQ3_S with MTP head   (12.12 GB)  dGPU server
#   small-4b  : Qwen3.8-4B-Distill Q4_K_M                            (2.78 GB)   iGPU server (tool calling 7/7)
#   small-2b  : Qwen3.8-2B-Distill Q4_K_M                            (1.31 GB)   CPU server (light chat only)
# env: MODELS_DIR (default ~/models)
set -euo pipefail
DIR=${MODELS_DIR:-$HOME/models}; mkdir -p "$DIR"
declare -A URL=(
 [swift-27b]="https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
 [small-4b]="https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF/resolve/main/Qwen3.8-4B-Q4_K_M.gguf"
 [small-2b]="https://huggingface.co/empero-ai/Qwen3.8-2B-Distill-GGUF/resolve/main/Qwen3.8-2B-Q4_K_M.gguf")
declare -A FILE=([swift-27b]=Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf [small-4b]=Qwen3.8-4B-Distill-Q4_K_M.gguf [small-2b]=Qwen3.8-2B-Distill-Q4_K_M.gguf)
want=${1:-swift-27b}; [ "$want" = all ] && want="swift-27b small-4b small-2b"
[ "$want" = -h ] || [ "$want" = --help ] && { sed -n 2,9p "$0"; exit 0; }
for k in $want; do
  [ -n "${URL[$k]:-}" ] || { echo "unknown model: $k" >&2; exit 2; }
  f="$DIR/${FILE[$k]}"
  size=$(curl -sIL "${URL[$k]}" | awk 'tolower($1)=="content-length:" {gsub("\r","",$2); s=$2} END {print s}')
  [ -n "$size" ] || { echo "$k: could not read the remote size (network / URL problem)" >&2; exit 1; }
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$have" = "$size" ]; then echo "$k: already complete ($(numfmt --to=iec "$size")), skipping"
  else
    free=$(df --output=avail -B1 "$DIR" | tail -1)
    [ "$free" -gt $((size - have + 1000000000)) ] || { echo "$k: not enough free disk space (need $(numfmt --to=iec $((size-have))) + 1 GB)" >&2; exit 1; }
    echo "$k: downloading $(numfmt --to=iec "$size") to $f"
    curl -L -C - --fail --progress-bar -o "$f" "${URL[$k]}"
    [ "$(stat -c %s "$f")" = "$size" ] || { echo "$k: size mismatch after download" >&2; exit 1; }
  fi
  if [ "$k" = swift-27b ]; then
    n=$(head -c 20000000 "$f" | grep -a -o 'blk\.64\.nextn\.eh_proj' | wc -l)   # no grep -q: early exit + pipefail = false negative
    if [ "$n" -gt 0 ]; then echo "$k: MTP head present (blk.64.nextn.*)"
    else echo "$k: WARNING - no MTP tensors found; speculative decoding (draft-mtp) will not work" >&2; fi
  fi
done
