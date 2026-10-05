#!/usr/bin/env bash
cd "$(dirname "$0")/.."; source runners/common.env
OUT=results-skipped.jsonl
UB="--ubatch-size 1024 --batch-size 2048"
NG="--spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16"
run() { python3 bench.py "$@" 2>&1 | tail -1 >> $OUT; }
# --- speculation variants (65k ctx, q8_0)
run S-ref        -- $COMMON $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 $NG
run S-n8         -- $COMMON $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 8 $NG
run S-ng8-8      -- $COMMON $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 --spec-ngram-mod-n-match 8  --spec-ngram-mod-n-min 8
run S-ng16-16    -- $COMMON $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 --spec-ngram-mod-n-match 16 --spec-ngram-mod-n-min 16
run S-ng24-48    -- $COMMON $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48
run S-k4v-mtp    -- $COMMON $UB --spec-type ngram-map-k4v,draft-mtp --spec-draft-n-max 4
run S-simple-mtp -- $COMMON $UB --spec-type ngram-simple,draft-mtp --spec-draft-n-max 4
# --- CPU threads for the server
for t in 4 6 12; do run T-threads$t -- $COMMON $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 $NG --threads $t; done
# --- TurboQuant port: turbo4 KV
TQ=~/turboquant/fork/build/bin/llama-server
TQC=$(echo "$COMMON" | sed 's/--cache-type-k q8_0 --cache-type-v q8_0/--cache-type-k turbo4 --cache-type-v turbo4/')
run TQ-turbo4 --bin $TQ -- $TQC $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 $NG
# --- prompt cache variants (agent loop)
python3 bench-ckpt.py K-cram16k-reuse256 -- $COMMON $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 $NG --cache-ram 16384 --cache-reuse 256 2>&1 | tail -1 >> results-ckpt.jsonl
# --- decode at depth: q4_0 KV and turbo4 KV vs the q8_0 reference (51/38/36/29 t/s)
C131=$(echo "$COMMON" | sed "s/-c 65536/-c 131072/")
C131Q4=$(echo "$C131" | sed 's/--cache-type-k q8_0 --cache-type-v q8_0/--cache-type-k q4_0 --cache-type-v q4_0/')
C131T=$(echo "$C131" | sed 's/--cache-type-k q8_0 --cache-type-v q8_0/--cache-type-k turbo4 --cache-type-v turbo4/')
python3 bench-deep.py Z-deep-q4 -- $C131Q4 $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 $NG 2>&1 | tail -1 >> results-deep.jsonl
BENCH_BIN=$TQ python3 bench-deep.py Z-deep-turbo4 -- $C131T $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 $NG 2>&1 | tail -1 >> results-deep.jsonl
echo done > skipped.done
