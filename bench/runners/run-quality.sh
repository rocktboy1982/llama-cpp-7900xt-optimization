#!/usr/bin/env bash
cd "$(dirname "$0")/.."; source runners/common.env
C2=$(echo "$COMMON" | sed "s/-c 65536/-c 131072/")
SP="--spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --ubatch-size 1024 --batch-size 2048"
python3 bench-quality.py Q-q8-nospec -- $C2 --ubatch-size 1024 --batch-size 2048 >> results-quality.jsonl 2>&1
python3 bench-quality.py Q-q8-spec -- $C2 $SP >> results-quality.jsonl 2>&1
C4=$(echo "$C2" | sed 's/--cache-type-k q8_0 --cache-type-v q8_0/--cache-type-k q4_0 --cache-type-v q4_0/')
python3 bench-quality.py Q-q4-spec -- $C4 $SP >> results-quality.jsonl 2>&1
echo done > quality.done
