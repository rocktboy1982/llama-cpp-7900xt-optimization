#!/usr/bin/env bash
cd "$(dirname "$0")/.."; source runners/common.env
UB="--ubatch-size 1024 --batch-size 2048"
for rep in 1 2 3; do
  python3 bench.py N-ref12_16-$rep   -- $COMMON $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 2>&1 | tail -1 >> results-ngram.jsonl
  python3 bench.py N-mod24_48-$rep   -- $COMMON $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 2>&1 | tail -1 >> results-ngram.jsonl
  python3 bench.py N-simple-$rep     -- $COMMON $UB --spec-type ngram-simple,draft-mtp --spec-draft-n-max 4 2>&1 | tail -1 >> results-ngram.jsonl
  python3 bench.py N-simple+mod-$rep -- $COMMON $UB --spec-type ngram-simple,ngram-mod,draft-mtp --spec-draft-n-max 4 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 2>&1 | tail -1 >> results-ngram.jsonl
  python3 bench.py N-mod16_16-$rep   -- $COMMON $UB --spec-type ngram-mod,draft-mtp --spec-draft-n-max 4 --spec-ngram-mod-n-match 16 --spec-ngram-mod-n-min 16 2>&1 | tail -1 >> results-ngram.jsonl
done
echo done > ngram.done
