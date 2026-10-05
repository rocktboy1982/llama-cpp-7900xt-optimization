#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python3 tabby-ctx.py D-q4-196k 196608 Q4 Q4 >> /dev/null 2>&1
python3 tabby-ctx.py C-q4-131k 131072 Q4 Q4 >> /dev/null 2>&1
python3 tabby-ctx.py B-q6-131k 131072 Q6 Q4 >> /dev/null 2>&1
python3 tabby-ctx.py A-q8-98k 98304 Q8 Q4 >> /dev/null 2>&1
echo done > tabby-ctx.done
