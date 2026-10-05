#!/usr/bin/env python3
"""Streaming benchmark for any OpenAI-compatible server. usage: bench-api.py BASE_URL MODEL ; env API_KEY
Workloads like bench.py: newcode (512 tok), edit (150-line file), prefill (~9k tokens). Prints JSON."""
import json, os, sys, time, urllib.request
base, model = sys.argv[1], sys.argv[2]; key = os.environ.get("API_KEY", ""); HOME = os.path.expanduser("~")
src = open(f"{HOME}/llama.cpp/convert_hf_to_gguf_update.py").read().splitlines()
big = open(f"{HOME}/llama.cpp/src/llama-kv-cache.cpp").read()[:30000]
work = {"newcode": ("Write a complete Python module implementing an LRU cache class with TTL support, type hints, docstrings and a small CLI demo. Code only.", 512),
        "edit": ("Return this file unchanged except rename the variable `args` to `cli_args` everywhere. Output only the full file.\n\n```python\n" + "\n".join(src[:150]) + "\n```", 1500),
        "prefill": ("Summarize in one sentence:\n\n" + big, 16)}
res = {"model": model}
for name, (prompt, mt) in work.items():
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": 0.2, "stream": True, "stream_options": {"include_usage": True}}
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = f"Bearer {key}"
    t0 = time.time(); tfirst = None; usage = None
    with urllib.request.urlopen(urllib.request.Request(f"{base}/chat/completions", json.dumps(body).encode(), h), timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")) and tfirst is None: tfirst = time.time()
    t1 = time.time(); pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    res[name] = {"prompt_n": pt, "ttft_s": round(tfirst - t0, 2), "prefill_tps": round(pt / (tfirst - t0), 1) if name == "prefill" else None,
                 "gen_n": ct, "decode_tps": round((ct - 1) / (t1 - tfirst), 1) if ct > 1 and t1 > tfirst else None}
print(json.dumps(res))
