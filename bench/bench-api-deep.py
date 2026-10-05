#!/usr/bin/env python3
"""Decode/prefill at depth through any OpenAI-compatible streaming API. usage: bench-api-deep.py BASE_URL MODEL [depths_k,...] ; env API_KEY"""
import glob, json, os, sys, time, urllib.request
base, model = sys.argv[1], sys.argv[2]; depths = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "14,28,56").split(",")]
key = os.environ.get("API_KEY", ""); HOME = os.path.expanduser("~")
files = sorted(glob.glob(f"{HOME}/llama.cpp/src/*.cpp") + glob.glob(f"{HOME}/llama.cpp/ggml/src/*.c*") + glob.glob(f"{HOME}/llama.cpp/tools/server/*.cpp") + glob.glob(f"{HOME}/llama.cpp/common/*.cpp"))
corpus = "".join(f"\n// FILE {os.path.basename(f)}\n" + open(f, errors="ignore").read() for f in files)
out = []
for k, d in enumerate(depths):
    text = f"// variant {k} {time.time()}\n" + corpus[: int(d * 1000 * 3.4)]
    body = {"model": model, "messages": [{"role": "user", "content": "Here is source code:\n\n" + text + "\n\nWrite a 200-word summary of what this code base does."}], "max_tokens": 300, "temperature": 0.2, "stream": True, "stream_options": {"include_usage": True}}
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = f"Bearer {key}"
    t0 = time.time(); tfirst = None; usage = None
    with urllib.request.urlopen(urllib.request.Request(f"{base}/chat/completions", json.dumps(body).encode(), h), timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            j = json.loads(line[5:])
            if j.get("usage"): usage = j["usage"]
            ch = j.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")) and tfirst is None: tfirst = time.time()
    t1 = time.time(); pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    out.append({"prompt_n": pt, "prefill_tps": round(pt / (tfirst - t0), 1), "decode_tps": round((ct - 1) / (t1 - tfirst), 1), "gen_n": ct})
print(json.dumps(out))
