#!/usr/bin/env python3
"""usage: tabby-ctx.py LABEL MAX_SEQ_LEN CACHE_MODE [DRAFT_CACHE_MODE]
Rewrites ~/tabbyAPI/config.yml, restarts TabbyAPI, measures VRAM, then (a) decode/prefill at 28k, 60% and 90% of the context and
(b) needle recall: a code hidden at 10% depth of the 90%-of-context prompt. Appends one JSON line to results-tabby-ctx.jsonl."""
import json, os, re, subprocess, sys, time, urllib.request
HOME = os.path.expanduser("~"); label, ctx, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]; dmode = sys.argv[4] if len(sys.argv) > 4 else "Q4"
URL = "http://127.0.0.1:11437"; KEY = open(f"{HOME}/.llama-api-key").read().strip(); MODEL = "Qwen3.8-27B-EXL3-3.5bpw"
RS = "/opt/rocm/core-10.0/bin/rocm-smi"
def vram():
    d = json.loads(subprocess.check_output([RS, "--showmeminfo", "vram", "--json"], text=True))["card0"]
    return round(int(d["VRAM Total Used Memory (B)"]) / 2**30, 2)
def stop_tabby():
    out = subprocess.run("ss -ltnp | grep ':11437' | grep -oP 'pid=\\K[0-9]+' | head -1", shell=True, capture_output=True, text=True).stdout.strip()
    if out:
        os.kill(int(out), 15)
        for _ in range(40):
            try: os.kill(int(out), 0); time.sleep(1)
            except OSError: break
cfgp = f"{HOME}/tabbyAPI/config.yml"; s = open(cfgp).read()
s = re.sub(r"(?m)^(  max_seq_len: ).*$", rf"\g<1>{ctx}", s); s = re.sub(r"(?m)^(  cache_size: ).*$", rf"\g<1>{ctx}", s)
s = re.sub(r"(?m)^(  cache_mode: ).*$", rf"\g<1>{mode}", s); s = re.sub(r"(?m)^(  draft_cache_mode: ).*$", rf"\g<1>{dmode}", s)
open(cfgp, "w").write(s)
stop_tabby(); time.sleep(3)
env = dict(os.environ, VIRTUAL_ENV=f"{HOME}/exllamav3-rocm/.venv-rocm", HIP_VISIBLE_DEVICES="0", PATH=f"{HOME}/exllamav3-rocm/.venv-rocm/bin:" + os.environ["PATH"])
log = open(f"{HOME}/bench/logs/tabby-{label}.log", "w")
proc = subprocess.Popen([f"{HOME}/exllamav3-rocm/.venv-rocm/bin/python", "main.py", "--config", "config.yml"], cwd=f"{HOME}/tabbyAPI", env=env, stdout=log, stderr=subprocess.STDOUT)
res = {"label": label, "ctx": ctx, "cache": mode, "draft_cache": dmode}
def done(extra):
    res.update(extra); print(json.dumps(res)); open(f"{HOME}/bench/results-tabby-ctx.jsonl", "a").write(json.dumps(res) + "\n"); sys.exit(0)
t0 = time.time()
while True:
    if proc.poll() is not None: done({"result": "FAIL", "error": "tabby exited during load (see log)"})
    try:
        if "healthy" in urllib.request.urlopen(URL + "/health", timeout=3).read().decode(): break
    except Exception: pass
    if time.time() - t0 > 480: proc.kill(); done({"result": "FAIL", "error": "load timeout"})
    time.sleep(3)
res["load_s"] = round(time.time() - t0); res["vram_loaded_gib"] = vram()
# corpus of source code
import glob
files = []
for root, _, fs in os.walk(f"{HOME}/llama.cpp"):
    if "/build" in root or "/.git" in root or "/models" in root: continue
    for f in fs:
        if f.endswith((".cpp", ".h", ".cu", ".cuh", ".c", ".py", ".md")): files.append(os.path.join(root, f))
files.sort(); corpus = ""
for f in files:
    corpus += f"\n// FILE {os.path.relpath(f, HOME)}\n" + open(f, errors="ignore").read()
    if len(corpus) > int(ctx * 6.0): break
def req(prompt, max_tokens=300):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.2, "chat_template_kwargs": {"reasoning_effort": "low"}, "stream": True, "stream_options": {"include_usage": True}}
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
    t0 = time.time(); tfirst = None; usage = None; text = ""
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), h), timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            j = json.loads(line[5:])
            if j.get("usage"): usage = j["usage"]
            ch = j.get("choices") or []
            if ch:
                d = ch[0].get("delta", {}); c = d.get("content") or ""; text += c + (d.get("reasoning_content") or "")
                if (c or d.get("reasoning_content")) and tfirst is None: tfirst = time.time()
    t1 = time.time(); pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    return {"prompt_n": pt, "prefill_tps": round(pt / (tfirst - t0), 1), "decode_tps": round((ct - 1) / max(1e-6, t1 - tfirst), 1), "gen_n": ct}, text
try:
    res["depth"] = []; code = f"{__import__('random').Random(ctx).randint(1000,9999)}-LIMA"
    cpt = 3.4  # chars per token, recalibrated from each request's reported prompt_tokens
    for frac, tok in ((None, 28000), (0.6, int(ctx * 0.6)), (0.9, int(ctx * 0.93))):
        if len(corpus) < int(tok * cpt) + 100000: raise RuntimeError("corpus too small for target")
        body = corpus[: int(tok * cpt)]
        needle = frac == 0.9
        if needle:
            cut = body.rfind("\n", 0, int(len(body) * 0.1)); body = body[:cut] + f"\n// NOTE: the vault access code is {code}.\n" + body[cut:]
        ask = "\n\nFirst state the vault access code from the notes (if any), then write a 150-word summary of what this code base does."
        m, text = req(f"// variant {tok} {time.time()}\n" + body + ask, 1500)
        m["target_tokens"] = tok; cpt = len(body) / m["prompt_n"]
        if needle: m["needle_ok"] = code in text
        res["depth"].append(m); res["vram_peak_gib"] = vram()
    done({"result": "OK"})
except Exception as e:
    res["vram_peak_gib"] = vram(); done({"result": "FAIL", "error": f"{type(e).__name__}: {str(e)[:150]}"})
