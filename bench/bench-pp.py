#!/usr/bin/env python3
"""warm-up + 3 runs at ~16k tokens (distinct prefixes). usage: bench-pp.py LABEL [--bin P] -- <server args>"""
import glob, json, os, signal, statistics as st, subprocess, sys, time, urllib.request
HOME = os.path.expanduser("~"); PORT = 8193
argv = sys.argv[1:]; label = argv[0]; sep = argv.index("--"); opts, srv = argv[1:sep], argv[sep+1:]
binp = opts[opts.index("--bin")+1] if "--bin" in opts else f"{HOME}/llama.cpp/build/bin/llama-server"
env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib")
log = open(f"{HOME}/bench/logs/{label}.log", "w")
proc = subprocess.Popen([binp, "--port", str(PORT), "--host", "127.0.0.1", *srv], stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
def post(b):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
def vram():
    try: return round(int(json.loads(subprocess.check_output(["/opt/rocm/core-10.0/bin/rocm-smi","--showmeminfo","vram","--json"],text=True))["card0"]["VRAM Total Used Memory (B)"])/2**30,2)
    except Exception: return None
files = sorted(glob.glob(f"{HOME}/llama.cpp/src/*.cpp")+glob.glob(f"{HOME}/llama.cpp/ggml/src/*.c*")+glob.glob(f"{HOME}/llama.cpp/common/*.cpp"))
corpus = "".join(f"\n// FILE {os.path.basename(f)}\n"+open(f,errors="ignore").read() for f in files)
res = {"label": label, "args": " ".join(srv)}
try:
    for _ in range(600):
        if proc.poll() is not None: raise SystemExit(json.dumps({**res, "result": "FAIL", "error": "server exited during load"}))
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok": break
        except Exception: time.sleep(1)
    runs = []
    for k, n in enumerate((4000, 16000, 16000, 16000)):
        t0 = time.time()
        r = post({"messages":[{"role":"user","content":f"// run {k} {time.time()}\n"+corpus[k*7000:k*7000+int(n*3.4)]+"\n\nIn one sentence, what is this?"}],"max_tokens":200,"temperature":0.2,"chat_template_kwargs":{"reasoning_effort":"low"}})
        t = r["timings"]; runs.append((t["prompt_n"], t["prompt_per_second"], t["predicted_per_second"]))
    m = runs[1:]
    f = lambda i: {"median": round(st.median(x[i] for x in m),1), "min": round(min(x[i] for x in m),1), "max": round(max(x[i] for x in m),1)}
    res.update(result="OK", prompt_n=int(st.median(x[0] for x in m)), prefill=f(1), decode=f(2), vram_gib=vram())
finally:
    try: os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=30)
    except Exception: os.killpg(proc.pid, signal.SIGKILL)
print(json.dumps(res))
