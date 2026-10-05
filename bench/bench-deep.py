#!/usr/bin/env python3
"""Decode/prefill speed at depth. usage: bench-deep.py LABEL -- <server args>"""
import glob, json, os, signal, subprocess, sys, time, urllib.request
HOME = os.path.expanduser("~"); PORT = 8192
BENCH = os.environ.get("BENCH_DIR", os.path.dirname(os.path.abspath(__file__)))  # logs/, quality/ and results go here
os.makedirs(f"{BENCH}/logs", exist_ok=True)
argv = sys.argv[1:]; label = argv[0]; srv_args = argv[argv.index("--") + 1:]
env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib")
log = open(f"{BENCH}/logs/{label}.log", "w")
proc = subprocess.Popen([os.environ.get("BENCH_BIN", f"{HOME}/llama.cpp/build/bin/llama-server"), "--port", str(PORT), "--host", "127.0.0.1", *srv_args], stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))
files = sorted(glob.glob(f"{HOME}/llama.cpp/src/*.cpp") + glob.glob(f"{HOME}/llama.cpp/ggml/src/*.c*") + glob.glob(f"{HOME}/llama.cpp/tools/server/*.cpp") + glob.glob(f"{HOME}/llama.cpp/common/*.cpp"))
corpus = "".join(f"\n// FILE {os.path.basename(f)}\n" + open(f, errors="ignore").read() for f in files)
res = {"label": label, "depths": []}
try:
    for _ in range(600):
        if proc.poll() is not None: raise SystemExit(json.dumps({**res, "error": "server exited during load"}))
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok": break
        except Exception: time.sleep(1)
    for k, target_tokens in enumerate((16000, 32000, 64000, 120000)):
        text = f"// variant {k}\n" + corpus[: int(target_tokens * 3.4)]
        t0 = time.time()
        r = post({"messages": [{"role": "user", "content": "Here is source code:\n\n" + text + "\n\nWrite a 200-word summary of what this code base does."}], "max_tokens": 300, "temperature": 0.2, "chat_template_kwargs": {"reasoning_effort": "low"}})
        t = r["timings"]
        res["depths"].append({"prompt_n": t["prompt_n"], "prefill_tps": round(t["prompt_per_second"], 1), "decode_tps": round(t["predicted_per_second"], 1), "gen_n": t["predicted_n"], "wall_s": round(time.time() - t0)})
finally:
    try: os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=30)
    except Exception: os.killpg(proc.pid, signal.SIGKILL)
print(json.dumps(res))
