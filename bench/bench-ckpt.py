#!/usr/bin/env python3
"""Agent-loop reuse test: usage bench-ckpt.py LABEL [--bin PATH] -- <llama-server args>
Step 1 sends a ~10k-token prompt; steps 2-5 append the previous reply plus a ~600-token 'tool output'.
Reports per step: tokens re-processed (prompt_n), prompt time (ms), total wall time. Lower is better."""
import json, os, signal, subprocess, sys, time, urllib.request
HOME = os.path.expanduser("~"); PORT = 8191
BENCH = os.environ.get("BENCH_DIR", os.path.dirname(os.path.abspath(__file__)))  # logs/, quality/ and results go here
os.makedirs(f"{BENCH}/logs", exist_ok=True)
argv = sys.argv[1:]; label = argv[0]; sep = argv.index("--")
opts, srv_args = argv[1:sep], argv[sep + 1:]
binpath = opts[opts.index("--bin") + 1] if "--bin" in opts else f"{HOME}/llama.cpp/build/bin/llama-server"
env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib")
log = open(f"{BENCH}/logs/{label}.log", "w")
proc = subprocess.Popen([binpath, "--port", str(PORT), "--host", "127.0.0.1", *srv_args], stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))
res = {"label": label, "args": " ".join(srv_args), "steps": []}
try:
    for _ in range(600):
        if proc.poll() is not None: raise SystemExit(json.dumps({**res, "error": "server exited during load"}))
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok": break
        except Exception: time.sleep(1)
    base = open(f"{HOME}/llama.cpp/src/llama-kv-cache.cpp").read()[:40000]
    chunks = [open(f"{HOME}/llama.cpp/src/{f}").read()[3000:5400] for f in ("llama-graph.cpp", "llama-model.cpp", "llama-context.cpp", "llama-batch.cpp")]
    msgs = [{"role": "user", "content": "You are reviewing this file. Reply with one short sentence.\n\n" + base}]
    for i in range(5):
        t0 = time.time()
        r = post({"messages": msgs, "max_tokens": 48, "temperature": 0.2, "chat_template_kwargs": {"reasoning_effort": "low"}})
        t = r["timings"]
        res["steps"].append({"step": i + 1, "prompt_n": t["prompt_n"], "cached": t.get("cache_n"), "prompt_ms": round(t["prompt_ms"]), "wall_s": round(time.time() - t0, 2)})
        msgs.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or "ok"})
        if i < 4: msgs.append({"role": "user", "content": "Tool output:\n" + chunks[i] + "\nReply with one short sentence."})
finally:
    try: os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=30)
    except Exception: os.killpg(proc.pid, signal.SIGKILL)
print(json.dumps(res))
