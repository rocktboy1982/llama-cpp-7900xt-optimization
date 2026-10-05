#!/usr/bin/env python3
"""Start llama-server with the given args, run 3 workloads, print one JSON line, stop the server.

usage: bench.py LABEL [--bin PATH] -- <llama-server args>
Workloads (first run of each config only; ngram-mod remembers repeats):
  newcode : write a 512-token python module
  edit    : return a 150-line file with a rename (reply copies the prompt)
  prefill : ~8k-token prompt, 16 output tokens
"""
import json, os, signal, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~")
BENCH = os.environ.get("BENCH_DIR", os.path.dirname(os.path.abspath(__file__)))  # logs/, quality/ and results go here
os.makedirs(f"{BENCH}/logs", exist_ok=True)
PORT = 8190
argv = sys.argv[1:]
label = argv[0]
sep = argv.index("--")
opts, srv_args = argv[1:sep], argv[sep + 1:]
binpath = f"{HOME}/llama.cpp/build/bin/llama-server"
if "--bin" in opts:
    binpath = opts[opts.index("--bin") + 1]

env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib")
log = open(f"{BENCH}/logs/{label}.log", "w")
proc = subprocess.Popen([binpath, "--port", str(PORT), "--host", "127.0.0.1", *srv_args],
                        stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)


def vram_gib():
    try:
        out = subprocess.check_output(["/opt/rocm/core-10.0/bin/rocm-smi", "--showmeminfo", "vram", "--json"], text=True)
        d = json.loads(out)
        return round(int(d["card0"]["VRAM Total Used Memory (B)"]) / 2**30, 2)
    except Exception:
        return None


def post(body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def stop():
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        os.killpg(proc.pid, signal.SIGKILL)


res = {"label": label, "args": " ".join(srv_args)}
try:
    for _ in range(600):
        if proc.poll() is not None:
            raise SystemExit(json.dumps({**res, "error": "server exited during load (see log)"}))
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok":
                break
        except Exception:
            time.sleep(1)
    else:
        raise SystemExit(json.dumps({**res, "error": "timeout waiting for health"}))
    res["vram_loaded_gib"] = vram_gib()

    src = open(f"{HOME}/llama.cpp/convert_hf_to_gguf_update.py").read().splitlines()
    edit_file = "\n".join(src[:150])
    big = open(f"{HOME}/llama.cpp/src/llama-kv-cache.cpp").read()[:30000]
    base = {"temperature": float(os.environ.get("BENCH_TEMP", "0.2")), "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"reasoning_effort": "low"}}
    work = {
        "newcode": dict(messages=[{"role": "user", "content": "Write a complete Python module implementing an LRU cache class with TTL support, type hints, docstrings and a small CLI demo. Code only."}], max_tokens=512),
        "edit": dict(messages=[{"role": "user", "content": "Return this file unchanged except rename the variable `args` to `cli_args` everywhere. Output only the full file.\n\n```python\n" + edit_file + "\n```"}], max_tokens=1500),
        "prefill": dict(messages=[{"role": "user", "content": "Summarize in one sentence:\n\n" + big}], max_tokens=16),
    }
    for name, body in work.items():
        t0 = time.time()
        r = post({**base, **body})
        t = r["timings"]
        res[name] = {
            "prompt_n": t["prompt_n"], "prefill_tps": round(t["prompt_per_second"], 1),
            "gen_n": t["predicted_n"], "decode_tps": round(t["predicted_per_second"], 1),
            "draft_n": t.get("draft_n", 0), "draft_acc": t.get("draft_n_accepted", 0),
            "wall_s": round(time.time() - t0, 1),
        }
    res["vram_peak_gib"] = vram_gib()
finally:
    stop()
print(json.dumps(res))
