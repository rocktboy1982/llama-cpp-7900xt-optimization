#!/usr/bin/env python3
"""Quality gates on one server config. usage: bench-quality.py LABEL -- <server args>
 needle : hide a random code in ~32k / ~64k tokens of source, at 10/50/90% depth; ask for it back
 greedy : temp 0 outputs for 3 prompts (incl. one ~24k tokens) -> saved to quality/LABEL.json for diffing"""
import glob, json, os, random, signal, subprocess, sys, time, urllib.request
HOME = os.path.expanduser("~"); PORT = 8194
argv = sys.argv[1:]; label = argv[0]; srv = argv[argv.index("--")+1:]
env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib")
os.makedirs(f"{HOME}/bench/quality", exist_ok=True)
log = open(f"{HOME}/bench/logs/{label}.log", "w")
proc = subprocess.Popen([f"{HOME}/llama.cpp/build/bin/llama-server", "--port", str(PORT), "--host", "127.0.0.1", *srv], stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
def post(b):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
files = sorted(glob.glob(f"{HOME}/llama.cpp/src/*.cpp")+glob.glob(f"{HOME}/llama.cpp/ggml/src/*.c*")+glob.glob(f"{HOME}/llama.cpp/common/*.cpp")+glob.glob(f"{HOME}/llama.cpp/tools/server/*.cpp"))
corpus = "".join(f"\n// FILE {os.path.basename(f)}\n"+open(f,errors="ignore").read() for f in files)
res = {"label": label, "args": " ".join(srv), "needle": [], "greedy": {}}
rng = random.Random(7)
try:
    for _ in range(600):
        if proc.poll() is not None: raise SystemExit(json.dumps({**res, "error": "server exited during load"}))
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok": break
        except Exception: time.sleep(1)
    for toks in (32000, 64000):
        for pos in (0.1, 0.5, 0.9):
            code = f"{rng.randint(1000,9999)}-{rng.choice(['KILO','LIMA','ECHO','TANGO'])}"
            body = corpus[: int(toks*3.4)]
            cut = body.rfind("\n", 0, int(len(body)*pos))
            doc = body[:cut] + f"\n// NOTE: the vault access code is {code}.\n" + body[cut:]
            r = post({"messages":[{"role":"user","content":doc+"\n\nWhat is the vault access code mentioned in the notes? Answer with just the code."}],"max_tokens":400,"temperature":0.0,"chat_template_kwargs":{"reasoning_effort":"low"}})
            ans = r["choices"][0]["message"].get("content") or ""
            res["needle"].append({"tokens": r["timings"]["prompt_n"], "pos": pos, "ok": code in ans, "answer": ans.strip()[:60]})
    prompts = {"short": "Write a Python function that parses ISO-8601 dates without libraries, with tests.",
               "mid": corpus[:14000] + "\n\nList the main data structures declared above, one per line.",
               "long24k": corpus[20000:20000+int(24000*3.4)] + "\n\nSummarize the purpose of this code in 5 bullet points."}
    for k, p in prompts.items():
        r = post({"messages":[{"role":"user","content":p}],"max_tokens":300,"temperature":0.0,"chat_template_kwargs":{"reasoning_effort":"low"}})
        res["greedy"][k] = {"text": (r["choices"][0]["message"].get("content") or "") + "|R|" + (r["choices"][0]["message"].get("reasoning_content") or ""), "prompt_n": r["timings"]["prompt_n"]}
finally:
    try: os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=30)
    except Exception: os.killpg(proc.pid, signal.SIGKILL)
json.dump(res, open(f"{HOME}/bench/quality/{label}.json", "w"))
print(json.dumps({"label": label, "needle_ok": f"{sum(n['ok'] for n in res['needle'])}/{len(res['needle'])}", "needle": [(n['tokens'], n['pos'], n['ok']) for n in res['needle']]}))
