#!/usr/bin/env python3
"""Tool-calling test. usage: bench-tools.py BASE_URL MODEL [effort] ; env API_KEY. Prints pass/fail per case."""
import json, os, sys, time, urllib.request
base, model = sys.argv[1], sys.argv[2]; effort = sys.argv[3] if len(sys.argv) > 3 else "low"
key = os.environ.get("API_KEY", "")
def T(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}}
tools = [
 T("get_weather", "Get current weather for a city", {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
 T("list_ec2_instances", "List EC2 instances in a region", {"region": {"type": "string"}, "state": {"type": "string", "enum": ["running", "stopped", "all"]}}, ["region"]),
 T("resize_volume", "Resize an EBS volume", {"volume_id": {"type": "string"}, "size_gb": {"type": "integer"}}, ["volume_id", "size_gb"]),
 T("create_alarm", "Create a CloudWatch alarm", {"metric": {"type": "string"}, "threshold": {"type": "number"}, "comparison": {"type": "string", "enum": ["GreaterThan", "LessThan"]}}, ["metric", "threshold", "comparison"]),
]
def call(msgs):
    body = {"model": model, "messages": msgs, "tools": tools, "tool_choice": "auto", "temperature": 0.2, "max_tokens": 1500, "chat_template_kwargs": {"reasoning_effort": effort}}
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = f"Bearer {key}"
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"{base}/chat/completions", json.dumps(body).encode(), h), timeout=900))
    return r["choices"][0]["message"], (r.get("usage") or {}).get("completion_tokens"), time.time() - t0
def tcs(m): return [(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")) for c in (m.get("tool_calls") or [])]
cases = []
def case(name, fn):
    try: ok, info, toks, secs = fn()
    except Exception as e: ok, info, toks, secs = False, f"EXC {type(e).__name__}: {e}"[:80], None, 0
    cases.append((name, ok, info, toks, secs)); print(f"{'PASS' if ok else 'FAIL'}  {name:18s} {toks!s:>5} tok {secs:5.1f}s  {info}")
def c1():
    m, n, s = call([{"role": "user", "content": "What's the weather in Paris right now, in celsius?"}]); t = tcs(m)
    return (len(t) == 1 and t[0][0] == "get_weather" and str(t[0][1].get("city", "")).lower() == "paris"), t, n, s
def c2():
    m, n, s = call([{"role": "user", "content": "Show me all running EC2 instances in eu-west-1."}]); t = tcs(m)
    return (len(t) == 1 and t[0][0] == "list_ec2_instances" and t[0][1].get("region") == "eu-west-1" and t[0][1].get("state") == "running"), t, n, s
def c3():
    m, n, s = call([{"role": "user", "content": "Grow volume vol-0abc123 to 200 GB."}]); t = tcs(m)
    return (len(t) == 1 and t[0][0] == "resize_volume" and t[0][1].get("volume_id") == "vol-0abc123" and t[0][1].get("size_gb") == 200), t, n, s
def c4():
    m, n, s = call([{"role": "user", "content": "What is 17 times 3? Answer directly."}]); t = tcs(m)
    return (len(t) == 0 and "51" in (m.get("content") or "")), (m.get("content") or "")[:50], n, s
def c5():
    msgs = [{"role": "user", "content": "How many running instances are in us-east-1? Use the tool."}]
    m, n, s = call(msgs); t = tcs(m)
    if not (t and t[0][0] == "list_ec2_instances"): return False, t, n, s
    msgs += [{"role": "assistant", "content": m.get("content") or "", "tool_calls": m["tool_calls"]}, {"role": "tool", "tool_call_id": m["tool_calls"][0]["id"], "content": json.dumps({"instances": ["i-1", "i-2", "i-3"]})}]
    m2, n2, s2 = call(msgs)
    return ("3" in (m2.get("content") or "") or "three" in (m2.get("content") or "").lower()), (m2.get("content") or "")[:60], (n or 0) + (n2 or 0), s + s2
def c6():
    m, n, s = call([{"role": "user", "content": "Alert me when CPUUtilization goes above 80 percent."}]); t = tcs(m)
    return (len(t) == 1 and t[0][0] == "create_alarm" and t[0][1].get("metric") == "CPUUtilization" and float(t[0][1].get("threshold", 0)) == 80 and t[0][1].get("comparison") == "GreaterThan"), t, n, s
def c7():
    m, n, s = call([{"role": "user", "content": "Get the weather in Tokyo and in Berlin."}]); t = tcs(m)
    return (len(t) == 2 and {str(x[1].get("city", "")).lower() for x in t} == {"tokyo", "berlin"}), t, n, s
for nme, f in [("single call", c1), ("enum args", c2), ("integer arg", c3), ("no tool needed", c4), ("multi-step", c5), ("number+enum", c6), ("parallel calls", c7)]: case(nme, f)
p = sum(c[1] for c in cases); tot = sum(c[3] or 0 for c in cases)
print(f"RESULT {p}/{len(cases)} passed | effort={effort} | total completion tokens={tot}")
