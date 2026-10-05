import json, os, sys
a, b = (json.load(open(os.path.expanduser(f"~/bench/quality/{x}.json"))) for x in sys.argv[1:3])
for k in a["greedy"]:
    x, y = a["greedy"][k]["text"], b["greedy"][k]["text"]
    n = next((i for i, (c, d) in enumerate(zip(x, y)) if c != d), None)
    print(f"{k:8s} prompt_n={a['greedy'][k]['prompt_n']:>6}  identical={x==y}  first_diff_char={n}  len={len(x)}/{len(y)}")
