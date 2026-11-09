import json, sys, os
TPL = 2048 + 256
def load(name):
    p = "result/mx/%s.json" % name
    if not os.path.exists(p): return None
    d = json.load(open(p)); s = d["summary"]
    reqs = [x for x in (d.get("rows") or []) if "noise" not in (x.get("sid") or "")]
    n = len(reqs) or 1
    intact = sum(1 for x in reqs if x["cached_tokens"] >= TPL) / n * 100
    tpl = sum(min(x["cached_tokens"], TPL) for x in reqs) / (n * TPL) * 100
    cap = sum(x["prompt_tokens"] - TPL for x in reqs) or 1
    priv = sum(max(0, x["cached_tokens"] - TPL) for x in reqs) / cap * 100
    return dict(intact=round(intact,1), tpl=round(tpl,1), priv=round(priv,1),
                hit=round(s["hit_pct"],2), ev=s["evictions"],
                picks=(s.get("slru") or {}).get("slru_picks"))
rows = []
for trace, pools in json.loads(sys.argv[1]):
    for pool in pools:
        for arm in ("A0","A1","A2","A3","A4"):
            r = load("%s_%s_%s" % (trace, arm, pool))
            if r: rows.append((trace, pool, arm, r))
print(json.dumps(rows))
