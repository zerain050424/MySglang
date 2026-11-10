#!/usr/bin/env python3
"""回放器不变量（与命中率无关的硬约束）。

1. 树内 token 数 == 分配器 used：两者不等就意味着同一段池索引出现在多个节点上，
   命中率会虚高（这类缺陷不会抛异常，只能靠账目对账发现）。
2. 在途锁定后仍不得出现「树内索引重复」。
"""
import json, sys
sys.path.insert(0, ".")
from wtl.trace import schedule_hybrid
from wtl.simclock import SimClock, install
from wtl.backends import make_backend

clock = SimClock(); install(clock)
reqs = schedule_hybrid(tasks_config="../mixexp/configs/mix_smoke.yaml", seed=42,
                       wait_scale=0.1, arrival_scale=1.0, concurrency=8)
cfg = json.loads([l.split("\t")[2] for l in open("../mixexp/arms.tsv", encoding="utf-8")
                  if l.startswith("hc_full")][0])
be = make_backend("real", pool_tokens=30000, policy="heter_cache", config=cfg)
for r in reqs:
    clock.set(r.ts); be.acquire(r)

seen, toks = set(), 0
st = [be.cache.root_node]
while st:
    n = st.pop(); st.extend(n.children.values())
    if n is be.cache.root_node:
        continue
    toks += len(n.value)
    for v in n.value.tolist():
        assert v not in seen, f"池索引 {v} 在多个节点中重复出现"
        seen.add(v)
used = be.used_tokens()
assert toks == used, f"树内 token {toks} != 分配器 used {used}"
print(f"invariants OK: tree={toks} used={used}")
