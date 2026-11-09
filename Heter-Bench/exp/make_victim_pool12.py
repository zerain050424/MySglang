#!/usr/bin/env python3
"""从 AgentBank alfred 全量数据切出 12 条 26 轮 victim，均分为 H/W/C 三组。

选片规则（保证三组可比）：
  - num_turns >= 26（只保留前 26 轮，prompt 不截断）
  - turn_26 的 prompt 长度落在 [2400, 2560]，压缩链长差异
  - 按长度排序后 round-robin 分到三组，避免某组系统性偏长
用法：python exp/make_victim_pool12.py
"""
import json
from pathlib import Path

SRC = Path("TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl")
OUT = Path("TaskData")
TURNS = 26
N = 12
LO, HI = 2400, 2560

cands = []
with open(SRC, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3000:            # 前 3000 行足够（实测 1484 条 >=26 轮）
            break
        sid, sess = next(iter(json.loads(line).items()))
        if sess["num_turns"] < TURNS:
            continue
        L = len(sess[f"turn_{TURNS}"]["input_ids"])
        if LO <= L <= HI:
            cands.append((L, sid, sess))

cands.sort(key=lambda x: x[0])
if len(cands) < N:
    raise SystemExit(f"候选不足：{len(cands)} < {N}，请放宽 [{LO},{HI}]")

# 均匀取 12 条（等间隔抽样，覆盖整个区间且不挑极值）
step = len(cands) / N
pick = [cands[int(i * step)] for i in range(N)]

for g, name in zip((0, 1, 2), ("H", "W", "C")):
    grp = pick[g::3]                     # round-robin：长度交错分配
    path = OUT / f"victim_pool12_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for L, sid, sess in grp:
            slim = {k: sess[k] for k in ["id", "num_turns"] + [f"turn_{t}" for t in range(1, TURNS + 1)]}
            slim["num_turns"] = TURNS
            f.write(json.dumps({sid: slim}, ensure_ascii=False) + "\n")
    print(f"{path}  {len(grp)} 条  turn{TURNS} 长度={[L for L, _, _ in grp]}  合计={sum(L for L,_,_ in grp)}")
print(f"12 条合计驻留 = {sum(L for L,_,_ in pick)} tokens")
