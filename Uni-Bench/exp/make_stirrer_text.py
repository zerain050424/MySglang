#!/usr/bin/env python3
"""切出「可调度」搅拌流：ShareGPT 多轮会话的第 1 轮，各成一个单轮会话。

用途：Cascade 实验的压力必须落在**可调度带**。单轮 apps/Codeparrot 属 one-shot 带，
会被优先清空，压力到不了 agentic 受害者（R3 首跑实测 evicted single 33,798 /
agentic 2,407，即证）。此处用 ShareGPT-multi-turn-* 前缀 → MULTI_TURN_TEXT（可调度）。

选片：turn_1 长度落在 [400, 900]，压窄 F 的可预测性；只保留 turn_1。
"""
import json
from pathlib import Path

SRC = Path("TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.jsonl")
OUT = Path("TaskData/stirrer_text.jsonl")
N = 60
LO, HI = 400, 900

out = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        sid, s = next(iter(json.loads(line).items()))
        L = len(s["turn_1"]["input_ids"])
        if LO <= L <= HI:
            out.append((sid, {"id": sid, "num_turns": 1, "turn_1": s["turn_1"]}))
            if len(out) >= N:
                break

with open(OUT, "w", encoding="utf-8") as f:
    for sid, sess in out:
        f.write(json.dumps({sid: sess}, ensure_ascii=False) + "\n")

L = [len(s["turn_1"]["input_ids"]) for _, s in out]
print(f"{OUT}  {len(out)} 条  turn1 均值={sum(L)/len(L):.0f}  范围={min(L)}..{max(L)}")
