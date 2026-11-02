#!/usr/bin/env python3
"""构造「两两轮转空闲」负载：12 条会话 = 6 对，任何时刻恰好 2 条空闲、10 条活跃。

时间轴（gap = 1.5 s，pause = 15 s，每对偏移 15 s）：
  pair i  : 活跃 10 轮 (15 s) → 空闲 15 s → 恢复 2 轮
  pair i+1: 比 pair i 晚 15 s 起跑
  ⇒ 窗口 i = [15i+15, 15i+30]，期间 pair i 空闲，pair i+1..i+5 活跃（=10 条）
  ⇒ 窗口首尾相接，任何时刻恰好 1 对（2 条）空闲

压力来源：10 条活跃会话在 15 s 内各写 10 轮 × ~200 token ≈ 20k，
足以把空闲那对的 ~13k 前缀清掉（LRU 会清，RR 应保留 κ 比例）。
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

APIGEN = Path("/root/data_processed/apigen-mt/qwen2.5-7b/APIGenMT_session_data.jsonl.gz")
OUT = Path("TaskData/burst")
PAUSE_TURN, TAIL = 10, 2
GAP_S, PAUSE_S = 1.5, 15.0
PAIRS = 6


def main() -> None:
    cand = []
    with gzip.open(APIGEN, "rt") as f:
        for line in f:
            sid, s = next(iter(json.loads(line).items()))
            n = s["num_turns"]
            if n < PAUSE_TURN + TAIL:
                continue
            L = len(s["turn_10"]["input_ids"]) + len(s["turn_10"]["response_ids"])
            if 3200 <= L <= 3800:
                cand.append((sid, L))
    print(f"候选 {len(cand)} 条（L@10 ∈ [3.2k, 3.8k]）")
    rng = random.Random(20251101)
    rng.shuffle(cand)
    picks = cand[: PAIRS * 2]
    want = {sid for sid, _ in picks}

    rows = {}
    with gzip.open(APIGEN, "rt") as f:
        for line in f:
            sid, s = next(iter(json.loads(line).items()))
            if sid not in want:
                continue
            s["num_turns"] = PAUSE_TURN + TAIL
            rows[sid] = json.dumps({sid: s}, ensure_ascii=False)

    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i in range(PAIRS):
        two = picks[2 * i : 2 * i + 2]
        p = OUT / f"roll2_pair{i}.jsonl"
        p.write_text("\n".join(rows[sid] for sid, _ in two) + "\n", encoding="utf-8")
        tasks.append((i, [L for _, L in two], GAP_S * PAUSE_TURN * i))
        print(f"  pair{i}: L@10 = {[L for _, L in two]}")

    yml = Path("tailshot/large/burst_roll2.yaml")
    yml.parent.mkdir(parents=True, exist_ok=True)
    body = [f"""# 两两轮转空闲：{PAIRS} 对 × 2 条，任何时刻恰好 2 条空闲
# gap={GAP_S}s、pause={PAUSE_S}s、每对偏移 {GAP_S * PAUSE_TURN}s
tasks:"""]
    for i, Ls, off in tasks:
        body.append(f"""- name: pair{i}
  task_id: {10 + i}
  data_path: {OUT.resolve()}/roll2_pair{i}.jsonl
  lambda_rate: 0
  seed: 42
  max_record: -1
  shuffle: false
  arrival: {{mode: fixed, offset_s: {off}, step_s: 0.05}}
  gap_mode: fixed
  gap_s: {GAP_S}
  pause_after_turn: {PAUSE_TURN}
  pause_s: {PAUSE_S}""")
    yml.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"→ {yml}")
    print(f"  ΣL@10 = {sum(L for _, L in picks)}（pool 35,975 → 溢出 {sum(L for _, L in picks)/35975:.2f}×）")


if __name__ == "__main__":
    main()
