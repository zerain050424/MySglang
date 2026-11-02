#!/usr/bin/env python3
"""台阶式空闲负载：8 条会话（4 对）**同时起跑**，全部在第 10 轮进入 pause，
但 pause 长度不同（15 / 30 / 45 / 60 s）→ 形成"由浅到深"的冷度台阶。

时间轴（gap = 1.5 s，10 轮 ≈ 15 s）：
  t=0        8 条同时起跑，turn1..10（每轮 1.5 s）
  t≈15 s     全部进入 pause
  t≈30 s     pair1（pause 15s）回归 → 之后它单独写 15 s
  t≈45 s     pair2（pause 30s）回归
  t≈60 s     pair3（pause 45s）回归
  t≈75 s     pair4（pause 60s）回归  ← 缺席最久，被削最狠

用途：检验策略是否"越冷越保护"，以及台阶末端的整链清零代价。
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

APIGEN = Path("/root/data_processed/apigen-mt/qwen2.5-7b/APIGenMT_session_data.jsonl.gz")
OUT = Path("TaskData/burst")
PAUSE_TURN, TAIL = 10, 2
GAP_S = 1.5
PAUSES = [15.0, 30.0, 45.0, 60.0]     # pair i 的 pause_s


def main() -> None:
    cand = []
    with gzip.open(APIGEN, "rt") as f:
        for line in f:
            sid, s = next(iter(json.loads(line).items()))
            n = s["num_turns"]
            if n < PAUSE_TURN + TAIL:
                continue
            L = len(s["turn_10"]["input_ids"]) + len(s["turn_10"]["response_ids"])
            if 5500 <= L <= 9000:
                cand.append((sid, L))
    print(f"候选 {len(cand)} 条（L@10 ∈ [5.5k, 9k]）")
    rng = random.Random(20251102)
    rng.shuffle(cand)
    picks = cand[: len(PAUSES) * 2]
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
    lines = ["# 台阶式空闲：8 条同时起跑，第 10 轮全部 pause，但时长 15/30/45/60 s",
             "# gap=1.5s；pair i 的 pause_s = 15+15i", "tasks:"]
    tot = 0
    for i, ps in enumerate(PAUSES):
        two = picks[2 * i: 2 * i + 2]
        p = OUT / f"stair_pair{i}.jsonl"
        p.write_text("\n".join(rows[sid] for sid, _ in two) + "\n", encoding="utf-8")
        tot += sum(L for _, L in two)
        print(f"  pair{i}: L@10 = {[L for _, L in two]}  pause_s = {ps}")
        lines.append(f"""- name: pair{i}
  task_id: {20 + i}
  data_path: {p.resolve()}
  lambda_rate: 0
  seed: 42
  max_record: -1
  shuffle: false
  arrival: {{mode: fixed, offset_s: 0.0, step_s: 0.05}}
  gap_mode: fixed
  gap_s: {GAP_S}
  pause_after_turn: {PAUSE_TURN}
  pause_s: {ps}""")
    yml = Path("tailshot/large/burst_stair.yaml")
    yml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"→ {yml}\n  ΣL@10 = {tot}（pool 35,975 → {tot/35975:.2f}×）")


if __name__ == "__main__":
    main()
