#!/usr/bin/env python3
"""构造「临界并发 + 稀疏活跃 + 回归浪潮」实验的两个会话池。

来源：APIGenMT（452 token/轮，48.8% 的轮有工具调用）
  idle 组：第 10 轮时链长 ∈ [5.5k, 9.5k]  ← 重前缀，冷回归代价 1.4 s
  active 组：第 10 轮时链长 ∈ [1.5k, 2.8k] ← 短链，持续写入但不吃满池子

两个池都截断到 12 轮（12 轮 burst + pause + 8 轮恢复）。
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

APIGEN = Path("/root/data_processed/apigen-mt/qwen2.5-7b/APIGenMT_session_data.jsonl.gz")
OUT = Path("TaskData/burst")
PAUSE_TURN = 10
TAIL = 2


def L_at(s: dict, t: int) -> int:
    return len(s[f"turn_{t}"]["input_ids"]) + len(s[f"turn_{t}"]["response_ids"])


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20251027)
    ap.add_argument("--n-active", type=int, default=8)
    ap.add_argument("--n-idle", type=int, default=8)
    ap.add_argument("--pause-s", type=int, default=60)
    ap.add_argument("--name", default="s")
    a = ap.parse_args()
    idle_cand, active_cand = [], []
    with gzip.open(APIGEN, "rt") as f:
        for line in f:
            sid, s = next(iter(json.loads(line).items()))
            n = s["num_turns"]
            if n < PAUSE_TURN + TAIL:
                continue
            L = L_at(s, PAUSE_TURN)
            if 5500 <= L <= 9500:
                idle_cand.append((sid, L))
            elif 1500 <= L <= 2800:
                active_cand.append((sid, L))

    print(f"候选：idle {len(idle_cand)} 条，active {len(active_cand)} 条")
    rng = random.Random(a.seed)
    rng.shuffle(idle_cand)
    rng.shuffle(active_cand)

    OUT.mkdir(parents=True, exist_ok=True)
    picks = {"idle": idle_cand[:a.n_idle], "active": active_cand[:a.n_active]}
    want = {sid for _, group in picks.items() for sid, _ in group}

    # 第二遍：只写出被选中的会话，截断到 PAUSE_TURN + TAIL 轮
    rows = {"idle": [], "active": []}
    with gzip.open(APIGEN, "rt") as f:
        for line in f:
            sid, s = next(iter(json.loads(line).items()))
            if sid not in want:
                continue
            for k in ("idle", "active"):
                if any(sid == x for x, _ in picks[k]):
                    s["num_turns"] = PAUSE_TURN + TAIL
                    rows[k].append(json.dumps({sid: s}, ensure_ascii=False))
                    break

    for k, v in rows.items():
        p = OUT / f"{k}_{len(v)}_{a.name}.jsonl"
        p.write_text("\n".join(v) + "\n", encoding="utf-8")
        Ls = [L for _, L in picks[k]]
        print(f"  {p}  {len(v)} 条，L@turn{PAUSE_TURN} = "
              f"{min(Ls)}–{max(Ls)}（均值 {sum(Ls)/len(Ls):.0f}）")

    yml = Path(f"tailshot/large/burst_{a.name}.yaml")
    yml.parent.mkdir(parents=True, exist_ok=True)
    yml.write_text(
        f"""# 格 {a.name}：active {len(rows["active"])} + idle {len(rows["idle"])}，pause {a.pause_s}s
# active 持续写入；idle 第 {PAUSE_TURN} 轮后统一 pause 60s → 回归浪潮
tasks:
- name: agentic_active
  task_id: 3
  data_path: ../../TaskData/burst/active_{len(rows["active"])}_{a.name}.jsonl
  lambda_rate: 0
  seed: 42
  max_record: -1
  shuffle: false
  arrival: {{mode: fixed, offset_s: 0.0, step_s: 0.05}}
  gap_mode: fixed
  gap_s: 6.0
- name: agentic_idle
  task_id: 4
  data_path: ../../TaskData/burst/idle_{len(rows["idle"])}_{a.name}.jsonl
  lambda_rate: 0
  seed: 42
  max_record: -1
  shuffle: false
  arrival: {{mode: fixed, offset_s: 0.0, step_s: 0.05}}
  gap_mode: fixed
  gap_s: 6.0
  pause_after_turn: {PAUSE_TURN}
  pause_s: {a.pause_s}
""", encoding="utf-8")
    print(f"  → {yml}")


if __name__ == "__main__":
    main()
