#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Heter-Bench 的 session_data.jsonl 转成 SGLang Simulator 的 Autobench trace。

【为什么需要这一层】
  `session_data.jsonl`（`tools/prepare_session_data.py` 的产物）是「一个会话一行、
  turn 内嵌」的结构，给 Heter-Bench 自己的 `scripts/bench_heap.py` 用。
  `tools/sglang-simulator` 吃的却是 Autobench 的**扁平 JSONL**：一行一个请求，
  带 `timestamp`（到达时刻，毫秒）。两者只差这一层摊平，token 不用重新算。

【session_id 怎么传进去】
  这是本脚本存在的核心理由。HeterCache 在 `radix_cache.py` 里用
  `getattr(req, "session_id", None)` 认会话；拿不到就只有一套空账本，队列分类、
  one-shot 判定、跨会话共享全部失效，而且**不会报错**。

  链路（已核对，无需改任何代码）：

    trace 行里的 "session_id"
      -> autobench.py `_RESERVED_FIELDS` 不含它，于是落进 DatasetRow.extra_request_body
      -> benchmark/serving.py 合并进请求体 `payload.update(extra_request_body)`
      -> `GenerateReqInput.session_id`（io_struct.py）
      -> `TokenizedGenerateReqInput.session_id`
      -> `req.session_id` -> SessionRegistry

  所以 `session_id` 必须是 trace 行的**顶层字段**，不要塞进 `extra_request_body`
  （那样也能通，但多一层，且 autobench 的 param_send 会覆盖同名字段）。

【时间轴】
  同一会话的多轮必须串行落树，否则后一轮在写树前就发出去了，前缀必然 miss。
  这里按「会话内每轮 +turn_gap_ms、会话之间 +session_stagger_ms」排时间段。
  配合 `bench_serving.py --max-concurrency`，把并发压到可控范围。

【用法】
  python Heter-Bench/tools/make_simulator_trace.py \\
      --session-data /tmp/simgpt/session_data.jsonl \\
      --out /tmp/simgpt/trace.jsonl \\
      --turn-gap-ms 1000 --session-stagger-ms 200 --output-cap 64
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterator, Tuple


def iter_sessions(path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """逐行读 session_data.jsonl，产出 (session_id, session_dict)。

    行格式（prepare_session_data.py 的输出）::

        {"<sid>": {"id": "<sid>", "num_turns": N,
                   "turn_1": {"input_ids": [...], "response_ids": [...], ...}, ...}}
    """
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from error
            if len(record) != 1:
                raise ValueError(
                    f"{path}:{line_number} 期望一行一个会话（单键），"
                    f"实际有 {len(record)} 个键"
                )
            session_id, session = next(iter(record.items()))
            yield session_id, session


def build_rows(
    session_data: str,
    *,
    turn_gap_ms: float,
    session_stagger_ms: float,
    output_cap: int,
) -> list[dict]:
    """摊平成 Autobench 行，并排好到达时刻。"""
    rows: list[dict] = []
    for session_index, (session_id, session) in enumerate(iter_sessions(session_data)):
        num_turns = int(session["num_turns"])
        base = session_index * session_stagger_ms
        for turn in range(1, num_turns + 1):
            payload = session[f"turn_{turn}"]
            prompt = payload["input_ids"]
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"{session_id} turn_{turn} 的 input_ids 为空")
            responses = payload.get("response_ids") or []
            output_len = min(len(responses), output_cap) if responses else output_cap
            rows.append(
                {
                    "prompt": prompt,
                    # 顶层字段：autobench 会把它当 extra_request_body 透传成
                    # GenerateReqInput.session_id，HeterCache 靠它认会话。
                    "session_id": session_id,
                    "output_len": max(int(output_len), 1),
                    "timestamp": base + (turn - 1) * turn_gap_ms,
                }
            )
    rows.sort(key=lambda row: (row["timestamp"], row["session_id"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="session_data.jsonl -> SGLang Simulator Autobench trace"
    )
    parser.add_argument("--session-data", required=True, help="session_data.jsonl 路径")
    parser.add_argument("--out", required=True, help="输出 trace.jsonl 路径")
    parser.add_argument(
        "--turn-gap-ms",
        type=float,
        default=1000.0,
        help="同一会话相邻两轮的间隔（毫秒）。太小会让后一轮在写树前发出。",
    )
    parser.add_argument(
        "--session-stagger-ms",
        type=float,
        default=200.0,
        help="相邻会话的起始错开量（毫秒）。0 表示全部会话同时起跑。",
    )
    parser.add_argument(
        "--output-cap",
        type=int,
        default=64,
        help="单请求 output_len 上限。真实 response 长度通常远超仿真所需。",
    )
    args = parser.parse_args()

    rows = build_rows(
        args.session_data,
        turn_gap_ms=args.turn_gap_ms,
        session_stagger_ms=args.session_stagger_ms,
        output_cap=args.output_cap,
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sessions = len({row["session_id"] for row in rows})
    span_ms = rows[-1]["timestamp"] - rows[0]["timestamp"] if rows else 0.0
    print(
        f"[make_simulator_trace] {len(rows)} 个请求 / {sessions} 个会话 "
        f"/ 时间跨度 {span_ms / 1000:.1f}s -> {args.out}"
    )
    print(
        "[make_simulator_trace] 每 session 的请求数（前 5）：",
        sorted(
            ((sid, sum(1 for r in rows if r["session_id"] == sid)) for sid in {r["session_id"] for r in rows}),
            key=lambda pair: -pair[1],
        )[:5],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
