#!/usr/bin/env python3
"""从 session_data.jsonl 校准 T-LRU 的 (threshold, next_prompt_estimate)。

T-LRU 的两个参数都是 token 数（论文里是 block，需乘 page_size）：

    budget   = max(L + Q_hat - threshold, 0)
    tel_safe = (本节点起点偏移 >= budget)

- Q_hat  = 下一轮 prompt 相对本轮的增量中位数（"下一轮会新增多少 token"）
- threshold = 允许放弃的尾部预算：越大越激进（保留 L 个以外的尾巴越少）

约束（CLI 启动校验见 kv_cache_hook）：threshold 必须 > Q_hat，否则没有任何节点
会 tel-safe，TLRU 退化成纯 LRU。

用法：
    python calibrate_tlru.py \
        --data-path ../TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl \
        [--out configs/arm_a3_native_tlru.json] [--fracs 0.10,0.25,0.50]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, List


def percentiles(xs: List[float]) -> dict:
    if not xs:
        return {}
    xs = sorted(xs)

    def pct(p: float) -> float:
        k = (len(xs) - 1) * p / 100.0
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {
        "n": len(xs),
        "min": round(xs[0], 1),
        "p10": round(pct(10), 1),
        "p50": round(pct(50), 1),
        "p90": round(pct(90), 1),
        "max": round(xs[-1], 1),
        "mean": round(statistics.fmean(xs), 1),
    }


def iter_sessions(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield list(json.loads(line).values())[0]


def prompt_len(turn: dict, tokenizer=None) -> int:
    """优先用已 tokenize 的 input_ids；退化时按字符数估。"""
    ids = turn.get("input_ids")
    if ids is not None:
        return len(ids)
    text = turn.get("input_text") or ""
    return len(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--max-sessions", type=int, default=-1)
    ap.add_argument("--fracs", default="0.10,0.25,0.50",
                    help="threshold = Q_hat + frac * median(L) 的候选比例")
    ap.add_argument("--out", default=None,
                    help="把默认（中间档）配置写成 eviction_policy_config JSON")
    args = ap.parse_args()

    path = Path(args.data_path)
    if not path.exists():
        print(f"[calibrate] data not found: {path}")
        return 2

    ls: List[float] = []          # 每轮 prompt 长度
    deltas: List[float] = []      # 轮间增量 = 下一轮 prompt - 本轮 prompt
    last_ls: List[float] = []     # 每个 session 的最后一轮长度
    n_sessions = n_turns = 0

    for sess in iter_sessions(path):
        num_turns = int(sess.get("num_turns", 0))
        if num_turns <= 0:
            continue
        prev = None
        for t in range(1, num_turns + 1):
            turn = sess.get(f"turn_{t}")
            if not turn:
                continue
            cur = prompt_len(turn)
            ls.append(cur)
            if prev is not None and cur > prev:
                deltas.append(cur - prev)
            prev = cur
            n_turns += 1
        if prev is not None:
            last_ls.append(prev)
        n_sessions += 1
        if args.max_sessions > 0 and n_sessions >= args.max_sessions:
            break

    if not deltas:
        print("[calibrate] no multi-turn deltas found; is this a multi-turn dataset?")
        return 1

    q_hat = int(statistics.median(deltas))
    l_p50 = int(statistics.median(last_ls)) if last_ls else int(statistics.median(ls))
    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]

    candidates = []
    for frac in fracs:
        threshold = q_hat + int(round(frac * l_p50))
        candidates.append({
            "frac_of_L": frac,
            "threshold": threshold,
            "next_prompt_estimate": q_hat,
            "config": {
                "threshold": threshold,
                "next_prompt_estimate": q_hat,
            },
        })

    report = {
        "data_path": str(path),
        "sessions": n_sessions,
        "turns": n_turns,
        "prompt_len": percentiles(ls),
        "session_final_prompt_len": percentiles(last_ls),
        "next_prompt_delta": percentiles(deltas),
        "q_hat": q_hat,
        "suggested_L_reference": l_p50,
        "candidates": candidates,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out:
        mid = candidates[len(candidates) // 2]["config"]
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(mid, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"\n[calibrate] 中间档配置已写入 {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
