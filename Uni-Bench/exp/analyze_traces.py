#!/usr/bin/env python3
"""从公开生产 trace 拟合「轮间隔」分布，用于校准 P_i 的先验 (mu, sigma)。

数据源（下载到 /root/traces/）：
  qwen_traceA_blksz_16.jsonl   阿里云百炼 to-C 聊天，含 chat_id/parent_chat_id/turn/timestamp
                               → 可还原多轮关系，得到真实的轮间隔
  qwen_traceB_blksz_16.jsonl   百炼 to-B API，全部 turn=1，只有到达过程
  mooncake_trace.jsonl         Mooncake arxiv trace，无多轮关系，且按 20 条/批、批间隔 3053s

用法：
  python exp/analyze_traces.py                       # 只跑 traceA（唯一有多轮关系的）
  python exp/analyze_traces.py --csv gap_fit.csv     # 同时导出逐 type 的拟合结果

输出：逐 type 的 p50/p90/p99 与 lognormal(mu, sigma)，并与 UniCache TASK_CONFIG 对照。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

TRACE_DIR = Path("/root/traces")
# UniCache 内置常数（evict_policy.TASK_CONFIG），用于对照
UNICACHE = {
    "text": (4.77, 1.279),
    "image": (4.4449, 1.09066),
    "search": (4.961663, 1.13877),
    "file": (5.07658, 1.06383),
}


def fit(gaps: list[float]) -> dict:
    logs = [math.log(x) for x in gaps]
    ss = sorted(gaps)
    q = lambda p: ss[min(int(p * len(ss)), len(ss) - 1)]
    return {
        "n": len(gaps),
        "p50": st.median(gaps),
        "p90": q(0.90),
        "p99": q(0.99),
        "mu": st.fmean(logs),
        "sigma": st.pstdev(logs),
    }


def load_bailian(path: Path) -> tuple[dict, dict]:
    """返回 (逐 type 的轮间隔, 全局统计)。轮间隔 = 子请求 ts − 父请求 ts。"""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["chat_id"]] = (
                float(r["timestamp"]),
                r.get("parent_chat_id", -1),
                int(r.get("turn", 1)),
                r.get("type", "?"),
            )
    gaps: dict[str, list[float]] = defaultdict(list)
    n_turn1 = n_multi = 0
    for ts, pid, turn, typ in rows.values():
        if turn <= 1:
            n_turn1 += 1
            continue
        parent = rows.get(pid)
        if parent is None:
            continue
        n_multi += 1
        d = ts - parent[0]
        if d > 0:
            gaps[typ].append(d)
    return dict(gaps), {"requests": len(rows), "turn1": n_turn1, "turn_gt1": n_multi}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(TRACE_DIR / "qwen_traceA_blksz_16.jsonl"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    gaps, stat = load_bailian(Path(args.trace))
    print(f"{Path(args.trace).name}: 请求 {stat['requests']}，turn=1 有 {stat['turn1']}，"
          f"turn>1 有 {stat['turn_gt1']}")

    rows = []
    print(f"\n{'type':8s}{'n':>8s}{'p50(s)':>10s}{'p90(s)':>10s}{'p99(s)':>10s}"
          f"{'mu':>7s}{'sigma':>7s}   UniCache(mu/sigma)")
    for typ, g in sorted(gaps.items(), key=lambda x: -len(x[1])):
        f = fit(g)
        ref = UNICACHE.get(typ)
        ref_s = f"{ref[0]:.3f}/{ref[1]:.3f}" if ref else "—"
        print(f"{typ:8s}{f['n']:>8d}{f['p50']:>10.1f}{f['p90']:>10.1f}{f['p99']:>10.1f}"
              f"{f['mu']:>7.2f}{f['sigma']:>7.2f}   {ref_s}")
        rows.append({"type": typ, **f,
                     "unicache_mu": ref[0] if ref else None,
                     "unicache_sigma": ref[1] if ref else None})

    allg = [x for g in gaps.values() for x in g]
    f = fit(allg)
    print(f"{'ALL':8s}{f['n']:>8d}{f['p50']:>10.1f}{f['p90']:>10.1f}{f['p99']:>10.1f}"
          f"{f['mu']:>7.2f}{f['sigma']:>7.2f}")
    rows.append({"type": "ALL", **f})

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n已写入 {args.csv}")


if __name__ == "__main__":
    main()
