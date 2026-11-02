#!/usr/bin/env python3
"""合并两个真实轨迹来源，导出逐工具执行时长的经验分布。

来源：
  tracelab   /root/agent-sessions/01-with-timestamps/tracelab/syfi_coding_trace.jsonl.gz
             665,453 轮 / 8,058 会话；tool_call 与 tool_result 按 tool_call_id 配对
  cachewise  /root/sg-unicache/cachewise-coding-traces/parsed_traces/*/*/events.json
             94 会话；tool_call 自带 execution_time_ms

输出 exp/tool_durations.json：
  tools[name] = {n, sources, min, p10/p50/p90/p99, max, q: [101 个分位点]}
  q[i] = 第 i 百分位的秒数，供下游按逆 CDF 采样（不落原始样本，文件小且不失真）。

用法: python exp/build_tool_durations.py [--out /tmp/x.json]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

TRACELAB = Path("/root/agent-sessions/01-with-timestamps/tracelab/syfi_coding_trace.jsonl.gz")
CACHEWISE = Path("/root/sg-unicache/cachewise-coding-traces/parsed_traces")
OUT = Path("exp/tool_durations.json")
MAX_S = 3600.0          # 超过视为挂起/失控，不进统计

# 语义分类：有些「工具」其实是人工等待，不能当机器耗时用
CLASS_OVERRIDE = {
    "AskUserQuestion": "human",     # 等用户回答
    "ExitPlanMode": "human",        # 等用户批准计划
    "SendUserFile": "human",
    "write_stdin": "interactive",   # 交互式终端等待
    "Monitor": "interactive",
}


def classify(tool: str, p50: float) -> str:
    if tool in CLASS_OVERRIDE:
        return CLASS_OVERRIDE[tool]
    if p50 < 0.2:
        return "fast"
    if p50 < 10:
        return "medium"
    return "slow"


def _ts(s: str) -> float:
    # fromisoformat 在 3.10 及以下不认尾部 "Z"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def load_tracelab(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    pending: dict[tuple[str, str], tuple[float, str]] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            sid = r["session_id"]
            for e in r.get("timing_events") or []:
                et, t = e.get("event_type"), e.get("timestamp")
                if not t:
                    continue
                if et == "tool_call":
                    pending[(sid, e.get("tool_call_id"))] = (_ts(t), e.get("tool_name") or "?")
                elif et == "tool_result":
                    hit = pending.pop((sid, e.get("tool_call_id")), None)
                    if hit is None:
                        continue
                    d = _ts(t) - hit[0]
                    if 0 <= d <= MAX_S:
                        out[hit[1]].append(d)
    return out


def load_cachewise(root: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for f in glob.glob(str(root / "*" / "*" / "events.json")):
        for e in json.load(open(f)):
            if e.get("event_type") != "tool_call":
                continue
            d = e.get("execution_time_ms")
            if d is None:
                continue
            s = float(d) / 1000.0
            if 0 <= s <= MAX_S:
                out[e.get("tool_name") or "?"].append(s)
    return out


def load_cachewise_buckets(root: Path) -> tuple[list[float], list[float]]:
    """cachewise 语义与我们的 driver 一致（gap = 工具耗时 + 残余）。

    返回 (nothink, eps)：
      nothink  上一轮没有工具调用时的轮间隔（人工等待）
      eps      上一轮有工具时，轮间隔扣掉工具耗时后的残余
    """
    nothink: list[float] = []
    eps: list[float] = []
    for f in glob.glob(str(root / "*" / "*" / "events.json")):
        ev = sorted(json.load(open(f)), key=lambda e: _ts(e["timestamp"]))
        seq: list[list] = []          # [llm 时刻, rid, [工具耗时]]
        cur = None
        for e in ev:
            et = e.get("event_type")
            if et == "llm_call":
                rid = e.get("request_id")
                if cur and cur[1] == rid:
                    continue
                cur = [_ts(e["timestamp"]), rid, []]
                seq.append(cur)
            elif et == "tool_call" and cur is not None:
                d = e.get("execution_time_ms")
                if d is not None:
                    cur[2].append(float(d) / 1000.0)
        for a, b in zip(seq, seq[1:]):
            gap = b[0] - a[0]
            if gap <= 0:
                continue
            if a[2]:
                if gap <= MAX_S:
                    eps.append(max(gap - sum(a[2]), 0.0))
            elif gap <= MAX_S:
                nothink.append(gap)
    return nothink, eps


# 分位网格：主体 0..99 每 1%，尾部 99.1..99.9 每 0.1%，再补 99.95 / 99.99 / 100。
# 尾部必须加密：等距 101 点时 [q99, q100] 跨 3 个数量级，线性插值会把 p99.9
# 放大 4–19 倍（实测 Bash 1304→3227、exec_command 156→3020），而本实验的
# 级联触发条件恰恰是尾部（最冷会话）。
LEVELS = (
    [float(i) for i in range(100)]                      # 0..99，步长 1%
    + [round(99.0 + 0.1 * i, 2) for i in range(1, 10)]  # 99.1..99.9，步长 0.1%
    + [round(99.9 + 0.01 * i, 2) for i in range(1, 10)] # 99.91..99.99，步长 0.01%
    + [99.995, 100.0]
)


def quantiles(v: list[float]) -> dict:
    """把样本压成 (p, q) 两张表；q 为分位点上的线性插值。

    v: 原始样本（秒）。返回 {n, min, p10/p50/p90/p99, max, p, q}，
       p/q 长度相同、p 严格递增，供下游 bisect 定位后做逆 CDF 采样。
    """
    if not v:
        return {"n": 0}
    v = sorted(v)
    n = len(v)

    def at(p: float) -> float:
        x = (p / 100.0) * (n - 1)
        i = int(x)
        return v[i] if i >= n - 1 else v[i] + (v[i + 1] - v[i]) * (x - i)

    q = [round(at(p), 6) for p in LEVELS]
    d = {"n": n, "min": round(v[0], 6), "max": round(v[-1], 6)}
    for p in (10, 50, 90, 99):
        d[f"p{p}"] = round(at(p), 6)
    d["p"] = LEVELS
    d["q"] = q
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    src = {"tracelab": load_tracelab(TRACELAB), "cachewise": load_cachewise(CACHEWISE)}
    merged: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, dict[str, int]] = defaultdict(dict)
    for name, per_tool in src.items():
        for tool, v in per_tool.items():
            merged[tool].extend(v)
            sources[tool][name] = len(v)

    tools = {}
    for tool, v in merged.items():
        d = quantiles(v)
        d["sources"] = dict(sorted(sources[tool].items()))
        d["class"] = classify(tool, d["p50"])
        tools[tool] = d
    tools = dict(sorted(tools.items(), key=lambda x: -x[1]["n"]))

    nothink, eps = load_cachewise_buckets(CACHEWISE)

    doc = {
        "meta": {
            "sources": {"tracelab": str(TRACELAB), "cachewise": str(CACHEWISE)},
            "n_tool_calls": sum(v["n"] for v in tools.values()),
            "n_tools": len(tools),
            "max_duration_s": MAX_S,
            "sampling": "按 (p, q) 逆 CDF 线性插值采样（p 尾部加密到 99.99）",
        },
        "tools": tools,
        # 两个轮级桶（cachewise 语义：gap = 工具耗时 + 残余）。
        # 工具表可以合并两个来源，但这两个桶只取 cachewise —— tracelab 的
        # round 间隔里工具是「轮内」的，语义不同，混进来会把 eps 拉成 0。
        "buckets": {
            "nothink": quantiles(nothink),
            "eps": quantiles(eps),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)

    print(f"{'tool':18s}{'class':>11s}{'n':>9s}{'p50':>9s}{'p90':>9s}{'p99':>10s}   来源")
    for tool, d in tools.items():
        print(f"{tool:18s}{d['class']:>11s}{d['n']:>9d}{d['p50']:>9.2f}"
              f"{d['p90']:>9.2f}{d['p99']:>10.2f}   {d['sources']}")
    print(f"\n合计 {doc['meta']['n_tool_calls']} 次工具调用 / {len(tools)} 个工具 → {args.out}")
    for k in ("nothink", "eps"):
        b = doc["buckets"][k]
        print(f"  桶 {k:8s} n={b['n']:>6} p50={b['p50']:>7.2f}s p90={b['p90']:>8.2f}s")


if __name__ == "__main__":
    main()
