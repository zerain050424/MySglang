#!/usr/bin/env python3
"""从 agent 会话 trace 中抽取「轮间间隔」与「工具调用」特征，产出统计表与图。

数据源
  tracelab  : 665k 轮 / 8,058 会话（claude + codex），每行一个 round，带全量
              timing_events 时间戳、token 结构与 first_input_event_type
  cachewise : 94 会话，llm_call / tool_call 事件交替

产出
  exp/data/rounds.parquet  每轮一行的中间表（供复算/复用）
  exp/agent_gap_stats.json 统计表
  exp/figs/fig*.png        图

用法: /root/sl_old/bin/python exp/trace_features.py [--reparse]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TRACELAB = Path("/root/agent-sessions/01-with-timestamps/tracelab/syfi_coding_trace.jsonl.gz")
CACHEWISE = Path("/root/sg-unicache/cachewise-coding-traces/parsed_traces")
OUT_DATA = Path("exp/data/rounds.parquet")
OUT_FIG = Path("exp/figs")
OUT_STATS = Path("exp/agent_gap_stats.json")
MAX_GAP = 86400.0     # 超过 1 天视为跨天离开，不进间隔统计
MAX_TOOL = 3600.0


def _ts(s: str) -> float:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def parse_tracelab(path: Path) -> pd.DataFrame:
    """逐行解析 tracelab：一行 = 一个 round。

    claude 的 tool_call 与 tool_result 分属相邻两轮（call 在轮尾、result 在下一轮开头），
    所以工具时长必须做**会话级**配对，并把时长记在"发起该调用的那一轮"名下，
    否则 claude 的 tool_sum 会被系统性算成 0。
    """
    rows: list[dict] = []
    idx_by_key: dict[tuple[str, int], int] = {}
    pending: dict[tuple[str, str], tuple[float, str, int]] = {}

    def bump(sid: str, ridx: int, d: float) -> None:
        i = idx_by_key.get((sid, ridx))
        if i is not None:
            rows[i]["tool_sum"] += d

    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            ev = r.get("timing_events") or []
            ts = [(_ts(e["timestamp"]), e) for e in ev if e.get("timestamp")]
            if not ts:
                continue
            ts.sort(key=lambda x: x[0])
            sid = r["session_id"]
            ridx = r.get("round_index", 0)
            names: list[str] = []
            for t, e in ts:
                et = e.get("event_type")
                if et == "tool_call":
                    cid = e.get("tool_call_id")
                    pending[(sid, cid)] = (t, e.get("tool_name") or "?", ridx)
                    names.append(e.get("tool_name") or "?")
                elif et == "tool_result":
                    hit = pending.pop((sid, e.get("tool_call_id")), None)
                    if hit:
                        d = t - hit[0]
                        if 0 <= d <= MAX_TOOL:
                            bump(sid, hit[2], d)
            rows.append(
                dict(
                    source=r.get("provider") or "?",
                    session_id=sid,
                    round_index=ridx,
                    t_start=ts[0][0],
                    t_end=ts[-1][0],
                    trigger=r.get("first_input_event_type") or "?",
                    n_tools=len(names),
                    tool_names="|".join(names),
                    tool_sum=0.0,
                    in_tok=r.get("input_tokens_total") or 0,
                    prefix_tok=r.get("prefix_tokens") or 0,
                    new_tok=r.get("newly_append_tokens") or 0,
                    out_tok=r.get("output_tokens") or 0,
                )
            )
            idx_by_key[(sid, ridx)] = len(rows) - 1
    return pd.DataFrame(rows)


def parse_cachewise(root: Path) -> pd.DataFrame:
    """cachewise：一轮 = 一次 llm_call，其后的 tool_call 触发下一轮。"""
    rows = []
    for fp in glob.glob(str(root / "*" / "*" / "events.json")):
        sid = Path(fp).parts[-2]
        ev = sorted(json.load(open(fp)), key=lambda e: _ts(e["timestamp"]))
        calls = []            # [[ts, rid, [工具耗时], [工具名]]]
        cur = None
        for e in ev:
            et = e.get("event_type")
            if et == "llm_call":
                rid = e.get("request_id")
                if cur and cur[1] == rid:
                    continue
                cur = [_ts(e["timestamp"]), rid, [], []]
                calls.append(cur)
            elif et == "tool_call" and cur is not None:
                d = e.get("execution_time_ms")
                cur[3].append(e.get("tool_name") or "?")
                if d is not None:
                    cur[2].append(float(d) / 1000.0)
        for i, c in enumerate(calls):
            rows.append(
                dict(
                    source="cachewise",
                    session_id=sid,
                    round_index=i,
                    t_start=c[0],
                    t_end=c[0] + (sum(c[2]) if c[2] else 0.0),
                    trigger="tool_result" if c[3] else "user_message",
                    n_tools=len(c[3]),
                    tool_names="|".join(c[3]),
                    tool_sum=float(min(sum(c[2]), MAX_TOOL)),
                    in_tok=np.nan,
                    prefix_tok=np.nan,
                    new_tok=np.nan,
                    out_tok=np.nan,
                )
            )
    return pd.DataFrame(rows)


def add_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """同一会话内按 round_index 排序，gap = 本轮开始 − 上一轮结束。"""
    df = df.sort_values(["session_id", "round_index"]).copy()
    prev_end = df.groupby("session_id")["t_end"].shift(1)
    df["gap"] = df["t_start"] - prev_end
    df.loc[(df["gap"] <= 0) | (df["gap"] > MAX_GAP), "gap"] = np.nan
    # 工具耗时算在上一轮的"离场时间"里（claude 的 result 落在下一轮开头），
    # 所以残余 eps = gap − 上一轮 tool_sum 才是「模型答完后到下次请求」的空档。
    df["prev_tool_sum"] = df.groupby("session_id")["tool_sum"].shift(1)
    df["eps"] = (df["gap"] - df["prev_tool_sum"].fillna(0)).clip(lower=0)
    df["n_rounds"] = df.groupby("session_id")["round_index"].transform("max") + 1
    df["pos"] = df["round_index"] / df["n_rounds"].clip(lower=1)
    return df


def q(v, p: float) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, p)) if len(v) else float("nan")


def describe(v) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {}
    return {
        "n": int(len(v)),
        "p10": q(v, .10), "p50": q(v, .50), "p90": q(v, .90),
        "p99": q(v, .99), "p999": q(v, .999),
        "mean": float(v.mean()),
        "gt60": float((v > 60).mean()), "gt300": float((v > 300).mean()),
    }


def acf(x: np.ndarray, lags: int = 12) -> list[float]:
    """会话内 log 间隔的自相关：先按会话去均值，再合并计算。"""
    out = []
    for k in range(1, lags + 1):
        num = den = 0.0
        for _, g in x:
            if len(g) <= k:
                continue
            a = g[:-k] - g.mean()
            b = g[k:] - g.mean()
            num += float((a * b).sum())
            den += float((a * a).sum())
        out.append(num / den if den else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reparse", action="store_true")
    args = ap.parse_args()

    if args.reparse or not OUT_DATA.exists():
        print("解析 tracelab ...", flush=True)
        tl = parse_tracelab(TRACELAB)
        print(f"  {len(tl)} 轮 / {tl.session_id.nunique()} 会话", flush=True)
        print("解析 cachewise ...", flush=True)
        cw = parse_cachewise(CACHEWISE)
        print(f"  {len(cw)} 轮 / {cw.session_id.nunique()} 会话", flush=True)
        df = add_gaps(pd.concat([tl, cw], ignore_index=True))
        OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_DATA, index=False)
        print(f"→ {OUT_DATA}  ({OUT_DATA.stat().st_size/1e6:.0f} MB)", flush=True)
    else:
        df = pd.read_parquet(OUT_DATA)

    with_gap = df[df.gap.notna()]
    print(f"总轮数 {len(df)}，有间隔 {len(with_gap)}，会话 {df.session_id.nunique()}")
    print("触发类型:", Counter(df.trigger).most_common())

    stats: dict = {"meta": {
        "rounds": int(len(df)),
        "sessions": int(df.session_id.nunique()),
        "rounds_with_gap": int(len(with_gap)),
        "sources": {k: int(v) for k, v in df.source.value_counts().items()},
    }}
    open(OUT_STATS, "w").write(json.dumps(stats, indent=1))
    print("统计表 →", OUT_STATS)


if __name__ == "__main__":
    main()
