#!/usr/bin/env python3
"""汇总 burst 实验：每个 arm 的 burst 窗口（idle 组回归那一轮）指标。

burst 窗口 = agentic_idle 组 turn == pause_after_turn+1（本实验固定 11）的请求。
主终点：burst 总重算、max 重算、TTFT p50/p99、max queue（队头阻塞）。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd


def one(path: str) -> dict | None:
    d = pd.read_csv(path)
    if "task" not in d or "turn" not in d:
        return None
    b = d[(d.task == "agentic_idle") & (d.turn == 11)]
    if not len(b):
        return None
    a = d[d.task == "agentic_active"]
    return dict(
        tag=os.path.basename(path).replace(".csv", ""),
        hit=1 - d.recompute_tokens.sum() / d.prompt_tokens.sum(),
        b_hit=1 - b.recompute_tokens.sum() / b.prompt_tokens.sum(),
        b_tot=int(b.recompute_tokens.sum()), b_max=int(b.recompute_tokens.max()),
        ttft50=float(b.ttft_ms.quantile(.5)), ttft99=float(b.ttft_ms.quantile(.99)),
        qmax=float(b.queue_time_ms.max()),
        a_hit=(1 - a.recompute_tokens.sum() / a.prompt_tokens.sum()) if len(a) else float("nan"),
        spread=float(b.ts.max() - b.ts.min()),
    )


def main() -> None:
    rows = [r for f in sorted(glob.glob("result/burst/*.csv")) if (r := one(f))]
    df = pd.DataFrame(rows)
    if not len(df):
        print("无数据")
        return
    print(f"{'arm':34s}{'全量命中':>9s}{'burst命中':>10s}{'burst总重算':>12s}{'max重算':>9s}"
          f"{'TTFT p50':>10s}{'TTFT p99':>10s}{'max queue':>11s}{'active命中':>11s}{'跨度s':>7s}")
    for _, r in df.iterrows():
        print(f"{r.tag:34s}{r.hit:>9.3f}{r.b_hit:>10.3f}{r.b_tot:>12d}{r.b_max:>9d}"
              f"{r.ttft50:>10.0f}{r.ttft99:>10.0f}{r.qmax:>11.0f}{r.a_hit:>11.3f}{r.spread:>7.2f}")


if __name__ == "__main__":
    main()
