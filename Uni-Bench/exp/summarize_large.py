#!/usr/bin/env python3
"""对 LargeScale 单格的逐请求 CSV 出核心指标（不依赖 burst 结构）。

指标
  hit_agentic      仅本负载的命中率
  max_recompute    单请求最大重算 token
  cold_hit         每会话最差一轮的 cached/prompt，取全会话最小值
  resume_p99       每会话第 2 轮起、gap>60s 之后的 TTFT（冷回归）
  deficit_max      账本缺口代理 = Σ(prompt − cached) 的会话级最大值
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, p)) if len(v) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--wait-scale", type=float, default=1.0,
                    help="CSV 里的 gap 是墙钟，乘 1/s 还原为逻辑间隔")
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    d["hit"] = 1 - d["recompute_tokens"] / d["prompt_tokens"].clip(lower=1)
    per = d.groupby("session_id").agg(
        n=("turn", "size"), hit=("hit", "mean"), worst_hit=("hit", "min"),
        recompute_sum=("recompute_tokens", "sum"), recompute_max=("recompute_tokens", "max"),
        ttft_p99=("ttft_ms", lambda x: q(x, .99))).reset_index()

    d["gap_logical"] = d["gap_since_prev_end_s"] / max(a.wait_scale, 1e-9)
    cold = d[(d.turn > 1) & (d["gap_logical"] > 60)]
    res = {
        "requests": int(len(d)),
        "sessions": int(d.session_id.nunique()),
        "hit_agentic": float(d.hit.mean()),
        "max_recompute": int(d.recompute_tokens.max()),
        "recompute_p99": q(d.recompute_tokens, .99),
        "cold_hit_min": float(per.worst_hit.min()),
        "cold_hit_p10": q(per.worst_hit, .10),
        "resume_n": int(len(cold)),
        "cold_requests": int((d.gap_logical > 60).sum()),
        "gap_max_logical": float(d.gap_logical.max()) if d.gap_logical.notna().any() else 0.0,
        "resume_p50_ms": q(cold.ttft_ms, .50),
        "resume_p99_ms": q(cold.ttft_ms, .99),
        "ttft_p99_all": q(d.ttft_ms, .99),
        "deficit_max": int(per.recompute_sum.max()),
        "deficit_gini": float(
            (2 * np.sum(np.arange(1, len(per) + 1) * np.sort(per.recompute_sum.values))
             / (len(per) * per.recompute_sum.sum()) - (len(per) + 1) / len(per))
            if per.recompute_sum.sum() > 0 else 0.0),
    }
    tag = a.csv.split("/")[-1].replace(".csv", "")
    print(f"===== {tag} =====")
    for k, v in res.items():
        print(f"  {k:18s}: {v:.4f}" if isinstance(v, float) else f"  {k:18s}: {v}")
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
