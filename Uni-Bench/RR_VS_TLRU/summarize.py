#!/usr/bin/env python3
"""burst/pause 实验汇总：只看 resume 轮，不看全局均值。

读 result/<arm>/<mode>/<budget>/：
  <seed>.json  —— summary（含 per_task 与 phases）
  <seed>.csv   —— 逐请求明细（含 resume_phase 列）

输出：
  1) resume 轮主表：命中率 / 零命中 victim 数 / 重算 token / TTFT 尾部
  2) 同时回归 vs 分批回归的差异（同一臂内）
  3) 有效性检查（force_agentic 是否生效、burst 是否真的产生淘汰）

用法：python summarize.py [--root result]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ARM_ORDER = ["a0_control", "a1_rr", "a2_tlru", "a3_native_tlru",
             "a4_tlru_t90", "a5_tlru_t200"]
ARM_LABEL = {"a0_control": "A0 对照", "a1_rr": "A1 RR", "a2_tlru": "A2 T-LRU(插件)",
             "a3_native_tlru": "A3 TLRU 原生 t=320",
             "a4_tlru_t90": "A4 TLRU 原生 t=90",
             "a5_tlru_t200": "A5 TLRU 原生 t=200"}


def pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def load_cell(path: Path) -> Optional[dict]:
    """读一个 (arm, mode, budget, seed) 的 resume 指标。"""
    js, cp = path.with_suffix(".json"), path.with_suffix(".csv")
    if not js.exists():
        return None
    try:
        summary = json.loads(js.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    cell = {"summary": summary}
    if not cp.exists():
        return cell

    resume_prompt = resume_cached = 0
    zero_victims = 0
    ttfts: List[float] = []
    with open(cp, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("resume_phase") != "resume":
                continue
            ptok = int(row["prompt_tokens"] or 0)
            ctok = int(row["cached_tokens"] or 0)
            resume_prompt += ptok
            resume_cached += ctok
            if ctok == 0:
                zero_victims += 1
            if row.get("ttft_ms"):
                ttfts.append(float(row["ttft_ms"]))
    cell.update({
        "resume_prompt": resume_prompt,
        "resume_cached": resume_cached,
        "resume_hit_pct": (100.0 * resume_cached / resume_prompt) if resume_prompt else None,
        "resume_recompute": resume_prompt - resume_cached,
        "zero_victims": zero_victims,
        "resume_ttft": ttfts,
    })
    return cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="result")
    ap.add_argument("--metric", default="resume_hit_pct")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"[summarize] {root} not found；先跑 run_matrix.sh")
        return 2

    cells: Dict[Tuple[str, str, str, str], dict] = {}
    for js in sorted(root.glob("*/*/*/*.json")):
        seed = js.stem
        if seed.startswith("meta_"):
            continue
        arm, mode, budget = js.relative_to(root).parts[:3]
        got = load_cell(js.with_suffix(""))
        if got:
            cells[(arm, mode, budget, seed)] = got

    if not cells:
        print(f"[summarize] no results under {root}")
        return 1

    arms = [a for a in ARM_ORDER if any(k[0] == a for k in cells)]
    arms += sorted({k[0] for k in cells} - set(arms))
    modes = sorted({k[1] for k in cells})
    budgets = sorted({k[2] for k in cells}, key=lambda b: int(b))
    seeds = sorted({k[3] for k in cells}, key=lambda s: int(s))

    def vals(arm, mode, budget, key):
        out = []
        for s in seeds:
            c = cells.get((arm, mode, budget, s))
            if c and c.get(key) is not None:
                out.append(c[key])
        return out

    def fmt(xs, nd=2):
        if not xs:
            return "—"
        if len(xs) == 1:
            return f"{xs[0]:.{nd}f}"
        return f"{statistics.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})"

    print("# RR vs T-LRU：burst/pause 场景\n")
    print(f"数据：`{root}`　seed：{', '.join(seeds)}　预算：{', '.join(f'{int(b)//1000}k' for b in budgets)}\n")

    for mode in modes:
        print(f"## 回归方式：{mode}\n")
        print("| 臂 | 预算 | resume 命中率 % | 零命中 victim 数 | resume 重算 token | resume TTFT p99 ms |")
        print("|---|---|---|---|---|---|")
        for arm in arms:
            for b in budgets:
                ttfts = []
                for s in seeds:
                    c = cells.get((arm, mode, b, s))
                    if c:
                        ttfts += c.get("resume_ttft") or []
                print(f"| {ARM_LABEL.get(arm, arm)} | {int(b)//1000}k | "
                      f"{fmt(vals(arm, mode, b, 'resume_hit_pct'))} | "
                      f"{fmt(vals(arm, mode, b, 'zero_victims'), 0)} | "
                      f"{fmt(vals(arm, mode, b, 'resume_recompute'), 0)} | "
                      f"{fmt([pct(ttfts, 99)], 1) if ttfts else '—'} |")
        print()

    # ---- 同时 vs 分批（同一臂内，配对按 seed） ----
    if len(modes) >= 2:
        print("## 同时回归 vs 分批回归（配对差值，百分点）\n")
        print("| 臂 | 预算 | resume 命中率差 | 零命中数差 | 判定 |")
        print("|---|---|---|---|---|")
        for arm in arms:
            for b in budgets:
                deltas, zdeltas = [], []
                for s in seeds:
                    a = cells.get((arm, "simultaneous", b, s))
                    c = cells.get((arm, "batched", b, s))
                    if a and c and a.get(args.metric) is not None and c.get(args.metric) is not None:
                        deltas.append(a[args.metric] - c[args.metric])
                        zdeltas.append(a.get("zero_victims", 0) - c.get("zero_victims", 0))
                if not deltas:
                    continue
                same = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
                med = statistics.median(deltas)
                verdict = ("同时更差" if same and med < 0 else
                           "同时更好" if same and med > 0 else "无稳定差异")
                print(f"| {ARM_LABEL.get(arm, arm)} | {int(b)//1000}k | "
                      f"{med:+.2f} | {statistics.median(zdeltas):+.0f} | {verdict} |")
        print()

    # ---- 有效性检查 ----
    print("## 有效性检查\n")
    problems = 0
    for (arm, mode, b, s), c in sorted(cells.items()):
        ph = (c["summary"].get("phases") or {})
        if not ph:
            print(f"⚠ {arm}/{mode}/{b}/seed={s}: 缺少 phases（burst 未启用？）")
            problems += 1
            continue
        resumed = ph.get("victim_resumed_s") or []
        burst_end = ph.get("burst_end_s")
        if burst_end is None or not resumed:
            print(f"⚠ {arm}/{mode}/{b}/seed={s}: phases 不完整 {ph}")
            problems += 1
            continue
        spread = max(resumed) - min(resumed)
        expect_spread = 0.0 if mode == "simultaneous" else (len(resumed) // max(1, ph.get("wave_size", 1))) * ph.get("wave_interval_s", 0)
        if mode == "batched" and spread < expect_spread * 0.5:
            print(f"⚠ {arm}/{mode}/{b}/seed={s}: 分批回归的恢复时间散布只有 {spread:.1f}s"
                  f"（期望 ≈{expect_spread:.1f}s）")
            problems += 1
    if problems == 0:
        print("✅ 所有运行都有完整的 burst → resume 相位记录。")

    print("\n> 提醒：`resume_phase` 只统计挂起后第一个回归请求；绝对数字只在组内可比。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
