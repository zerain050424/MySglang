#!/usr/bin/env python3
"""汇总 TLRU vs RR 实验矩阵，输出 Markdown 表 + 配对差值。

读 result/<arm>/<budget>/<seed>.json（bench_unicache.py 的输出）。
每个单元格是 3 个 seed 的 中位数（min–max）。

判据（跑之前已写死）：
  1. 主结论看配对差 a1 − a2（同 seed、同 budget）；
  2. 3 个 seed 的差值同号才算结论；
  3. |Δ| < 1 个百分点视为持平。

用法：python summarize.py [--root result] [--metric cache_hit_rate_pct]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ARM_ORDER = ["a0_control", "a1_rr", "a2_tlru"]
ARM_LABEL = {
    "a0_control": "A0 对照（无插件）",
    "a1_rr": "A1 RR-EVICT",
    "a2_tlru": "A2 T-LRU",
}


def load(root: Path, metric: str) -> Dict[Tuple[str, str, str], dict]:
    out: Dict[Tuple[str, str, str], dict] = {}
    for path in sorted(root.glob("*/*/*.json")):
        parts = path.relative_to(root).parts
        if len(parts) != 3:
            continue
        arm, budget, seed = parts[0], parts[1], path.stem
        if not seed.isdigit():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[summarize] skip unreadable {path}")
            continue
        if metric not in data:
            continue
        out[(arm, budget, seed)] = data
    return out


def fmt_cell(vals: List[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.median(vals):.2f} ({min(vals):.2f}–{max(vals):.2f})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="result")
    ap.add_argument("--metric", default="cache_hit_rate_pct")
    ap.add_argument("--ttft", default="ttft_p99_ms")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[summarize] {root} not found; 先跑 run_matrix.sh")
        return 2

    data = load(root, args.metric)
    if not data:
        print(f"[summarize] no results under {root}")
        return 1

    arms = [a for a in ARM_ORDER if any(k[0] == a for k in data)]
    arms += sorted({k[0] for k in data} - set(arms))
    budgets = sorted({k[1] for k in data}, key=lambda b: int(b))
    seeds = sorted({k[2] for k in data}, key=lambda s: int(s))

    print(f"# TLRU vs RR-EVICT（Agentic）\n")
    print(f"数据：`{root}`　指标：`{args.metric}`　seed：{', '.join(seeds)}\n")

    # ---- 主表 ----
    header = "| 臂 | " + " | ".join(f"{int(b)//1000}k" for b in budgets) + " |"
    print(header)
    print("|---|" + "---|" * len(budgets))
    for arm in arms:
        cells = []
        for b in budgets:
            vals = [data[(arm, b, s)][args.metric]
                    for s in seeds if (arm, b, s) in data]
            cells.append(fmt_cell(vals))
        print(f"| {ARM_LABEL.get(arm, arm)} | " + " | ".join(cells) + " |")

    # ---- TTFT 辅表 ----
    ttft_rows = []
    for arm in arms:
        cells = []
        for b in budgets:
            vals = [data[(arm, b, s)][args.ttft]
                    for s in seeds
                    if (arm, b, s) in data and data[(arm, b, s)].get(args.ttft) is not None]
            cells.append(fmt_cell(vals))
        ttft_rows.append((arm, cells))
    if any(c != "—" for _, cells in ttft_rows for c in cells):
        print(f"\n**{args.ttft}（ms）**\n")
        print(header)
        print("|---|" + "---|" * len(budgets))
        for arm, cells in ttft_rows:
            print(f"| {ARM_LABEL.get(arm, arm)} | " + " | ".join(cells) + " |")

    # ---- 配对差值 ----
    def paired(a: str, b: str) -> Optional[Tuple[List[float], bool]]:
        deltas: List[float] = []
        for budget in budgets:
            for seed in seeds:
                ka, kb = (a, budget, seed), (b, budget, seed)
                if ka in data and kb in data:
                    deltas.append(data[ka][args.metric] - data[kb][args.metric])
        if not deltas:
            return None
        same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        return deltas, same_sign

    print("\n## 配对差值（百分点）\n")
    print("| 对比 | 逐点差值 | 中位 | 同号？ | 判定 |")
    print("|---|---|---|---|---|")
    for a, b, label in (("a1_rr", "a2_tlru", "A1 − A2"),
                        ("a0_control", "a1_rr", "A0 − A1"),
                        ("a0_control", "a2_tlru", "A0 − A2")):
        got = paired(a, b)
        if got is None:
            continue
        deltas, same_sign = got
        med = statistics.median(deltas)
        if same_sign and abs(med) >= 1.0:
            verdict = "有差异"
        elif same_sign:
            verdict = "同号但 < 1pt，视为持平"
        else:
            verdict = "无显著差异（符号不一致）"
        shown = ", ".join(f"{d:+.2f}" for d in deltas)
        print(f"| {label} | {shown} | {med:+.2f} | {'是' if same_sign else '否'} | {verdict} |")

    # ---- 权威性提示 ----
    missing = []
    for arm in arms:
        for b in budgets:
            for s in seeds:
                if (arm, b, s) not in data:
                    missing.append(f"{arm}/{b}/{s}")
    if missing:
        print(f"\n> 缺 {len(missing)} 个点（例如 {missing[0]}），配对结论不完整。")

    print("\n> 提醒：本表的绝对命中率只可组内对比；跨实验/历史数字不可比。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
