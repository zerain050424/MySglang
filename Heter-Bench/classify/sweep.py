#!/usr/bin/env python3
"""阈值 / 证据口径扫描：准确率到底由哪条规则决定。

每个变体只改一个维度，其余保持生产缺省。输出 <out>/sweep.md + sweep.json。

    /root/sl_old/bin/python sweep.py --source trace
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from dataset_build import SG_ROOT, add_common_args, build, config_from_args
from session_data import assign_gaps
from metrics import LABELS, accuracy, ceiling, confusion_matrix, macro_f1, per_class_prf
from replay import replay


def evaluate(sessions, cfg) -> dict:
    results = list(replay(sessions, cfg).values())
    cm = confusion_matrix(results)
    prf = per_class_prf(cm)
    ceil = ceiling(results, cfg.multi_turn_min)
    acc = accuracy(results)
    return {
        "accuracy": round(acc, 4),
        "macro_f1": macro_f1(prf),
        "ceiling": ceil["ceiling"],
        "achievement": round(acc / ceil["ceiling"], 4) if ceil["ceiling"] else 0.0,
        "recall": {lab: prf[lab]["recall"] for lab in LABELS},
        "precision": {lab: prf[lab]["precision"] for lab in LABELS},
        "unreachable": ceil["unreachable_by_class"],
    }


VARIANTS = [
    ("baseline（生产缺省）", {}),
    ("structure=off（无跨会话证据）", {"structure": "off"}),
    ("min_share=32", {"min_share_tokens": 32}),
    ("min_share=64", {"min_share_tokens": 64}),
    ("min_share=128", {"min_share_tokens": 128}),
    ("min_share=512", {"min_share_tokens": 512}),
    ("gap_split=5s", {"gap_split_s": 5.0}),
    ("gap_split=10s", {"gap_split_s": 10.0}),
    ("gap_split=20s", {"gap_split_s": 20.0}),
    ("gap_split=30s", {"gap_split_s": 30.0}),
    ("multi_turn_min=3", {"multi_turn_min": 3}),
    ("multi_turn_min=4", {"multi_turn_min": 4}),
    ("multi_turn_min=6", {"multi_turn_min": 6}),
    ("tool_call=off（只靠间隔/增长）", {"tool_call_mode": "off"}),
    ("tool_call=same（含当前轮，泄漏）", {"tool_call_mode": "same"}),
    ("growth=500", {"growth_tokens": 500}),
    ("growth=3000", {"growth_tokens": 3000}),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out", type=Path, default=SG_ROOT / "result" / "classify")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    sessions = build(args)
    base = config_from_args(args)
    rows = []
    scenarios = [("", VARIANTS)]
    if args.source == "task":
        # 第二个场景：把轮间隔整体压成 20s（模拟真实 agent 的工具往返/人工确认），
        # 这时间隔不再能区分 chat 与 agentic，tool_call 证据是否生效才会暴露出来。
        scenarios.append(
            (
                "slow-gap(20s) ",
                [
                    ("baseline", {}),
                    ("tool_call=off", {"tool_call_mode": "off"}),
                    ("min_share=64", {"min_share_tokens": 64}),
                ],
            )
        )

    for prefix, variants in scenarios:
        if prefix:
            assign_gaps(sessions, "fixed:20")
        rows.extend(_run(sessions, base, prefix, variants))

    header = (
        "| 变体 | acc | macro-F1 | 上限 | 达成率 | struct R | random R | chat R | agentic R |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    body = "".join(
        f"| {r['variant']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['ceiling']:.4f} | "
        f"{r['achievement']:.4f} | "
        + " | ".join(f"{r['recall'][lab]:.3f}" for lab in LABELS)
        + " |\n"
        for r in rows
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    (out / f"sweep{suffix}.md").write_text(
        f"# 阈值与证据口径扫描\n\n{header}{body}", encoding="utf-8"
    )
    (out / f"sweep{suffix}.json").write_text(
        json.dumps({"n_sessions": len(sessions), "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run(sessions, base, prefix, variants):
    rows = []
    for name, override in variants:
        cfg = replace(base, **override)
        name = f"{prefix}{name}"
        row = {"variant": name, "config": override or "production default", **evaluate(sessions, cfg)}
        rows.append(row)
        print(
            f"{name:36s} acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f} "
            f"ceil={row['ceiling']:.4f} hit={row['achievement']:.4f} "
            + " ".join(f"{lab}R={row['recall'][lab]:.2f}" for lab in LABELS)
        )
    return rows


if __name__ == "__main__":
    main()
