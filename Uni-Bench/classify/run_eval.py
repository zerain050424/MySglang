#!/usr/bin/env python3
"""会话级分类准确率主实验。

用法（在 Uni-Bench/classify 下）：
    /root/sl_old/bin/python run_eval.py --source trace
    /root/sl_old/bin/python run_eval.py --source task --max-per-workload 2000 \
        --gap-model per_workload

输出：<out>/report.md、<out>/metrics.json、<out>/sessions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_build import SG_ROOT, add_common_args, build, config_from_args
from metrics import render_markdown, summarize, write_sessions_csv
from replay import replay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out", type=Path, default=SG_ROOT / "result" / "classify")
    ap.add_argument("--tag", default="", help="输出文件名后缀，便于区分多组配置")
    args = ap.parse_args()

    sessions = build(args)
    cfg = config_from_args(args)
    results = list(replay(sessions, cfg).values())
    target = f"source={args.source}" + (f", trace={args.trace_csv.name}" if args.source == "trace" else f", gap={args.gap_model}")
    rep = summarize(results, cfg, target)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    (out / f"report{suffix}.md").write_text(render_markdown(rep), encoding="utf-8")
    payload = {
        "target": rep.target,
        "config": rep.config,
        "n_sessions": rep.n_sessions,
        "by_class": rep.by_class,
        "accuracy": rep.accuracy,
        "macro_f1": rep.macro_f1,
        "prf": rep.prf,
        "confusion": rep.confusion,
        "curve": rep.curve,
        "convergence": rep.convergence,
        "ceiling": rep.ceiling,
        "structure_stats": rep.structure_stats,
    }
    (out / f"metrics{suffix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_sessions_csv(out / f"sessions{suffix}.csv", results)
    print(render_markdown(rep))


if __name__ == "__main__":
    main()
