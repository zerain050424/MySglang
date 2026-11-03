"""命令行参数与数据集装配（run_eval / sweep 共用）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from session_data import (
    assign_gaps,
    default_task_paths,
    load_task_sessions,
    load_trace_sessions,
    trace_session_ids,
)

SG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE = SG_ROOT / "result" / "phased_load_prediction_C.csv"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        choices=("trace", "task"),
        default="trace",
        help="trace=按 bench CSV 的真实时间轴重放；task=用任务数据 + gap 模型构造时间轴",
    )
    parser.add_argument("--trace-csv", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--turns-limit", type=int, default=None, help="只用每条会话的前 N 轮")
    parser.add_argument(
        "--task-data",
        type=Path,
        nargs="*",
        default=None,
        help="任务数据 jsonl，缺省为 TaskData/*/Qwen2.5-7B-Instruct/*_session_data.jsonl",
    )
    parser.add_argument("--max-per-workload", type=int, default=None)
    parser.add_argument(
        "--gap-model",
        default="per_workload",
        help="task 模式的轮间隔：per_workload | fixed:<s> | lognormal:<mu>:<sigma>",
    )
    parser.add_argument("--seed", type=int, default=42)

    # 分类器参数（缺省即生产值，改动只影响本次评测）
    parser.add_argument("--multi-turn-min", type=int, default=5)
    parser.add_argument("--gap-split", type=float, default=15.0)
    parser.add_argument("--growth", type=int, default=1500)

    # 结构证据口径
    parser.add_argument("--structure", choices=("prefix", "off"), default="prefix")
    parser.add_argument("--structure-turns", choices=("first", "all"), default="first")
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--min-share-tokens", type=int, default=1)

    # tool_call 传递口径
    parser.add_argument("--tool-call-mode", choices=("prior", "same", "off"), default="prior")


def build(args) -> dict:
    """按参数装配会话表（含时间轴）。"""
    paths = args.task_data or default_task_paths(SG_ROOT)
    if not paths:
        raise SystemExit("没有找到任务数据 jsonl，用 --task-data 指定")

    if args.source == "trace":
        ids = trace_session_ids(args.trace_csv)
        index = load_task_sessions(paths, only_ids=ids)
        sessions = load_trace_sessions(args.trace_csv, index, turns_limit=args.turns_limit)
        missing = sum(1 for r in sessions.values() if r.turns and r.turns[0].tokens is None)
        if missing:
            print(f"[warn] {missing} 条 trace 会话在任务数据里找不到 token，结构证据按 0 处理")
    else:
        sessions = load_task_sessions(
            paths,
            max_sessions=args.max_per_workload,
            seed=args.seed,
            keep_all_tokens=args.structure_turns == "all",
        )
        assign_gaps(sessions, args.gap_model, seed=args.seed)
    return sessions


def config_from_args(args):
    from replay import ReplayConfig

    return ReplayConfig(
        multi_turn_min=args.multi_turn_min,
        gap_split_s=args.gap_split,
        growth_tokens=args.growth,
        structure=args.structure,
        structure_turns=args.structure_turns,
        block=args.block,
        min_share_tokens=args.min_share_tokens,
        tool_call_mode=args.tool_call_mode,
    )
