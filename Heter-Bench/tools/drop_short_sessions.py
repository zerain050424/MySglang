#!/usr/bin/env python3
"""丢掉轮数 < MULTI_TURN_MIN 的多轮会话，让 TRANS 只可能来自单轮负载。

为什么要这一步
--------------
``SessionRegistry._classify`` 的判定顺序是「轮数 → 工具调用 → 间隔」：

    turn_count < MULTI_TURN_MIN(=5)  →  TRANS（未发生跨会话共享时）

派生的后果有两个，性质完全不同：

1. 真正的单轮负载（apps、Codeparrot）判 TRANS 是**对**的：一条请求、无复用。
2. ShareGPT / alfred 里那些**总共只有 1–4 轮**的会话，终身类型都是 TRANS。
   它们每一轮都在复用自己上一轮的前缀，却被当成一次性流量优先清掉——这就是
   M1 机制把多轮负载的均匀命中率压下去的主要来源（实测 LRU 与 hc 相差 ~20pp，
   ShareGPT turn-1 39.7%→7.5%，见 test/manual/test_slru_one_shot.py [9]）。

本工具的作用范围只有第 2 类：把「终身都是 TRANS」的会话整条丢掉，于是
**TRANS 带里只剩单轮会话和长会话的前几轮**。单轮负载本身不经过它。

口径边界（别误读）
------------------
过滤后**仍不是**「TRANS 100% 来自单轮」的严格口径：6 轮会话的第 1–4 轮照样判
TRANS，活到第 5 轮才切走。要做到逐请求口径，必须让分类器提前知道会话总轮数
（``GenerateReqInput`` 增加元数据 → ``SessionRegistry`` 记录 → ``_classify``
用总轮数而不是已见轮数判 TRANS）。那是分类器改动，不在本工具职责内。

用法
----
    # MULTI_TURN_MIN 从 heter_utils.py 读；输出 <in>.mt5.jsonl
    python tools/drop_short_sessions.py --in \\
        TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.nosys.jsonl

    # 只看会丢多少，不落盘
    python tools/drop_short_sessions.py --in <path> --dry-run

    # 显式指定阈值（与代码常量脱钩时用，结果不可比）
    python tools/drop_short_sessions.py --in <path> --min-turns 5

    # 只打印阈值（脚本批量调用时取同一个来源，别在 shell 里再写一个 5）
    python tools/drop_short_sessions.py --print-min-turns

输入是 ``mixexp/prepare_data.sh`` 产出的 ``*.nosys.jsonl``；输出沿用同一套后缀，
追加 ``.mt<N>`` 表示「已剔除轮数 < N 的会话」。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
ROOT_DIR = BENCH_DIR.parent
HETER_UTILS = ROOT_DIR / "python/sglang/srt/mem_cache/heter_utils.py"

_CONST_RE = re.compile(r"^\s*MULTI_TURN_MIN\s*=\s*(\d+)", re.MULTILINE)


def multi_turn_min_from_source(path: Path = HETER_UTILS) -> Optional[int]:
    """从 ``heter_utils.py`` 源码读 ``MULTI_TURN_MIN``。

    只解析文本、不 import：本工具要在没有 torch / sglang 依赖链的机器上跑
    （``import sglang.srt.mem_cache.heter_utils`` 会拉起整条 sglang 导入）。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _CONST_RE.search(text)
    return int(match.group(1)) if match else None


def num_turns_of(record: dict) -> int:
    """取会话轮数；缺失或非法一律按 0 处理（会被过滤器丢掉）。"""
    try:
        return int(record.get("num_turns") or 0)
    except (TypeError, ValueError):
        return 0


def iter_lines(path: Path) -> Iterator[Tuple[int, str, dict]]:
    """逐行产出 (行号, 原始行, 会话对象)。每行只含一条会话。"""
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} 不是合法 JSON：{exc}")
            if not isinstance(obj, dict) or not obj:
                raise SystemExit(f"{path}:{line_no} 不是 {{session_id: session}} 对象")
            session = next(iter(obj.values()))
            if not isinstance(session, dict):
                raise SystemExit(f"{path}:{line_no} 的值不是会话对象")
            yield line_no, line, session


def filter_sessions(
    path: Path, min_turns: int
) -> Tuple[List[str], int, List[int], List[int]]:
    """返回 (保留的原始行, 丢掉的会话数, 保留会话的轮数列表, 丢掉会话的轮数列表)。

    原样返回输入行而不重新序列化：这一步不该改动任何 prompt 内容。
    """
    kept: List[str] = []
    kept_turns: List[int] = []
    dropped_turns: List[int] = []
    for _line_no, line, session in iter_lines(path):
        turns = num_turns_of(session)
        if turns >= min_turns:
            kept.append(line if line.endswith("\n") else line + "\n")
            kept_turns.append(turns)
        else:
            dropped_turns.append(turns)
    return kept, len(dropped_turns), kept_turns, dropped_turns


def _summary(turns: List[int]) -> str:
    if not turns:
        return "无"
    ordered = sorted(turns)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return f"最少 {ordered[0]} / 中位 {median:g} / 最多 {ordered[-1]}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="剔除轮数 < MULTI_TURN_MIN 的会话，使 TRANS 只来自单轮负载",
    )
    parser.add_argument("--in", dest="src", help="输入 session_data.jsonl")
    parser.add_argument("--out", dest="dst", help="输出路径；缺省 <输入>.mt<N>.jsonl")
    parser.add_argument(
        "--min-turns",
        type=int,
        default=None,
        help="保留的最小轮数；缺省读 heter_utils.py 的 MULTI_TURN_MIN",
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    parser.add_argument(
        "--print-min-turns",
        action="store_true",
        help="只打印 MULTI_TURN_MIN 并退出（供 prepare_data.sh 复用同一来源）",
    )
    args = parser.parse_args()

    min_turns = args.min_turns
    source_of_min = "--min-turns"
    if min_turns is None:
        min_turns = multi_turn_min_from_source()
        source_of_min = f"{HETER_UTILS.name} 的 MULTI_TURN_MIN"
    if args.print_min_turns:
        if min_turns is None:
            print(f"读不到 {HETER_UTILS} 的 MULTI_TURN_MIN", file=sys.stderr)
            return 2
        print(min_turns)
        return 0

    if not args.src:
        print("缺少 --in（或用 --print-min-turns 只取阈值）", file=sys.stderr)
        return 2
    src = Path(args.src)
    if not src.is_file():
        print(f"输入不存在：{src}", file=sys.stderr)
        return 2
    if min_turns is None:
        print(
            f"读不到 {HETER_UTILS} 的 MULTI_TURN_MIN；请显式传 --min-turns。",
            file=sys.stderr,
        )
        return 2
    if min_turns < 2:
        print(f"--min-turns={min_turns} 会把单轮负载一起丢掉，拒绝执行。", file=sys.stderr)
        return 2

    dst = (
        Path(args.dst)
        if args.dst
        else src.with_suffix(f".mt{min_turns}.jsonl")
    )

    kept, dropped, kept_turns, dropped_turns = filter_sessions(src, min_turns)
    total = len(kept) + dropped
    print(f"输入 {src}")
    print(f"阈值 {min_turns} 轮（来自 {source_of_min}）")
    print(
        f"会话 {total} → 保留 {len(kept)}，剔除 {dropped}"
        + (f"（{dropped / total:.1%}）" if total else "")
    )
    print(f"保留会话轮数：{_summary(kept_turns)}")
    if dropped_turns:
        print(f"剔除会话轮数：{_summary(dropped_turns)}")

    if not kept:
        print(
            "全部会话都被剔除：这看起来是单轮数据集，本工具只用于多轮源。",
            file=sys.stderr,
        )
        return 1
    if dropped and dropped / total > 0.5:
        print(
            f"[warn] 剔除比例 {dropped / total:.1%}：源数据以短会话为主，"
            "过滤后规模与组成都会明显变化，别和过滤前的数字直接对比。",
            file=sys.stderr,
        )

    if args.dry_run:
        print("dry-run：未写入")
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(kept), encoding="utf-8")
    print(f"已写入 {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
