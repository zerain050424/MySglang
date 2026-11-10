#!/usr/bin/env python3
"""去掉 session_data.jsonl 里每轮 prompt 开头的 ChatML system 段。

为什么需要这一步
----------------
Qwen 的默认 system prompt 只有 24 个 token，却足以让
``SessionRegistry.record_cross_session_hit`` 把**任意**单轮会话标成 STRUCT——
运行时的跨会话共享判定没有 token 门槛（``heter_utils`` 里没有对应常量）。
后果是标签分布严重偏斜：离线消融实测 random 召回 0.000，apps 全被钉在 struct
（见 ``docs/reports/classify_system_prompt_ablation.md``）。

只剥掉这一段即可恢复，不需要改分类器阈值：

    臂 A  生产缺省         acc 0.5791   random 召回 0.000
    臂 C  只去 Qwen system acc 0.7697   random 召回 0.993

剥离逻辑复用 ``classify/session_data.strip_chatml_system``，只剥**首段**，
其余对话与 turn 结构原样保留。

用法
----
    # 生成新文件（推荐），文件名追加 .nosys
    python tools/strip_qwen_system.py \\
        --in  TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.jsonl

    # 原地改写
    python tools/strip_qwen_system.py --in <path> --in-place

    # 先看会剥掉多少，不落盘
    python tools/strip_qwen_system.py --in <path> --dry-run

输入格式：每行一个 ``{session_id: session}`` 对象，session 内含 ``num_turns``
与 ``turn_1..turn_N``（``turn_N.input_ids`` 才是运行时真正发送的 prompt）。

``--mode qwen``（默认）只剥 Qwen 默认 system 段；``--mode all`` 连数据集自带的
领域 system prompt（如 Codeparrot 的）一起剥——后者会降低结构复用，
仅在专门验证「无 system 前缀」假设时使用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR / "classify"))

from session_data import strip_chatml_system  # noqa: E402

MODES = ("qwen", "all")


def strip_session(record: dict, mode: str) -> tuple[int, int, bool]:
    """就地剥离一条会话的全部轮次。

    Returns:
        (改写轮数, 去掉的 token 数, 是否出现过「剥完为空」的轮次)
    """
    turns_changed = tokens_removed = 0
    emptied = False
    for turn_index in range(1, int(record.get("num_turns") or 0) + 1):
        turn = record.get(f"turn_{turn_index}")
        if not turn:
            continue
        ids = turn.get("input_ids")
        if not ids:
            continue
        stripped = strip_chatml_system(list(ids), mode)
        if stripped == ids:
            continue
        if not stripped:
            # 整轮只有 system 段：保留原样，否则后续会以空 prompt 发请求。
            emptied = True
            continue
        turn["input_ids"] = stripped
        turns_changed += 1
        tokens_removed += len(ids) - len(stripped)
    return turns_changed, tokens_removed, emptied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="去掉 session_data.jsonl 中每轮 prompt 开头的 ChatML system 段"
    )
    parser.add_argument("--in", dest="src", required=True, help="输入 session_data.jsonl")
    parser.add_argument("--out", help="输出路径；缺省为 <输入>.nosys.jsonl")
    parser.add_argument("--in-place", action="store_true", help="原地改写输入文件")
    parser.add_argument("--mode", choices=MODES, default="qwen", help="剥离范围，默认 qwen")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_file():
        print(f"输入不存在：{src}", file=sys.stderr)
        return 2
    if args.in_place and args.out:
        print("--in-place 与 --out 互斥", file=sys.stderr)
        return 2
    dst = src if args.in_place else Path(args.out) if args.out else src.with_suffix(".nosys.jsonl")

    sessions = turns_total = turns_changed = tokens_removed = emptied_sessions = 0
    records = []
    with src.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"{src}:{line_no} 不是合法 JSON：{exc}", file=sys.stderr)
                return 1
            # 每行是 {session_id: session}，num_turns / turn_N 在内层。
            if not isinstance(obj, dict) or not all(isinstance(v, dict) for v in obj.values()):
                print(f"{src}:{line_no} 不是 {{session_id: session}} 结构", file=sys.stderr)
                return 1
            for sess in obj.values():
                changed, removed, emptied = strip_session(sess, args.mode)
                sessions += 1
                turns_total += int(sess.get("num_turns") or 0)
                turns_changed += changed
                tokens_removed += removed
                emptied_sessions += int(emptied)
            if not args.dry_run:
                records.append(json.dumps(obj, ensure_ascii=False))

    print(f"会话 {sessions}，轮次 {turns_total}")
    print(f"改写轮次 {turns_changed}，去掉 {tokens_removed} token "
          f"（平均 {tokens_removed / turns_changed:.1f}/轮）" if turns_changed
          else "没有可剥离的 system 段——数据可能已经处理过")
    if emptied_sessions:
        print(f"[warn] {emptied_sessions} 条会话存在「仅含 system 段」的轮次，已跳过")

    if args.dry_run:
        print("dry-run：未写入")
        return 0

    # 无变化（数据已剥过 system 段）不算失败：--in-place 时没必要重写文件，
    # 生成副本时仍要落盘，否则下游按约定找 *.nosys.jsonl 会扑空。
    if not turns_changed and dst == src:
        print("无变化：原地模式，文件未改动")
        return 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(records) + "\n", encoding="utf-8")
    print(f"已写入 {dst}" if turns_changed else f"无变化，已原样写出 {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
