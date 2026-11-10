#!/usr/bin/env python3
"""build_exp3_data.py 的自测：分桶、struct 前缀共享、类别隔离、输出格式。

不依赖 transformers：用一个鸭子类型的假 tokenizer 把文本编码成固定长度的 id 序列。

    python mixexp/tools/tests/test_build_exp3_data.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_exp3_data as b3  # noqa: E402


class FakeTokenizer:
    """把文本编码成“每个字符一个 id”的假 tokenizer。"""

    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        return self._Out([1 + (ord(ch) % 200) for ch in text])


def make_session(sid: str, num_turns: int, tokens: int = 10) -> dict:
    session = {"id": sid, "num_turns": num_turns}
    for i in range(1, num_turns + 1):
        session[f"turn_{i}"] = {
            "input_text": f"prompt-{sid}-{i}",
            "input_ids": list(range(tokens)),
            "response_text": "ok",
            "response_ids": [7, 8],
        }
    return session


def write_source(path: Path, sessions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps({session["id"]: session}) + "\n")


def check_chat(tmp: Path, failures) -> None:
    """chat 分桶正确，且不注入任何前缀（类别隔离）。"""
    taskdata, outdir = tmp / "TaskData", tmp / "exp3"
    sessions = [make_session(f"chat-{i}", 7) for i in range(20)]
    sessions += [make_session(f"chat-long-{i}", 12) for i in range(8)]
    write_source(taskdata / b3.SPECS["chat"].source, sessions)

    args = type("A", (), dict(taskdata=str(taskdata), out_dir=str(outdir),
                              limit=0, struct_sessions=0,
                              struct_prefix_tokens=2000, struct_variants=8))
    spec = b3.SPECS["chat"]
    spec.target = 10
    with redirect_stdout(io.StringIO()):
        info = b3.build_one(spec, args, FakeTokenizer())

    rows = [json.loads(line) for line in Path(info["output"]).read_text().splitlines()]
    if len(rows) != 10:
        failures.append(f"chat 会话条数应为 10，实际 {len(rows)}")
    turns = [row[next(iter(row))]["num_turns"] for row in rows]
    if sum(1 for t in turns if t >= 10) != 2:
        failures.append("chat 长会话应为 2 条（20%）")

    session = rows[0][next(iter(rows[0]))]
    if len(session["turn_1"]["input_ids"]) != 10:
        failures.append("chat 的 input_ids 被改动了（不应注入前缀）")
    if not session["turn_1"]["input_text"].startswith("prompt-"):
        failures.append("chat 的 input_text 被改动了（不应注入前缀）")
    if session["turn_1"]["response_ids"] != [7, 8]:
        failures.append("response_ids 被改动")


def check_apps(tmp: Path, failures) -> None:
    """apps：真实会话不足目标时按真实长度分布补合成会话，且合成会话互不共享。"""
    taskdata, outdir = tmp / "TaskData", tmp / "exp3"
    real = []
    for i in range(6):
        session = make_session(f"apps-{i}", 1, tokens=100 + 10 * i)
        real.append(session)
    write_source(taskdata / b3.SPECS["apps"].source, real)

    args = type("A", (), dict(taskdata=str(taskdata), out_dir=str(outdir), limit=0,
                              apps_sessions=30, struct_sessions=0,
                              struct_prefix_tokens=2000, struct_variants=8))
    spec = b3.SPECS["apps"]
    with redirect_stdout(io.StringIO()):
        info = b3.build_one(spec, args, FakeTokenizer())

    rows = [json.loads(line) for line in Path(info["output"]).read_text().splitlines()]
    if len(rows) != 30:
        failures.append(f"apps 会话数应为 30，实际 {len(rows)}")
    if info["real"] != 6 or info["synthetic"] != 24:
        failures.append(f"apps 真实/合成应为 6/24，实际 {info['real']}/{info['synthetic']}")

    ids = [tuple(next(iter(r.values()))["turn_1"]["input_ids"]) for r in rows]
    if len(set(ids)) != len(ids):
        failures.append("apps 合成会话之间出现了重复 body（会制造假共享）")
    lengths = {len(i) for i in ids}
    if not lengths <= {100, 110, 120, 130, 140, 150}:
        failures.append(f"合成会话长度应取自真实分布，实际出现 {sorted(lengths - {100,110,120,130,140,150})}")
    if any(len(row) != 1 for row in rows):
        failures.append("apps 每行必须是单键 {session_id: session}")


def check_struct(tmp: Path, failures) -> None:
    """struct：变体内前缀唯一且等长，变体间不共享，私有正文互不相同。"""
    outdir = tmp / "exp3"
    args = type("A", (), dict(out_dir=str(outdir), struct_sessions=40,
                              struct_prefix_tokens=64, struct_variants=4))
    spec = b3.SPECS["struct"]
    with redirect_stdout(io.StringIO()):
        info = b3.build_one(spec, args, FakeTokenizer())

    rows = [json.loads(line) for line in Path(info["output"]).read_text().splitlines()]
    if len(rows) != 40:
        failures.append(f"struct 会话条数应为 40，实际 {len(rows)}")

    by_variant: dict[str, list[list[int]]] = {}
    bodies = set()
    for row in rows:
        sid, session = next(iter(row.items()))
        ids = session["turn_1"]["input_ids"]
        variant = session["id"].split("-")[1]
        by_variant.setdefault(variant, []).append(ids[:64])
        bodies.add(tuple(ids[64:]))

    if len(by_variant) != 4:
        failures.append(f"struct 变体数应为 4，实际 {len(by_variant)}")
    for variant, prefixes in by_variant.items():
        if len({tuple(p) for p in prefixes}) != 1:
            failures.append(f"变体 {variant} 内前缀不一致（应完全共享）")
        if len(prefixes[0]) != 64:
            failures.append(f"变体 {variant} 前缀长度应为 64，实际 {len(prefixes[0])}")
    distinct_prefixes = {tuple(prefixes[0]) for prefixes in by_variant.values()}
    if len(distinct_prefixes) != 4:
        failures.append("不同变体的前缀应互不相同")
    if len(bodies) != 40:
        failures.append("私有正文应每会话唯一，避免产生额外的共享节点")


def run_case() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        check_chat(tmp, failures)
        check_apps(tmp, failures)
        check_struct(tmp, failures)

    if failures:
        print("失败：")
        for item in failures:
            print("  -", item)
        return 1
    print("自测通过：分桶、apps 合成冲刷补足、struct 前缀共享/变体隔离、其他类别无前缀、"
          "字段保留、输出格式")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_case())
