"""``tools/drop_short_sessions.py`` 的自检：只丢「终身都是 TRANS」的短会话。

    python -m pytest Heter-Bench/tools/tests/test_drop_short_sessions.py -q

本文件同时充当口径守卫：过滤器阈值必须等于 ``SessionRegistry.MULTI_TURN_MIN``。
常量一改，这条断言就红——提醒重新生成 ``*.mt<N>.jsonl``，别拿旧数据跑新分类器。
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.abspath(os.path.join(_HERE, "..", "drop_short_sessions.py"))
_HETER_UTILS = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "python", "sglang", "srt", "mem_cache", "heter_cache/utils.py")
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


drop_tool = _load("drop_short_sessions", _TOOL)
# heter_utils 只依赖标准库，不需要 torch / sglang 导入链，可直接按路径加载。
heter_utils = _load("heter_utils", _HETER_UTILS)
MULTI_TURN_MIN = heter_utils.SessionRegistry.MULTI_TURN_MIN


def _session(sid, num_turns):
    record = {"session_id": sid, "num_turns": num_turns}
    for index in range(1, num_turns + 1):
        record[f"turn_{index}"] = {"input_ids": list(range(index, index + 8))}
    return record


def _write(tmp_path, sessions):
    import json

    path = tmp_path / "session_data.nosys.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps({session["session_id"]: session}) + "\n")
    return path


def test_threshold_tracks_registry_constant():
    """阈值只有一个来源：SessionRegistry.MULTI_TURN_MIN。"""

    parsed = drop_tool.multi_turn_min_from_source()
    assert parsed == MULTI_TURN_MIN


def test_keeps_sessions_at_or_above_threshold(tmp_path):
    sessions = [_session(f"s{n}", n) for n in (1, 2, MULTI_TURN_MIN - 1, MULTI_TURN_MIN, 12)]
    path = _write(tmp_path, sessions)

    kept, dropped, kept_turns, dropped_turns = drop_tool.filter_sessions(path, MULTI_TURN_MIN)

    assert dropped == 3
    assert sorted(kept_turns) == [MULTI_TURN_MIN, 12]
    assert sorted(dropped_turns) == [1, 2, MULTI_TURN_MIN - 1]
    assert len(kept) == 2


def test_input_lines_are_copied_verbatim(tmp_path):
    """过滤不能改动 prompt：保留行必须是原始字节。"""

    sessions = [_session("long", MULTI_TURN_MIN + 3), _session("short", 2)]
    path = _write(tmp_path, sessions)
    original = path.read_text(encoding="utf-8").splitlines()

    kept, *_ = drop_tool.filter_sessions(path, MULTI_TURN_MIN)

    assert [line.rstrip("\n") for line in kept] == [original[0]]


def test_missing_num_turns_is_dropped(tmp_path):
    import json

    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps({"x": {"session_id": "x"}}) + "\n", encoding="utf-8")

    kept, dropped, _, dropped_turns = drop_tool.filter_sessions(path, MULTI_TURN_MIN)

    assert kept == [] and dropped == 1 and dropped_turns == [0]


def test_all_short_input_is_refused(tmp_path, capsys, monkeypatch):
    """单轮数据集（apps / Codeparrot）全部会话都是 1 轮：必须拒绝，别静默清空。"""

    path = _write(tmp_path, [_session("apps_1", 1), _session("apps_2", 1)])
    monkeypatch.setattr(
        sys, "argv", ["drop_short_sessions.py", "--in", str(path), "--dry-run"]
    )

    assert drop_tool.main() == 1
    assert "全部会话都被剔除" in capsys.readouterr().err
