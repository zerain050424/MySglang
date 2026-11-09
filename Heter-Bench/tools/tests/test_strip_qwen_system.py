"""``tools/strip_qwen_system.py`` 的自检：只剥 Qwen 默认 system 段，别的一律不动。

    python -m pytest Heter-Bench/tools/tests/test_strip_qwen_system.py -q
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.abspath(os.path.join(_HERE, "..", "strip_qwen_system.py"))
_CLASSIFY = os.path.abspath(os.path.join(_HERE, "..", "..", "classify"))

sys.path.insert(0, _CLASSIFY)
from session_data import _IM_END, _IM_START, _SYSTEM, QWEN_SYSTEM_IDS  # noqa: E402

_spec = importlib.util.spec_from_file_location("strip_qwen_system", _TOOL)
strip_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strip_tool)

BODY = list(range(200, 230))
QWEN_PREFIX = [_IM_START, _SYSTEM, 198, *QWEN_SYSTEM_IDS, _IM_END, 198]
# 结构相同但正文不同：模拟 Codeparrot 这类自带领域 system prompt 的数据
OTHER_PREFIX = [_IM_START, _SYSTEM, 198, 999, 888, _IM_END, 198]


def _session(sid, turns):
    record = {"session_id": sid, "num_turns": len(turns)}
    for index, ids in enumerate(turns, 1):
        record[f"turn_{index}"] = {"input_ids": list(ids)}
    return record


def test_qwen_prefix_stripped_on_every_turn():
    record = _session("ShareGPT_a", [QWEN_PREFIX + BODY, QWEN_PREFIX + BODY + [300]])
    changed, removed, emptied = strip_tool.strip_session(record, "qwen")

    assert (changed, emptied) == (2, False)
    assert removed == 2 * len(QWEN_PREFIX)
    assert record["turn_1"]["input_ids"] == BODY
    assert record["turn_2"]["input_ids"] == BODY + [300]


def test_other_system_kept_under_qwen_mode():
    record = _session("Code-b", [OTHER_PREFIX + BODY])
    changed, removed, _ = strip_tool.strip_session(record, "qwen")

    assert (changed, removed) == (0, 0)
    assert record["turn_1"]["input_ids"] == OTHER_PREFIX + BODY


def test_other_system_stripped_under_all_mode():
    record = _session("Code-b", [OTHER_PREFIX + BODY])
    changed, removed, _ = strip_tool.strip_session(record, "all")

    assert (changed, removed) == (1, len(OTHER_PREFIX))
    assert record["turn_1"]["input_ids"] == BODY


def test_prompt_without_system_is_untouched():
    record = _session("apps_c", [list(BODY)])
    changed, removed, _ = strip_tool.strip_session(record, "qwen")

    assert (changed, removed) == (0, 0)
    assert record["turn_1"]["input_ids"] == BODY


def test_system_only_turn_is_skipped_not_emptied():
    """整轮只有 system 段时保留原样，否则会发出空 prompt。"""

    record = _session("edge_d", [QWEN_PREFIX])
    changed, removed, emptied = strip_tool.strip_session(record, "qwen")

    assert changed == 0 and removed == 0
    assert emptied is True
    assert record["turn_1"]["input_ids"] == QWEN_PREFIX
