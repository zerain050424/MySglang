"""R1 验收：`session_data.jsonl → List[Session]` 与 per-task 指标。

用手写的最小 jsonl（不依赖任何数据集）钉住三件对齐 emitter 的事：

1. ``prompt`` 取整段 ``turn_i.input_ids``（不是自己拼历史）；
2. ``out`` 取 ``len(response_ids)``，response 内容不进树；
3. ``tool_call`` 取**上一轮**的值（emitter 的 payload 口径）。

    PYTHONPATH=$PWD/python python -m pytest -q test/manual/test_sim_pipeline.py
"""

import json
import os
import sys
from dataclasses import replace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sim.core.config import SimConfig  # noqa: E402
from sim.metrics import RequestRecord, summarize  # noqa: E402
from sim.simulator import Simulator  # noqa: E402
from sim.workload.session_data import load_task, load_tasks  # noqa: E402
from sim.workload.trace import Session  # noqa: E402


def _session(sid, num_turns, turns):
    payload = {"id": sid, "num_turns": num_turns}
    for idx, (prompt, response, tool) in enumerate(turns, start=1):
        payload[f"turn_{idx}"] = {
            "input_ids": prompt,
            "response_ids": response,
            "tool_call": tool,
        }
    return {sid: payload}


@pytest.fixture()
def trace_path(tmp_path):
    rows = [
        _session(
            "ShareGPT-0",
            2,
            [
                (list(range(100, 164)), [7, 8, 9], True),
                (list(range(100, 164)) + [7, 8, 9] + list(range(200, 220)), [1, 2], None),
            ],
        ),
        _session("apps_0", 1, [(list(range(300, 364)), [], None)]),
    ]
    path = tmp_path / "session_data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def test_load_task_matches_emitter_semantics(trace_path):
    sessions = load_task(trace_path, task_name="ShareGPT", max_record=-1, shuffle=False)
    assert [s.session_id for s in sessions] == ["ShareGPT-0", "apps_0"]
    assert all(s.task_name == "ShareGPT" for s in sessions)

    multi = sessions[0]
    assert multi.n_turns == 2
    # prompt 是整段历史，且第二轮含上一轮的 response（64 + 3 = 67）
    assert list(multi.turns[0].prompt) == list(range(100, 164))
    assert list(multi.turns[1].prompt)[:67] == list(range(100, 164)) + [7, 8, 9]
    # out 是 response 长度，response token 本身也保留下来供回放
    assert multi.turns[0].out == 3
    assert multi.turns[1].out == 2
    assert list(multi.turns[0].response) == [7, 8, 9]
    assert list(multi.turns[1].response) == [1, 2]
    assert multi.turns[0].response_len == 3
    # tool_call 取上一轮（第一轮没有上一轮 -> False；第二轮取第一轮的 True）
    assert multi.turns[0].tool_call is False
    assert multi.turns[1].tool_call is True

    single = sessions[1]
    assert single.turns[0].out == 0


def test_load_task_respects_max_record_and_min_turns(trace_path):
    assert len(load_task(trace_path, task_name="t", max_record=1, shuffle=False)) == 1
    short = load_task(trace_path, task_name="t", shuffle=False, min_turns=2)
    assert [s.session_id for s in short] == ["ShareGPT-0"]
    # 打乱用 seed，同 seed 可复现
    a = load_task(trace_path, task_name="t", seed=7)
    b = load_task(trace_path, task_name="t", seed=7)
    assert [s.session_id for s in a] == [s.session_id for s in b]


def test_load_tasks_relative_paths(tmp_path, trace_path):
    tasks = [
        {"name": "chat", "data_path": "session_data.jsonl", "task_id": 2, "seed": 42},
    ]
    sessions = load_tasks(tasks, base_dir=str(tmp_path))
    assert {s.task_name for s in sessions} == {"chat"}
    assert len(sessions) == 2


def test_dataset_response_is_reused_across_turns(trace_path):
    """Turn.response makes the next turn hit prev prompt + prev response."""

    sessions = load_task(trace_path, task_name="ShareGPT", shuffle=False)
    result = Simulator(SimConfig(capacity=4096, seed=3)).run_replay(sessions)
    chat = [r for r in result.records if r.session_id == "ShareGPT-0"]
    # turn 1 writes 64 prompt + 3 response; turn 2 matches all 67.
    assert chat[1].hit == 67


def test_sentinel_mode_misses_response_suffix(trace_path):
    """Without response tokens the replay is deliberately pessimistic."""

    sessions = load_task(trace_path, task_name="ShareGPT", shuffle=False)
    stripped = [
        Session(
            session.session_id,
            tuple(replace(turn, response=None) for turn in session.turns),
            session.task_name,
        )
        for session in sessions
    ]
    result = Simulator(SimConfig(capacity=4096, seed=3)).run_replay(stripped)
    chat = [r for r in result.records if r.session_id == "ShareGPT-0"]
    # turn 2 still stops at the 64-token prompt: sentinel != dataset response.
    assert chat[1].hit == 64


def test_per_task_metrics_grouped_separately(trace_path):
    sessions = load_tasks(
        [{"name": "chat", "data_path": "session_data.jsonl", "seed": 1}],
        base_dir=os.path.dirname(trace_path),
    )
    sim = Simulator(SimConfig(capacity=4096, seed=3))
    result = sim.run_replay(sessions)
    summary = result.summary

    assert summary["per_task"], "per-task 指标没生成"
    chat = summary["per_task"]["chat"]
    assert chat["requests"] == 3
    assert chat["prompt_tokens"] == sum(r.prompt_len for r in result.records)
    # 第一轮 0 命中、第二轮命中整段历史前缀
    assert result.records[1].hit > 0
    assert 0.0 < summary["hit_rate"] <= 1.0


def test_record_task_name_defaults_to_empty():
    record = RequestRecord(
        t=0.0,
        session_id="s",
        rid="r",
        prompt_len=10,
        out_len=0,
        hit=0,
        need=10,
        evicted=0,
        used=10,
        session_type="trans",
    )
    assert record.task_name == ""
    assert summarize(
        [record], capacity=10, final_used=10, sessions=1, profiler=None, occupancy=None
    )["per_task"] == {}
