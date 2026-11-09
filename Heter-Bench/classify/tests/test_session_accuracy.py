"""评测脚本的自检：口径映射、trace 回填、重放语义、指标计算。

    /root/sl_old/bin/python -m pytest Uni-Bench/classify/tests/test_session_accuracy.py -q
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics import ceiling, confusion_matrix, per_class_prf, accuracy_curve, converged_turn
from replay import ReplayConfig, replay
from session_data import (
    PRED_LABEL,
    SessionRecord,
    Turn,
    gt_of_session,
    load_task_sessions,
    load_trace_sessions,
)

SYS_PROMPT = list(range(1000, 1024))  # 24 token 的公共 system prompt


def make_session(sid, gt, n_turns, gap, prompt_len=200, growth=0, tool_call=False, tokens=None):
    turns = []
    for i in range(n_turns):
        turns.append(
            Turn(
                index=i + 1,
                start=i * gap,
                prompt_len=prompt_len + i * growth,
                prior_tool_call=tool_call and i > 0,
                same_turn_tool_call=tool_call,
                tokens=tokens if i == 0 else None,
            )
        )
    return SessionRecord(session_id=sid, gt=gt, turns=turns)


def test_gt_and_pred_label_mapping():
    assert gt_of_session("Code-abc-1") == "struct"
    assert gt_of_session("apps_644") == "random"
    assert gt_of_session("ShareGPT-multi-turn-x_0") == "chat"
    assert gt_of_session("alfred_93-abc") == "agentic"
    assert gt_of_session("unknown") is None
    assert set(PRED_LABEL) == {"trans", "struct", "chat", "agentic"}
    assert PRED_LABEL["trans"] == "random"


def test_replay_semantics_for_the_four_buckets():
    sessions = {
        "chat5": make_session("chat5", "chat", 5, gap=20.0, growth=100),
        "chat4": make_session("chat4", "chat", 4, gap=20.0, growth=100),
        "agent": make_session("agent", "agentic", 5, gap=20.0, growth=100, tool_call=True),
        "single": make_session("single", "random", 1, gap=0.0),
    }
    res = replay(sessions, ReplayConfig(structure="off"))
    assert res["chat5"].final == "chat"
    assert res["chat4"].final == "random"  # >4 轮门槛：短 chat 不可判
    assert res["agent"].final == "agentic"
    assert res["single"].final == "random"


def test_short_session_with_shared_system_prompt_becomes_struct():
    sessions = {
        "a": make_session("a", "random", 1, gap=0.0, tokens=SYS_PROMPT + [1, 2]),
        "b": make_session("b", "random", 1, gap=0.5, tokens=SYS_PROMPT + [3, 4]),
    }
    default = replay(sessions, ReplayConfig())
    assert default["b"].final == "struct"  # 24 token 公共前缀也算跨会话共享

    strict = replay(sessions, ReplayConfig(min_share_tokens=64))
    assert strict["b"].final == "random"  # 提高到 64 token 后不再误判


def test_long_shared_code_prefix_survives_share_threshold():
    code = list(range(2000)) + [7]
    sessions = {
        "c1": make_session("c1", "struct", 1, gap=0.0, tokens=code),
        "c2": make_session("c2", "struct", 1, gap=1.0, tokens=code),
    }
    res = replay(sessions, ReplayConfig(min_share_tokens=64))
    assert res["c2"].final == "struct"
    assert res["c2"].shared_len >= 900


def test_replay_is_deterministic():
    sessions = {
        f"s{i}": make_session(f"s{i}", "chat", 6, gap=20.0 + i, growth=300)
        for i in range(5)
    }
    a = replay(sessions, ReplayConfig())
    b = replay(sessions, ReplayConfig())
    assert {k: v.preds for k, v in a.items()} == {k: v.preds for k, v in b.items()}


def test_metrics_confusion_and_ceiling():
    sessions = {
        "c1": make_session("c1", "chat", 5, gap=20.0, growth=100),  # 正确
        "c2": make_session("c2", "chat", 3, gap=20.0, growth=100),  # 不可达
        "r1": make_session("r1", "random", 1, gap=0.0),  # 正确
        "a1": make_session("a1", "agentic", 5, gap=20.0, growth=100),  # 错判 chat
    }
    results = list(replay(sessions, ReplayConfig(structure="off")).values())
    cm = confusion_matrix(results)
    assert sum(map(sum, cm)) == 4
    assert cm[2][2] == 1  # chat -> chat
    assert cm[3][2] == 1  # agentic -> chat（没有 tool_call、间隔像 chat）
    prf = per_class_prf(cm)
    assert prf["chat"]["support"] == 2 and prf["chat"]["recall"] == 0.5
    assert ceiling(results, 5)["unreachable_by_class"]["chat"] == 1
    assert converged_turn(results[0]) >= 1
    assert accuracy_curve(results)[0]["n"] == 4


def test_trace_loader_backfills_tokens_and_prior_tool_call(tmp_path):
    csv_path = tmp_path / "trace.csv"
    csv_path.write_text(
        "task,phase,session_id,turn,prompt_tokens,cached_tokens,recompute_tokens,"
        "ttft_ms,e2e_ms,gap_since_prev_end_s,queue_time_ms,prefill_ms,ts\n"
        "t,p,alfred_1,1,120,0,120,10,20,,1,2,1000.0\n"
        "t,p,alfred_1,2,150,120,30,10,20,5.0,1,2,1003.0\n",
        encoding="utf-8",
    )
    data = tmp_path / "task.jsonl"
    data.write_text(
        '{"alfred_1": {"id": "alfred_1", "num_turns": 2,'
        ' "turn_1": {"input_ids": [1, 2, 3], "tool_call": {"action": "x"}},'
        ' "turn_2": {"input_ids": [1, 2, 3, 4], "tool_call": null}}}\n',
        encoding="utf-8",
    )
    index = load_task_sessions([data], only_ids=["alfred_1"])
    sessions = load_trace_sessions(csv_path, index)
    rec = sessions["alfred_1"]
    assert [t.prompt_len for t in rec.turns] == [120, 150]
    assert [t.start for t in rec.turns] == [1000.0, 1003.0]
    assert rec.turns[0].prior_tool_call is False
    assert rec.turns[1].prior_tool_call is True  # 上一轮的 tool_call 回填
    assert rec.turns[0].tokens == [1, 2, 3]
