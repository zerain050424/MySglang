"""堆发射器的纯逻辑单测（不需要 GPU / torch / 网络）。

被测对象：`Heter-Bench/scripts/bench_heap.py`
  ArrivalHeap.pop_due      —— 只弹到期事件、按 (due_ts, seq) 排序
  _resolve_prompt_ids      —— 返回的 prompt 恒等于数据集的完整历史

两条不变量各自防一类静默退化：

1. 同刻事件的弹出顺序由压入序号决定。若把它换成 `time.monotonic()` 或集合迭代，
   同一 seed 的到达流不再可复现，而实验里所有臂的对比都依赖「同一份到达流」。
2. 客户端拼接的历史只在**与数据集逐 token 相等**时才可用。Llama 模板在 response
   之前插入 assistant 头，response_ids 不是字面后缀；此时若仍返回拼接结果，
   前缀共享会无声失效（缓存命中率塌陷，而请求本身照常成功）。
"""

import os
import sys

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "Heter-Bench", "scripts"
)
sys.path.insert(0, _SCRIPTS)

import bench_heap  # noqa: E402


def _turn(session_id, turn_idx=1):
    return bench_heap.PendingTurn(
        task_name="t",
        task_cfg={"name": "t"},
        session={"id": session_id},
        turn_idx=turn_idx,
    )


def test_pop_due_only_returns_due_events_in_time_order():
    heap = bench_heap.ArrivalHeap()
    for due, sid in ((30.0, "c"), (10.0, "a"), (20.0, "b")):
        heap.push(due_ts=due, turn=_turn(sid))

    assert heap.next_due_ts == 10.0
    assert [s.session["id"] for _, s in heap.pop_due(now=20.0)] == ["a", "b"]
    assert len(heap) == 1, "未到期的 c 必须留在堆里"
    assert [s.session["id"] for _, s in heap.pop_due(now=30.0)] == ["c"]
    assert not heap


def test_same_timestamp_pops_in_push_order():
    heap = bench_heap.ArrivalHeap()
    for sid in ("first", "second", "third"):
        heap.push(due_ts=5.0, turn=_turn(sid))

    assert [s.session["id"] for _, s in heap.pop_due(now=5.0)] == [
        "first",
        "second",
        "third",
    ]


def test_earlier_event_pushed_later_becomes_next_due():
    """主循环靠这个性质决定要不要提前醒来：新压入的事件可以早于原堆顶。"""
    heap = bench_heap.ArrivalHeap()
    heap.push(due_ts=100.0, turn=_turn("far"))
    heap.push(due_ts=1.0, turn=_turn("near"))

    assert heap.next_due_ts == 1.0
    assert [s.session["id"] for _, s in heap.pop_due(now=1.0)] == ["near"]


def test_resolve_prompt_uses_dataset_history_when_response_is_literal_suffix():
    # Qwen 式：assistant 头在 response 之前，response_ids 是字面后缀
    session = {
        "num_turns": 2,
        "turn_1": {"input_ids": [1, 10], "response_ids": [11, 12]},
        "turn_2": {"input_ids": [1, 10, 11, 12, 20]},
    }

    ids, concat_ok = bench_heap._resolve_prompt_ids(session=session, turn_idx=2)

    assert concat_ok is True
    assert ids == [1, 10, 11, 12, 20]


def test_resolve_prompt_falls_back_to_dataset_when_template_interleaves_header():
    # Llama 式：prompt + response 之后还有 assistant 头，两个序列不共享前缀
    session = {
        "num_turns": 2,
        "turn_1": {"input_ids": [1, 10], "response_ids": [11, 12]},
        "turn_2": {"input_ids": [1, 10, 91, 92, 11, 12, 20]},
    }

    ids, concat_ok = bench_heap._resolve_prompt_ids(session=session, turn_idx=2)

    assert concat_ok is False, "拼接与数据集不一致时必须报告"
    assert ids == session["turn_2"]["input_ids"], "不得发送与数据集不同的 token 序列"
