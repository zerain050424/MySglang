"""Acceptance for the SGLang-style engine-step driver (``sim.engine``).

The tests pin the behaviour that distinguishes engine steps from the earlier
event-by-event driver:

1. one step admits a whole prefill batch up to the token budget;
2. same-batch requests cannot hit each other's freshly written prefix;
3. a later batch can hit the prefix written when the previous batch finished;
4. ``chunked_prefill_size`` splits one request across steps;
5. ``max_running_requests`` / ``max_prefill_bs`` gate prefill admission.

Run with::

    PYTHONPATH=$PWD/python python -m pytest -q test/manual/test_sim_engine.py
"""

import os
import sys
from array import array

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sim.engine import EngineConfig
from sim.runner import SimConfig, Simulator
from sim.schedule import ServiceModel, TaskSpec
from sim.trace import Session, Turn

# 100 tok/s prefill and 10 tok/s decode keep the expected wall times integral.
SERVICE = ServiceModel(
    prefill_base_s=0.0, prefill_tokens_per_s=100.0, decode_tokens_per_s=10.0
)


def _spec(**kw) -> TaskSpec:
    base = {
        "name": "chat",
        "arrival_mode": "fixed",
        "arrival_offset_s": 0.0,
        "arrival_step_s": 0.0,
    }
    base.update(kw)
    return TaskSpec(**base)


def _session(sid: str, prompt, out: int = 0) -> Session:
    return Session(
        sid, (Turn(t=0.0, prompt=array("q", prompt), out=out),), task_name="chat"
    )


def _run(sessions, scheduler: EngineConfig, **kw):
    return Simulator(SimConfig(capacity=4096, seed=1)).run_engine(
        sessions,
        [_spec()],
        scheduler=scheduler,
        service=SERVICE,
        arrival_scale=0.0,
        **kw,
    )


def test_step_admits_batch_up_to_token_budget():
    """SGLang commits the request that drains the budget, then stops."""

    sessions = [
        _session(f"ShareGPT-{i}", range(1000 * i, 1000 * i + 100)) for i in range(4)
    ]
    result = _run(sessions, EngineConfig(max_prefill_tokens=250))

    emit = result.summary["emit"]
    assert emit["mode"] == "engine_step"
    assert emit["prefill_steps"] == 2
    assert emit["prefill_requests"] == 4
    # 100 + 100 + 100 crosses 250; the third is committed before budget_state
    # returns OTHER, which is exactly what PrefillAdder.add_one_req does.
    assert emit["max_prefill_batch_tokens"] == 300
    assert emit["max_prefill_batch_size"] == 3
    assert [round(r.t, 3) for r in result.records] == [0.0, 0.0, 0.0, 3.0]
    assert emit["wall_s"] == pytest.approx(4.0)


def test_same_batch_requests_race_on_shared_prefix():
    prefix = list(range(64))
    sessions = [_session(f"apps_{i}", prefix + [900 + i] * 32) for i in range(3)]
    result = _run(sessions, EngineConfig(max_prefill_tokens=4096))

    assert [r.hit for r in result.records] == [0, 0, 0]
    emit = result.summary["emit"]
    assert emit["prefill_steps"] == 1
    assert emit["prefill_requests"] == 3
    # The two later inserts rediscover the same 64-token prefix.
    assert emit["duplicate_insert_tokens"] == 2 * 64
    assert emit["inflight_token_peak"] == 3 * 96


def test_next_batch_hits_prefix_written_by_previous_batch():
    prefix = list(range(64))
    sessions = [_session(f"apps_{i}", prefix + [900 + i] * 32) for i in range(3)]
    result = _run(sessions, EngineConfig(max_prefill_tokens=100))

    assert [r.hit for r in result.records] == [0, 0, 64]
    emit = result.summary["emit"]
    assert emit["prefill_steps"] == 2
    # first batch: 2 requests * 96 tokens; second batch: 32 new tokens.
    assert emit["max_prefill_batch_tokens"] == 192


def test_chunked_prefill_spans_steps():
    sessions = [_session("ShareGPT-0", range(300))]
    result = _run(
        sessions,
        EngineConfig(max_prefill_tokens=1000, chunked_prefill_size=128),
    )

    emit = result.summary["emit"]
    assert emit["chunked_requests"] == 1
    assert emit["prefill_steps"] == 3  # 128 + 128 + 44
    assert emit["max_prefill_batch_tokens"] == 128
    assert emit["wall_s"] == pytest.approx(3.0)
    assert result.records[0].prompt_len == 300


def test_max_running_requests_gates_prefill():
    sessions = [
        _session(f"ShareGPT-{i}", range(1000 * i, 1000 * i + 100), out=10)
        for i in range(3)
    ]
    result = _run(
        sessions,
        EngineConfig(max_prefill_tokens=4096, max_running_requests=1),
    )

    emit = result.summary["emit"]
    assert emit["prefill_steps"] == 3
    assert emit["decode_steps"] == 30
    assert emit["running_peak"] == 1
    # Each request is prefilled (1.0s) and decoded (10 * 0.1s) before the next
    # one starts.
    assert [round(r.t, 3) for r in result.records] == [0.0, 2.0, 4.0]


def test_max_prefill_bs_limits_one_step():
    sessions = [
        _session(f"ShareGPT-{i}", range(1000 * i, 1000 * i + 100)) for i in range(4)
    ]
    result = _run(
        sessions,
        EngineConfig(max_prefill_tokens=4096, max_prefill_bs=2),
    )

    emit = result.summary["emit"]
    assert emit["prefill_steps"] == 2
    assert emit["max_prefill_batch_size"] == 2


# ---------------------------------------------------------------------------
# LPM 的双重匹配：生产匹配两次，但 rid 相同
# ---------------------------------------------------------------------------
def _spec_named(name: str, offset: float) -> TaskSpec:
    return TaskSpec(
        name=name, arrival_mode="fixed", arrival_offset_s=offset, arrival_step_s=0.0
    )


def _lpm_warmup_run(schedule_policy: str):
    """t=0 一条预热请求写 64 token 前缀，t=10 三条冷请求同时命中。

    LPM 每步对每条 waiting **匹配两次**（`_compute_prefix_matches` 排序一次，
    `init_next_round_input` 准入一次）——生产也是两次，但两次用的是**同一个
    ``Req`` 对象，rid 相同**，所以 `SessionRegistry.observe_request` 与
    `CacheTelemetry.on_hit` 的 rid 去重会吸收第二次。仿真若每次现编 rid，
    这两处去重就失效了。
    """

    shared = list(range(64))
    warm = Session(
        "warm-0",
        (Turn(t=0.0, prompt=array("q", shared + [9000]), out=0),),
        task_name="warm",
    )
    cold = [
        Session(
            f"chat-{i}",
            (Turn(t=0.0, prompt=array("q", shared + [9100 + i] * 4), out=0),),
            task_name="chat",
        )
        for i in range(3)
    ]
    sim = Simulator(SimConfig(capacity=4096, seed=1))
    result = sim.run_engine(
        [warm, *cold],
        [_spec_named("warm", 0.0), _spec_named("chat", 10.0)],
        scheduler=EngineConfig(
            max_prefill_tokens=4096, schedule_policy=schedule_policy
        ),
        service=SERVICE,
    )
    return sim, result


@pytest.mark.parametrize("policy", ["fcfs", "lpm"])
def test_lpm_matches_each_request_once_per_rid(policy):
    """LPM 的排序匹配不能算成第二个请求：分类证据与命中账本各只记一次。"""

    sim, result = _lpm_warmup_run(policy)

    # 命中本身与排序策略无关：预热 miss，三条冷请求各命中 64。
    assert [r.hit for r in result.records] == [0, 64, 64, 64]

    # 症状 1：CacheTelemetry.on_hit 按 rid 去重 -> 同一请求不能计两次命中。
    queue_hit = sum(result.summary["queue_hit"].values())
    assert queue_hit == 3 * 64, (
        f"{policy}: queue_hit={queue_hit}，期望 192"
        "（LPM 的排序匹配用了新 rid，去重失效）"
    )

    # 症状 2：observe_request 按 rid 去重 -> turn_count 与轮间 gap 都不能翻倍。
    registry = sim.cache.session_registry
    for sid in ("warm-0", "chat-0", "chat-1", "chat-2"):
        state = registry.get(sid)
        assert state is not None, f"{sid} 没有会话账本"
        assert state.turn_count == 1, (
            f"{policy}: {sid} turn_count={state.turn_count}，期望 1"
        )
        # 第二次观察的 now 与第一次相同；若没去重会塞进一个 0 秒 gap，
        # 把 CHAT/AGENTIC 的 GAP_SPLIT_S=15s 判定拉偏。
        assert list(state.classifier_gaps) == [], (
            f"{policy}: {sid} classifier_gaps={list(state.classifier_gaps)}，期望空"
        )
