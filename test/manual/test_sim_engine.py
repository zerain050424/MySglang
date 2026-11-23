"""Acceptance for the SGLang-style engine-step driver (``sim.drivers.engine``).

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

from sim.core.config import SimConfig
from sim.drivers.engine import EngineConfig
from sim.simulator import Simulator
from sim.workload.arrival import TaskSpec
from sim.workload.service import ServiceModel
from sim.workload.trace import Session, Turn

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


def test_prefill_prefix_visible_while_request_is_decoding():
    """Prefill 完成后进入 decode 前即增量落树，同刻或稍后到达的请求可直接命中。"""
    prefix = list(range(64))
    sessions = [
        _session("apps_0", prefix + [900] * 36, out=50),
        _session("apps_1", prefix + [901] * 36, out=0),
    ]
    spec = _spec(arrival_mode="fixed", arrival_offset_s=0.0, arrival_step_s=1.0)

    progressive = Simulator(SimConfig(capacity=4096, seed=1)).run_engine(
        sessions,
        [spec],
        scheduler=EngineConfig(max_prefill_tokens=4096, progressive_cache=True),
        service=SERVICE,
        arrival_scale=1.0,
    )
    assert [r.hit for r in progressive.records] == [64, 0] or [
        (r.session_id, r.hit) for r in progressive.records
    ] == [("apps_1", 64), ("apps_0", 0)]
    assert progressive.summary["emit"]["duplicate_insert_tokens"] == 0

    legacy = Simulator(SimConfig(capacity=4096, seed=1)).run_engine(
        sessions,
        [spec],
        scheduler=EngineConfig(max_prefill_tokens=4096, progressive_cache=False),
        service=SERVICE,
        arrival_scale=1.0,
    )
    assert [(r.session_id, r.hit) for r in legacy.records] == [
        ("apps_1", 0),
        ("apps_0", 0),
    ]
    assert legacy.summary["emit"]["duplicate_insert_tokens"] == 64


def test_chunked_prefill_commits_intermediate_chunks_to_tree():
    """Chunked prefill 每完成一块即提交前缀到树并保持持锁，后续请求可直接命中已算块。"""

    sessions = [
        _session("apps_0", list(range(256)), out=0),
        _session("apps_1", list(range(128)) + [999] * 32, out=0),
    ]
    spec = _spec(arrival_mode="fixed", arrival_offset_s=0.0, arrival_step_s=1.28)
    result = Simulator(SimConfig(capacity=4096, seed=1)).run_engine(
        sessions,
        [spec],
        scheduler=EngineConfig(max_prefill_tokens=128, chunked_prefill_size=128),
        service=SERVICE,
        arrival_scale=1.0,
    )
    assert [(r.session_id, r.hit) for r in result.records] == [
        ("apps_0", 0),
        ("apps_1", 128),
    ]
    assert result.summary["emit"]["duplicate_insert_tokens"] == 0

    # 验证单步 chunked prefill 完成第 1 块后，树里已有 128 token 且被锁定保护
    from sim.core.allocator import tree_tokens
    from sim.drivers.engine import EngineScheduler

    sim = Simulator(SimConfig(capacity=4096, seed=1))
    lc = sim.lifecycle
    lc.prepare(width=256)
    single = [_session("apps_0", list(range(256)), out=0)]
    eng = EngineScheduler(
        lc,
        single,
        [_spec()],
        scheduler=EngineConfig(max_prefill_tokens=128, chunked_prefill_size=128),
        service=SERVICE,
    )
    del eng
    matched = lc.match(single[0].turns[0], "apps_0", rid="r0", task_name="chat")
    lc.lock(matched)
    pending = lc.new_pending(matched, skipped=False, evicted=0)
    lc.reserve_upto(pending, 128)
    lc.commit_upto(pending, 128, chunked=True)
    assert tree_tokens(sim.cache) == 128
    assert sim.cache.protected_size() == 128
    assert sim.inflight_tokens == 0
    lc.finish(pending)
    assert tree_tokens(sim.cache) == 256
    assert sim.cache.protected_size() == 0


def test_lazy_decode_alloc_defers_output_allocation_and_eviction_to_decode_steps():
    """准入时仅占 prompt 槽位；output 槽位与触发的淘汰推迟到 decode 步进时刻发生。"""
    from sim.drivers.engine import EngineScheduler

    sessions = [
        _session("apps_0", list(range(100)), out=0),
        _session("apps_1", list(range(500, 580)), out=40),
    ]
    spec = _spec(arrival_mode="fixed", arrival_offset_s=0.0, arrival_step_s=1.0)

    # 1) lazy_decode_alloc=True（默认）：s1 prefill 结束时仅用 100+80=180<=200，不淘汰 s0
    sim_lazy = Simulator(SimConfig(capacity=200, seed=1))
    evict_times_lazy = []
    orig_prepare = sim_lazy.lifecycle.prepare

    def hooked_prepare(width):
        orig_prepare(width)
        real_evict = sim_lazy.cache.evict

        def _spy(params):
            evict_times_lazy.append(round(sim_lazy.clock(), 3))
            return real_evict(params)

        sim_lazy.cache.evict = _spy

    sim_lazy.lifecycle.prepare = hooked_prepare
    res_lazy = EngineScheduler(
        sim_lazy.lifecycle,
        sessions,
        [spec],
        scheduler=EngineConfig(max_prefill_tokens=4096, lazy_decode_alloc=True),
        service=SERVICE,
    ).run(arrival_scale=1.0, wait_scale=1.0)

    assert res_lazy.records[1].session_id == "apps_1"
    assert res_lazy.records[1].evicted == 100
    assert res_lazy.records[1].used == 120
    assert res_lazy.records[1].arrival_t == pytest.approx(1.0)
    # s1 在 t=1.0 开始 prefill，t=1.8 prefill 结束；21 个 decode token 后于 t=3.9 才越界淘汰
    assert evict_times_lazy and all(t > 1.8 for t in evict_times_lazy)

    # 2) 对照 lazy_decode_alloc=False：在 s1 准入瞬间 t=1.0 就立刻触发淘汰
    sim_eager = Simulator(SimConfig(capacity=200, seed=1))
    evict_times_eager = []
    orig_prepare_e = sim_eager.lifecycle.prepare

    def hooked_prepare_e(width):
        orig_prepare_e(width)
        real_evict = sim_eager.cache.evict

        def _spy(params):
            evict_times_eager.append(round(sim_eager.clock(), 3))
            return real_evict(params)

        sim_eager.cache.evict = _spy

    sim_eager.lifecycle.prepare = hooked_prepare_e
    EngineScheduler(
        sim_eager.lifecycle,
        sessions,
        [spec],
        scheduler=EngineConfig(max_prefill_tokens=4096, lazy_decode_alloc=False),
        service=SERVICE,
    ).run(arrival_scale=1.0, wait_scale=1.0)
    assert evict_times_eager == [1.0]
