"""R3 验收：并发窗口（同刻 k 条在飞、锁各自持到结束、竞态）。

钉住四件事：

1. **同刻到点不互相命中**：begin 只 match 不写树，所以前缀相同的并发请求
   会都 miss（racing），与生产一致；顺序模型（``max_inflight=1``）才会命中。
2. **名额上限**：``max_inflight=k`` 的派发时刻与完成后补齐的顺序。
3. **锁窗口跨在飞**：begin 之后、end 之前，命中的前缀 ``lock_ref>0`` 且不在
   ``evictable_leaves``；在此期间别的请求逼出的淘汰不能把它删掉。
4. **确定性**：同 seed 两次并发 run 指纹相同。

    PYTHONPATH=$PWD/python python -m pytest -q test/manual/test_sim_concurrency.py
"""

import os
import sys
from array import array

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sim.core.allocator import tree_tokens  # noqa: E402
from sim.core.config import SimConfig  # noqa: E402
from sim.simulator import Simulator  # noqa: E402
from sim.workload.arrival import TaskSpec  # noqa: E402
from sim.workload.service import ServiceModel  # noqa: E402
from sim.workload.trace import Session, Turn  # noqa: E402

# 100 tok/s prefill + 10 tok/s decode：50 token 的请求恰好 0.5s
SERVICE = ServiceModel(
    prefill_base_s=0.0, prefill_tokens_per_s=100.0, decode_tokens_per_s=10.0
)


def _spec(**kw) -> TaskSpec:
    base = dict(
        name="chat", arrival_mode="fixed", arrival_offset_s=0.0, arrival_step_s=0.0
    )
    base.update(kw)
    return TaskSpec(**base)


def _one(sid, prompt, out=0, task="chat") -> Session:
    return Session(sid, (Turn(t=0.0, prompt=array("q", prompt), out=out),), task_name=task)


def test_same_instant_requests_race():
    """同刻到点 + 同前缀：并发全 miss，顺序则后到者命中。"""

    prefix = list(range(64))
    sessions = [
        _one(f"apps_{i}", prefix + [900 + i] * 32) for i in range(3)
    ]

    concurrent = Simulator(SimConfig(capacity=4096)).run_closed_loop(
        sessions, [_spec()], service=SERVICE, max_inflight=0
    )
    assert [r.hit for r in concurrent.records] == [0, 0, 0]
    assert concurrent.summary["emit"]["inflight_peak"] == 3
    # 后两条各自把已存在的 64 token 前缀又写了一遍
    assert concurrent.summary["emit"]["duplicate_insert_tokens"] == 2 * 64
    assert concurrent.summary["emit"]["mode"] == "closed_loop_concurrent"

    sequential = Simulator(SimConfig(capacity=4096)).run_closed_loop(
        sessions, [_spec()], service=SERVICE, max_inflight=1
    )
    assert [r.hit for r in sequential.records] == [0, 64, 64]
    assert sequential.summary["emit"]["duplicate_insert_tokens"] == 0
    assert sequential.summary["emit"]["mode"] == "closed_loop_sequential"


def test_max_inflight_caps_dispatch():
    sessions = [_one(f"apps_{i}", [1000 + i] * 50) for i in range(5)]
    result = Simulator(SimConfig(capacity=4096)).run_closed_loop(
        sessions, [_spec()], service=SERVICE, max_inflight=2
    )
    # 0s 发两条 -> 0.5s 完成并补发两条 -> 1.0s 补最后一条
    assert [round(r.t, 3) for r in result.records] == [0.0, 0.0, 0.5, 0.5, 1.0]
    assert result.summary["emit"]["inflight_peak"] == 2
    assert result.summary["emit"]["send_lag_max_ms"] == pytest.approx(1000.0)


def test_lock_window_spans_inflight():
    prefix = array("q", range(160))
    sim = Simulator(SimConfig(capacity=4096))
    sim.lifecycle.prepare(width=176)
    sim.lifecycle.serve(Turn(t=0.0, prompt=prefix, out=0), "apps_x", task_name="chat")

    pending = sim.lifecycle.start(
        Turn(t=1.0, prompt=prefix + array("q", [7] * 16), out=0),
        "apps_a",
        task_name="chat",
    )
    try:
        assert pending.hit == 160
        assert pending.node.lock_ref == 1
        assert pending.node not in sim.cache.evictable_leaves
        # 在飞未落树：池账 == 树 + inflight
        assert sim.alloc.used == tree_tokens(sim.cache) + sim.inflight_tokens
        assert sim.inflight_tokens == 16
    finally:
        sim.lifecycle.finish(pending)

    assert pending.node.lock_ref == 0
    assert sim.inflight_tokens == 0
    assert sim.alloc.used == tree_tokens(sim.cache)


def test_concurrent_eviction_cannot_drop_locked_prefix():
    """A 在飞期间锁住 P；B 逼出的淘汰只能清 junk，P 必须活到下一轮命中。"""

    prefix = array("q", range(160))
    junk = array("q", range(5000, 5200))
    sim = Simulator(SimConfig(capacity=520, seed=1))
    sim.lifecycle.prepare(width=400)

    sim.lifecycle.serve(Turn(t=0.0, prompt=prefix, out=0), "apps_x", task_name="chat")
    sim.lifecycle.serve(Turn(t=0.5, prompt=junk, out=0), "apps_y", task_name="chat")
    assert tree_tokens(sim.cache) == 360

    a = sim.lifecycle.start(
        Turn(t=1.0, prompt=prefix + array("q", [7] * 16), out=0),
        "apps_a",
        task_name="chat",
    )
    b = sim.lifecycle.start(
        Turn(t=1.0, prompt=array("q", range(9000, 9200)), out=0),
        "apps_b",
        task_name="chat",
    )
    try:
        assert b.evicted >= 56, f"B 应当逼出淘汰，实际 {b.evicted}"
    finally:
        sim.lifecycle.finish(a)
        sim.lifecycle.finish(b)

    # P 仍在树上：再请求一次应命中 160
    record = sim.lifecycle.serve(Turn(t=2.0, prompt=prefix, out=0), "apps_c", task_name="chat")
    assert record.hit == 160


def test_concurrent_runs_are_deterministic():
    sessions = []
    for i in range(12):
        prompt = list(range(i % 3 * 40, i % 3 * 40 + 40)) + [2000 + i] * 20
        turns = (
            Turn(t=0.0, prompt=array("q", prompt), out=8),
            Turn(t=0.0, prompt=array("q", prompt + [9] * 8 + [3000 + i] * 10), out=8),
        )
        sessions.append(Session(f"ShareGPT-{i}", turns, task_name="chat"))

    spec = _spec(
        lambda_rate=2.0,
        mu=1.0,
        sigma=0.5,
        seed=5,
        arrival_mode="poisson",
    )
    runs = [
        Simulator(SimConfig(capacity=2048, seed=7)).run_closed_loop(
            sessions, [spec], service=SERVICE, max_inflight=0
        ).fingerprint()
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
