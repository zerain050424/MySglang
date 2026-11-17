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
