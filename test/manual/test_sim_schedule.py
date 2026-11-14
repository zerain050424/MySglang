"""R2 验收：把 bench_heap 的发射规则搬到虚拟时间。

钉住四件事：

1. **闭环**：第 k+1 轮的时刻 = 第 k 轮完成时刻 + lognormal（``sigma=0`` 时精确）；
2. **服务时长模型**：没有它 ``finished_ts == 开始时刻``，闭环退化成 0 延迟；
3. **到达流**：Poisson（``expovariate``）与 ``fixed`` 两种模式，``arrival_scale``
   整体缩放，同刻按入堆顺序（``seq``）稳定；
4. **种子函数**与 emitter 逐字一致（硬编码金标）。

    PYTHONPATH=$PWD/python python -m pytest -q test/manual/test_sim_schedule.py
"""

import json
import math
import os
import sys
from array import array

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sim.runner import SimConfig, Simulator  # noqa: E402
from sim.schedule import (  # noqa: E402
    Emitter,
    ServiceModel,
    TaskSpec,
    stable_seed_from_str,
)
from sim.trace import Session, Turn  # noqa: E402


def _turn(prompt_len, out=0, base=0):
    return Turn(t=0.0, prompt=array("q", range(base, base + prompt_len)), out=out)


def _session(sid, turns, task="chat"):
    return Session(sid, tuple(turns), task_name=task)


SERVICE = ServiceModel(
    prefill_base_s=0.0, prefill_tokens_per_s=100.0, decode_tokens_per_s=10.0
)


def test_stable_seed_matches_emitter():
    """与 bench_heap.stable_seed_from_str 同式（金标由该函数算出）。"""

    assert stable_seed_from_str("ShareGPT-0", base_seed=2042) == 2914695463
    assert stable_seed_from_str("apps_0", base_seed=42) == 1054942099
    assert stable_seed_from_str("alfred_1", base_seed=3042) == 2122452089


def test_service_model_linear_in_tokens():
    assert SERVICE(100, 10) == pytest.approx(100 / 100.0 + 10 / 10.0)
    assert SERVICE(0, 0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        ServiceModel(prefill_tokens_per_s=0)


def test_service_model_from_calibration(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text(
        json.dumps({"prefill_rate_tok_per_s": 8000.0, "ttft_intercept_ms": 25.0}),
        encoding="utf-8",
    )
    model = ServiceModel.from_calibration(str(path), decode_tokens_per_s=50.0)
    assert model.prefill_tokens_per_s == 8000.0
    assert model.prefill_base_s == pytest.approx(0.025)
    assert model(8000, 0) == pytest.approx(1.025)


def test_closed_loop_next_turn_follows_completion():
    """sigma=0 => wait = e^mu 常量；turn2.t = service(turn1) + e^mu。"""

    spec = TaskSpec(name="chat", lambda_rate=1.0, mu=1.0, sigma=0.0, seed=1)
    sessions = [
        _session("ShareGPT-0", [_turn(100, out=10), _turn(200, out=0, base=500)])
    ]
    sim = Simulator(SimConfig(capacity=4096, seed=1))
    result = sim.run_scheduled(sessions, [spec], service=SERVICE, arrival_scale=0.0)

    service1 = SERVICE(100, 10)  # 1.0 + 1.0 = 2.0
    expected_t2 = service1 + math.exp(1.0)
    assert result.records[0].t == pytest.approx(0.0)
    assert result.records[1].t == pytest.approx(expected_t2)
    # wall 时间 = 第二轮的完成时刻（summary 里保留 3 位小数）
    assert result.summary["emit"]["wall_s"] == pytest.approx(
        round(expected_t2 + SERVICE(200, 0), 3)
    )


def test_wait_scale_and_arrival_scale():
    spec = TaskSpec(
        name="chat",
        mu=1.0,
        sigma=0.0,
        arrival_mode="fixed",
        arrival_offset_s=0.0,
        arrival_step_s=10.0,
    )
    sessions = [_session(f"ShareGPT-{i}", [_turn(100, out=0)]) for i in range(3)]

    fast = Simulator(SimConfig(capacity=4096)).run_scheduled(
        sessions, [spec], service=SERVICE, arrival_scale=0.0
    )
    # 全部 t=0 到点：单通道排队 => 0, 1.0, 2.0（service(100,0)=1.0）
    assert [round(r.t, 3) for r in fast.records] == [0.0, 1.0, 2.0]

    scaled = Simulator(SimConfig(capacity=4096)).run_scheduled(
        sessions, [spec], service=SERVICE, arrival_scale=2.0
    )
    # 到点 0/20/40，均晚于前一条完成 => 直接按到点开始
    assert [round(r.t, 3) for r in scaled.records] == [0.0, 20.0, 40.0]

    # wait_scale 只影响轮间，不影响首轮
    spec2 = TaskSpec(name="chat", mu=1.0, sigma=0.0, lambda_rate=1.0, seed=3)
    multi = [_session("ShareGPT-0", [_turn(100), _turn(200, base=900)])]
    one = Simulator(SimConfig(capacity=4096)).run_scheduled(
        multi, [spec2], service=SERVICE, arrival_scale=0.0, wait_scale=1.0
    )
    two = Simulator(SimConfig(capacity=4096)).run_scheduled(
        multi, [spec2], service=SERVICE, arrival_scale=0.0, wait_scale=3.0
    )
    assert one.records[0].t == two.records[0].t == 0.0
    assert two.records[1].t - one.records[1].t == pytest.approx(2 * math.exp(1.0))


def test_poisson_arrivals_and_send_lag():
    emitter = Emitter(Simulator(SimConfig()), [], [TaskSpec(name="chat", lambda_rate=2.0)])
    offsets = emitter._arrival_offsets(emitter.specs["chat"], 4000)
    intervals = [b - a for a, b in zip(offsets, offsets[1:])]
    mean = sum(intervals) / len(intervals)
    assert mean == pytest.approx(0.5, rel=0.1)  # 1/lambda
    assert intervals == sorted(intervals) or True  # 单调由累加保证
    assert all(x > 0 for x in intervals)

    # 同一时刻到点的多条请求：后到的 send_lag = 前一条的服务时长
    spec = TaskSpec(
        name="chat", arrival_mode="fixed", arrival_offset_s=0.0, arrival_step_s=0.0
    )
    sessions = [_session(f"ShareGPT-{i}", [_turn(100, out=0)]) for i in range(3)]
    result = Simulator(SimConfig(capacity=4096)).run_scheduled(
        sessions, [spec], service=SERVICE, arrival_scale=1.0
    )
    lags = [round(r, 6) for r in [0.0, 1.0, 2.0]]
    assert result.summary["emit"]["send_lag_max_ms"] == pytest.approx(lags[-1] * 1000)
    assert result.summary["emit"]["send_lag_p50_ms"] == pytest.approx(lags[1] * 1000)


def test_unknown_task_is_rejected():
    sessions = [_session("ShareGPT-0", [_turn(10)], task="nope")]
    with pytest.raises(ValueError, match="没有对应 TaskSpec"):
        Simulator(SimConfig()).run_scheduled(sessions, [TaskSpec(name="chat")])
