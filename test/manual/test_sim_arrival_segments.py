"""``arrival.mode="segments"``（分段常数强度的非齐次 Poisson）回归。

构成漂移实验（`drift3`）靠它把同一个 task 的注入速率按时段改掉：struct / chat /
agentic 轮流做主导，`trans` 背景速率是唯一可控变量。这里只测到达流本身——
策略与缓存行为另有测试。

    PYTHONPATH=$PWD:$PWD/python python -m pytest -q test/manual/test_sim_arrival_segments.py
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, ROOT)

schedule = pytest.importorskip("sim.schedule")
TaskSpec = schedule.TaskSpec
check_segments = schedule.check_segments


def _spec(segments, *, seed=42, task_id=0):
    return TaskSpec.from_config(
        {
            "name": "T",
            "seed": seed,
            "task_id": task_id,
            "arrival": {
                "mode": "segments",
                "segments": [
                    {"until_s": until, "lambda_rate": rate} for until, rate in segments
                ],
            },
        }
    )


def _offsets(spec, n):
    # _arrival_offsets 只依赖 spec，不需要会话；借 Emitter 的未绑定方法。
    return schedule.Emitter._arrival_offsets(None, spec, n)


# ---------------------------------------------------------------------------
# 语义
# ---------------------------------------------------------------------------
def test_rate_matches_each_segment():
    """每段的经验速率必须贴近配置（±25%，泊松样本量 400 足够）。"""

    segments = [(100.0, 0.5), (200.0, 2.0)]
    spec = _spec(segments)
    n = 400
    offsets = _offsets(spec, n)
    assert len(offsets) == n
    assert offsets == sorted(offsets), "偏移必须升序"

    first = [t for t in offsets if t < 100.0]
    second = [t for t in offsets if 100.0 <= t < 200.0]
    # 期望 0.5*100=50 与 2.0*100=200，各留 ±25%
    assert 0.75 * 50 <= len(first) <= 1.25 * 50, len(first)
    assert 0.75 * 200 <= len(second) <= 1.25 * 200, len(second)


def test_segment_boundary_is_respected():
    """λ 变大后段内不再残留慢速率：第 2 段单位时间到达数明显多于第 1 段。"""

    spec = _spec([(100.0, 0.2), (200.0, 4.0)])
    offsets = _offsets(spec, 600)
    first = len([t for t in offsets if t < 100.0])
    second = len([t for t in offsets if 100.0 <= t < 200.0])
    assert second > 4 * first, (first, second)


def test_zero_rate_segment_is_silent():
    """λ=0 的段不产生任何到达（构成漂移里的"该相不注入"）。"""

    spec = _spec([(100.0, 1.0), (200.0, 0.0), (300.0, 1.0)])
    offsets = _offsets(spec, 300)
    assert not [t for t in offsets if 100.0 <= t < 200.0], "静默段出现了到达"


def test_continuation_after_last_segment():
    """会话数用不完时，时间轴结束后沿用最后强度继续放，不会卡死。"""

    spec = _spec([(10.0, 1.0)])
    offsets = _offsets(spec, 200)
    assert len(offsets) == 200
    assert max(offsets) > 10.0
    tail = [t for t in offsets if t >= 10.0]
    rate = len(tail) / (max(tail) - 10.0)
    assert 0.7 <= rate <= 1.4, rate


def test_deterministic_for_same_seed():
    a = _offsets(_spec([(100.0, 0.5), (200.0, 2.0)], seed=42), 200)
    b = _offsets(_spec([(100.0, 0.5), (200.0, 2.0)], seed=42), 200)
    c = _offsets(_spec([(100.0, 0.5), (200.0, 2.0)], seed=43), 200)
    assert a == b, "同 seed 必须逐位一致"
    assert a != c, "换 seed 必须不同（否则不是独立抽样）"


def test_task_id_changes_stream():
    a = _offsets(_spec([(100.0, 0.5)], seed=42, task_id=0), 100)
    b = _offsets(_spec([(100.0, 0.5)], seed=42, task_id=4), 100)
    assert a != b


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def test_validation_errors():
    with pytest.raises(ValueError, match="非空"):
        check_segments(())
    with pytest.raises(ValueError, match="严格大于"):
        check_segments(((100.0, 1.0), (100.0, 1.0)))
    with pytest.raises(ValueError, match="不能为负"):
        check_segments(((100.0, -1.0),))
    with pytest.raises(ValueError, match="全为 0"):
        check_segments(((100.0, 0.0), (200.0, 0.0)))
    # 通过 from_config 也要拦
    with pytest.raises(ValueError, match="严格大于"):
        TaskSpec.from_config(
            {
                "name": "T",
                "arrival": {
                    "mode": "segments",
                    "segments": [
                        {"until_s": 100, "lambda_rate": 1},
                        {"until_s": 50, "lambda_rate": 1},
                    ],
                },
            }
        )


def test_default_mode_unchanged():
    """不写 arrival 时仍是常数 λ 的 Poisson（不能悄悄改成 segments）。"""

    spec = TaskSpec.from_config({"name": "T"})
    assert spec.arrival_mode == "poisson"
    assert spec.segments == ()
    assert len(_offsets(spec, 50)) == 50
