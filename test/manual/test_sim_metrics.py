"""Acceptance for compute-saving metrics, cost models and skipped requests.

PrefixSim showed that block hit ratio and prefill compute saving can rank
policies differently. These tests pin our version of the metric:

1. ``uniform`` cost makes compute saving equal hit rate;
2. ``position`` / ``linear`` weight deeper blocks more;
3. ``page_size`` turns hit tokens into hit blocks;
4. skipped requests are excluded from hit-rate denominators;
5. every normal run keeps the self-eviction diagnostic at zero.

Run with::

    PYTHONPATH=$PWD/python python -m pytest -q test/manual/test_sim_metrics.py
"""

import os
import sys
from array import array

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sim.cost import CostModel  # noqa: E402
from sim.runner import SimConfig, Simulator  # noqa: E402
from sim.trace import Session, Turn  # noqa: E402


def _tok(base: int, n: int) -> array:
    return array("q", range(base, base + n))


def _two_turns(page_size: int = 1):
    first = _tok(1000, 16)
    second = first + _tok(2000, 16)
    return [
        Session(
            "ShareGPT-0",
            (Turn(0.0, first, out=0), Turn(10.0, second, out=0)),
            task_name="chat",
        )
    ]


@pytest.mark.parametrize("page_size", [1, 8])
def test_uniform_cost_saving_equals_hit_rate(page_size):
    prompt = _tok(1000, 10)
    sessions = [
        Session(
            "ShareGPT-0",
            (Turn(0.0, prompt, out=0), Turn(10.0, prompt, out=0)),
            task_name="chat",
        )
    ]
    result = Simulator(
        SimConfig(capacity=4096, page_size=page_size, cost_model="uniform")
    ).run(sessions)

    assert result.compute_saving_ratio == pytest.approx(result.hit_rate)
    assert result.summary["compute_saving_ratio"] == pytest.approx(result.hit_rate)


def test_position_cost_weights_deeper_hits():
    result = Simulator(
        SimConfig(capacity=4096, page_size=1, cost_model="position")
    ).run(_two_turns())

    second = result.records[1]
    assert second.total_cost == sum(range(1, 33))
    assert second.saved_cost == sum(range(1, 17))
    assert second.saved_cost / second.total_cost == pytest.approx(136 / 528)


def test_page_size_weights_tokens_by_block_cost():
    first = _tok(1000, 8)
    second = first + _tok(2000, 8)
    sessions = [
        Session(
            "ShareGPT-0",
            (Turn(0.0, first, out=0), Turn(10.0, second, out=0)),
            task_name="chat",
        )
    ]
    result = Simulator(
        SimConfig(capacity=4096, page_size=8, cost_model="position")
    ).run(sessions)

    second = result.records[1]
    assert second.hit == 8
    assert second.total_cost == 8 * 1 + 8 * 2
    assert second.saved_cost == 8 * 1


def test_cost_model_parse():
    assert CostModel.parse("uniform").cost(10) == 1
    assert CostModel.parse("position").cost(3) == 4
    assert CostModel.parse("linear:865,2").cost(100) == 1065
    assert CostModel.parse("qwen3coder30b_blksz_16").cost(100) == 1065


def test_table_cost_model(tmp_path):
    path = tmp_path / "costs.txt"
    path.write_text("# measured per-block costs\n10\n20\n", encoding="utf-8")
    model = CostModel.parse(f"table:{path}")
    assert model.cost(0) == 10
    assert model.cost(1) == 20
    assert model.cost(50) == 20


def test_invalid_cost_models_raise(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("# no values\n", encoding="utf-8")
    for spec in ("nope", "linear:0,1", "linear:1,-1", f"table:{empty}"):
        with pytest.raises(ValueError):
            CostModel.parse(spec)


def test_skipped_requests_do_not_enter_rates():
    sessions = [
        Session("apps_0", (Turn(0.0, _tok(0, 500), out=0),), task_name="chat"),
        Session("apps_1", (Turn(0.0, _tok(1000, 100), out=0),), task_name="chat"),
    ]
    result = Simulator(
        SimConfig(capacity=100, seed=1, page_size=1, oversize="skip")
    ).run(sessions)

    assert [record.skipped for record in result.records] == [True, False]
    assert result.summary["requests"] == 1
    assert result.summary["total_requests"] == 2
    assert result.summary["skipped_requests"] == 1
    assert result.summary["skipped_prompt_tokens"] == 500
    assert result.summary["prompt_tokens"] == 100
    assert result.requests == 1
    assert result.skipped_requests == 1


def test_self_eviction_diagnostics_are_clean():
    result = Simulator(SimConfig(capacity=4096, page_size=1)).run(_two_turns())
    assert result.summary["self_evictions"] == 0
    assert result.summary["unexpected_evictions"] == 0
    assert result.summary["verify"] is True
