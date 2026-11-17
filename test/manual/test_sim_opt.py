"""OPT 上界（落点 A）的回归测试。

两部分：
1. **策略语义**（不需要 torch）：优先级 = `-下次使用时刻`，无未来 → `-inf`
   最先淘汰；真值来自 use map 的二分查询，策略必须**只读无状态**。
2. **与缓存的接线**（需要 torch，base conda 下自动 skip）：use map 由
   `sim.opt_map.build_use_map` 生成，hash 必须与 `TreeNode.hash_value` 对得上
   （否则 OPT 静默退化成「全 +∞」），且同 trace 上 `opt.hit_rate > lru.hit_rate`。

    PYTHONPATH=$PWD/python python -m pytest -q test/manual/test_sim_opt.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, ROOT)

evict_mod = pytest.importorskip("sglang.srt.mem_cache.evict_policy")
OPTStrategy = evict_mod.OPTStrategy
load_use_map = evict_mod.load_use_map


class FakeNode:
    """只要 hash_value / last_access_time / id / parent 就能定优先级。"""

    def __init__(self, hashes, last_access=0.0, nid=0, parent=None):
        self.hash_value = hashes
        self.last_access_time = last_access
        self.id = nid
        self.parent = parent
        self.key = None


def _write_map(tmp_path, table):
    path = tmp_path / "use_map.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# 1. 策略语义
# ---------------------------------------------------------------------------
def test_priority_is_negative_next_use(tmp_path):
    path = _write_map(
        tmp_path,
        {
            "future-soon": [110.0],
            "future-far": [900.0],
            "never-again": [50.0],  # 只有过去 → 视为不再使用
        },
    )
    now = [100.0]
    opt = OPTStrategy(use_map_path=path, block_size=1, now=lambda: now[0])

    soon = opt.get_priority(FakeNode(["future-soon"], last_access=10.0, nid=1))
    far = opt.get_priority(FakeNode(["future-far"], last_access=10.0, nid=2))
    never = opt.get_priority(FakeNode(["never-again"], last_access=10.0, nid=3))

    assert soon[0] == -110.0
    assert far[0] == -900.0
    # heapq 取最小 = 最先淘汰：far（-900）< soon（-110）< never（-inf）? 反了
    assert never[0] == -math.inf
    assert min(soon, far, never) == never, "无未来使用时必须最先淘汰"

    # 时间推进后「soon」的使用已经发生 → 不再算未来。
    now[0] = 120.0
    assert opt.get_priority(FakeNode(["future-soon"]))[0] == -math.inf


def test_priority_is_read_only_and_repeatable(tmp_path):
    path = _write_map(tmp_path, {"a": [110.0], "b": [120.0, 130.0]})
    opt = OPTStrategy(use_map_path=path, block_size=1, now=lambda: 100.0)
    node = FakeNode(["a", "b"])
    # 节点跨多个 page 时取最早的下次使用（保守侧）。
    first = opt.get_priority(node)
    for _ in range(5):
        assert opt.get_priority(node) == first
    assert first[0] == -110.0


def test_unknown_hash_counts_as_inf(tmp_path):
    path = _write_map(tmp_path, {"known": [110.0]})
    opt = OPTStrategy(use_map_path=path, block_size=1, now=lambda: 100.0)
    assert opt.get_priority(FakeNode(["absent"]))[0] == -math.inf


def test_missing_or_empty_use_map_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        OPTStrategy(use_map_path="", block_size=1)
    with pytest.raises(ValueError, match="no usable use times"):
        OPTStrategy(use_map_path=_write_map(tmp_path, {}), block_size=1)
    with pytest.raises(ValueError, match="no usable use times"):
        # 全是空序列也算空表，不能静默退化成任意序。
        OPTStrategy(use_map_path=_write_map(tmp_path, {"h": []}), block_size=1)


def test_block_size_must_match_page_size(tmp_path):
    path = _write_map(tmp_path, {"h": [1.0]})
    opt = OPTStrategy(use_map_path=path, block_size=8)
    opt.bind_page_size(8)  # 一致：放行
    with pytest.raises(ValueError, match="does not match cache page_size"):
        opt.bind_page_size(1)


def test_load_use_map_sorts_and_dedups(tmp_path):
    path = _write_map(tmp_path, {"h": [3.0, 1.0, 3.0, 2.0]})
    assert load_use_map(path)["h"] == (1.0, 2.0, 3.0)


def test_unknown_param_is_rejected(tmp_path):
    path = _write_map(tmp_path, {"h": [1.0]})
    with pytest.raises(ValueError, match="unknown opt params"):
        evict_mod.OptParams.from_any({"use_map_path": path, "oops": 1})
    with pytest.raises(ValueError, match="use_map_path"):
        evict_mod.OptParams.from_any(None)
    with pytest.raises(ValueError, match="must be > 0"):
        evict_mod.OptParams.from_any({"use_map_path": path, "block_size": 0})


# ---------------------------------------------------------------------------
# 2. 与缓存接线（需要 torch）
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="需要 torch 才能建真缓存")

from array import array  # noqa: E402

from sim.opt_map import build_use_map, write_use_map  # noqa: E402
from sim.runner import SimConfig, Simulator  # noqa: E402
from sim.trace import Session, Turn  # noqa: E402


def _tok(base, n):
    return array("i", [base + i for i in range(n)])


def _trace():
    """A 在 t=100 复用；U1/U2 是一次性噪声，用来制造容量压力。"""

    a = _tok(1000, 32)
    b = _tok(3000, 32)
    u1 = _tok(5000, 40)
    u2 = _tok(7000, 40)
    return [
        Session("s1", (Turn(0.0, a, 1), Turn(100.0, array("i", list(a) + list(b)), 1))),
        Session("s2", (Turn(10.0, u1, 1), Turn(20.0, u2, 1))),
    ]


def test_use_map_generation_matches_cache_hash(tmp_path):
    """生成器与缓存层必须算出同一组 hash；否则 OPT 静默全 +inf。"""

    sessions = _trace()
    use_map = build_use_map(sessions, page_size=8)
    assert use_map, "use map 不能为空"
    assert all(len(h) == 64 for h in use_map), "SHA256 hex 应为 64 字符"

    path = tmp_path / "use_map.json"
    write_use_map(str(path), use_map)

    sim = Simulator(
        SimConfig(
            capacity=4096,
            page_size=8,
            eviction_policy="opt",
            opt_use_map=str(path),
        )
    )
    result = sim.run(sessions)
    strategy = sim.cache.eviction_strategy
    assert type(strategy).__name__ == "OPTStrategy"
    # 关键：没有大面积 miss，说明 hash 与 TreeNode.hash_value 同源。
    assert strategy._misses == 0, (
        f"use map 与缓存 hash 不一致：{strategy._misses}/{strategy._lookups} 未命中"
    )
    assert result.hit_rate > 0.0


def test_opt_beats_lru_on_same_trace(tmp_path):
    sessions = _trace()
    path = tmp_path / "use_map.json"
    write_use_map(str(path), build_use_map(sessions, page_size=8))

    lru = Simulator(SimConfig(capacity=96, page_size=8, eviction_policy="lru"))
    lru_result = lru.run(sessions)

    opt = Simulator(
        SimConfig(capacity=96, page_size=8, eviction_policy="opt", opt_use_map=str(path))
    )
    opt_result = opt.run(sessions)

    assert opt_result.hit_rate > lru_result.hit_rate, (
        f"OPT 应优于 LRU：opt={opt_result.hit_rate:.3f} lru={lru_result.hit_rate:.3f}"
    )


def test_dataset_response_gets_finite_next_use(tmp_path):
    """With Turn.response the next turn hits the response block, and OPT sees it."""

    a = _tok(1000, 32)
    r = _tok(2000, 16)
    b = _tok(3000, 16)
    sessions = [
        Session(
            "s1",
            (
                Turn(0.0, a, out=len(r), response=r),
                Turn(100.0, array("i", list(a) + list(r) + list(b)), out=0),
            ),
        )
    ]
    path = tmp_path / "use_map.json"
    write_use_map(str(path), build_use_map(sessions, page_size=8))

    sim = Simulator(
        SimConfig(
            capacity=4096,
            page_size=8,
            eviction_policy="opt",
            opt_use_map=str(path),
        )
    )
    result = sim.run(sessions)
    assert [record.hit for record in result.records] == [0, len(a) + len(r)]
    assert sim.cache.eviction_strategy._misses == 0

    # Same trace without response tokens: turn 2 stops at the 32-token prompt.
    sentinel = [
        Session(
            "s1",
            (
                Turn(0.0, a, out=len(r)),
                Turn(100.0, array("i", list(a) + list(r) + list(b)), out=0),
            ),
        )
    ]
    miss = Simulator(
        SimConfig(
            capacity=4096,
            page_size=8,
            eviction_policy="opt",
            opt_use_map=str(path),
        )
    ).run(sentinel)
    assert [record.hit for record in miss.records] == [0, len(a)]


def test_opt_requires_use_map():
    with pytest.raises(ValueError, match="opt_use_map"):
        Simulator(SimConfig(eviction_policy="opt", page_size=8)).run(_trace())
