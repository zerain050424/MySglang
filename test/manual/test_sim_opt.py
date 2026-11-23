"""OPT 上界（落点 A）的回归测试。

两部分：
1. **策略语义**（不需要 torch）：优先级 = `-下次使用时刻`，无未来 → `-inf`
   最先淘汰；真值来自 use map 的二分查询，策略必须**只读无状态**。
2. **与缓存的接线**（需要 torch，base conda 下自动 skip）：use map 由
   `sim.analysis.opt_map.build_use_map` 生成，hash 必须与 `TreeNode.hash_value` 对得上
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


def test_miss_ratio_counts_every_lookup(tmp_path):
    """``_misses/_lookups`` 必须是真实未命中率，而不是恒 1.0。

    回归点：``_lookups`` 曾经只在 miss 分支（``_maybe_warn`` 内）自增，
    两次自增落在同一条路径上 → 比率恒为 100%，健康负载也报「OPT 已退化」。
    这里 90% 命中、20 万次查询，真实 miss≈3%，不应告警。
    """
    path = _write_map(tmp_path, {f"present-{i}": [110.0] for i in range(1)})
    opt = OPTStrategy(use_map_path=path, block_size=1, now=lambda: 100.0)

    total = 200_000
    miss_every = 33  # ≈3% 未命中
    for i in range(total):
        node = FakeNode(["absent"] if i % miss_every == 0 else ["present-0"])
        opt.get_priority(node)

    expected_misses = len(range(0, total, miss_every))
    assert opt._lookups == total, "每一次查询都必须计入 lookups"
    assert opt._misses == expected_misses
    ratio = opt._misses / opt._lookups
    assert 0.0 < ratio < 0.05, ratio
    assert opt._warned is False, "3% 未命中不该触发退化告警"


def test_miss_ratio_warns_on_real_mismatch(tmp_path, caplog):
    """真·粒度错配（这里 60% 查不到）必须告警，且打印的是真实比率。"""
    path = _write_map(tmp_path, {"present-0": [110.0]})
    opt = OPTStrategy(use_map_path=path, block_size=1, now=lambda: 100.0)

    with caplog.at_level("WARNING", logger="sglang.srt.mem_cache.evict_policy"):
        for i in range(2000):
            opt.get_priority(FakeNode(["absent"] if i % 5 < 3 else ["present-0"]))

    assert opt._lookups == 2000
    assert opt._misses == 1200
    assert opt._warned is True
    assert "60.0%" in caplog.text, caplog.text
    assert "100.0%" not in caplog.text


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

from sim.analysis.opt_map import build_use_map, write_use_map  # noqa: E402
from sim.core.config import SimConfig  # noqa: E402
from sim.simulator import Simulator  # noqa: E402
from sim.workload.trace import Session, Turn  # noqa: E402


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


def test_use_map_capacity_excludes_oversized_turns():
    """A turn that can never be resident must not shape the OPT oracle."""

    small = _tok(1000, 16)   # 2 blocks at page_size=8
    large = _tok(2000, 80)   # 10 blocks, over capacity_blocks=4
    sessions = [
        Session("s1", (Turn(0.0, small, 0),)),
        Session("s2", (Turn(10.0, large, 0),)),
    ]
    full = build_use_map(sessions, page_size=8)
    filtered = build_use_map(sessions, page_size=8, capacity_blocks=4)
    expected = build_use_map([sessions[0]], page_size=8)

    assert len(full) > len(filtered)
    assert filtered == expected
    with pytest.raises(ValueError, match="capacity_blocks"):
        build_use_map(sessions, page_size=8, capacity_blocks=0)


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
    result = sim.run_replay(sessions)
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
    lru_result = lru.run_replay(sessions)

    opt = Simulator(
        SimConfig(capacity=96, page_size=8, eviction_policy="opt", opt_use_map=str(path))
    )
    opt_result = opt.run_replay(sessions)

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
    result = sim.run_replay(sessions)
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
    ).run_replay(sentinel)
    assert [record.hit for record in miss.records] == [0, len(a)]


def test_opt_requires_use_map():
    with pytest.raises(ValueError, match="opt_use_map"):
        Simulator(SimConfig(eviction_policy="opt", page_size=8)).run_replay(_trace())


# ---------------------------------------------------------------------------
# 3. 闭环调度的 OPT 上界（冻结时间轴 + 覆盖 response 页）
# ---------------------------------------------------------------------------
from sim.analysis.opt_map import use_map_from_run  # noqa: E402
from sim.core.request import MAX_SENTINEL_REQUESTS, OUTPUT_SENTINEL_BASE  # noqa: E402
from sim.workload.arrival import TaskSpec  # noqa: E402
from sim.workload.service import ServiceModel  # noqa: E402


def test_sentinel_fits_hash_and_int32():
    """response 哨兵必须同时安全于 C++ hash 的 uint32 与 opt_map 的 array('i')。"""

    top = OUTPUT_SENTINEL_BASE + MAX_SENTINEL_REQUESTS
    assert OUTPUT_SENTINEL_BASE > 1 << 20, "必须远高于任何真实词表（Qwen2.5=151k）"
    assert top < 1 << 31, "int32 有符号（sim/opt_map._page_hashes 用 array('i')）"
    assert top < 1 << 32, "hash_binding.cpp::checked_u32"


def _closed_loop_trace():
    """两轮会话：第 2 轮重读第 1 轮的 32 token，外加一次性噪声制造压力。"""

    a = _tok(1000, 32)
    b = _tok(3000, 32)
    u1 = _tok(5000, 40)
    u2 = _tok(7000, 48)
    return [
        Session("s1", (Turn(0.0, a, 8), Turn(0.0, array("i", list(a) + list(b)), 8)),
                "T"),
        Session("s2", (Turn(0.0, u1, 8),), "T"),
        # 一次性大请求：把 96 的池子顶爆，强制走淘汰，OPT 才有决策可比。
        Session("s3", (Turn(0.0, u2, 8),), "T"),
    ]


def test_closed_loop_freeze_and_opt_upper_bound(tmp_path):
    """闭环 run 的时刻不写回 Turn：必须 freeze 后再建表，否则 OPT 静默退化。"""

    sessions = _closed_loop_trace()
    specs = [TaskSpec.from_config({"name": "T", "lambda_rate": 1.0, "max_record": 10})]
    service = ServiceModel(prefill_tokens_per_s=8000.0, decode_tokens_per_s=40.0)
    # 96 = 刚好让 s2 触发一次淘汰（s1 的两轮占 72），OPT 才有决策可做。
    cfg = dict(capacity=96, page_size=8, seed=0, oversize="skip")

    lru = Simulator(SimConfig(eviction_policy="lru", **cfg))
    run = lru.run_closed_loop(sessions, specs, service=service, max_inflight=0)

    assert sum(r.evicted for r in run.records) > 0, "这份 trace 必须有淘汰，否则测不到选择"

    # 阻碍 1：原始 sessions 的 Turn.t 仍是加载值 0（闭环不写回）
    assert all(t.t == 0.0 for s in sessions for t in s.turns)
    # 出口：run.dispatched 记下了真实派发时刻
    assert len(run.dispatched) == sum(len(s.turns) for s in sessions)
    frozen = run.freeze_sessions(sessions)
    assert all(t.t > 0.0 for s in frozen for t in s.turns), "冻结后必须有真实时刻"

    # 阻碍 3：不带 response 页的表会大面积未命中；带上之后应为 0
    plain = build_use_map(frozen, page_size=8)
    full = use_map_from_run(run, sessions, page_size=8)
    assert len(full) > len(plain), "response 页必须使表变大"

    path = tmp_path / "use_map.json"
    write_use_map(str(path), full)
    opt = Simulator(
        SimConfig(eviction_policy="opt", opt_use_map=str(path), **cfg)
    )
    opt_result = opt.run_replay(frozen)
    strategy = opt.cache.eviction_strategy

    assert strategy._misses == 0, (
        f"表与树的 token 口径不一致：{strategy._misses}/{strategy._lookups} 未命中"
    )
    assert opt_result.hit_rate >= run.hit_rate, (
        f"OPT 应不低于同口径 LRU：opt={opt_result.hit_rate:.3f} lru={run.hit_rate:.3f}"
    )


def test_freeze_keeps_real_response_tokens(tmp_path):
    """``freeze_sessions`` 不能把 ``Turn.response`` 丢掉。

    丢了就等于退回 fallback 哨兵：树里写的不是真实 answer，下一轮 prompt
    （= 上一轮 prompt + 上一轮 response）那一截永远命不中，OPT 也看不到它的下次
    使用。这里用「表里必须出现 a+r 的每一页」来验证，不依赖淘汰是否发生。
    """

    from sim.analysis.opt_map import _page_hashes
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.utils import get_storage_hash_str

    a = _tok(1000, 16)
    r = _tok(2000, 8)
    sessions = [Session("s1", (Turn(0.0, a, out=len(r), response=r),), "T")]
    specs = [TaskSpec.from_config({"name": "T", "lambda_rate": 1.0})]
    service = ServiceModel(prefill_tokens_per_s=8000.0, decode_tokens_per_s=40.0)

    run = Simulator(
        SimConfig(capacity=4096, page_size=8, eviction_policy="lru")
    ).run_closed_loop(sessions, specs, service=service, max_inflight=0)

    frozen = run.freeze_sessions(sessions)
    assert list(frozen[0].turns[0].response) == list(r), "冻结丢了真实 response"

    table = use_map_from_run(run, sessions, page_size=8)
    pages = _page_hashes(
        list(a) + list(r), 8, lambda key, prior: get_storage_hash_str(key, prior),
        RadixKey,
    )
    assert pages, "至少一页"
    missing = [h for h in pages if h not in table]
    assert not missing, f"真实 response 的页没进 use map：{len(missing)}/{len(pages)}"


def test_use_map_from_run_requires_a_closed_loop_result(tmp_path):
    """开环 run 也能冻结（时刻就是 Turn.t），但缺 response 哨兵的表必须报错。"""

    sessions = _closed_loop_trace()
    open_loop = Simulator(SimConfig(capacity=4096, page_size=8)).run_replay(
        [
            Session(s.session_id, tuple(Turn(t=i + 1.0, prompt=tt.prompt, out=tt.out)
                                        for i, tt in enumerate(s.turns)), s.task_name)
            for s in sessions
        ]
    )
    frozen = open_loop.freeze_sessions(
        [
            Session(s.session_id, tuple(Turn(t=i + 1.0, prompt=tt.prompt, out=tt.out)
                                        for i, tt in enumerate(s.turns)), s.task_name)
            for s in sessions
        ]
    )
    assert all(t.t > 0.0 for s in frozen for t in s.turns)

    with pytest.raises(ValueError, match="response_tokens"):
        build_use_map(frozen, page_size=8, with_response=True)
