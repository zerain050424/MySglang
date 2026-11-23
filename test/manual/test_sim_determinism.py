"""仿真的决定性：同 seed 两次 run 必须逐位一致（fingerprint 相同）。

回归点：``RadixCache.evict`` 原先用 ``list(self.evictable_leaves)``，而那是 **set**
——迭代序由对象地址决定。仿真里大量节点的 ``last_access_time`` 完全相同（同一
虚拟时刻到达 / 同一请求命中的整条路径），平票时 heapq 比较 nodes（``__lt__`` 比
``last_access_time``，相等），于是输入序一变淘汰序列就变，同进程连续两次 run 的
``fingerprint()`` 都不同。drift3 的 OPT 固定点迭代要求 ``dispatched`` 逐条相等，
必须先把「平票选谁」钉住（按单调的 ``TreeNode.id`` 排序）。

    PYTHONPATH=$PWD:$PWD/python python -m pytest -q test/manual/test_sim_determinism.py
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, ROOT)

pytest.importorskip("torch", reason="需要 torch 才能建真缓存")

from array import array  # noqa: E402

from sim.core.config import SimConfig  # noqa: E402
from sim.simulator import Simulator  # noqa: E402
from sim.workload.trace import Session, Turn  # noqa: E402

PREFIX = 256          # 每轮独有的 token 数（保证不被去重）
SESSIONS = 20
CAPACITY = 3072       # 远小于 20 会话 × 2 轮 × 256 token → 强制大量淘汰


def _trace():
    """**同一虚拟时刻**到达 + 每会话独有内容 ⇒ 大量 last_access_time 平票。"""

    sessions = []
    for i in range(SESSIONS):
        rows = []
        for turn in range(2):
            base = 10_000 + i * 100_000 + turn * 10_000
            # 第 2 轮复用第 1 轮的前缀（制造内部节点与更深的分裂）
            toks = list(range(base, base + PREFIX))
            if turn == 1:
                toks = list(range(base - 10_000, base - 10_000 + PREFIX)) + toks
            rows.append(Turn(t=0.0, prompt=array("i", toks), out=0))
        sessions.append(Session(f"s{i}", tuple(rows), "T"))
    return sessions


def _run(policy):
    # oversize="skip"：单条可能装不下，但淘汰仍会发生（skip 的请求不落树，不影响平票）
    sim = Simulator(
        SimConfig(capacity=CAPACITY, page_size=8, eviction_policy=policy,
                  seed=42, oversize="skip")
    )
    return sim, sim.run_replay(_trace())


@pytest.mark.parametrize("policy", ["lru", "heter_cache"])
def test_two_runs_in_one_process_are_identical(policy):
    _s1, a = _run(policy)
    _s2, b = _run(policy)
    assert [r.used for r in a.records] == [r.used for r in b.records], (
        f"{policy}: 水位序列不一致（淘汰顺序漂移）"
    )
    assert [r.evicted for r in a.records] == [r.evicted for r in b.records]
    assert a.fingerprint() == b.fingerprint(), (
        f"{policy}: 同 seed 两次 run 不一致 {a.fingerprint()} vs {b.fingerprint()}"
    )


def test_evict_is_insensitive_to_set_iteration_order():
    """精确回归：把所有优先级打成平票，淘汰序必须与 ``evictable_leaves`` 的
    迭代序无关（实现里先按 ``TreeNode.id`` 排序）。

    不依赖地址复用的运气：这里直接给两次相同 run 的缓存喂**顺/逆**两种容器序。
    """

    class _OrderedLeaves:
        """set 的最小替身，但迭代序可控（照抄 evict 路径用到的接口）。"""

        def __init__(self, items, reverse):
            self._items = list(reversed(items)) if reverse else list(items)

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            return len(self._items)

        def __contains__(self, item):
            return item in self._items

        def add(self, item):
            if item not in self._items:
                self._items.append(item)

        def discard(self, item):
            if item in self._items:
                self._items.remove(item)

        def remove(self, item):
            self._items.remove(item)

    def evicted_ids(reverse: bool):
        sim, _res = _run("lru")
        cache = sim.cache
        leaves = sorted(cache.evictable_leaves, key=lambda n: n.id)
        assert len(leaves) >= 4, "叶子太少，测不出平票敏感性"
        # 全部平票：淘汰只可能由 leaves 的顺序决定
        cache.eviction_strategy.get_priority = lambda node: 0.0
        cache.evictable_leaves = _OrderedLeaves(leaves, reverse)

        # TreeNode.id 是**跨 run 的全局计数器**，两次 run 的绝对值必然不同，
        # 所以比较「在按 id 排序后的名次」而不是 id 本身。
        rank = {node.id: i for i, node in enumerate(leaves)}
        seen = []
        orig = cache._delete_leaf

        def spy(node):
            seen.append(rank[node.id])
            return orig(node)

        cache._delete_leaf = spy
        sym = sim.sym
        cache.evict(sym["EvictParams"](num_tokens=len(leaves[0].value) * 2 + 1))
        return seen

    forward = evicted_ids(False)
    backward = evicted_ids(True)
    assert forward and forward == backward, (
        f"淘汰序随 set 迭代序漂移：{forward[:6]} vs {backward[:6]}"
    )


def test_eviction_really_happens():
    """没有淘汰就没有平票，上面的测试会变成空转。"""

    _sim, res = _run("lru")
    assert sum(r.evicted for r in res.records) > 0
    assert max(r.used for r in res.records) > 0.9 * CAPACITY
