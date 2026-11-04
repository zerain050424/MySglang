"""SimpleHybridRadixCache.evict() 淘汰循环的回归测试（不需要 torch）。

用 stub 顶替 `radix_cache` / `base_prefix_cache` / `registry` 这三个会拉起
torch 的依赖，只加载 `simple_hybrid_radix_cache.py` 本体，再挂一棵假树。
钉住四件事：

1. **没有 one-shot 叶子时不会崩**：基础候选走 `victim_key`，绝不退化成 None
   （历史 bug：`victim` 初始化在循环外 + 基础候选被注释掉 → `_evict_one(None)`）。
2. **候选不跨轮复用**：每轮重新计算，已删除的节点不会被二次删除
   （历史 bug：复用上一轮 victim → 二次 `free_segment` + `_delete_leaf` 断言失败）。
3. **one-shot 永远先走**：`one_shot_key` 口径（结束偏移最深者先）。
4. **SLRU 只在受保护的 struct 叶子上改写候选**，并把来源记成 `qstruct`。

    python test/manual/test_evict_loop.py
"""

import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "..", "..", "python")

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


# --------------------------------------------------------------------------
# 用 stub 绕开 torch：只把 simple_hybrid_radix_cache 真正需要的东西装出来
# --------------------------------------------------------------------------


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(PY, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_stubs():
    for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m

    base = types.ModuleType("sglang.srt.mem_cache.base_prefix_cache")

    class EvictParams:
        def __init__(self, num_tokens):
            self.num_tokens = num_tokens

    class EvictResult:
        def __init__(self, num_tokens_evicted=0):
            self.num_tokens_evicted = num_tokens_evicted

    base.EvictParams = EvictParams
    base.EvictResult = EvictResult
    sys.modules[base.__name__] = base

    rc = types.ModuleType("sglang.srt.mem_cache.radix_cache")

    class RadixCache:  # 只需存在，不参与本测试
        def evict(self, params):
            raise AssertionError(
                "SimpleHybridStrategy must not fall back to RadixCache.evict()"
            )

    rc.RadixCache = RadixCache
    sys.modules[rc.__name__] = rc

    reg = types.ModuleType("sglang.srt.mem_cache.registry")
    reg.register_radix_cache_backend = lambda *a, **k: None
    sys.modules[reg.__name__] = reg


# --------------------------------------------------------------------------
# 假的树 / 缓存
# --------------------------------------------------------------------------


class FakeNode:
    _next_id = 0

    def __init__(
        self,
        session_id,
        tag,
        tokens=16,
        abs_start=0,
        last_access=0.0,
        share_sessions=0,
        is_decode=False,
    ):
        FakeNode._next_id += 1
        self.id = FakeNode._next_id
        self.ckey = f"k{self.id}"          # 代替 RadixKey.child_key 的字典键
        self.value = [0] * tokens
        self.key = [0] * tokens
        self.session_id = session_id
        self.queue_tag = tag
        self.share_sessions = share_sessions
        self.session_bloom = 0
        self.absolute = abs_start
        self.abs_start = abs_start
        self.last_access_time = last_access
        self.hit_count = 0
        self.is_decode = is_decode
        self.parent = None
        self.children = {}
        self.lock_ref = 0
        self.evicted = False
        self.priority = 0

    def __repr__(self):
        return f"<{self.ckey}>"


class FakeAllocator:
    def __init__(self):
        self.freed = []

    def free_segment(self, value, *, start_pos=0):
        self.freed.append((tuple(value), start_pos))


class FakeEvents:
    def record_remove(self, node):
        pass


class FakeRadixKey:
    """供 TreeNode.key.child_key() 调用（_delete_leaf 里用）。"""

    def __init__(self, node):
        self._node = node

    def child_key(self, page_size):
        return self._node.ckey


def make_cache(cache_cls, strategy, nodes):
    cache = cache_cls.__new__(cache_cls)
    cache.disable = False
    cache.eviction_strategy = strategy
    cache.session_registry = strategy.session_registry
    cache.victim_overrides = tuple(strategy.victim_overrides)
    cache._alpha_updater = None
    cache.token_to_kv_pool_allocator = FakeAllocator()
    cache.kv_events = FakeEvents()
    cache.deleted = []
    cache.metrics = []
    cache.root_node = FakeNode(None, strategy.tagState and None or None)
    cache.root_node.ckey = "root"
    cache.evictable_leaves = set()
    for n in nodes:
        n.parent = cache.root_node
        cache.root_node.children[n.ckey] = n
        cache.evictable_leaves.add(n)
        # _delete_leaf 会用 node.key.child_key(page_size)；假树里用轻量替身
        n.key = FakeRadixKey(n)
    return cache


def _delete_leaf(self, node):
    if node.parent is not None:
        got = node.parent.children.pop(node.ckey, None)
        assert got is node, f"double delete of {node.ckey}"
    if node in self.evictable_leaves:
        self.evictable_leaves.remove(node)
    assert node not in self.deleted, f"double delete of {node.ckey}"
    self.deleted.append(node)


def _update_eviction_metrics(self, num_evicted, start_time):
    self.metrics.append(num_evicted)


def _profile_token_counts(self):
    return {}, {}


# --------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------


def _bootstrap():
    _install_stubs()
    _load("sglang.srt.mem_cache.uni_utils", "sglang/srt/mem_cache/uni_utils.py")
    _load("sglang.srt.mem_cache.victim_override", "sglang/srt/mem_cache/victim_override.py")
    _load("sglang.srt.mem_cache.tag_lru", "sglang/srt/mem_cache/tag_lru.py")
    ep = _load("sglang.srt.mem_cache.evict_policy", "sglang/srt/mem_cache/evict_policy.py")
    shc = _load(
        "sglang.srt.mem_cache.simple_hybrid_radix_cache",
        "sglang/srt/mem_cache/simple_hybrid_radix_cache.py",
    )
    cls = shc.SimpleHybridRadixCache
    cls._delete_leaf = _delete_leaf
    cls.update_eviction_metrics = _update_eviction_metrics
    cls._profile_token_counts = _profile_token_counts
    return ep, shc, sys.modules["sglang.srt.mem_cache.base_prefix_cache"]


def _strategy(ep, sessions, **tiny_lru):
    st = ep.SimpleHybridStrategy(tiny_lru=tiny_lru or None)
    reg = sys.modules["sglang.srt.mem_cache.uni_utils"].SessionRegistry()
    uni = sys.modules["sglang.srt.mem_cache.uni_utils"]
    for sid, stype in sessions.items():
        reg._sessions[sid] = uni.SessionState(session_id=sid, session_type=stype)
    st.session_registry = reg
    st.slru_evict.session_registry = reg
    return st


def test_no_one_shot_leaves_does_not_crash(ep, shc, base):
    """历史 P0：没有 one-shot 叶子时 victim 曾是 None / 陈旧节点。"""
    print("[1] no one-shot leaves: base candidate via victim_key, no crash")
    uni = sys.modules["sglang.srt.mem_cache.uni_utils"]
    st = _strategy(ep, {"a1": uni.SessionType.AGENTIC})
    nodes = [
        FakeNode("a1", uni.SimpleHybridQueueTag.QAGENT, tokens=t)
        for t in (16, 32, 48)
    ]
    cache = make_cache(shc.SimpleHybridRadixCache, st, nodes)

    try:
        result = cache.evict(base.EvictParams(num_tokens=10_000))
    except Exception as exc:  # noqa: BLE001
        check(False, f"evict() raised {type(exc).__name__}: {exc}")
        return

    check(len(cache.deleted) == 3, f"all leaves evicted, got {len(cache.deleted)}")
    check(result.num_tokens_evicted == 96,
          f"released tokens, got {result.num_tokens_evicted}")
    check(st._victim_sources.get("global") == 96,
          f"source must be 'global', got {st._victim_sources}")
    check(not st._victim_sources.get("one_shot"),
          "no one-shot band should have been used")


def test_victim_not_reused_across_rounds(ep, shc, base):
    """历史 P0：第二轮复用上一轮已删节点 → 二次释放 / 断言失败。"""
    print("[2] victim must be recomputed every round (no double delete)")
    uni = sys.modules["sglang.srt.mem_cache.uni_utils"]
    st = _strategy(ep, {"a1": uni.SessionType.AGENTIC})
    nodes = [
        FakeNode("a1", uni.SimpleHybridQueueTag.QAGENT, tokens=16) for _ in range(4)
    ]
    cache = make_cache(shc.SimpleHybridRadixCache, st, nodes)

    # 每轮只淘汰 16 token -> 必须连跑 4 轮
    for i in range(4):
        try:
            cache.evict(base.EvictParams(num_tokens=16))
        except Exception as exc:  # noqa: BLE001
            check(False, f"round {i} raised {type(exc).__name__}: {exc}")
            return
    check(len(cache.deleted) == 4, f"4 distinct victims, got {len(cache.deleted)}")
    check(len({id(n) for n in cache.deleted}) == 4, "no node evicted twice")
    check(cache.token_to_kv_pool_allocator.freed and
          len(cache.token_to_kv_pool_allocator.freed) == 4,
          f"each node freed exactly once, got {len(cache.token_to_kv_pool_allocator.freed)}")


def test_one_shot_band_first(ep, shc, base):
    """one-shot 叶子先走，且按结束偏移最深优先。"""
    print("[3] one-shot leaves are drained first (deepest end offset wins)")
    uni = sys.modules["sglang.srt.mem_cache.uni_utils"]
    st = _strategy(
        ep,
        {"t1": uni.SessionType.TRANS, "s1": uni.SessionType.STRUCT},
    )
    # 一个未共享的 TRANS 节点（one-shot） + 一个共享的 STRUCT 节点（受保护）
    shallow = FakeNode("t1", uni.SimpleHybridQueueTag.QTRANS, tokens=16, abs_start=0)
    deep = FakeNode("t1", uni.SimpleHybridQueueTag.QTRANS, tokens=16, abs_start=256)
    shared = FakeNode(
        "s1", uni.SimpleHybridQueueTag.QSTRUCT, tokens=16, abs_start=0,
        share_sessions=st.slru_evict.struct_share_min,
    )
    check(st.is_one_shot(deep) is True, "sanity: trans node is one-shot")
    check(st.is_one_shot(shared) is False, "sanity: shared struct is protected")

    cache = make_cache(shc.SimpleHybridRadixCache, st, [shallow, deep, shared])
    cache.evict(base.EvictParams(num_tokens=16))

    check(cache.deleted == [deep],
          f"deepest one-shot node must go first, got {cache.deleted}")
    check(st._victim_sources.get("one_shot") == 16,
          f"source must be 'one_shot', got {st._victim_sources}")


def test_slru_rewrites_struct_candidate(ep, shc, base):
    """只有受保护的 struct 叶子时，来源应是 SLRU，且挑最久未访问的那片。"""
    print("[4] SLRU rewrites the candidate among protected struct leaves")
    uni = sys.modules["sglang.srt.mem_cache.uni_utils"]
    st = _strategy(ep, {"s1": uni.SessionType.STRUCT})
    thr = st.slru_evict.struct_share_min
    old = FakeNode(
        "s1", uni.SimpleHybridQueueTag.QSTRUCT, tokens=16,
        last_access=1_000.0, share_sessions=thr,
    )
    new = FakeNode(
        "s1", uni.SimpleHybridQueueTag.QSTRUCT, tokens=16,
        last_access=9_999_999.0, share_sessions=thr,
    )
    cache = make_cache(shc.SimpleHybridRadixCache, st, [old, new])

    cache.evict(base.EvictParams(num_tokens=16))

    check(cache.deleted == [old], f"LRU-oldest must be picked, got {cache.deleted}")
    check(st._victim_sources.get("qstruct") == 16,
          f"source must be the SLRU queue, got {st._victim_sources}")
    check(st.slru_evict.stats()["slru_picks"] >= 1, "SLRU must have been consulted")


def test_slru_actually_rewrites_with_tag_off(ep, shc, base):
    """tiny_lru=off：基础候选按位置打分，SLRU 把它改写成年限最久的那个。

    这条证明 SLRU 不是只改个 source 标签 —— 默认 tag 模式下 victim_key 的
    rank-2 分支已经是 struct 内 LRU，与 SLRU 的挑选重合，看不出差别。
    """
    print("[4b] with tiny_lru=off, SLRU rewrites the base candidate")
    uni = sys.modules["sglang.srt.mem_cache.uni_utils"]
    st = _strategy(ep, {"s1": uni.SessionType.STRUCT}, policy="off")
    thr = st.slru_evict.struct_share_min
    # 位置打分偏向「结束偏移小」的节点 -> 基础候选会挑 deep（offset 越界被截成 0 分）
    old_lru = FakeNode(
        "s1", uni.SimpleHybridQueueTag.QSTRUCT, tokens=16,
        abs_start=0, last_access=1_000.0, share_sessions=thr,
    )
    deep_recent = FakeNode(
        "s1", uni.SimpleHybridQueueTag.QSTRUCT, tokens=16,
        abs_start=8000, last_access=9_999_999.0, share_sessions=thr,
    )
    cache = make_cache(shc.SimpleHybridRadixCache, st, [old_lru, deep_recent])

    base_pick = min(cache.evictable_leaves, key=st.victim_key)
    check(base_pick is deep_recent,
          f"sanity: base candidate should be the deep node, got {base_pick}")

    cache.evict(base.EvictParams(num_tokens=16))
    check(cache.deleted == [old_lru],
          f"SLRU must override the base candidate, got {cache.deleted}")
    check(st._victim_sources.get("qstruct") == 16,
          f"source must be the SLRU queue, got {st._victim_sources}")


def test_empty_tree_is_noop(ep, shc, base):
    """空树：直接返回 0，不报错。"""
    print("[5] empty tree returns 0")
    uni = sys.modules["sglang.srt.mem_cache.uni_utils"]
    st = _strategy(ep, {})
    cache = make_cache(shc.SimpleHybridRadixCache, st, [])
    result = cache.evict(base.EvictParams(num_tokens=128))
    check(result.num_tokens_evicted == 0, f"got {result.num_tokens_evicted}")
    check(cache.deleted == [], "nothing to delete")


def main():
    ep, shc, base = _bootstrap()
    tests = [
        test_no_one_shot_leaves_does_not_crash,
        test_victim_not_reused_across_rounds,
        test_one_shot_band_first,
        test_slru_rewrites_struct_candidate,
        test_slru_actually_rewrites_with_tag_off,
        test_empty_tree_is_noop,
    ]
    for t in tests:
        try:
            t(ep, shc, base)
        except Exception as exc:  # noqa: BLE001
            FAILS.append(f"{t.__name__} raised {type(exc).__name__}: {exc}")
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)} problems)")
        for f in FAILS:
            print("  -", f)
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
