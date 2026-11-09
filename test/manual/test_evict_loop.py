"""HeterCacheRadixCache.evict() 淘汰循环的回归测试（不需要 torch）。

用 stub 顶替 `radix_cache` / `base_prefix_cache` / `registry` 这三个会拉起
torch 的依赖，只加载 `heter_cache_radix_cache.py` 本体，再挂一棵假树。
钉住四件事：

1. **没有 one-shot 叶子时不会崩**：配额仲裁或随机兜底必须给出候选
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
# 用 stub 绕开 torch：只把 heter_cache_radix_cache 真正需要的东西装出来
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
                "HeterCacheStrategy must not fall back to RadixCache.evict()"
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
    """容量恒为「已满 20% 空闲」的假分配器：配额需要 size/available_size。"""

    size = 1000

    def __init__(self):
        self.freed = []

    def free_segment(self, value, *, start_pos=0):
        self.freed.append((tuple(value), start_pos))

    def available_size(self):
        return 800


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
    cache.node_observers = tuple(strategy.node_observers)
    # evict() 依赖这三个由 __init__ 派生的字段；假缓存需要手工接上
    cache._needs_capacity = strategy.needs_capacity
    cache.page_size = 1
    cache.evictable_size_ = 0
    # 观测钩子：本测试只关心淘汰控制流，不采样占用
    cache._occupancy_observer = None
    cache._hit_recorder = None
    cache._record_insert = None
    cache.force_agentic = False
    cache.token_to_kv_pool_allocator = FakeAllocator()
    cache.kv_events = FakeEvents()
    cache.deleted = []
    cache.metrics = []
    hu = sys.modules["sglang.srt.mem_cache.heter_utils"]
    cache.root_node = FakeNode(None, hu.HeterCacheNodeTag.QTRANS)
    cache.root_node.ckey = "root"
    cache.evictable_leaves = set()
    for n in nodes:
        n.parent = cache.root_node
        cache.root_node.children[n.ckey] = n
        cache.evictable_leaves.add(n)
        # _delete_leaf 会用 node.key.child_key(page_size)；假树里用轻量替身
        n.key = FakeRadixKey(n)
    # 真实缓存在建节点时会调 record_insert；假缓存手工补这一步，
    # 否则 _occ 恒 0，配额排名永远为空。
    for n in nodes:
        if not strategy.is_one_shot(n):
            strategy.record_insert(n)
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


def _per_queue_token_count(self):
    return {}


# --------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------


def _bootstrap():
    _install_stubs()
    _load("sglang.srt.mem_cache.heter_utils", "sglang/srt/mem_cache/heter_utils.py")
    # evict_policy 现在直接依赖 quota_selector 与 heter_logging（观测门面）：
    # 这两者必须先于它装载，否则包路径被 stub 掉后无法按名解析。
    _load("sglang.srt.mem_cache.quota_selector", "sglang/srt/mem_cache/quota_selector.py")
    _load("sglang.srt.mem_cache.heter_logging", "sglang/srt/mem_cache/heter_logging.py")
    _load("sglang.srt.mem_cache.victim_override", "sglang/srt/mem_cache/victim_override.py")
    ep = _load("sglang.srt.mem_cache.evict_policy", "sglang/srt/mem_cache/evict_policy.py")
    shc = _load(
        "sglang.srt.mem_cache.heter_cache_radix_cache",
        "sglang/srt/mem_cache/heter_cache_radix_cache.py",
    )
    cls = shc.HeterCacheRadixCache
    cls._delete_leaf = _delete_leaf
    cls.update_eviction_metrics = _update_eviction_metrics
    cls._per_queue_token_count = _per_queue_token_count
    return ep, shc, sys.modules["sglang.srt.mem_cache.base_prefix_cache"]


CAPS = {"caps": {"struct": 0.3, "agentic": 0.4, "chat": 0.3}}


def _strategy(ep, sessions, **slru):
    st = ep.HeterCacheStrategy(slru=slru or None, quota=dict(CAPS))
    reg = sys.modules["sglang.srt.mem_cache.heter_utils"].SessionRegistry()
    hu = sys.modules["sglang.srt.mem_cache.heter_utils"]
    for sid, stype in sessions.items():
        reg._sessions[sid] = hu.SessionState(session_id=sid, session_type=stype)
    st.session_registry = reg
    st.slru_evict.session_registry = reg
    return st


def _seed_share(st, node, n):
    """把节点的跨会话共享数直接记进 SLRU 账本（share_of 只认这本账）。"""
    for i in range(n):
        st.slru_evict.on_node_hit(node, f"seed-{id(node)}-{i}")


def test_no_one_shot_leaves_does_not_crash(ep, shc, base):
    """历史 P0：没有 one-shot 叶子时 victim 曾是 None / 陈旧节点。"""
    print("[1] no one-shot leaves: quota path still yields a victim")
    hu = sys.modules["sglang.srt.mem_cache.heter_utils"]
    st = _strategy(ep, {"a1": hu.SessionType.AGENTIC})
    nodes = [
        FakeNode("a1", hu.HeterCacheNodeTag.QAGENT, tokens=t)
        for t in (16, 32, 48)
    ]
    cache = make_cache(shc.HeterCacheRadixCache, st, nodes)

    try:
        result = cache.evict(base.EvictParams(num_tokens=10_000))
    except Exception as exc:  # noqa: BLE001
        check(False, f"evict() raised {type(exc).__name__}: {exc}")
        return

    check(len(cache.deleted) == 3, f"all leaves evicted, got {len(cache.deleted)}")
    check(result.num_tokens_evicted == 96,
          f"released tokens, got {result.num_tokens_evicted}")
    quota_srcs = {k: v for k, v in st.profiler._victim_sources.items() if k.startswith("quota:")}
    check(quota_srcs.get("quota:agentic", 0) + quota_srcs.get("quota:fallback", 0) == 96,
          f"all 96 tokens must come from the quota path, got {st.profiler._victim_sources}")
    check(not st.profiler._victim_sources.get("one_shot"),
          "no one-shot band should have been used")


def test_victim_not_reused_across_rounds(ep, shc, base):
    """历史 P0：第二轮复用上一轮已删节点 → 二次释放 / 断言失败。"""
    print("[2] victim must be recomputed every round (no double delete)")
    hu = sys.modules["sglang.srt.mem_cache.heter_utils"]
    st = _strategy(ep, {"a1": hu.SessionType.AGENTIC})
    nodes = [
        FakeNode("a1", hu.HeterCacheNodeTag.QAGENT, tokens=16) for _ in range(4)
    ]
    cache = make_cache(shc.HeterCacheRadixCache, st, nodes)

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
    """one-shot 叶子先走：不管位置深浅，一律先于受保护节点被清。"""
    print("[3] one-shot leaves are drained before protected ones")
    hu = sys.modules["sglang.srt.mem_cache.heter_utils"]
    st = _strategy(
        ep,
        {"t1": hu.SessionType.TRANS, "s1": hu.SessionType.STRUCT},
    )
    # 一个未共享的 TRANS 节点（one-shot） + 一个共享的 STRUCT 节点（受保护）
    shallow = FakeNode(
        "t1", hu.HeterCacheNodeTag.QTRANS, tokens=16, abs_start=0, last_access=1.0
    )
    deep = FakeNode(
        "t1", hu.HeterCacheNodeTag.QTRANS, tokens=16, abs_start=256, last_access=2.0
    )
    shared = FakeNode("s1", hu.HeterCacheNodeTag.QSTRUCT, tokens=16, abs_start=0)
    _seed_share(st, shared, st.slru_evict.share_min)
    check(st.is_one_shot(deep) is True, "sanity: trans node is one-shot")
    check(st.is_one_shot(shared) is False, "sanity: shared struct is protected")

    cache = make_cache(shc.HeterCacheRadixCache, st, [shallow, deep, shared])
    cache.evict(base.EvictParams(num_tokens=16))

    # 带内排序是 one_shot_key（LRU）：shallow 的 last_access 更早，先走
    check(cache.deleted == [shallow],
          f"LRU-oldest one-shot node must go first, got {cache.deleted}")
    check(st.profiler._victim_sources.get("one_shot") == 16,
          f"source must be 'one_shot', got {st.profiler._victim_sources}")


def test_slru_rewrites_struct_candidate(ep, shc, base):
    """只有受保护的 struct 叶子时，配额点名 struct 队，队内按段内 LRU 挑最久未访问的那片。"""
    print("[4] struct queue: quota names it, struct_tier picks the LRU-oldest")
    hu = sys.modules["sglang.srt.mem_cache.heter_utils"]
    st = _strategy(ep, {"s1": hu.SessionType.STRUCT})
    thr = st.slru_evict.share_min
    old = FakeNode("s1", hu.HeterCacheNodeTag.QSTRUCT, tokens=16, last_access=1_000.0)
    new = FakeNode(
        "s1", hu.HeterCacheNodeTag.QSTRUCT, tokens=16, last_access=9_999_999.0
    )
    _seed_share(st, old, thr)
    _seed_share(st, new, thr)
    cache = make_cache(shc.HeterCacheRadixCache, st, [old, new])

    cache.evict(base.EvictParams(num_tokens=16))

    check(cache.deleted == [old], f"LRU-oldest must be picked, got {cache.deleted}")
    check(st.profiler._victim_sources.get("quota:struct") == 16,
          f"source must be the struct queue, got {st.profiler._victim_sources}")
    # struct 队的选点由 struct_tier 直接完成（share 分级 + 段内 LRU），
    # 不再经过 SlruEvictPolicy.pick，所以这里不校验 slru_picks。


def test_empty_tree_is_noop(ep, shc, base):
    """空树：直接返回 0，不报错。"""
    print("[5] empty tree returns 0")
    hu = sys.modules["sglang.srt.mem_cache.heter_utils"]
    st = _strategy(ep, {})
    cache = make_cache(shc.HeterCacheRadixCache, st, [])
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
