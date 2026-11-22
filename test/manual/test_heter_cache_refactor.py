"""HeterCache 重构回归测试（不依赖 torch / pytest，直接 python3 运行）。"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_HC = os.path.join(_REPO, "python/sglang/srt/mem_cache/heter_cache")


def _load(name: str, rel_or_abs: str):
    path = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(_REPO, "python", rel_or_abs)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap():
    for pkg in ("sglang", "sglang.srt"):
        m = types.ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m
    mc = types.ModuleType("sglang.srt.mem_cache")
    mc.__path__ = []
    sys.modules["sglang.srt.mem_cache"] = mc
    hc = types.ModuleType("sglang.srt.mem_cache.heter_cache")
    hc.__path__ = [_HC]
    sys.modules["sglang.srt.mem_cache.heter_cache"] = hc

    base = types.ModuleType("sglang.srt.mem_cache.base_prefix_cache")

    class BasePrefixCache:
        pass

    class EvictParams:
        def __init__(self, num_tokens=0):
            self.num_tokens = num_tokens

    class EvictResult:
        def __init__(self, num_tokens_evicted=0):
            self.num_tokens_evicted = num_tokens_evicted

    base.BasePrefixCache = BasePrefixCache
    base.EvictParams = EvictParams
    base.EvictResult = EvictResult
    sys.modules["sglang.srt.mem_cache.base_prefix_cache"] = base

    rc = types.ModuleType("sglang.srt.mem_cache.radix_cache")

    class TreeNode:
        pass

    class RadixCache:
        def __init__(self, params=None):
            pass

        def evict(self, params):
            return EvictResult(0)

    rc.TreeNode = TreeNode
    rc.RadixCache = RadixCache
    sys.modules["sglang.srt.mem_cache.radix_cache"] = rc

    reg = types.ModuleType("sglang.srt.mem_cache.registry")
    reg.register_radix_cache_backend = lambda *a, **k: None
    sys.modules["sglang.srt.mem_cache.registry"] = reg

    utils = _load("sglang.srt.mem_cache.heter_cache.utils", f"{_HC}/utils.py")
    _load("sglang.srt.mem_cache.heter_cache.occupancy", f"{_HC}/occupancy.py")
    _load("sglang.srt.mem_cache.heter_cache.value", f"{_HC}/value.py")
    selector = _load("sglang.srt.mem_cache.heter_cache.selector", f"{_HC}/selector.py")
    _load("sglang.srt.mem_cache.heter_cache.bench_bridge", f"{_HC}/bench_bridge.py")
    _load("sglang.srt.mem_cache.heter_cache.node_share", f"{_HC}/node_share.py")
    regret_gate = _load("sglang.srt.mem_cache.heter_cache.regret_gate", f"{_HC}/regret_gate.py")
    override = _load("sglang.srt.mem_cache.heter_cache.override", f"{_HC}/override.py")
    ep = _load("sglang.srt.mem_cache.evict_policy", "sglang/srt/mem_cache/evict_policy.py")
    hrc = _load("sglang.srt.mem_cache.heter_cache.radix_cache", f"{_HC}/radix_cache.py")
    return utils, selector, regret_gate, override, ep, hrc


utils, selector, regret_gate, override, ep, hrc = _bootstrap()


class FakeNode:
    def __init__(
        self,
        nid: int,
        tokens: int,
        *,
        tag=None,
        lat: float = 0.0,
        session_id: str | None = None,
        reuse_count: int = 0,
        parent=None,
    ):
        self.id = nid
        self.key = list(range(tokens))
        self.value = list(range(tokens))
        self.queue_tag = tag or utils.HeterCacheNodeTag.QCHAT
        self.last_access_time = lat
        self.session_id = session_id
        self.reuse_count = reuse_count
        self.parent = parent
        self.children = {}


class TestHeterCacheRefactor(unittest.TestCase):
    def test_median(self):
        self.assertEqual(utils.median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(utils.median([10.0, 1.0, 3.0, 2.0]), 2.5)

    def test_session_registry_and_probe_order(self):
        clock = [0.0]
        seen = []

        class Probe(utils.SessionProbe):
            def on_charge(self, node, state, num_tokens):
                seen.append((state.d, state.L, num_tokens))

        reg = utils.SessionRegistry(probe=Probe(), now=lambda: clock[0])
        reg.observe_request("s1", rid="r1", prompt_len=100, now=1.0)
        reg.touch("s1", rid="r1", prompt_len=100, now=1.0)
        reg.attach("s1", 150)
        node = FakeNode(1, 150, session_id="s1")
        reg.charge(node)
        # on_charge 必须在 d 被截断到 L 之前看到原始 d=150
        self.assertEqual(seen, [(150, 100, 150)])
        self.assertEqual(reg.get("s1").d, 100)

    def test_strategy_invariants(self):
        clock = [100.0]
        st = ep.HeterCacheStrategy(force_agentic=True, now=lambda: clock[0])
        node = FakeNode(1, 64, tag=utils.HeterCacheNodeTag.QCHAT, lat=10.0)
        self.assertEqual(st.node_tag(node), utils.HeterCacheNodeTag.QAGENT)
        with self.assertRaises(NotImplementedError):
            st.get_priority(node)

    def test_regret_gate_window_and_abstain(self):
        clock = [0.0]
        params = utils.RegretGateParams(
            enabled=True, window_s=10.0, k=1.0, r_min=0.3, min_evicted=100, retain_s=20.0
        )
        gate = regret_gate.RegretGate(params)
        gate.use_clock(lambda: clock[0])

        gate.on_evict(key=101, queue="chat", span=200)
        gate.on_evict(key=102, queue="struct", span=200)
        gate.on_request(key=101)
        # 同一 stamp 只归因一次
        gate.on_request(key=101)
        self.assertAlmostEqual(gate.r("chat"), 1.0)
        self.assertAlmostEqual(gate.r("struct"), 0.0)
        self.assertTrue(gate.abstain("chat", ["chat", "struct"]))
        self.assertFalse(gate.abstain("struct", ["chat", "struct"]))

        clock[0] = 15.0  # 超过 window_s=10
        self.assertIsNone(gate.r("chat"))

    def test_release_node_order(self):
        events = []
        cache = hrc.HeterCacheRadixCache.__new__(hrc.HeterCacheRadixCache)
        cache.eviction_strategy = types.SimpleNamespace(
            record_evict=lambda n: events.append("record_evict")
        )
        cache.token_to_kv_pool_allocator = types.SimpleNamespace(
            free_segment=lambda v, start_pos=0: events.append("free_segment")
        )
        cache.kv_events = types.SimpleNamespace(
            record_remove=lambda n: events.append("record_remove")
        )
        cache._delete_leaf = lambda n: events.append("_delete_leaf")

        released = cache._release_node(FakeNode(1, 32))
        self.assertEqual(released, 32)
        self.assertEqual(
            events, ["record_evict", "free_segment", "record_remove", "_delete_leaf"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
