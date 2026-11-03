"""T-LRU override 插件：高水位记账 + tel-safe 排序 + 与 TLRUStrategy 对拍。

被测对象：
  victim_override.TlruEvictPolicy —— QAGENT 的尾巴优先修正
  uni_utils.TlruParams            —— 参数与校验
  radix_cache                      —— 插入/分裂时回调 on_node_inserted（本文件用替身模拟）

最后两例把插件的 tel_safe 判定与 evict_policy.TLRUStrategy.get_priority 逐点对拍，
保证两边公式没有漂移。
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)

from sglang.srt.mem_cache.evict_policy import (  # noqa: E402
    SimpleHybridStrategy,
    TLRUStrategy,
)
from sglang.srt.mem_cache.uni_utils import (  # noqa: E402
    SimpleHybridQueueTag,
    TlruParams,
)
from sglang.srt.mem_cache.victim_override import (  # noqa: E402
    TlruEvictPolicy,
    VictimOverride,
)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


class FakeNode:
    """TreeNode 的最小子集：key 长度、父子链、标签、时间和 _tlru_* 字段。"""

    _next_id = 0

    def __init__(self, tokens=0, parent=None, tag=SimpleHybridQueueTag.QAGENT):
        FakeNode._next_id += 1
        self.id = FakeNode._next_id
        self.key = list(range(tokens))
        self.value = [0] * tokens
        self.children = {}
        self.parent = parent
        self.queue_tag = tag
        self.last_access_time = 0.0
        self._tlru_cached_prefix_len = 0
        self._tlru_history_len = 0
        if parent is not None:
            parent.children[self.id] = self

    def __lt__(self, other):  # heapq / tuple 比较用
        return self.last_access_time < other.last_access_time


def policy(threshold=4096, q_hat=1024):
    return TlruEvictPolicy(
        TlruParams(enabled=True, threshold=threshold, next_prompt_estimate=q_hat)
    )


def tel_safe_of(pol, node):
    """复算插件内部的 tel_safe 判定（仅供断言用）。"""
    budget = max(node._tlru_history_len + pol.params.next_prompt_estimate - pol.params.threshold, 0)
    cached_without = node._tlru_cached_prefix_len - len(node.key)
    return cached_without >= budget


def test_bookkeeping_chain():
    """插入链：cached_prefix 累加，高水位抬到最深并上抬所有祖先。"""
    print("[1] insert bookkeeping: cached_prefix accumulates, history raised to ancestors")
    root = FakeNode()
    pol = policy()

    a = FakeNode(100, root)
    pol.on_node_inserted(a)
    b = FakeNode(200, a)
    pol.on_node_inserted(b)

    check(a._tlru_cached_prefix_len == 100,
          f"a.cached={a._tlru_cached_prefix_len}, want 100")
    check(b._tlru_cached_prefix_len == 300,
          f"b.cached={b._tlru_cached_prefix_len}, want 300")
    check(b._tlru_history_len == 300, f"b.history={b._tlru_history_len}, want 300")
    check(a._tlru_history_len == 300,
          f"ancestor a.history={a._tlru_history_len}, want 300 (raised)")
    check(root._tlru_history_len == 300,
          f"root.history={root._tlru_history_len}, want 300 (raised)")


def test_bookkeeping_split():
    """分裂：新父节点位于 split 处，并继承更深子分支的高水位。"""
    print("[2] split bookkeeping: new parent inherits the child's high-water")
    root = FakeNode()
    pol = policy()
    a = FakeNode(100, root)
    pol.on_node_inserted(a)
    child = FakeNode(200, a)          # 深度 300，高水位 300
    pol.on_node_inserted(child)

    # 100..150 处分裂：新父节点 key=50，唯一子树是 child
    new = FakeNode(50, a)
    new.children = {child.id: child}
    child.parent = new
    pol.on_node_inserted(new)

    check(new._tlru_cached_prefix_len == 150,
          f"split node cached={new._tlru_cached_prefix_len}, want 150")
    check(new._tlru_history_len == 300,
          f"split node history={new._tlru_history_len}, want 300 (inherited)")
    check(child._tlru_cached_prefix_len == 300,
          "child keeps its own depth across the split")


def test_history_never_decreases():
    """高水位单调：删掉深尾后再插浅分支，水位不回落（T-LRU 的关键性质）。"""
    print("[3] history is monotone: trimming the tail must not lower it")
    root = FakeNode()
    pol = policy()
    deep = FakeNode(8000, root)
    pol.on_node_inserted(deep)
    check(root._tlru_history_len == 8000, "precondition: raised to 8000")

    # 模拟深尾被淘汰后重新插入一条更短的分支
    deep.parent.children.clear()
    short = FakeNode(600, root)
    pol.on_node_inserted(short)

    check(root._tlru_history_len == 8000,
          f"root.history={root._tlru_history_len}, want 8000 (must not shrink)")
    check(short._tlru_history_len == 600,
          f"new short branch history={short._tlru_history_len}, want 600")
    check(short._tlru_cached_prefix_len == 600,
          f"new short branch cached={short._tlru_cached_prefix_len}, want 600")


def test_tel_safe_beats_recency():
    """tel-safe 的尾巴先走，即使它比另一个候选更新（phase 1 压过 phase 2）。"""
    print("[4] tel-safe branch drains before recency (phase 1 over phase 2)")
    pol = policy(threshold=4096, q_hat=1024)
    root = FakeNode()

    # L=8192 -> budget = 8192 + 1024 - 4096 = 5120
    deep = FakeNode(1024, root)       # cached=1024, key=1024
    deep._tlru_history_len = 8192
    deep._tlru_cached_prefix_len = 8192
    check(tel_safe_of(pol, deep), "precondition: deep node is tel-safe")

    shallow = FakeNode(512, root)     # cached=512 -> cached_without=512 < 5120
    shallow._tlru_history_len = 8192
    shallow._tlru_cached_prefix_len = 512
    check(not tel_safe_of(pol, shallow), "precondition: shallow node is not tel-safe")

    deep.last_access_time = 900.0     # 更新
    shallow.last_access_time = 1.0    # 更旧

    got = pol.pick([deep, shallow], registry=None)
    check(got is deep,
          "tel-safe node must win even though the other is older")
    check(pol.tel_safe_picks == 1 and pol.lru_picks == 0,
          f"counters: tel_safe={pol.tel_safe_picks} lru={pol.lru_picks}")

    got2 = pol.pick([shallow], registry=None)
    check(got2 is shallow, "falls back to the only remaining candidate")
    check(pol.lru_picks == 1, f"lru_picks={pol.lru_picks}, want 1")


def test_tel_safe_boundary():
    """budget 边界：cached_without 恰好等于 budget 时算 tel-safe（>= 而非 >）。"""
    print("[5] tel-safe boundary is >= budget")
    pol = policy(threshold=4096, q_hat=1024)
    node = FakeNode(0)
    node._tlru_history_len = 5120          # budget = 5120 + 1024 - 4096 = 2048
    node._tlru_cached_prefix_len = 2048 + 100
    node.key = list(range(100))            # cached_without = 2048 == budget
    check(tel_safe_of(pol, node), "cached_without == budget must be tel-safe")

    # 注意方向：key 越长，cached_without = cached_prefix - len(key) 越小
    node.key = list(range(101))
    check(not tel_safe_of(pol, node), "cached_without < budget must not be tel-safe")


def test_scope_is_agentic_only():
    print("[6] applies_to only covers the agentic queue")
    pol = policy()
    check(pol.applies_to(FakeNode(10, tag=SimpleHybridQueueTag.QAGENT)),
          "QAGENT belongs to the override")
    for tag in (SimpleHybridQueueTag.QCHAT, SimpleHybridQueueTag.QSTRUCT,
                SimpleHybridQueueTag.QTRANS):
        check(not pol.applies_to(FakeNode(10, tag=tag)),
              f"{tag.value} must stay outside the override")
    check(not pol.applies_to(None), "applies_to(None) is False")


def test_no_candidate_falls_back():
    """没有 QAGENT 叶子时返回 None（交回全局候选），并计数。"""
    print("[7] no agentic leaf -> None + fallback counter")
    pol = policy()
    chat = FakeNode(10, tag=SimpleHybridQueueTag.QCHAT)
    check(pol.pick([chat], registry=None) is None, "pick must return None")
    check(pol.fallbacks == 1, f"fallbacks={pol.fallbacks}, want 1")


def test_params_validation():
    print("[8] param validation mirrors the CLI guard")
    try:
        TlruParams(enabled=True, threshold=1024, next_prompt_estimate=1024)
        check(False, "threshold == q_hat must be rejected when enabled")
    except ValueError as exc:
        check("must be > tlru.next_prompt_estimate" in str(exc),
              f"unexpected message: {exc}")

    # 关闭时不校验次序，允许保留占位值
    off = TlruParams(enabled=False, threshold=0, next_prompt_estimate=0)
    check(off.enabled is False, "disabled params are accepted")

    try:
        TlruParams(enabled=True, threshold=4096, next_prompt_estimate=-1)
        check(False, "negative q_hat must be rejected")
    except ValueError:
        check(True, "negative q_hat rejected")


def test_strategy_wiring():
    print("[9] strategy wiring and mutual exclusion with fair_rr")
    on = SimpleHybridStrategy(
        tlru={"enabled": True, "threshold": 4096, "next_prompt_estimate": 1024}
    )
    check(isinstance(on.tlru_evict, TlruEvictPolicy), "tlru_evict built")
    check(on.tlru_evict in on.victim_overrides, "registered as an override")
    check(all(isinstance(o, VictimOverride) for o in on.victim_overrides),
          "every override honours the contract")

    off = SimpleHybridStrategy()
    check(off.tlru_evict is None, "disabled by default")
    check(off.victim_overrides == (), "no override when disabled")

    try:
        SimpleHybridStrategy(
            rr={"fair_rr": True},
            tlru={"enabled": True, "threshold": 4096, "next_prompt_estimate": 1024},
        )
        check(False, "fair_rr + tlru must be rejected")
    except ValueError as exc:
        check("mutually exclusive" in str(exc), f"unexpected message: {exc}")


def test_matches_tlru_strategy_exactly():
    """与 TLRUStrategy.get_priority 逐点对拍：tel_safe 与两档优先级完全一致。"""
    print("[10] tel_safe agrees with evict_policy.TLRUStrategy on a grid")
    threshold, q_hat = 4096, 1024
    pol = policy(threshold=threshold, q_hat=q_hat)
    ref = TLRUStrategy(threshold=threshold, next_prompt_estimate=q_hat)

    mismatches = []
    for history in (0, 1024, 4096, 5120, 6144, 8192, 16384):
        for cached_prefix in (0, 512, 2048, 4096, 5120, 6144, 8192, 16384):
            for key_len in (128, 512, 1024):
                node = FakeNode(0)
                node._tlru_history_len = history
                node._tlru_cached_prefix_len = cached_prefix
                node.key = list(range(key_len))
                node.last_access_time = 42.0

                want = ref.get_priority(node)          # (-1|0, last_access_time)
                # 插件内部用 (0|1) 编码，排序等价；对拍时映射回 (-1|0)
                got = (-1 if tel_safe_of(pol, node) else 0, node.last_access_time)
                if got != want:
                    mismatches.append((history, cached_prefix, key_len, want, got))

    check(not mismatches, f"{len(mismatches)} mismatches vs TLRUStrategy, e.g. {mismatches[:2]}")


def main():
    tests = [
        test_bookkeeping_chain,
        test_bookkeeping_split,
        test_history_never_decreases,
        test_tel_safe_beats_recency,
        test_tel_safe_boundary,
        test_scope_is_agentic_only,
        test_no_candidate_falls_back,
        test_params_validation,
        test_strategy_wiring,
        test_matches_tlru_strategy_exactly,
    ]
    for t in tests:
        try:
            t()
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
