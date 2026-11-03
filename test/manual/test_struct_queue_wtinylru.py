"""Characterization tests: Q_struct 队列状态机 + 淘汰适配器。

拆分（state machine / victim override）之前先钉住行为：

  StructQueueState      成员判定 / 三区 / 频数准入 / aging / 归属查询
                        —— 只回答「谁在队列里、在哪个区」，不回答「淘汰谁」
  StructTinyLruPolicy 把「区头」暴露成 VictimOverride 候选
                        —— 自身无状态，全部转发给状态机

这些用例全部走公开路径（on_* / eviction_candidate / owns / stats / snapshot），
不直接调私有方法，因此重构内部结构也不会让它们失真。
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)

from sglang.srt.mem_cache.evict_policy import SimpleHybridStrategy  # noqa: E402
from sglang.srt.mem_cache.struct_queue_policy import (  # noqa: E402
    REGION_NONE,
    REGION_PARKED,
    REGION_PROBATION,
    REGION_PROTECTED,
    REGION_WINDOW,
    StructQueueState,
)
from sglang.srt.mem_cache.uni_utils import SimpleHybridQueueTag  # noqa: E402
from sglang.srt.mem_cache.victim_override import (  # noqa: E402
    StructTinyLruPolicy,
    VictimOverride,
)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


class FakeNode:
    """TreeNode 的最小替身：只带 struct_queue_policy 读写的字段。"""

    _next_id = 0

    def __init__(self, tokens=16, parent=None, share=0):
        FakeNode._next_id += 1
        self.id = FakeNode._next_id
        self.value = [0] * tokens
        self.children = {}
        self.parent = parent
        self.lock_ref = 0
        self.evicted = False
        self.queue_tag = SimpleHybridQueueTag.QSTRUCT
        self.share_count = share
        self.rid_bloom = 0
        self.struct_ever_branch = False
        self.struct_region = REGION_NONE
        if parent is not None:
            parent.children[self.id] = self


def forest():
    """root ─┬─ 两个空叶子（让 root 成为分叉点）。"""
    root = FakeNode()
    FakeNode(parent=root)
    FakeNode(parent=root)
    return root


def fresh_leaf(root, share=0):
    return FakeNode(parent=root, share=share)


def state(**kw):
    cfg = dict(enabled=True, k=3, window_entries=64)
    cfg.update(kw)
    return StructQueueState(**cfg)


def test_region_by_share_count():
    """入队分档：share<2 且非分叉 -> window 不成立，分叉叶子进 window；
    share==2 -> probation；share>=k -> protected。"""
    print("[1] band mapping by share_count")
    root = forest()
    st = state()

    w = fresh_leaf(root, share=0)
    w.struct_ever_branch = True  # 曾经分叉、现在成叶子 -> window 成员
    st.on_became_leaf(w)
    check(w.struct_region == REGION_WINDOW, f"share=0 -> region={w.struct_region}, want window")

    p = fresh_leaf(root, share=2)
    st.on_became_leaf(p)
    check(p.struct_region == REGION_PROBATION, f"share=2 -> region={p.struct_region}, want probation")

    r = fresh_leaf(root, share=3)
    st.on_became_leaf(r)
    check(r.struct_region == REGION_PROTECTED, f"share=k -> region={r.struct_region}, want protected")

    # 非成员：share<2 且从未分叉
    n = fresh_leaf(root, share=1)
    st.on_became_leaf(n)
    check(n.struct_region == REGION_NONE, f"share=1 non-branch -> region={n.struct_region}, want none")


def test_member_detection():
    """成员判定：分叉点、或 share>=2 的主干（父链最终落到根）。"""
    print("[2] member detection: branch or shared trunk")
    root = forest()
    st = state()

    # 分叉节点本身：值是 None 时不可淘汰，但 owns() 只看 struct_region
    branch = FakeNode(parent=root)
    FakeNode(parent=branch)
    FakeNode(parent=branch)
    st.on_children_added(branch)
    check(branch.struct_ever_branch, "branch node must get sticky struct_ever_branch")
    check(branch.struct_region == REGION_PARKED,
          f"non-leaf branch is a member but not evictable -> parked, got {branch.struct_region}")

    # share>=2 的主干叶子：成员
    trunk = fresh_leaf(root, share=2)
    st.on_became_leaf(trunk)
    check(st.owns(trunk), "share>=2 leaf must be owned")

    # share<2 普通叶子：不是成员
    plain = fresh_leaf(root, share=1)
    st.on_became_leaf(plain)
    check(not st.owns(plain), "share<2 non-branch leaf must not be owned")


def test_owns_respects_enabled():
    print("[3] owns() is O(1) and respects enabled")
    root = forest()
    st = state()
    n = fresh_leaf(root, share=3)
    st.on_became_leaf(n)
    check(st.owns(n), "enabled state owns member")
    check(not st.owns(None), "owns(None) is False")
    off = state(enabled=False)
    check(not off.owns(n), "disabled state owns nothing")
    check(off.eviction_candidate() is None, "disabled state has no candidate")


def test_eviction_candidate_region_order():
    """淘汰顺序 = 区顺序 window -> probation -> protected；只剩 protected 时记 forced。"""
    print("[4] eviction_candidate follows window -> probation -> protected")
    root = forest()
    st = state()

    w = fresh_leaf(root, share=0)
    w.struct_ever_branch = True
    st.on_became_leaf(w)
    p = fresh_leaf(root, share=2)
    st.on_became_leaf(p)
    r = fresh_leaf(root, share=3)
    st.on_became_leaf(r)

    check(st.eviction_candidate() is w, "window drains first")
    st.on_delete(w)
    check(st.eviction_candidate() is p, "then probation")
    st.on_delete(p)
    check(st.stats["forced"] == 0, f"forced must stay 0, got {st.stats['forced']}")
    check(st.eviction_candidate() is r, "then protected")
    check(st.stats["forced"] == 1, f"protected-only eviction must count forced, got {st.stats['forced']}")
    st.on_delete(r)
    check(st.eviction_candidate() is None, "empty queue has no candidate")


def test_eviction_candidate_skips_locked_leaf():
    """上锁/有未淘汰子的节点不能被选中；索引里的陈旧条目惰性清理。"""
    print("[5] eviction_candidate skips non-evictable leaves")
    root = forest()
    st = state()
    # 注意：on_became_leaf 走 origin="inherit"，probation 成员插到区头。
    # 后建的在区头，所以先建 free 再建 locked，让 locked 成为区头。
    free = fresh_leaf(root, share=2)
    st.on_became_leaf(free)
    locked = fresh_leaf(root, share=2)
    st.on_became_leaf(locked)

    locked.lock_ref = 1  # 上锁后不再可淘汰
    check(st.eviction_candidate() is free, "locked leaf must be skipped")
    check(locked.struct_region == REGION_NONE, "stale index entry must be lazily cleaned")


def test_admission_rejects_once_only_document():
    """准入：window 超配额且候选频数低于区头 -> 降级 QTRANS，交给 one-shot 带处决。"""
    print("[6] admission rejects the single-use document segment")
    root = forest()
    st = state(window_entries=1)

    victim = fresh_leaf(root, share=2)  # probation 区头，频数 2
    st.on_became_leaf(victim)

    first = fresh_leaf(root, share=0)
    first.struct_ever_branch = True
    st.on_became_leaf(first)
    check(first.struct_region == REGION_WINDOW, "first window entry admitted")
    check(st.stats["reject"] == 0, "no rejection while window fits")

    second = fresh_leaf(root, share=0)
    second.struct_ever_branch = True
    st.on_became_leaf(second)  # 溢出 -> 与 probation 区头比较 -> 拒绝

    rejected = [n for n in (first, second) if n.struct_region == REGION_NONE]
    check(len(rejected) == 1, f"exactly one window entry must be rejected, got {len(rejected)}")
    loser = rejected[0]
    check(loser.queue_tag == SimpleHybridQueueTag.QTRANS,
          f"rejected member must be downgraded to QTRANS, got {loser.queue_tag}")
    check(not st.owns(loser), "rejected member must not be owned")
    check(st.stats["reject"] == 1, f"reject counter, got {st.stats['reject']}")
    check(st.owns(second) or st.owns(first), "winner stays a member")


def test_park_and_reinherit():
    """不可淘汰（长出子节点/被上锁）-> 挂起；重新成为叶子 -> 按继承表落区。"""
    print("[7] park on no-longer-leaf, re-place on became-leaf")
    root = forest()
    st = state()
    n = fresh_leaf(root, share=3)
    st.on_became_leaf(n)
    check(n.struct_region == REGION_PROTECTED, "precondition: protected member")

    st.on_no_longer_leaf(n)
    check(n.struct_region == REGION_PARKED, f"parked, got {n.struct_region}")
    check(st.stats["park"] == 1, f"park counter, got {st.stats['park']}")

    st.on_became_leaf(n)
    check(n.struct_region == REGION_PROTECTED, "re-inherits protected band")
    check(n.share_count == 3, f"share must be kept across parking, got {n.share_count}")


def test_aging_halves_and_retires():
    """aging：share 减半、bloom 清空；减到 <2 且非分叉 -> 退队并降级。"""
    print("[8] aging halves share_count and retires weak members")
    root = forest()
    # 入区本身也算一个事件，所以 epoch_events=2 时：入区=1，再一个事件触发 aging。
    st = state(epoch_events=2, k=4)

    member = fresh_leaf(root, share=4)
    st.on_became_leaf(member)
    check(member.struct_region == REGION_PROTECTED, "precondition: protected member")
    check(st.stats["ages"] == 0, f"placement must not age yet, got {st.stats['ages']}")

    trigger = fresh_leaf(root, share=0)  # 用 delete 事件推进纪元
    st.on_delete(trigger)
    check(st.stats["ages"] == 1, f"one aging epoch, got {st.stats['ages']}")
    check(member.share_count == 2, f"share halved, got {member.share_count}")
    check(member.struct_region == REGION_PROBATION, f"4->2 moves to probation, got {member.struct_region}")

    st.on_delete(trigger)
    st.on_delete(trigger)  # 攒满一个纪元
    check(st.stats["ages"] == 2, f"second aging epoch, got {st.stats['ages']}")
    # _retire() 把 share_count / rid_bloom / ever_branch 一并清掉，所以是 0 而不是 1。
    check(member.share_count == 0, f"retire zeroes share_count, got {member.share_count}")
    check(member.struct_region == REGION_NONE, "retired after dropping below MIN_SHARE")
    check(member.queue_tag == SimpleHybridQueueTag.QTRANS, "retired member downgraded to QTRANS")
    # 已知缺陷（本次重构不修）：_age 先置 struct_region=NONE 再调 _retire，
    # _retire 里 `was_member` 恒为假，所以 retire 计数器永远不会涨。钉住现状，
    # 修的时候必须同步改这里。
    check(st.stats["retire"] == 0, f"known: retire counter stays 0, got {st.stats['retire']}")


def test_snapshot_and_reset():
    print("[9] snapshot reports regions and counters; reset clears the index")
    root = forest()
    st = state()
    n = fresh_leaf(root, share=3)
    st.on_became_leaf(n)

    snap = st.snapshot()
    for key in ("window", "probation", "protected", "parked", "window_tokens",
                "protected_tokens", "epoch_events", "admit", "reject"):
        check(key in snap, f"snapshot missing key {key!r}")
    check(snap["protected"] == 1, f"snapshot protected={snap['protected']}")
    check(snap["protected_tokens"] == 16, f"snapshot protected_tokens={snap['protected_tokens']}")

    st.reset()
    check(n.struct_region == REGION_NONE, "reset must clear node region")
    check(not st.owns(n), "reset must clear ownership")
    check(st.snapshot()["protected"] == 0, "reset must clear region index")


def test_state_is_not_an_override():
    """拆分契约：状态机不是插件，缓存层依赖的是它，不是 override。"""
    print("[10] state machine must not be a VictimOverride")
    check(not issubclass(StructQueueState, VictimOverride),
          "StructQueueState must stay independent of the override contract")


def test_override_delegates_to_state():
    """插件只有转发，没有自己的状态。"""
    print("[11] override delegates applies_to / pick / stats to the state machine")
    root = forest()
    st = state()
    ov = StructTinyLruPolicy(st)

    check(isinstance(ov, VictimOverride), "adapter must implement VictimOverride")
    check(ov.queue is SimpleHybridQueueTag.QSTRUCT, f"queue={ov.queue}, want QSTRUCT")

    m = fresh_leaf(root, share=3)
    st.on_became_leaf(m)
    plain = fresh_leaf(root, share=0)

    check(ov.applies_to(m) is True, "applies_to must match owns")
    check(ov.applies_to(plain) is False, "applies_to must reject non-members")
    check(ov.pick() is st.eviction_candidate(), "pick must match eviction_candidate")

    got = ov.stats()
    check(isinstance(got, dict), f"stats() must return dict, got {type(got).__name__}")
    check(got == st.stats, "stats() must mirror state counters")
    got["admit"] = 999  # 必须是副本：改返回值不得污染状态机
    check(st.stats["admit"] != 999, "stats() must return a copy, not the live dict")

    ov.reset()
    check(not st.owns(m), "override.reset() must reach the state machine")


def test_strategy_wiring():
    """策略层：struct 指向状态机（缓存层用它），TinyLru_evict 进 victim_overrides。"""
    print("[12] strategy wiring: state for cache, override for eviction")
    on = SimpleHybridStrategy(tiny_lru={"policy": "wtinylru"})
    check(isinstance(on.structState, StructQueueState),
          f"strategy.structState must be the state machine, got {type(on.structState).__name__}")
    check(isinstance(on.TinyLru_evict, StructTinyLruPolicy),
          f"strategy.TinyLru_evict must be the adapter, got {type(on.TinyLru_evict).__name__}")
    check(on.TinyLru_evict.state is on.structState, "adapter must wrap that same state machine")
    check(on.TinyLru_evict in on.victim_overrides, "adapter must be registered as an override")
    check(all(isinstance(o, VictimOverride) for o in on.victim_overrides),
          "every registered override must implement VictimOverride")
    check(hasattr(on.structState, "owns"),
          "cache keeps the state machine for the one-shot dead-chain gate")

    off = SimpleHybridStrategy(tiny_lru={"policy": "off"})
    check(off.structState is None and off.TinyLru_evict is None,
          "policy=off must build neither state machine nor adapter")
    check(off.victim_overrides == (), f"policy=off must register no override, got {off.victim_overrides}")


def main():
    tests = [
        test_region_by_share_count,
        test_member_detection,
        test_owns_respects_enabled,
        test_eviction_candidate_region_order,
        test_eviction_candidate_skips_locked_leaf,
        test_admission_rejects_once_only_document,
        test_park_and_reinherit,
        test_aging_halves_and_retires,
        test_snapshot_and_reset,
        test_state_is_not_an_override,
        test_override_delegates_to_state,
        test_strategy_wiring,
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
