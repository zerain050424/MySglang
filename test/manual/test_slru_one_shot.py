"""SLRU / one-shot 判定的 characterization tests。

钉住四件事：

1. **一次性判定只看「会话类型 + 跨会话共享数」**：
   `one_shot = type ∈ {TRANS, STRUCT} and share_sessions < share_min`
2. **共享数达到阈值就退出一次性带**，即使创建者当初被判成 TRANS。
3. **`queue_tag` 不参与判定**：QSTRUCT 标签救不了私有的 TRANS 节点，
   QTRANS 标签也不会把已被多次复用的节点拉回一次性带。
4. **阈值同源**：`is_one_shot` 与 `SlruEvictPolicy.applies_to` 用同一个
   `share_min`，且共用同一本会话账。

    python test/manual/test_slru_one_shot.py
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)

from sglang.srt.mem_cache.evict_policy import SimpleHybridStrategy  # noqa: E402
from sglang.srt.mem_cache.uni_utils import (  # noqa: E402
    DEFAULT_SHARE_MIN,
    SessionRegistry,
    SessionState,
    SessionType,
    SimpleHybridQueueTag,
)
from sglang.srt.mem_cache.victim_override import SlruEvictPolicy  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


class FakeNode:
    """TreeNode 的最小替身：只带 one-shot / SLRU 判定读的字段。"""

    def __init__(self, session_id=None, tag=SimpleHybridQueueTag.QTRANS, decode=False,
                 share_sessions=0):
        self.value = [0] * 16
        self.key = [0] * 16
        self.session_id = session_id
        self.queue_tag = tag
        self.is_decode = decode
        self.share_sessions = share_sessions
        self.session_bloom = 0
        self.abs_start = 0
        self.last_access_time = 0.0


def make_strategy(session_types, *, force_agentic=False, **slru):
    """构造策略 + 私有会话账（不污染进程级单例）。"""
    st = SimpleHybridStrategy(
        slru=slru or None, force_agentic=force_agentic
    )
    reg = SessionRegistry()
    for sid, stype in session_types.items():
        reg._sessions[sid] = SessionState(session_id=sid, session_type=stype)
    # 策略与 SLRU 插件必须读同一本账
    st.session_registry = reg
    st.slru_evict.session_registry = reg
    return st, reg


def test_trans_unshared_vs_shared():
    """TRANS：没被别的会话复用过 -> one-shot；被 >=2 个会话复用过 -> 不再是。"""
    print("[1] TRANS: unshared is one-shot, shared is protected")
    st, _ = make_strategy({"t1": SessionType.TRANS})
    n = FakeNode(session_id="t1")

    check(st.is_one_shot(n) is True, "TRANS with share=0 must be one-shot")

    for i in range(st.slru_evict.share_min):
        st.slru_evict.on_node_hit(n, f"other-{i}")
    check(n.share_sessions == st.slru_evict.share_min,
          f"share_sessions, got {n.share_sessions}")
    check(st.is_one_shot(n) is False,
          "TRANS node reused by enough sessions must leave the one-shot band")


def test_struct_threshold():
    """STRUCT：share < 阈值 -> one-shot；达到阈值 -> 退出。"""
    print("[2] STRUCT: threshold is share_min")
    st, _ = make_strategy({"s1": SessionType.STRUCT})
    n = FakeNode(session_id="s1")
    thr = st.slru_evict.share_min
    check(thr == 2, f"expected threshold 2 (share < 2 <=> sessions <= 2), got {thr}")

    for k in range(thr):
        check(st.is_one_shot(n) is True, f"STRUCT share={k} must be one-shot")
        st.slru_evict.on_node_hit(n, f"other-{k}")
    check(n.share_sessions == thr, f"share_sessions, got {n.share_sessions}")
    check(st.is_one_shot(n) is False, "STRUCT at threshold must be protected")


def test_multiturn_never_one_shot():
    """CHAT / AGENTIC：共享数无关，永远不进一次性带。"""
    print("[3] CHAT / AGENTIC never one-shot")
    st, _ = make_strategy(
        {"c1": SessionType.CHAT, "a1": SessionType.AGENTIC}
    )
    for sid in ("c1", "a1"):
        n = FakeNode(session_id=sid, decode=True)
        check(st.is_one_shot(n) is False, f"{sid} share=0 must not be one-shot")
        for i in range(3):
            st.slru_evict.on_node_hit(n, f"x{i}")
        check(st.is_one_shot(n) is False, f"{sid} share=3 must not be one-shot")


def test_queue_tag_is_irrelevant():
    """标签不再参与判定：两个方向各验一次。"""
    print("[4] queue_tag must not influence the decision")
    st, _ = make_strategy({"t1": SessionType.TRANS, "s1": SessionType.STRUCT})

    # 方向 A：打了 QSTRUCT 标签的私有 TRANS 节点，仍是一次性内容
    private = FakeNode(session_id="t1", tag=SimpleHybridQueueTag.QSTRUCT)
    check(private.queue_tag == SimpleHybridQueueTag.QSTRUCT, "sanity: tag set")
    check(st.is_one_shot(private) is True,
          "a QSTRUCT-tagged but unshared TRANS node is still one-shot")

    # 方向 B：打着 QTRANS 标签、已被多次复用的节点，不再是
    reused = FakeNode(session_id="t1", tag=SimpleHybridQueueTag.QTRANS)
    for i in range(st.slru_evict.share_min):
        st.slru_evict.on_node_hit(reused, f"y{i}")
    check(st.is_one_shot(reused) is False,
          "a QTRANS-tagged but widely reused node must leave the one-shot band")
    check(reused.queue_tag == SimpleHybridQueueTag.QTRANS,
          "sanity: the tag itself was never consulted")


def test_creator_counted_at_insert():
    """插入即记创建者：share 0->1，且自读去重（私有上下文不会误达标）。"""
    print("[4b] creator is counted at insert; self re-reads dedup")
    st, _ = make_strategy({"s1": SessionType.STRUCT})
    node = FakeNode(session_id="s1")

    st.slru_evict.on_node_inserted(node)
    check(node.share_sessions == 1, f"insert must count creator, got {node.share_sessions}")
    check(node.session_bloom != 0, "creator's bloom bits must be written at insert")
    check(st.is_one_shot(node) is True, "share=1 (<min) is still one-shot")

    # 创建者后续轮次重读：必须去重
    for _ in range(3):
        st.slru_evict.on_node_hit(node, "s1")
    check(node.share_sessions == 1, f"self re-reads must dedup, got {node.share_sessions}")
    check(st.is_one_shot(node) is True, "private context must stay one-shot")
    check(st.slru_evict.applies_to(node) is False, "private context must not be protected")

    # 1 次外部命中即达阈值（这正是本次修改的目的）
    st.slru_evict.on_node_hit(node, "s2")
    check(node.share_sessions == 2, f"one foreign hit must reach min, got {node.share_sessions}")
    check(st.is_one_shot(node) is False, "cross-session template leaves the band")
    check(st.slru_evict.applies_to(node) is True, "cross-session template is protected")

    # 分裂出的父节点继承计数后不重复计
    split_parent = FakeNode(session_id="s1", share_sessions=1)
    split_parent.session_bloom = node.session_bloom
    st.slru_evict.on_node_inserted(split_parent)
    check(split_parent.share_sessions == 1,
          f"split parent must not double count, got {split_parent.share_sessions}")

    # 无 session 的节点不记
    orphan = FakeNode(session_id=None)
    st.slru_evict.on_node_inserted(orphan)
    check(orphan.share_sessions == 0, "node without session must not be counted")


def test_no_session_is_not_one_shot():
    """无会话证据（未带 session_id / 账本被 sweep）：不判 one-shot。"""
    print("[5] nodes without session evidence are not one-shot")
    st, _ = make_strategy({})
    n = FakeNode(session_id=None, tag=SimpleHybridQueueTag.QTRANS, decode=True)
    check(st.is_one_shot(n) is False, "no session -> not one-shot")
    ghost = FakeNode(session_id="never-registered", decode=True)
    check(st.is_one_shot(ghost) is False, "swept session -> not one-shot")


def test_is_one_shot_and_applies_to_agree():
    """同一阈值 + 同一本账：STRUCT 节点上两者互补。"""
    print("[6] is_one_shot and SlruEvictPolicy.applies_to are complementary")
    st, _ = make_strategy({"s1": SessionType.STRUCT})
    node = FakeNode(session_id="s1")

    check(st.slru_evict.applies_to(node) is False, "share=0 must not be SLRU-scoped")
    check(st.is_one_shot(node) is True, "share=0 must be one-shot")

    for i in range(st.slru_evict.share_min):
        st.slru_evict.on_node_hit(node, f"z{i}")
    check(st.slru_evict.applies_to(node) is True,
          "at threshold the node becomes SLRU-scoped")
    check(st.is_one_shot(node) is False, "at threshold the node leaves the band")


def test_works_under_force_agentic():
    """force_agentic 实验开关下，判定与账本仍然一致。"""
    print("[7] decision and ledger hold under force_agentic")
    for kwargs in (
        {},
        {"force_agentic": True},
    ):
        st, _ = make_strategy({"t1": SessionType.TRANS}, **kwargs)
        n = FakeNode(session_id="t1", tag=SimpleHybridQueueTag.QAGENT)
        check(st.is_one_shot(n) is True, f"{kwargs}: unshared TRANS is one-shot")
        for i in range(st.slru_evict.share_min):
            st.slru_evict.on_node_hit(n, f"w{i}")
        check(n.share_sessions == st.slru_evict.share_min,
              f"{kwargs}: ledger must still count, got {n.share_sessions}")
        check(st.is_one_shot(n) is False, f"{kwargs}: reused TRANS is protected")


def test_share_min_is_single_knob():
    """share_min 是唯一旋钮：配置 / one-shot / SLRU 三处同值。"""
    print("[8] share_min drives config, one-shot and SLRU together")
    st, _ = make_strategy({"s1": SessionType.STRUCT}, share_min=3)
    check(st.slru_evict.share_min == 3,
          f"SLRU threshold must follow config, got {st.slru_evict.share_min}")
    check(st.slru.share_min == 3, "SlruParams must carry the configured value")

    node = FakeNode(session_id="s1")
    for k in (0, 1):
        check(st.is_one_shot(node) is True, f"share={k} < 3 must be one-shot")
        st.slru_evict.on_node_hit(node, f"o{k}")
    check(node.share_sessions == 2, f"share_sessions, got {node.share_sessions}")
    check(st.is_one_shot(node) is True, "share=2 is still one-shot when min=3")
    check(st.slru_evict.applies_to(node) is False, "share=2 is not SLRU-scoped yet")

    st.slru_evict.on_node_hit(node, "o2")
    check(node.share_sessions == 3, f"share_sessions, got {node.share_sessions}")
    check(st.is_one_shot(node) is False, "share=3 leaves the one-shot band")
    check(st.slru_evict.applies_to(node) is True, "share=3 is SLRU-scoped")

    # 默认值：配置不写时两边都回落到 DEFAULT_SHARE_MIN
    d, _ = make_strategy({"s1": SessionType.STRUCT})
    check(d.slru_evict.share_min == DEFAULT_SHARE_MIN,
          "default threshold must come from DEFAULT_SHARE_MIN")
    check(DEFAULT_SHARE_MIN == 2, "documented default is 2 (sessions <= 2)")

    # 配置非法值由 SlruParams 拦住（<2 没有意义）
    try:
        SimpleHybridStrategy(slru={"share_min": 1})
    except ValueError as exc:
        check("share_min" in str(exc), f"unexpected message: {exc}")
    else:
        check(False, "share_min=1 must be rejected")


def main():
    tests = [
        test_trans_unshared_vs_shared,
        test_struct_threshold,
        test_multiturn_never_one_shot,
        test_queue_tag_is_irrelevant,
        test_creator_counted_at_insert,
        test_no_session_is_not_one_shot,
        test_is_one_shot_and_applies_to_agree,
        test_works_under_force_agentic,
        test_share_min_is_single_knob,
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
