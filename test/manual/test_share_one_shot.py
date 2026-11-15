"""共享保护段 / one-shot 判定的 characterization tests。

钉住四件事：

1. **一次性判定只看「会话类型 + 跨会话共享数 + 是否被匹配过」**：
   `one_shot = type ∈ {TRANS, STRUCT} and share_sessions < share_min and hit_count == 0`
2. **共享数达到阈值就退出一次性带**，即使创建者当初被判成 TRANS。
3. **`queue_tag` 不参与判定**：QSTRUCT 标签救不了私有的 TRANS 节点，
   QTRANS 标签也不会把已被多次复用的节点拉回一次性带。
4. **阈值同源**：`is_one_shot` 与共享保护段用同一个
   `share_min`，且共用同一本会话账。
5. **时间豁免（轴一）**：归属会话仍活跃（最近请求或节点最近访问不足
   `ONE_SHOT_IDLE_S`）的未匹配节点不进带；搁置满窗口后原判据生效。

    python test/manual/test_share_one_shot.py
"""

import os
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)

from sglang.srt.mem_cache.evict_policy import HeterCacheStrategy  # noqa: E402
from sglang.srt.mem_cache.heter_cache.utils import (  # noqa: E402
    DEFAULT_SHARE_MIN,
    ONE_SHOT_IDLE_S,
    SessionRegistry,
    SessionState,
    SessionType,
    HeterCacheNodeTag,
)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


class FakeNode:
    """TreeNode 的最小替身：只带 one-shot / 共享保护段判定读的字段。"""

    def __init__(self, session_id=None, tag=HeterCacheNodeTag.QTRANS, decode=False,
                 share_sessions=0, hit_count=0):
        self.value = [0] * 16
        self.key = [0] * 16
        self.session_id = session_id
        self.queue_tag = tag
        self.is_decode = decode
        self.share_sessions = share_sessions
        self.session_bloom = 0
        self.abs_start = 0
        self.last_access_time = 0.0
        self.hit_count = hit_count


def _in_share_scope(strategy, node) -> bool:
    """节点是否落在共享保护段内。

    判据即 is_aged / is_one_shot 里的那一对条件（会话类型 STRUCT 且共享数
    达标）；测试直接断言判据本身，不再借道一个无人调用的选点器。
    """
    st = strategy.session_registry.get(getattr(node, "session_id", None))
    if st is None:
        return False
    return (
        st.session_type == SessionType.STRUCT
        and strategy.node_share.share_of(node) >= strategy.node_share.share_min
    )


def make_strategy(session_types, *, force_agentic=False, now=None, **share):
    """构造策略 + 私有会话账（不污染进程级单例）。

    now: 时间源注入（轴一的时间豁免用假时钟控制窗口推进）；缺省用真实时钟，
    此时测试里的 SessionState.last_active 与 FakeNode.last_access_time 都是
    0.0，锚点为 0，豁免窗口天然满足，旧用例行为不变。
    """
    st = HeterCacheStrategy(
        share=share or None,
        force_agentic=force_agentic,
        now=now,
    )
    reg = SessionRegistry()
    for sid, stype in session_types.items():
        reg._sessions[sid] = SessionState(session_id=sid, session_type=stype)
    # one-shot 与老化判据都从策略持有的会话账读证据
    st.session_registry = reg
    return st, reg


def test_trans_unshared_vs_shared():
    """TRANS：没被别的会话复用过 -> one-shot；被 >=2 个会话复用过 -> 不再是。"""
    print("[1] TRANS: unshared is one-shot, shared is protected")
    st, _ = make_strategy({"t1": SessionType.TRANS})
    n = FakeNode(session_id="t1")

    check(st.is_one_shot(n) is True, "TRANS with share=0 must be one-shot")

    for i in range(st.node_share.share_min):
        st.node_share.on_hit(n, f"other-{i}")
    check(n.share_sessions == st.node_share.share_min,
          f"share_sessions, got {n.share_sessions}")
    check(st.is_one_shot(n) is False,
          "TRANS node reused by enough sessions must leave the one-shot band")


def test_struct_threshold():
    """STRUCT：share < 阈值 -> one-shot；达到阈值 -> 退出。"""
    print("[2] STRUCT: threshold is share_min")
    st, _ = make_strategy({"s1": SessionType.STRUCT})
    n = FakeNode(session_id="s1")
    thr = st.node_share.share_min
    check(thr == 2, f"expected threshold 2 (share < 2 <=> sessions <= 2), got {thr}")

    for k in range(thr):
        check(st.is_one_shot(n) is True, f"STRUCT share={k} must be one-shot")
        st.node_share.on_hit(n, f"other-{k}")
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
            st.node_share.on_hit(n, f"x{i}")
        check(st.is_one_shot(n) is False, f"{sid} share=3 must not be one-shot")


def test_queue_tag_is_irrelevant():
    """标签不再参与判定：两个方向各验一次。"""
    print("[4] queue_tag must not influence the decision")
    st, _ = make_strategy({"t1": SessionType.TRANS, "s1": SessionType.STRUCT})

    # 方向 A：打了 QSTRUCT 标签的私有 TRANS 节点，仍是一次性内容
    private = FakeNode(session_id="t1", tag=HeterCacheNodeTag.QSTRUCT)
    check(private.queue_tag == HeterCacheNodeTag.QSTRUCT, "sanity: tag set")
    check(st.is_one_shot(private) is True,
          "a QSTRUCT-tagged but unshared TRANS node is still one-shot")

    # 方向 B：打着 QTRANS 标签、已被多次复用的节点，不再是
    reused = FakeNode(session_id="t1", tag=HeterCacheNodeTag.QTRANS)
    for i in range(st.node_share.share_min):
        st.node_share.on_hit(reused, f"y{i}")
    check(st.is_one_shot(reused) is False,
          "a QTRANS-tagged but widely reused node must leave the one-shot band")
    check(reused.queue_tag == HeterCacheNodeTag.QTRANS,
          "sanity: the tag itself was never consulted")


def test_ledger_semantics():
    """账本语义：计数只来自「命中」；自读去重，首次外部命中即达阈值。"""
    print("[4b] ledger: first self-hit counts 1, self re-reads dedup, 1 foreign hit promotes")
    st, _ = make_strategy({"s1": SessionType.STRUCT})
    node = FakeNode(session_id="s1")

    check(node.share_sessions == 0, "a fresh node has no counted session")
    check(st.is_one_shot(node) is True, "share=0 is one-shot")

    # 创建者第一次重读：计入 1
    st.node_share.on_hit(node, "s1")
    check(node.share_sessions == 1, f"first self-hit must count, got {node.share_sessions}")
    check(node.session_bloom != 0, "session bloom must be written on the first counted hit")
    check(st.is_one_shot(node) is True, "share=1 is still one-shot")

    # 创建者后续轮次重读：去重
    for _ in range(3):
        st.node_share.on_hit(node, "s1")
    check(node.share_sessions == 1, f"self re-reads must dedup, got {node.share_sessions}")
    check(_in_share_scope(st, node) is False, "private context must not be protected")

    # 1 次外部命中即达阈值
    st.node_share.on_hit(node, "s2")
    check(node.share_sessions == 2, f"one foreign hit must reach min, got {node.share_sessions}")
    check(st.is_one_shot(node) is False, "cross-session template leaves the band")
    check(_in_share_scope(st, node) is True, "cross-session template is protected")

    # 无 session 的节点不记
    orphan = FakeNode(session_id=None)
    st.node_share.on_hit(orphan, None)
    check(orphan.share_sessions == 0, "node without session must not be counted")


def test_no_session_is_not_one_shot():
    """无会话证据（未带 session_id / 账本被 sweep）：不判 one-shot。"""
    print("[5] nodes without session evidence are not one-shot")
    st, _ = make_strategy({})
    n = FakeNode(session_id=None, tag=HeterCacheNodeTag.QTRANS, decode=True)
    check(st.is_one_shot(n) is False, "no session -> not one-shot")
    ghost = FakeNode(session_id="never-registered", decode=True)
    check(st.is_one_shot(ghost) is False, "swept session -> not one-shot")


def test_is_one_shot_and_pick_scope_agree():
    """同一阈值 + 同一本账：STRUCT 节点上两者互补。"""
    print("[6] is_one_shot and share scope are complementary")
    st, _ = make_strategy({"s1": SessionType.STRUCT})
    node = FakeNode(session_id="s1")

    check(_in_share_scope(st, node) is False, "share=0 must not be protected")
    check(st.is_one_shot(node) is True, "share=0 must be one-shot")

    for i in range(st.node_share.share_min):
        st.node_share.on_hit(node, f"z{i}")
    check(_in_share_scope(st, node) is True,
          "at threshold the node becomes protected")
    check(st.is_one_shot(node) is False, "at threshold the node leaves the band")


def test_works_under_force_agentic():
    """force_agentic 实验开关下，判定与账本仍然一致。"""
    print("[7] decision and ledger hold under force_agentic")
    for kwargs in (
        {},
        {"force_agentic": True},
    ):
        st, _ = make_strategy({"t1": SessionType.TRANS}, **kwargs)
        n = FakeNode(session_id="t1", tag=HeterCacheNodeTag.QAGENT)
        # 过 90s 豁免但未到 3600s 老化 TTL：两种判据都不该干扰本条
        n.last_access_time = time.monotonic() - 200
        check(st.is_one_shot(n) is True, f"{kwargs}: unshared TRANS is one-shot")
        for i in range(st.node_share.share_min):
            st.node_share.on_hit(n, f"w{i}")
        check(n.share_sessions == st.node_share.share_min,
              f"{kwargs}: ledger must still count, got {n.share_sessions}")
        check(st.is_one_shot(n) is False, f"{kwargs}: reused TRANS is protected")


def test_share_min_is_single_knob():
    """share_min 是唯一旋钮：配置 / one-shot / 保护段三处同值。"""
    print("[8] share_min drives config, one-shot and protection together")
    st, _ = make_strategy({"s1": SessionType.STRUCT}, share_min=3)
    check(st.node_share.share_min == 3,
          f"share threshold must follow config, got {st.node_share.share_min}")
    check(st.share.share_min == 3, "ShareParams must carry the configured value")

    node = FakeNode(session_id="s1")
    for k in (0, 1):
        check(st.is_one_shot(node) is True, f"share={k} < 3 must be one-shot")
        st.node_share.on_hit(node, f"o{k}")
    check(node.share_sessions == 2, f"share_sessions, got {node.share_sessions}")
    check(st.is_one_shot(node) is True, "share=2 is still one-shot when min=3")
    check(_in_share_scope(st, node) is False, "share=2 is not protected yet")

    st.node_share.on_hit(node, "o2")
    check(node.share_sessions == 3, f"share_sessions, got {node.share_sessions}")
    check(st.is_one_shot(node) is False, "share=3 leaves the one-shot band")
    check(_in_share_scope(st, node) is True, "share=3 is protected")

    # 默认值：配置不写时两边都回落到 DEFAULT_SHARE_MIN
    d, _ = make_strategy({"s1": SessionType.STRUCT})
    check(d.node_share.share_min == DEFAULT_SHARE_MIN,
          "default threshold must come from DEFAULT_SHARE_MIN")
    check(DEFAULT_SHARE_MIN == 2, "documented default is 2 (sessions <= 2)")

    # 配置非法值由 ShareParams 拦住（<2 没有意义）
    try:
        HeterCacheStrategy(share={"share_min": 1})
    except ValueError as exc:
        check("share_min" in str(exc), f"unexpected message: {exc}")
    else:
        check(False, "share_min=1 must be rejected")


def test_once_read_leaves_one_shot_band():
    """F1：hit_count>=1 的 TRANS/STRUCT 节点离开 one-shot 带。

    机制背景：会话前 MULTI_TURN_MIN 轮创建的节点也带 TRANS/STRUCT 标签，
    仅凭类型+共享数判定会让它们在创建者下一轮重读到来之前就被扫描流清掉。
    上游 SLRU 的 hit_count>=2 晋升同理（创建者自己的重读是合法证据）。
    """
    print("[9] F1: once-read nodes leave the one-shot band")
    st, _ = make_strategy({"t1": SessionType.TRANS, "s1": SessionType.STRUCT})

    # TRANS share=0：未被匹配过 -> one-shot；被匹配过一次 -> 不再是
    fresh = FakeNode(session_id="t1", hit_count=0)
    check(st.is_one_shot(fresh) is True, "TRANS hit_count=0 must stay one-shot")
    read = FakeNode(session_id="t1", hit_count=1)
    check(st.is_one_shot(read) is False,
          "TRANS node matched once must leave the one-shot band")
    read2 = FakeNode(session_id="t1", hit_count=3)
    check(st.is_one_shot(read2) is False, "multiple matches also promote")

    # STRUCT share=0：同样被一次匹配拯救（跨会话共享内容不再因带内冲刷早死）
    s_fresh = FakeNode(session_id="s1", tag=HeterCacheNodeTag.QSTRUCT, hit_count=0)
    check(st.is_one_shot(s_fresh) is True,
          "STRUCT hit_count=0 share=0 must stay one-shot")
    s_read = FakeNode(session_id="s1", tag=HeterCacheNodeTag.QSTRUCT, hit_count=1)
    check(st.is_one_shot(s_read) is False,
          "STRUCT node matched once must leave the one-shot band")

    # hit_count 不得推翻其它豁免：共享数达标的节点本来就不在带内
    shared = FakeNode(session_id="t1", hit_count=0)
    st.node_share.on_hit(shared, "other-0")
    st.node_share.on_hit(shared, "other-1")
    check(st.is_one_shot(shared) is False,
          "share>=min exemption must survive the hit_count rule")


def test_active_session_exempt_until_idle():
    """轴一：活跃会话的未匹配节点不进 one-shot 带，搁置满窗口后才进。

    机制依据：hit_count 证据的成立前提是节点先活过一个完整重访间隔，而
    Stage 0 的清理节奏是秒级、重访间隔是 6.1s（alfred）到 30~63s（chat）——
    仅凭 F1 的 hit_count==0 豁免，latch 在扫描流下几乎永不发生。
    """
    print("[10] idle exemption: live sessions keep unread nodes out of the band")
    clock = {"t": 1000.0}
    st, reg = make_strategy(
        {"t1": SessionType.TRANS, "s1": SessionType.STRUCT},
        now=lambda: clock["t"],
    )
    reg._sessions["t1"].last_active = 980.0
    reg._sessions["s1"].last_active = 980.0

    n = FakeNode(session_id="t1", hit_count=0)
    check(st.is_one_shot(n) is False,
          "a live session's unread node must be exempt from the band")
    clock["t"] = 980.0 + ONE_SHOT_IDLE_S - 1
    check(st.is_one_shot(n) is False, "still inside the idle window")
    clock["t"] = 980.0 + ONE_SHOT_IDLE_S
    check(st.is_one_shot(n) is True,
          "past the window the node re-enters the band")

    # 锚点取 max(session.last_active, node.last_access_time)：请求间隙里
    # 新建的节点比会话账本新，由节点自身时间兜底。
    m = FakeNode(session_id="t1", hit_count=0)
    m.last_access_time = 1050.0
    clock["t"] = 1050.0 + ONE_SHOT_IDLE_S - 1
    check(st.is_one_shot(m) is False,
          "anchor must follow the newer of session/node timestamps")
    clock["t"] = 1050.0 + ONE_SHOT_IDLE_S
    check(st.is_one_shot(m) is True, "node-side anchor expires too")

    # 豁免只延缓不推翻：过窗后原判据照常生效（F1 与共享豁免不变）
    reg._sessions["s1"].last_active = 0.0
    clock["t"] = 10_000.0
    r = FakeNode(session_id="s1", tag=HeterCacheNodeTag.QSTRUCT, hit_count=0)
    check(st.is_one_shot(r) is True,
          "idle STRUCT share=0 hit=0 is one-shot again")
    r.hit_count = 1
    check(st.is_one_shot(r) is False,
          "hit evidence still overrides after expiry (F1 unchanged)")
    shared = FakeNode(session_id="t1", hit_count=0)
    for i in range(st.node_share.share_min):
        st.node_share.on_hit(shared, f"other-{i}")
    check(st.is_one_shot(shared) is False,
          "share exemption still overrides after expiry")


def main():
    tests = [
        test_trans_unshared_vs_shared,
        test_struct_threshold,
        test_multiturn_never_one_shot,
        test_queue_tag_is_irrelevant,
        test_ledger_semantics,
        test_no_session_is_not_one_shot,
        test_is_one_shot_and_pick_scope_agree,
        test_works_under_force_agentic,
        test_share_min_is_single_knob,
        test_once_read_leaves_one_shot_band,
        test_active_session_exempt_until_idle,
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
