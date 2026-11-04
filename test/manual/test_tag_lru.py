"""标签版 TinyLRU（`tiny_lru.policy="tag_lru"`）的 characterization tests。

钉住五件事：

1. **分类即晋升**：STRUCT 会话下，节点在共享数达到阈值的命中当场从 QTRANS → QSTRUCT，
   不需要等它变成「可淘汰叶子」（这是三区版最难等的条件）。
2. **共享数 = 不同会话数**：同一会话重复命中不重复计数；账本归 `SlruEvictPolicy`，
   `TagLruPolicy` 只读。
3. **非 STRUCT 会话不参与晋升**：chat/agentic 的节点保持自己的队列。
4. **类间顺序 + 类内 LRU**：`victim_key` 让 QTRANS（ranks 0）最先、chat/agentic（1）
   其次、QSTRUCT（2）最后，且 QSTRUCT 内部按 `last_access_time` 排（不是位置打分）。
5. **接线**：tag 模式不构造三区状态机；命中路径的账本由 SLRU 的 `on_node_hit` 负责。

    python test/manual/test_tag_lru.py
"""

import os
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)

from sglang.srt.mem_cache.evict_policy import SimpleHybridStrategy  # noqa: E402
from sglang.srt.mem_cache.tag_lru import TagLruPolicy  # noqa: E402
from sglang.srt.mem_cache.uni_utils import (  # noqa: E402
    SESSION_SHARE_MAX,
    SessionType,
    SimpleHybridQueueTag,
)
from sglang.srt.mem_cache.victim_override import SlruEvictPolicy  # noqa: E402

FAILS = []


def hit(share, tag_policy, node, session_id, session_type):
    """命中路径的真实顺序：先由 SLRU 记共享数，再由 tag_lru 贴标签。"""
    share.on_node_hit(node, session_id)
    tag_policy.observe_hit(node, session_id, session_type)


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


class FakeNode:
    """TreeNode 的最小替身：只带标签版策略与 victim_key 读写的字段。"""

    def __init__(self, tokens=16, tag=SimpleHybridQueueTag.QTRANS, last_access=None):
        self.value = [0] * tokens
        self.children = {}
        self.parent = None
        self.lock_ref = 0
        self.evicted = False
        self.queue_tag = tag
        self.share_count = 0
        self.share_sessions = 0
        self.rid_bloom = 0
        self.session_bloom = 0
        self.last_access_time = time.monotonic() if last_access is None else last_access
        self.abs_start = 0


def test_initial_tag_by_session_type():
    """新节点出生标签：STRUCT 会话先给 QTRANS（它只是本会话私有的一段）。"""
    print("[1] initial tag by session type")
    p = TagLruPolicy()  # 默认 local_hot="chat"
    check(p.initial_tag(SessionType.STRUCT) == SimpleHybridQueueTag.QCHAT,
          "STRUCT new node starts as QCHAT under the default (local_hot=chat)")
    check(TagLruPolicy(local_hot="one_shot").initial_tag(SessionType.STRUCT)
          == SimpleHybridQueueTag.QTRANS,
          "local_hot=one_shot starts STRUCT nodes on the one-shot band")
    check(p.initial_tag(SessionType.CHAT) == SimpleHybridQueueTag.QCHAT,
          "CHAT -> QCHAT")
    check(p.initial_tag(SessionType.AGENTIC) == SimpleHybridQueueTag.QAGENT,
          "AGENTIC -> QAGENT")
    check(p.initial_tag(SessionType.TRANS) == SimpleHybridQueueTag.QTRANS,
          "TRANS -> QTRANS")
    check(p.stats["born"] == 4, f"born counter, got {p.stats['born']}")


def test_promotion_on_second_session():
    """分类即晋升：共享数达到阈值时当场 QTRANS -> QSTRUCT。"""
    print("[2] promotion happens when the share count reaches the threshold")
    p = TagLruPolicy()
    share = SlruEvictPolicy()
    node = FakeNode()

    hit(share, p, node, "sid-A", SessionType.STRUCT)
    check(node.share_sessions == 1, f"share_sessions, got {node.share_sessions}")
    # 默认 local_hot="chat"：命中即证明「正在被复用」，不能落回 one-shot 带
    check(node.queue_tag == SimpleHybridQueueTag.QCHAT,
          f"one session -> QCHAT (not one-shot), got {node.queue_tag}")
    check(p.stats["promote"] == 0, "no QSTRUCT promotion yet")

    # 同一会话重复命中：去重，不涨共享数
    hit(share, p, node, "sid-A", SessionType.STRUCT)
    check(node.share_sessions == 1, f"same session must dedup, got {node.share_sessions}")
    check(share.stats()["share_dedup"] == 1,
          f"dedup counter, got {share.stats()['share_dedup']}")

    # 第二个会话命中：晋升
    hit(share, p, node, "sid-B", SessionType.STRUCT)
    check(node.share_sessions == 2, f"share_sessions, got {node.share_sessions}")
    check(node.queue_tag == SimpleHybridQueueTag.QSTRUCT,
          f"two sessions -> QSTRUCT, got {node.queue_tag}")
    check(p.stats["promote"] == 1, f"promote counter, got {p.stats['promote']}")
    check(p.owns(node), "owns() must be true for QSTRUCT")


def test_share_sessions_owned_by_slru():
    """共享会话数账本：SlruEvictPolicy 独占写入 + 去重 + 饱和。"""
    print("[2b] share_sessions ledger is owned by SlruEvictPolicy")
    share = SlruEvictPolicy()
    node = FakeNode()

    share.on_node_hit(node, "sid-A")
    check(node.share_sessions == 1, f"first hit, got {node.share_sessions}")
    share.on_node_hit(node, "sid-A")
    check(node.share_sessions == 1, f"same session dedups, got {node.share_sessions}")
    check(share.stats()["share_dedup"] == 1, "dedup counter")
    check(share.stats()["share_counted"] == 1, "counted counter")

    share.on_node_hit(node, "sid-B")
    check(node.share_sessions == 2, f"second session, got {node.share_sessions}")

    # 无 session_id：不计数（无法去重）
    before = node.share_sessions
    share.on_node_hit(node, None)
    share.on_node_hit(None, "sid-Z")
    check(node.share_sessions == before, "missing session/node must not count")

    # 64 bit bloom（4 探针）在 ~10 个会话后就基本饱和，之后新会话会被误判成
    # 「已计入」而跳过。所以 SESSION_SHARE_MAX=255 并不是实际可达的上界。
    for i in range(400):
        share.on_node_hit(node, f"bulk-{i}")
    check(node.share_sessions <= SESSION_SHARE_MAX,
          f"share must never exceed {SESSION_SHARE_MAX}, got {node.share_sessions}")
    check(share.stats()["share_dedup"] > 0, "bloom saturates -> dedup dominates")

    # 上界的显式保护：直接构造「已到顶 + 空 bloom」的状态
    node.session_bloom = 0
    node.share_sessions = SESSION_SHARE_MAX
    share.on_node_hit(node, "cap-probe")
    check(node.share_sessions == SESSION_SHARE_MAX,
          f"must cap at {SESSION_SHARE_MAX}, got {node.share_sessions}")

    share.reset()
    check(share.stats()["share_counted"] == 0 and share.stats()["share_dedup"] == 0,
          "reset() must clear the ledger counters")


def test_non_struct_session_does_not_promote():
    """chat/agentic 会话不参与晋升：共享数再高也保持自己的队列标签。"""
    print("[3] non-STRUCT sessions never promote to QSTRUCT")
    p = TagLruPolicy()
    share = SlruEvictPolicy()
    node = FakeNode(tag=SimpleHybridQueueTag.QCHAT)
    for sid in ("c1", "c2", "c3"):
        hit(share, p, node, sid, SessionType.CHAT)
    check(node.share_sessions == 3, f"share_sessions, got {node.share_sessions}")
    check(node.queue_tag == SimpleHybridQueueTag.QCHAT,
          f"CHAT tag must be untouched, got {node.queue_tag}")
    check(p.stats["promote"] == 0, "no promotion for chat sessions")
    check(not p.owns(node), "chat node is not struct-owned")


def test_struct_share_min_threshold():
    """struct_share_min 可调：=3 时前两个会话仍留在 QTRANS。"""
    print("[4] struct_share_min threshold")
    p = TagLruPolicy(struct_share_min=3)
    share = SlruEvictPolicy()
    node = FakeNode()
    hit(share, p, node, "s1", SessionType.STRUCT)
    hit(share, p, node, "s2", SessionType.STRUCT)
    check(node.queue_tag == SimpleHybridQueueTag.QCHAT,
          f"2 sessions with min=3 -> QCHAT (session-local), got {node.queue_tag}")
    hit(share, p, node, "s3", SessionType.STRUCT)
    check(node.queue_tag == SimpleHybridQueueTag.QSTRUCT,
          f"3 sessions with min=3 -> QSTRUCT, got {node.queue_tag}")


def test_local_hot_modes():
    """会话内热点：one_shot 掉进一次性带；chat 保留复用语义（rank 1）。"""
    print("[4b] local_hot modes for session-local reuse")
    check(TagLruPolicy().local_hot == "chat", "default local_hot must be 'chat'")
    one = TagLruPolicy(local_hot="one_shot")
    share = SlruEvictPolicy()
    node = FakeNode()
    hit(share, one, node, "s1", SessionType.STRUCT)      # share=1, rids=1
    check(node.queue_tag == SimpleHybridQueueTag.QTRANS,
          f"one_shot mode: single-session -> QTRANS, got {node.queue_tag}")
    node.share_count = 2                                  # 同一会话第二轮
    one.apply_tag(node, SessionType.STRUCT)
    check(node.queue_tag == SimpleHybridQueueTag.QTRANS,
          f"one_shot mode ignores rids, got {node.queue_tag}")

    chat = TagLruPolicy(local_hot="chat")
    n2 = FakeNode()
    check(chat.initial_tag(SessionType.STRUCT) == SimpleHybridQueueTag.QCHAT,
          "chat mode: new struct-session node starts as QCHAT (not one-shot)")
    hit(share, chat, n2, "s2", SessionType.STRUCT)
    n2.share_count = 2
    chat.apply_tag(n2, SessionType.STRUCT)
    check(n2.queue_tag == SimpleHybridQueueTag.QCHAT,
          f"chat mode keeps session-local hot nodes off the one-shot band, got {n2.queue_tag}")
    # 第二个会话仍会晋升
    hit(share, chat, n2, "s3", SessionType.STRUCT)
    check(n2.queue_tag == SimpleHybridQueueTag.QSTRUCT,
          f"second session still promotes in chat mode, got {n2.queue_tag}")


def test_victim_key_class_rank_and_lru():
    """victim_key：QTRANS(0) < chat/agentic(1) < QSTRUCT(2)，QSTRUCT 内部 LRU。"""
    print("[5] victim_key: class rank + LRU inside QSTRUCT")
    st = SimpleHybridStrategy(tiny_lru={"policy": "tag_lru"})
    check(st.tagState is not None, "tag mode must construct TagLruPolicy")

    now = time.monotonic()
    t = FakeNode(tag=SimpleHybridQueueTag.QTRANS, last_access=now)
    c = FakeNode(tag=SimpleHybridQueueTag.QCHAT, last_access=now)
    a = FakeNode(tag=SimpleHybridQueueTag.QAGENT, last_access=now)
    old_s = FakeNode(tag=SimpleHybridQueueTag.QSTRUCT, last_access=now - 100)
    new_s = FakeNode(tag=SimpleHybridQueueTag.QSTRUCT, last_access=now)

    kt, kc, ka = st.victim_key(t), st.victim_key(c), st.victim_key(a)
    ko, kn = st.victim_key(old_s), st.victim_key(new_s)

    check(kt[0] == 0, f"QTRANS rank, got {kt[0]}")
    check(kc[0] == 1 and ka[0] == 1, f"chat/agentic rank, got {kc[0]}/{ka[0]}")
    check(ko[0] == 2 and kn[0] == 2, f"QSTRUCT rank, got {ko[0]}/{kn[0]}")
    check(kt < kc < ko, f"class order must be QTRANS < chat < QSTRUCT: {kt} {kc} {ko}")
    check(ko < kn, f"QSTRUCT is LRU: older must sort first ({ko} vs {kn})")

    # 全局 min() 的语义：一次性先走、其次 chat、最后共享前缀
    victim = min([new_s, a, t, c, old_s], key=st.victim_key)
    check(victim is t, f"min() must pick the QTRANS node, got {victim.queue_tag}")

    # 关闭时保持原键（不受 class rank 影响）
    off_mode = SimpleHybridStrategy(tiny_lru={"policy": "off"})
    check(off_mode.tagState is None, "off must not build tagState")
    check(len(off_mode.victim_key(t)) == 2,
          f"off keeps (score, -tokens), got {off_mode.victim_key(t)}")


def test_wiring_and_dead_chain_gate():
    """接线：tag 模式不建三区状态机；SLRU 账本插件始终在场；owns_struct 走标签。"""
    print("[6] wiring + dead-chain gate")
    st = SimpleHybridStrategy(tiny_lru={"policy": "tag_lru"})
    check(st.tagState is not None, "tag mode must build TagLruPolicy")
    check(not hasattr(st, "structState"), "three-region state machine must be gone")
    check(not hasattr(st, "TinyLru_evict"), "struct override must be gone")
    # 共享会话数账本与 tag_lru 解耦：即使 tag 关闭，SLRU 仍在 victim_overrides 里
    check(st.victim_overrides == (st.slru_evict,),
          f"only the SLRU override is registered, got {st.victim_overrides}")
    check(hasattr(st.slru_evict, "on_node_hit"),
          "SLRU must expose the share-session ledger hook")

    shared = FakeNode(tag=SimpleHybridQueueTag.QSTRUCT)
    private = FakeNode(tag=SimpleHybridQueueTag.QTRANS)
    check(st.owns_struct(shared), "owns_struct(QSTRUCT) must be True")
    check(not st.owns_struct(private), "owns_struct(QTRANS) must be False")

    # off：两者都不建
    off = SimpleHybridStrategy(tiny_lru={"policy": "off"})
    check(off.tagState is None, "off builds nothing")
    check(not off.owns_struct(shared), "off owns nothing")


def test_tag_and_tlru_are_mutually_exclusive():
    """tag_lru 与 T-LRU 同时开会互相遮蔽，必须启动期报错。"""
    print("[7] tag_lru + tlru must fail fast")
    try:
        SimpleHybridStrategy(
            tiny_lru={"policy": "tag_lru"},
            # threshold 必须 > next_prompt_estimate，否则 TlruParams 自己就会先报错
            tlru={"enabled": True, "threshold": 1024, "next_prompt_estimate": 128},
        )
    except ValueError as exc:
        check("mutually exclusive" in str(exc), f"unexpected message: {exc}")
    else:
        check(False, "expected ValueError for tag_lru + tlru")


def main():
    tests = [
        test_initial_tag_by_session_type,
        test_promotion_on_second_session,
        test_share_sessions_owned_by_slru,
        test_non_struct_session_does_not_promote,
        test_struct_share_min_threshold,
        test_local_hot_modes,
        test_victim_key_class_rank_and_lru,
        test_wiring_and_dead_chain_gate,
        test_tag_and_tlru_are_mutually_exclusive,
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
