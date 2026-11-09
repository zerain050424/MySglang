"""force_agentic：把全部节点直达 QAGENT 队列。

实验前提（RR vs T-LRU 的 burst/pause 场景）：RrEvictPolicy 的
队内策略只认 QAGENT，而状态机默认会把新节点标成 QTRANS（one-shot）、把间隔
≥8s 的命中标成 QCHAT。force_agentic 关掉这套分流：

  1. 新节点直接 QAGENT（不再 QTRANS → 不被 one-shot 带先清）
  2. 命中不再按 8s 间隔翻牌（长挂起会话不会掉进 QCHAT）
  3. 分叉父节点不再被打成 QSTRUCT（不再走位置项评分）
  4. 策略层 _task_type_of 短路，评分/one-shot 判定统一按 agentic
并保留 session_type 不变（分类器自身统计仍可测）。

Run: /root/sl_old/bin/python test/manual/test_force_agentic.py
"""

import os
import sys
from array import array

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)

import torch  # noqa: E402

from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.evict_policy import HeterCacheStrategy  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.heter_cache_radix_cache import (  # noqa: E402
    HeterCacheRadixCache,
)
from sglang.srt.mem_cache.heter_utils import HeterCacheNodeTag  # noqa: E402

FAILS = []
QAGENT = HeterCacheNodeTag.QAGENT
QTRANS = HeterCacheNodeTag.QTRANS
QSTRUCT = HeterCacheNodeTag.QSTRUCT


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


class Alloc:
    device = torch.device("cpu")

    def free_segment(self, indices=None, start_pos=0):
        return None


def _in_slru_scope(strategy, node) -> bool:
    """节点是否落在 SLRU 的保护范围内。

    这就是原 ``SlruEvictPolicy.pick`` 的准入条件（会话类型 STRUCT 且共享数
    达标），pick 删除后测试直接断言判据本身，不再借道一个无人调用的选点器。
    """
    st = strategy.slru_evict.session_registry.get(getattr(node, "session_id", None))
    if st is None:
        return False
    return (
        st.session_type == SessionType.STRUCT
        and strategy.slru_evict.share_of(node) >= strategy.slru_evict.share_min
    )


def build(force_agentic):
    # 配额必填：HeterCache 只走配额仲裁
    cfg = {"quota": {"caps": {"struct": 0.3, "agentic": 0.4, "chat": 0.3}}}
    if force_agentic:
        cfg["force_agentic"] = True
    return HeterCacheRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Alloc(),
            page_size=1,
            eviction_policy="heter_cache",
            eviction_policy_config=cfg,
        )
    )


def insert(cache, tokens, session_id="s0"):
    return cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=torch.tensor(tokens, dtype=torch.int64),
            session_id=session_id,
        )
    )


def match(cache, tokens, session_id="s0"):
    return cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(array("q", tokens)),
            req=type("Req", (), {"rid": f"{session_id}_t", "session_id": session_id})(),
        )
    )


def all_nodes(cache):
    out, stack = [], [cache.root_node]
    while stack:
        n = stack.pop()
        if n is not cache.root_node:
            out.append(n)
        stack.extend(n.children.values())
    return out


def test_flag_plumbed():
    print("[1] config -> strategy -> cache")
    on = build(True)
    check(getattr(on.eviction_strategy, "force_agentic", None) is True,
          "strategy.force_agentic must be True")
    check(getattr(on, "force_agentic", None) is True,
          "cache.force_agentic must be True (read from the strategy)")

    off = build(False)
    check(getattr(off.eviction_strategy, "force_agentic", None) is False,
          "default must be False")
    check(getattr(off, "force_agentic", None) is False, "cache default False")


def test_new_node_is_agentic():
    print("[2] fresh insert is QAGENT (not QTRANS)")
    cache = build(True)
    insert(cache, list(range(100)), session_id="s-forced")
    nodes = all_nodes(cache)
    check(nodes, "tree must have nodes")
    bad = [(n.id, n.queue_tag) for n in nodes if n.queue_tag is not QAGENT]
    check(not bad, f"all nodes must be QAGENT, offenders={bad[:3]}")

    # 对照：不开开关时新节点是 QTRANS（且被判为 one-shot）
    ctrl = build(False)
    insert(ctrl, list(range(100)), session_id="s-control")
    ctrl_nodes = all_nodes(ctrl)
    check(ctrl_nodes and all(n.queue_tag is QTRANS for n in ctrl_nodes),
          "control: fresh nodes must stay QTRANS")
    check(all(ctrl.eviction_strategy.is_one_shot(n) for n in ctrl_nodes),
          "control: QTRANS nodes are one-shot (this is what force_agentic removes)")


def test_no_one_shot_anywhere():
    print("[3] with force_agentic, nothing is one-shot (one-shot band stays empty)")
    cache = build(True)
    insert(cache, list(range(50, 150)), session_id="s1")
    for turn in range(2, 5):
        # 每轮先命中再追加，制造命中/分叉/多轮
        match(cache, list(range(0, 50 + 25 * turn)), session_id="s1")
        insert(cache, list(range(50 + 25 * turn, 120 + 25 * turn)), session_id="s1")

    leaves = list(cache.evictable_leaves)
    check(leaves, "must have evictable leaves")
    oneshot = [n.id for n in leaves if cache.eviction_strategy.is_one_shot(n)]
    check(not oneshot, f"no leaf may be one-shot, offenders={oneshot[:5]}")
    check(
        all(cache.eviction_strategy._node_tag(n) is HeterCacheNodeTag.QAGENT for n in leaves),
        "every leaf must carry QAGENT under force_agentic",
    )


def test_hit_does_not_flip_to_chat():
    print("[4] a long-gap hit must not flip tags to QCHAT")
    cache = build(True)
    insert(cache, list(range(200)), session_id="s2")
    for n in all_nodes(cache):
        n.last_access_time -= 60.0      # 伪造 60s 空闲（远超 8s 阈值）
    match(cache, list(range(150)), session_id="s2")

    flipped = [(n.id, n.queue_tag) for n in all_nodes(cache) if n.queue_tag is not QAGENT]
    check(not flipped,
          f"60s-gap hit must keep QAGENT (no QCHAT), offenders={flipped[:3]}")

    ctrl = build(False)
    insert(ctrl, list(range(200)), session_id="s3")
    for n in all_nodes(ctrl):
        n.last_access_time -= 60.0
    match(ctrl, list(range(150)), session_id="s3")
    check(any(n.queue_tag != QAGENT for n in all_nodes(ctrl)),
          "control: 60s-gap hit must flip away from QAGENT")


def test_branch_parent_not_struct():
    print("[5] a forked parent must not become QSTRUCT")
    cache = build(True)
    insert(cache, list(range(0, 100)), session_id="a")
    insert(cache, [0] * 20 + list(range(100, 180)), session_id="b")  # 共享 20 前缀后分叉
    parents = [n for n in all_nodes(cache) if len(n.children) >= 2]
    check(parents, "expected a forked parent")
    check(all(p.queue_tag is QAGENT for p in parents),
          f"forked parent must stay QAGENT, got {[p.queue_tag for p in parents]}")

    ctrl = build(False)
    insert(ctrl, list(range(0, 100)), session_id="a")
    insert(ctrl, [0] * 20 + list(range(100, 180)), session_id="b")
    ctrl_parents = [n for n in all_nodes(ctrl) if len(n.children) >= 2]
    check(ctrl_parents and all(p.queue_tag is QSTRUCT for p in ctrl_parents),
          "control: forked parent becomes QSTRUCT")


def test_plugins_see_everything():
    print("[6] QAGENT-only plugins now cover all leaves")
    cfg = {
        "force_agentic": True,
        "slru": {"share_min": 2},
    }
    cache = HeterCacheRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Alloc(),
            page_size=1,
            eviction_policy="heter_cache",
            eviction_policy_config=cfg,
        )
    )
    insert(cache, list(range(0, 300)), session_id="p")
    insert(cache, [0] * 10 + list(range(300, 600)), session_id="q")
    match(cache, list(range(0, 200)), session_id="p")

    strategy = cache.eviction_strategy
    slru = strategy.slru_evict
    leaves = list(cache.evictable_leaves)
    covered = [n.id for n in leaves if _in_slru_scope(strategy, n) is True]
    check(covered, f"no leaf reaches the SLRU plugin, leaves={len(leaves)}")


def main():
    tests = [
        test_flag_plumbed,
        test_new_node_is_agentic,
        test_no_one_shot_anywhere,
        test_hit_does_not_flip_to_chat,
        test_branch_parent_not_struct,
        test_plugins_see_everything,
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
