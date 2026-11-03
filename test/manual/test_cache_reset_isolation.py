"""缓存 reset / flush 的跨 run 隔离性。

实验加速的关键是**复用同一个 server 跑多个 seed**，每次之间调 `/flush_cache`。
`/flush_cache` 会走 `scheduler.flush_cache()` → `tree_cache.reset()`，因此
reset 必须把「上一次 run 的状态」全部清掉，否则会串味：

  SessionRegistry 的 d/L/gaps  ->  RR 的 loss_cap 直接读它们
  策略的窗口/累计/来源计数     ->  [SimpleHybrid profile] 会打印混合值
  插件的计数器                 ->  tel_safe/lru 比例失真
  树上节点的 _tlru_* 高水位    -> 随树一起丢弃

Run: /root/sl_old/bin/python test/manual/test_cache_reset_isolation.py
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
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.simple_hybrid_radix_cache import (  # noqa: E402
    SimpleHybridRadixCache,
)
from sglang.srt.mem_cache.uni_utils import SimpleHybridQueueTag  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


class Alloc:
    device = torch.device("cpu")

    def free_segment(self, indices=None, start_pos=0):
        return None


CFG = {
    "force_agentic": True,
    "tlru": {"enabled": True, "threshold": 943, "next_prompt_estimate": 71},
}


def build(cfg=None):
    return SimpleHybridRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Alloc(),
            page_size=1,
            eviction_policy="simple_hybrid",
            eviction_policy_config=CFG if cfg is None else cfg,
        )
    )


class Req:
    def __init__(self, rid, sid):
        self.rid = rid
        self.session_id = sid


def insert(cache, tokens, sid="s"):
    return cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=torch.tensor(tokens, dtype=torch.int64),
            session_id=sid,
        )
    )


def match(cache, tokens, sid="s", rid="r"):
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)), req=Req(rid, sid))
    )


def all_nodes(cache):
    out, stack = [], [cache.root_node]
    while stack:
        n = stack.pop()
        if n is not cache.root_node:
            out.append(n)
        stack.extend(n.children.values())
    return out


def dirty(cache):
    """模拟一次完整 run：命中 + 淘汰记账 + 插件计数。"""
    insert(cache, list(range(0, 200)), sid="sessA")
    match(cache, list(range(0, 120)), sid="sessA", rid="rid-1")
    insert(cache, list(range(0, 150)) + list(range(200, 260)), sid="sessB")

    strategy = cache.eviction_strategy
    # 人为把记账/计数打脏（等价于一轮真实 run 之后的状态）
    strategy._hit_tokens["agentic"] = 1234.0
    strategy._total_hit_tokens["agentic"] = 5678.0
    strategy._total_evicted_tokens["agentic"] = 4321.0
    strategy._total_evicted_one_shot = 99.0
    strategy._victim_sources["qagent"] = 777.0
    strategy._seen_rids.add("rid-1")
    strategy._rid_order.append("rid-1")
    if strategy.tlru_evict is not None:
        strategy.tlru_evict.tel_safe_picks = 5
        strategy.tlru_evict.history_raises = 42

    # 会话账本：touch 会写 d/L
    cache.session_registry.touch("sessA", "rid-1", prompt_len=200)
    st = cache.session_registry.get("sessA")
    if st is not None:
        st.d = 777
    return strategy


def test_reset_clears_everything():
    print("[1] reset() clears session ledger, strategy counters and plugins")
    cache = build()
    strategy = dirty(cache)

    check(len(cache.session_registry) > 0, "precondition: registry has sessions")
    check(strategy._hit_tokens, "precondition: hit window non-empty")

    cache.reset()

    check(len(cache.session_registry) == 0,
          f"SessionRegistry must be empty after reset, got {len(cache.session_registry)}")
    check(not strategy._hit_tokens, "hit window must be cleared")
    check(not strategy._total_hit_tokens, "total hit tokens must be cleared")
    check(not strategy._total_evicted_tokens, "total evicted tokens must be cleared")
    check(strategy._total_evicted_one_shot == 0.0, "one-shot scalar must be zeroed")
    check(not strategy._victim_sources, "victim sources must be cleared")
    check(not strategy._seen_rids and not strategy._rid_order,
          "request dedup set must be cleared")
    check(not all_nodes(cache), "tree must be empty after reset")
    if strategy.tlru_evict is not None:
        s = strategy.tlru_evict.stats()
        check(all(v == 0 for v in s.values()),
              f"plugin counters must be zeroed, got {s}")


def test_second_run_equals_fresh_run():
    """同一 server 上跑「run A → reset → run B」与直接新建 cache 跑 B，状态必须一致。"""
    print("[2] run A -> reset -> run B  ==  fresh cache -> run B")

    reused = build()
    dirty(reused)
    reused.reset()
    insert(reused, list(range(0, 300)), sid="sessB")
    match(reused, list(range(0, 180)), sid="sessB", rid="b-1")

    fresh = build()
    insert(fresh, list(range(0, 300)), sid="sessB")
    match(fresh, list(range(0, 180)), sid="sessB", rid="b-1")

    def fingerprint(cache):
        return sorted(
            (
                n._tlru_cached_prefix_len,
                n._tlru_history_len,
                n.struct_region,
                getattr(n.queue_tag, "value", str(n.queue_tag)),
                0 if n.value is None else len(n.value),
            )
            for n in all_nodes(cache)
        )

    check(fingerprint(reused) == fingerprint(fresh),
          "tree fingerprint differs: reuse is contaminated")
    check(reused.root_node._tlru_history_len == fresh.root_node._tlru_history_len,
          f"root high-water {reused.root_node._tlru_history_len} != "
          f"fresh {fresh.root_node._tlru_history_len}")

    st_reused = reused.session_registry.get("sessB")
    st_fresh = fresh.session_registry.get("sessB")
    check((st_reused is None) == (st_fresh is None), "session presence differs")
    if st_reused is not None and st_fresh is not None:
        check(st_reused.d == st_fresh.d and st_reused.L == st_fresh.L,
              f"session state differs: d={st_reused.d}/{st_fresh.d} L={st_reused.L}/{st_fresh.L}")


def test_tags_rebuilt_from_scratch():
    print("[3] tags are rebuilt from scratch (no leakage of the previous run)")
    cache = build()
    insert(cache, list(range(0, 100)), sid="x")
    # 污染：把某个节点改成 QCHAT，模拟旧 run 的残留
    for n in all_nodes(cache):
        n.queue_tag = SimpleHybridQueueTag.QCHAT
    cache.reset()
    insert(cache, list(range(0, 100)), sid="y")
    bad = [n.queue_tag for n in all_nodes(cache) if n.queue_tag is not SimpleHybridQueueTag.QAGENT]
    check(not bad, f"with force_agentic every fresh node must be QAGENT, got {bad[:3]}")


def main():
    tests = [
        test_reset_clears_everything,
        test_second_run_equals_fresh_run,
        test_tags_rebuilt_from_scratch,
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
