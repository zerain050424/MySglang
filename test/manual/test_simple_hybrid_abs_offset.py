"""Proof tests: SimpleHybrid "offset" must be the ABSOLUTE token offset.

UniCache's heap key is (-num_hashed_tokens, ...), where num_hashed_tokens is a
block's end offset in the sequence. The sglang port used len(node.value)
(radix node segment length) instead, which is an artifact of sharing topology,
not of position in the sequence. These tests pin down the intended semantics:

  node.abs_start                     == absolute offset of the segment's first token
  node.abs_start + len(node.value)   == absolute end offset ("num_hashed_tokens")
  child.abs_start == parent.abs_start + len(parent.value)

and that SimpleHybridStrategy._reuse_score() ranks by that absolute end offset.

Run:  /root/sl_old/bin/python test/manual/test_simple_hybrid_abs_offset.py
"""

import os
import time
import sys
from array import array
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

import torch  # noqa: E402

from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.simple_hybrid_radix_cache import (  # noqa: E402
    SimpleHybridRadixCache,
)
from sglang.srt.mem_cache.evict_policy import SimpleHybridStrategy  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey  # noqa: E402
from sglang.srt.mem_cache.uni_utils import SimpleHybridTaskType  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


def new_cache(page_size=1):
    return RadixCache.create_simulated(page_size=page_size)


def ins(cache, tokens):
    return cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=torch.tensor(tokens, dtype=torch.int64),
        )
    )


def walk(cache, label):
    """Verify abs_start bookkeeping over the whole tree."""
    ok = True
    stack = [(cache.root_node, 0)]
    while stack:
        node, expected_start = stack.pop()
        got = getattr(node, "abs_start", None)
        if not check(
            got == expected_start,
            f"{label}: node(depth-tokens={expected_start}) abs_start={got}, "
            f"expected {expected_start}",
        ):
            ok = False
            continue
        for child in node.children.values():
            if child.evicted:
                continue
            # child.abs_start == this node's abs_start + len(this node's segment)
            stack.append((child, got + len(node.value)))
    return ok


def _find_len(cache, n):
    for node in cache.root_node.children.values():
        if len(node.value) == n:
            return node
    return None


def test_fresh_insert_starts_at_zero():
    print("[1] fresh insert -> abs_start == 0")
    cache = new_cache()
    ins(cache, list(range(10)))
    leaf = _find_len(cache, 10)
    check(leaf is not None, "expected a 10-token node")
    if leaf is not None:
        check(getattr(leaf, "abs_start", None) == 0, "fresh node abs_start should be 0")
    walk(cache, "fresh")


def test_split_via_insert():
    print("[2] split via second insert -> suffix abs_start == split point")
    cache = new_cache()
    ins(cache, list(range(10)))
    ins(cache, list(range(5)) + [99])
    walk(cache, "split-insert")
    mid = _find_len(cache, 5)
    check(mid is not None, "expected a 5-token prefix node after split")
    if mid is not None:
        check(mid.abs_start == 0, f"prefix node abs_start={mid.abs_start}, expected 0")
        starts = sorted(c.abs_start for c in mid.children.values())
        check(starts == [5, 5], f"suffix children abs_start={starts}, expected [5, 5]")


def test_split_via_match():
    print("[3] split via match_prefix -> same invariants")
    cache = new_cache()
    ins(cache, list(range(10)))
    cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", list(range(5))))))
    walk(cache, "split-match")
    mid = _find_len(cache, 5)
    check(mid is not None, "match should have split the 10-token node at 5")
    if mid is not None:
        check(mid.abs_start == 0, f"prefix abs_start={mid.abs_start}, expected 0")
        suffix = next(iter(mid.children.values()))
        check(suffix.abs_start == 5, f"suffix abs_start={suffix.abs_start}, expected 5")


def test_deep_tree_offsets():
    print("[4] deep chain -> offsets accumulate down the path")
    cache = new_cache()
    for n in range(1, 6):
        ins(cache, list(range(n)))
    walk(cache, "deep")
    offsets = []
    node = cache.root_node
    while node.children:
        node = next(iter(node.children.values()))
        offsets.append((node.abs_start, len(node.value)))
    check(
        offsets == [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
        f"chain offsets={offsets}, expected [(0,1),(1,1),(2,1),(3,1),(4,1)]",
    )


def _mk_node(n_tokens, abs_start, task_type=SimpleHybridTaskType.SINGLE_TURN):
    return SimpleNamespace(
        value=[0] * n_tokens,
        key=[0] * n_tokens,
        abs_start=abs_start,
        task_type=task_type,
        is_decode=False,
        last_access_time=time.monotonic(),
        hit_count=0,
    )


def test_reuse_score_uses_absolute_offset():
    print("[5] _reuse_score: same segment length, shallower offset scores higher")
    strategy = SimpleHybridStrategy(max_offset=2048)
    shallow = _mk_node(32, abs_start=0)
    deep = _mk_node(32, abs_start=3000)
    s_shallow = strategy._reuse_score(shallow)
    s_deep = strategy._reuse_score(deep)
    check(s_shallow > 0.9, f"shallow(0..32) score={s_shallow:.4f}, expected ~0.98")
    check(s_deep == 0.0, f"deep(3000..3032) score={s_deep:.4f}, expected 0.0")
    check(s_shallow > s_deep, "shallower offset must score strictly higher")
    # Segment-length-only scoring gave BOTH nodes 1 - 32/2048; the deep node
    # must now differ sharply from that value.
    old_segment_only = 1.0 - 32 / 2048
    check(
        abs(s_deep - old_segment_only) > 0.5,
        f"deep score={s_deep:.4f} must differ from segment-length-only "
        f"{old_segment_only:.4f}",
    )


def test_no_abs_start_fallback():
    print("[6] node without abs_start falls back to segment length (no crash)")
    strategy = SimpleHybridStrategy(max_offset=2048)
    node = _mk_node(32, abs_start=0)
    del node.abs_start
    s = strategy._reuse_score(node)
    check(
        abs(s - (1.0 - 32 / 2048)) < 1e-9,
        f"fallback score={s:.6f}, expected {1.0 - 32 / 2048:.6f}",
    )


def test_randomized_invariants():
    print("[7] randomized inserts (heavy sharing/splitting) keep invariants")
    import random

    for page_size in (1, 4):
        rng = random.Random(1234)
        cache = new_cache(page_size=page_size)
        for i in range(300):
            n = rng.randint(1, 40)
            toks = [rng.randint(0, 6) for _ in range(n)]
            ins(cache, toks)
            if i % 50 == 0 and not walk(cache, f"random-p{page_size}-i{i}"):
                return
        walk(cache, f"random-p{page_size}-final")
        # deepest resident node must never exceed the longest inserted sequence
        max_end = 0
        stack = [cache.root_node]
        while stack:
            node = stack.pop()
            max_end = max(max_end, getattr(node, "abs_start", 0) + (len(node.value) if node.value is not None else 0))
            stack.extend(c for c in node.children.values() if not c.evicted)
        check(max_end <= 300 * 40, f"max end offset {max_end} exploded")


def test_eviction_prefers_deep_single_turn_node():
    """End-to-end: cross-queue eviction must spend KV on the shallow prefix.

    Deep   : single-turn node, abs_start=2000, 32 tokens  (end offset 2032)
             -> new score 0.000 ; old (segment-length) score 0.984
    Rival  : agentic node, 32 tokens, idle ~10s -> score ~0.33

    Old code kept the deep single-turn node and evicted the agentic one;
    correct code evicts the deep single-turn node instead.
    """
    print("[8] cross-queue evict: deep single-turn node goes before a mid-score agentic node")

    class Alloc:
        device = torch.device("cpu")

        def free_segment(self, indices=None, start_pos=0):
            return None

    cache = SimpleHybridRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Alloc(),
            page_size=1,
            eviction_policy="simple_hybrid",
        )
    )

    def insert(tokens, task_type):
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=torch.tensor(tokens, dtype=torch.int64),
                task_type=task_type,
            )
        )

    # deep: [0..1999] then extend to [0..2031] -> 32-token child at abs_start=2000
    insert(list(range(0, 2000)), SimpleHybridTaskType.SINGLE_TURN)
    insert(list(range(0, 2032)), SimpleHybridTaskType.SINGLE_TURN)
    # rival: separate 32-token agentic node hanging off the root
    insert(list(range(5000, 5032)), SimpleHybridTaskType.MULTI_TURN_AGENTIC)

    deep = rival = None
    for node in cache.evictable_leaves:
        if len(node.value) != 32:
            continue
        if node.task_type == SimpleHybridTaskType.SINGLE_TURN:
            deep = node
        elif node.task_type == SimpleHybridTaskType.MULTI_TURN_AGENTIC:
            rival = node
    check(deep is not None, "expected a 32-token single-turn node")
    check(rival is not None, "expected a 32-token agentic node")
    if deep is None or rival is None:
        return

    check(deep.abs_start == 2000, f"deep.abs_start={deep.abs_start}, expected 2000")
    check(rival.abs_start == 0, f"rival.abs_start={rival.abs_start}, expected 0")

    # make the rival idle ~10s so its lognormal reuse score sits mid-range
    rival.last_access_time = time.monotonic() - 10.0

    s_deep = cache.eviction_strategy._reuse_score(deep)
    s_rival = cache.eviction_strategy._reuse_score(rival)
    print(f"      score(deep single-turn, end=2032)={s_deep:.4f}")
    print(f"      score(rival agentic, idle 10s)    ={s_rival:.4f}")
    check(s_deep < s_rival,
          f"deep single-turn score {s_deep:.4f} must be below agentic {s_rival:.4f}")

    result = cache.evict(EvictParams(num_tokens=32))
    check(result.num_tokens_evicted == 32,
          f"expected 32 tokens evicted, got {result.num_tokens_evicted}")
    still_there = {n for p in cache.evictable_leaves for n in [p]}
    check(rival in still_there, "agentic rival should survive")
    check(deep not in still_there, "deep single-turn node should be evicted")


def test_same_queue_picks_lowest_score():
    """#2: intra-queue candidacy must use the same key as the cross-queue compare.

    Same queue (single), so the old per-queue heap_key (largest segment first)
    and the score-based key disagree:

      A = deep   : abs_start=2000, len=32   -> score(0.008), heap_key(-32)
      B = shallow: abs_start=0,    len=2000 -> score(0.023), heap_key(-2000)

    Old: heap_key 推举 B -> 淘汰 B（2000 token）
    New: score     推举 A -> 淘汰 A（32 token）
    """
    print("[9] same-queue evict: lowest score wins, not largest segment")

    class Alloc:
        device = torch.device("cpu")

        def free_segment(self, indices=None, start_pos=0):
            return None

    cache = SimpleHybridRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Alloc(),
            page_size=1,
            eviction_policy="simple_hybrid",
        )
    )

    def insert(tokens, task_type=SimpleHybridTaskType.SINGLE_TURN):
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=torch.tensor(tokens, dtype=torch.int64),
                task_type=task_type,
            )
        )

    insert(list(range(0, 2000)))          # shallow 2000-token single node
    insert(list(range(0, 2032)))          # splits it: 32-token child at abs_start=2000
    insert(list(range(5000, 7000)))       # second 2000-token single node (shallow)

    deep = shallow_short = None
    for node in cache.evictable_leaves:
        if node.abs_start == 2000 and len(node.value) == 32:
            deep = node
    check(deep is not None, "expected a 32-token node at abs_start=2000")
    if deep is None:
        return

    s_deep = cache.eviction_strategy._reuse_score(deep)
    check(0.0 < s_deep < 0.02, f"deep score={s_deep:.4f}, expected ~0.008")

    result = cache.evict(EvictParams(num_tokens=32))
    print(f"      evicted {result.num_tokens_evicted} tokens")
    check(
        result.num_tokens_evicted == 32,
        f"expected the 32-token deep node to be evicted, got "
        f"{result.num_tokens_evicted} tokens (largest-segment-first behaviour)",
    )


def test_one_shot_still_drained_first():
    """#2 regression guard: one-shot content must still go before everything."""
    print("[10] one-shot band still drained first")

    class Alloc:
        device = torch.device("cpu")

        def free_segment(self, indices=None, start_pos=0):
            return None

    cache = SimpleHybridRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Alloc(),
            page_size=1,
            eviction_policy="simple_hybrid",
        )
    )

    def insert(tokens, task_type):
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=torch.tensor(tokens, dtype=torch.int64),
                task_type=task_type,
            )
        )

    # random single-turn -> one-shot, low reuse value
    insert(list(range(0, 100)), SimpleHybridTaskType.SINGLE_TURN_RANDOM)
    # normal single-turn, shallow -> high reuse score
    insert(list(range(9000, 9032)), SimpleHybridTaskType.SINGLE_TURN)

    result = cache.evict(EvictParams(num_tokens=32))
    print(f"      evicted {result.num_tokens_evicted} tokens")
    check(
        result.num_tokens_evicted == 100,
        f"expected the 100-token one-shot node, got "
        f"{result.num_tokens_evicted} tokens",
    )
    survivors = [n for n in cache.evictable_leaves if n.task_type == SimpleHybridTaskType.SINGLE_TURN]
    check(len(survivors) == 1, "the shallow single-turn node must survive")


def test_one_shot_order_by_tree_offset():
    """one-shot 带内按树上结束偏移降序（对齐 UniCache -num_hashed_tokens）。

    两个单轮请求（独立分支）：response 段 A=end 160/len 60，B=end 540/len 40。
    按片段长度排序会先淘汰 A；按树上结束偏移应先淘汰 B（位置更深），
    且 B 上方的 prompt 死链（500 token，无其他子树）随 one-shot 一次删净。
    """
    print("[10b] one-shot ordering uses tree end offset, not segment length")

    class Alloc:
        device = torch.device("cpu")

        def free_segment(self, indices=None, start_pos=0):
            return None

    cache = SimpleHybridRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Alloc(),
            page_size=1,
            eviction_policy="simple_hybrid",
        )
    )

    def insert(tokens, decode_start=None):
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=torch.tensor(tokens, dtype=torch.int64),
                task_type=SimpleHybridTaskType.SINGLE_TURN,
                decode_start=decode_start,
            )
        )

    # 请求 1：prompt 100 token + response 60 token（one-shot 叶子 end=160, len=60）
    insert(list(range(0, 100)))
    insert(list(range(0, 160)), decode_start=100)
    # 请求 2（内容不同 → 树上独立分支）：prompt 500 + response 40（one-shot 叶子 end=540, len=40）
    insert(list(range(1000, 1500)))
    insert(list(range(1000, 1540)), decode_start=500)

    one_shot = [n for n in cache.evictable_leaves
                if cache.eviction_strategy.is_one_shot(n)]
    ends = sorted(cache.eviction_strategy.one_shot_key(n)[0] for n in one_shot)
    check(ends == [-540, -160], f"one-shot 叶子 end offset 应为 160/540, got {ends}")

    result = cache.evict(EvictParams(num_tokens=40))
    print(f"      evicted {result.num_tokens_evicted} tokens")
    # end=540 的 response(len 40) 先走，且连带把其上 500-token 的 prompt 死链
    # 一次删净（40 + 500 = 540）；按旧片段长度键会错杀 len 60 的 end=160 块。
    check(result.num_tokens_evicted == 540,
          f"one-shot 应连带死链共删 540, got {result.num_tokens_evicted}")
    survivors = sorted(len(n.value) for n in cache.evictable_leaves)
    check(survivors == [60], f"survivors={survivors}, expected [60]（死链不得再出现）")
    left_os = sorted(-cache.eviction_strategy.one_shot_key(n)[0]
                     for n in cache.evictable_leaves
                     if cache.eviction_strategy.is_one_shot(n))
    check(left_os == [160], f"幸存的 one-shot 叶子 end offset 应为 160, got {left_os}")


def _find_node(cache, pred):
    stack = [cache.root_node]
    while stack:
        n = stack.pop()
        if n is not cache.root_node and pred(n):
            return n
        stack.extend(n.children.values())
    return None


def test_one_shot_chain_stops_at_branch_and_lock():
    """one-shot 链式删除的停止条件：分支点与上锁节点不得连带删除。

    场景 A（分支）：共享 prompt 0..100 挂两个 response（end 130 / end 160）。
      最深者（end=160, 30→60 token）先淘汰，链在分支点停：
      prompt 与另一个 response 必须存活。
    场景 B（锁）：锁住 prompt 后删 response，链在锁处停，prompt 留在树上。
    """
    print("[10c] one-shot chain stops at branch points and locked nodes")

    class Alloc:
        device = torch.device("cpu")

        def free_segment(self, indices=None, start_pos=0):
            return None

    def build():
        return SimpleHybridRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=Alloc(),
                page_size=1,
                eviction_policy="simple_hybrid",
            )
        )

    # ---- 场景 A：分支点 ----
    cache = build()
    for tokens, ds in ((list(range(0, 100)), None),
                       (list(range(0, 130)), 100),
                       (list(range(0, 100)) + list(range(500, 560)), 100)):
        cache.insert(InsertParams(
            key=RadixKey(array("q", tokens)),
            value=torch.tensor(tokens, dtype=torch.int64),
            task_type=SimpleHybridTaskType.SINGLE_TURN,
            decode_start=ds,
        ))
    prompt = _find_node(cache, lambda n: len(n.value) == 100)
    check(prompt is not None and len(prompt.children) == 2,
          f"prompt 应挂两个 response, got {len(prompt.children) if prompt else None}")

    # 最深者先淘汰：end=160（len 60）先于 end=130（len 30）
    result = cache.evict(EvictParams(num_tokens=60))
    print(f"      [branch] evicted {result.num_tokens_evicted} tokens")
    check(result.num_tokens_evicted == 60,
          f"应淘汰 end=160 的 response(60), got {result.num_tokens_evicted}")
    check(_find_node(cache, lambda n: len(n.value) == 100) is not None,
          "共享 prompt（分支点）不得被连带删除")
    resp1 = _find_node(cache, lambda n: len(n.value) == 30)
    check(resp1 is not None and resp1 in cache.evictable_leaves,
          "另一个 response(end=130) 必须存活且仍可淘汰")

    # ---- 场景 B：上锁 ----
    cache = build()
    cache.insert(InsertParams(
        key=RadixKey(array("q", list(range(0, 100)))),
        value=torch.tensor(list(range(0, 100)), dtype=torch.int64),
        task_type=SimpleHybridTaskType.SINGLE_TURN,
    ))
    cache.insert(InsertParams(
        key=RadixKey(array("q", list(range(0, 130)))),
        value=torch.tensor(list(range(0, 130)), dtype=torch.int64),
        task_type=SimpleHybridTaskType.SINGLE_TURN,
        decode_start=100,
    ))
    prompt = _find_node(cache, lambda n: len(n.value) == 100)
    cache.inc_lock_ref(prompt)

    result = cache.evict(EvictParams(num_tokens=30))
    print(f"      [lock] evicted {result.num_tokens_evicted} tokens")
    check(result.num_tokens_evicted == 30,
          f"锁处必须停：只删 response(30), got {result.num_tokens_evicted}")
    prompt = _find_node(cache, lambda n: len(n.value) == 100)
    check(prompt is not None and prompt.lock_ref == 1,
          "上锁的 prompt 不得被连带删除")
    check(len(cache.evictable_leaves) == 0,
          f"锁下的树不可淘汰, evictable={len(cache.evictable_leaves)}")


def test_alpha_symmetry_stays_neutral():
    """#4: 三队列效率/容量完全对称时，alpha 必须保持 1.0。

    旧实现恒等于下界 0.001：eff = (h/Σh)/(cap+1e-3) 的量级是 1e-5，被绝对量
    eps=1e-3 主导 -> r ~1e-2 -> temperature=0.2 即 5 次幂 -> ~1e-10 ->
    lower(=0.001) > upper(~1e-10) -> max(lower, min(upper, x)) 退化成常数。
    """
    print("[11] alpha symmetry: identical efficiency must keep alpha == 1.0")
    strategy = SimpleHybridStrategy()
    hits = {"single": 1000.0, "text": 1000.0, "agentic": 1000.0}
    cap = {"single": 30000.0, "text": 30000.0, "agentic": 30000.0}
    for _ in range(6):
        strategy._hit_tokens.clear()
        strategy._hit_tokens.update(hits)
        strategy.update_alpha_balance(cap)
    got = {q: round(strategy.alpha_balance[q], 6) for q in hits}
    print(f"      alpha={got}")
    for q in hits:
        a = strategy.alpha_balance[q]
        check(abs(a - 1.0) < 1e-6, f"{q} alpha={a}, expected 1.0 (neutral)")


def test_alpha_rewards_productive_queue():
    """#4: 单位容量命中产出更高的队列 alpha 更高，且差异由数据驱动。"""
    print("[12] alpha rewards the productive queue")
    strategy = SimpleHybridStrategy()
    hits = {"single": 500.0, "text": 2000.0, "agentic": 120000.0}
    cap = {"single": 38104.0, "text": 18759.0, "agentic": 26982.0}
    for _ in range(6):
        strategy._hit_tokens.clear()
        strategy._hit_tokens.update(hits)
        strategy.update_alpha_balance(cap)
    a = strategy.alpha_balance
    print(f"      alpha={ {q: round(a[q], 6) for q in hits} }")
    check(a["agentic"] > 1.0, f"agentic alpha={a['agentic']:.6f}, expected > 1")
    check(a["single"] < 1.0, f"single alpha={a['single']:.6f}, expected < 1")
    check(a["agentic"] > a["single"] * 10.0,
          f"agentic({a['agentic']:.4f}) must exceed single({a['single']:.4f}) by 10x")
    for q in hits:
        check(0.001 <= a[q] <= 10.0, f"{q} alpha={a[q]} out of bounds")


def test_alpha_punishes_zero_hit_capacity():
    """#4: 占着容量但零命中的队列必须被降权。

    旧实现 `if h > 0 and cap > 0` 直接把「有容量、无命中」的队列排除在统计外，
    于是它的 alpha 原地不动 —— single 堆到 38% 容量却只贡献 0.48% 命中时，
    恰恰拿不到任何惩罚。
    """
    print("[13] alpha punishes a queue that holds capacity with zero hits")
    strategy = SimpleHybridStrategy()
    hits = {"text": 10000.0}          # single 有容量，本窗口零命中
    cap = {"single": 30000.0, "text": 30000.0}
    # smooth=0.7 + 单步限速后收敛更慢，给足迭代次数
    for _ in range(20):
        strategy._hit_tokens.clear()
        strategy._hit_tokens.update(hits)
        strategy.update_alpha_balance(cap)
    a = strategy.alpha_balance
    print(f"      alpha={ {q: round(a[q], 6) for q in cap} }")
    check(a["single"] < 0.1,
          f"zero-hit single alpha={a['single']:.6f}, expected pushed below 0.1")
    check(a["text"] > a["single"] * 10.0,
          f"text({a['text']:.4f}) must exceed single({a['single']:.4f}) by 10x")


def test_single_score_is_time_invariant():
    """#3: 单轮分数不再含杜撰的时间项 exp(-t/600)。

    bench 里每个 session 只发一次（build_hybrid_arrival_schedule 一 session
    一到达），单轮没有「下一轮」，600s 这个常数没有对应的生成过程。
    单轮的队列级权重交给 alpha（由实测命中效率驱动），节点级只保留位置项。
    """
    print("[14] single-turn score must not depend on idle time")
    strategy = SimpleHybridStrategy(max_offset=2048)
    fresh = _mk_node(32, abs_start=0)
    fresh.last_access_time = time.monotonic()
    idle = _mk_node(32, abs_start=0)
    idle.last_access_time = time.monotonic() - 600.0
    s_fresh = strategy._reuse_score(fresh)
    s_idle = strategy._reuse_score(idle)
    print(f"      score(idle 0s)={s_fresh:.4f}  score(idle 600s)={s_idle:.4f}")
    check(
        abs(s_fresh - s_idle) < 1e-12,
        f"score must not depend on idle time: {s_fresh:.6f} vs {s_idle:.6f}",
    )
    check(
        abs(s_idle - (1.0 - 32 / 2048)) < 1e-9,
        f"idle-600s score={s_idle:.6f}, expected {1.0 - 32 / 2048:.6f} "
        f"(no fabricated exp decay)",
    )


def test_starved_queue_does_not_explode():
    """容量被压到极小的队列，效率不能因为分母趋零而爆炸。

    实测到的锯齿：single 容量掉到 786/100000（0.8%）时，eff = 命中占比/容量占比
    的分母趋零 -> r**5 顶到上界 10 -> 下一窗口它没有容量自然没命中 -> EMA 以
    ×0.1 衰减回 0.001，形成 ~2.5 分钟周期的极限环（62 分钟内 alpha 跨度 10000×）。
    对策：容量份额加下限（min_cap_share），使 eff 有界。
    """
    print("[15] starved queue must not explode to the upper bound")
    strat = SimpleHybridStrategy()
    hits = {"single": 300.0, "agentic": 100000.0}   # single 极少命中
    cap = {"single": 50.0, "agentic": 100000.0}     # single 容量 0.05%
    for _ in range(20):
        strat._hit_tokens.clear()
        strat._hit_tokens.update(hits)
        strat.update_alpha_balance(cap)
    a = strat.alpha_balance["single"]
    print(f"      alpha(single) after 20 windows = {a:.4f}")
    check(a < 0.1, f"starved single alpha={a:.4f}, expected < 0.1 (no explosion)")


def test_alpha_step_is_rate_limited():
    """单次更新不得把 alpha 变化超过 max_alpha_step 倍，避免 Bang-Bang 跳变。"""
    print("[16] per-update alpha change is rate limited")
    strat = SimpleHybridStrategy()
    strat.alpha_balance["single"] = 0.001
    strat._hit_tokens.update({"single": 100000.0, "agentic": 1.0})
    strat.update_alpha_balance({"single": 50000.0, "agentic": 50000.0})
    a = strat.alpha_balance["single"]
    limit = 0.001 * strat.max_alpha_step * 1.0001
    print(f"      0.001 -> {a:.6f}  (limit {limit:.6f})")
    check(a <= limit, f"alpha jumped 0.001 -> {a:.6f}, expected <= {limit:.6f}")


def test_alpha_converges_under_steady_load():
    """稳态负载下 alpha 必须收敛（尾部窗口不再变化），而不是持续极限环。"""
    print("[17] alpha converges under a steady workload")
    strat = SimpleHybridStrategy()
    hits = {"single": 500.0, "text": 2000.0, "agentic": 120000.0}
    cap = {"single": 38104.0, "text": 18759.0, "agentic": 26982.0}
    hist = []
    for _ in range(80):
        strat._hit_tokens.clear()
        strat._hit_tokens.update(hits)
        strat.update_alpha_balance(cap)
        hist.append(dict(strat.alpha_balance))
    tail = hist[-5:]
    got = {q: round(tail[-1][q], 6) for q in hits}
    print(f"      converged alpha={got}")
    for q in hits:
        vals = [h[q] for h in tail]
        check(
            max(vals) - min(vals) < 1e-6,
            f"{q} still oscillating in last 5 windows: {[round(v,6) for v in vals]}",
        )


def _sh_cache():
    class Alloc:
        device = torch.device("cpu")

        def free_segment(self, indices=None, start_pos=0):
            return None

    return SimpleHybridRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Alloc(),
            page_size=1,
            eviction_policy="simple_hybrid",
        )
    )


def test_workload_and_band_are_orthogonal():
    """profile 口径：工作负载归属 与 淘汰带 是两个正交维度。

    两个维度：
      - workload：response 段属于 agentic（记账不能丢）；
      - one-shot：**多轮**的 response 不是一次性内容 —— 它会成为下一轮
                  prompt 的前缀（对齐 UniCache：按会话类型归队，没有
                  「decode 就先进 one-shot」这条规则）。
    只有单轮负载的生成段才是一次性的。
    """
    print("[18] workload label vs eviction band must be orthogonal")
    st = SimpleHybridStrategy()
    prompt = _mk_node(32, 0, task_type=SimpleHybridTaskType.MULTI_TURN_AGENTIC)
    resp = _mk_node(32, 32, task_type=SimpleHybridTaskType.MULTI_TURN_AGENTIC)
    resp.is_decode = True
    rnd = _mk_node(32, 0, task_type=SimpleHybridTaskType.SINGLE_TURN_RANDOM)
    code = _mk_node(32, 0, task_type=SimpleHybridTaskType.SINGLE_TURN)

    check(st.workload_of(prompt) == "agentic", "agentic prompt workload")
    check(
        st.workload_of(resp) == "agentic" and not st.is_one_shot(resp),
        "多轮 agentic 的 response 应留在 agentic 队列，而不是 one-shot",
    )
    check(st.is_one_shot(rnd), "random request is one-shot")
    check(st.workload_of(rnd) == "random", "apps/mind2web must not fold into 'single'")
    check(st.workload_of(code) == "single", "codeparrot workload = single")


def test_workload_occ_accounting_adds_up():
    """占用统计：Σ(total−one_shot) 必须等于 alpha 可调度分母（同一遍历推导）。"""
    print("[19] workload occupancy accounting must add up")
    cache = _sh_cache()

    def ins(tokens, task, decode_start=None):
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=torch.tensor(tokens, dtype=torch.int64),
                task_type=task,
                decode_start=decode_start,
            )
        )

    ins(list(range(0, 1000)), SimpleHybridTaskType.MULTI_TURN_AGENTIC, decode_start=1000)
    ins(list(range(0, 2000)), SimpleHybridTaskType.MULTI_TURN_AGENTIC, decode_start=1000)
    ins(list(range(5000, 5100)), SimpleHybridTaskType.SINGLE_TURN_RANDOM)
    ins(list(range(9000, 9100)), SimpleHybridTaskType.SINGLE_TURN)

    wl = cache._per_workload_token_count()
    cap = cache._schedulable_per_alpha_key(wl)
    print(f"      capacity={ {k:int(v) for k,v in cap.items()} }")
    print(f"      workload={ {k:{kk:int(vv) for kk,vv in v.items()} for k,v in wl.items()} }")

    ag = wl.get("agentic", {})
    check(ag.get("total") == 2000, f"agentic total={ag.get('total')}, expected 2000")
    # 多轮 response 不算 one-shot -> agentic 的 2000 全部可调度
    check(ag.get("one_shot") == 0,
          f"agentic one_shot={ag.get('one_shot')}, expected 0 (多轮 response 不算一次性)")
    rn = wl.get("random", {})
    check(rn.get("total") == 100, f"random total={rn.get('total')}, expected 100")
    check(rn.get("one_shot") == 100, "random 全部是一次性内容")
    cd = wl.get("single", {})
    check(cd.get("one_shot") == 0, f"single one_shot={cd.get('one_shot')}, expected 0")

    tot_total = sum(v["total"] for v in wl.values())
    tot_os = sum(v["one_shot"] for v in wl.values())
    check(abs((tot_total - tot_os) - sum(cap.values())) < 1e-9,
          f"schedulable({tot_total - tot_os}) != alpha capacity({sum(cap.values())})")
    check(tot_total == 2200, f"total={tot_total}, expected 2200")
    check("random" not in cap, "random 整行一次性，不得进入 alpha 分母")


def test_record_hit_uses_workload_label():
    """命中记账按 workload；策略维度 _hit_tokens 仍按 alpha_key（行为不变）。"""
    print("[20] record_hit: profile 按 workload 记账，策略仍按 alpha_key")
    st = SimpleHybridStrategy()
    req = SimpleNamespace(rid="r1", session_id="apps_123", extra_key=None)
    st.record_hit(req, 500)
    check(st._total_hit_tokens.get("random") == 500,
          f"profile 记到 {list(st._total_hit_tokens)}, expected random=500")
    check(st._hit_tokens.get("single") == 500,
          f"策略维度应保持 alpha_key=single, got {st._hit_tokens}")


def test_record_evict_one_shot_scalar():
    """淘汰记账：workload 一本账 + one-shot 一个标量（decode 带可推导）。"""
    print("[21] record_evict: workload 账本 + one-shot 标量")
    st = SimpleHybridStrategy()
    resp = _mk_node(64, 32, task_type=SimpleHybridTaskType.MULTI_TURN_AGENTIC)
    resp.is_decode = True
    st.record_evict(resp)
    check(st._total_evicted_tokens.get("agentic") == 64,
          f"workload 记账={st._total_evicted_tokens}, expected agentic=64")
    check(st._total_evicted_one_shot == 0,
          f"多轮 response 不算 one-shot，实际={st._total_evicted_one_shot}")

    rnd = _mk_node(32, 0, task_type=SimpleHybridTaskType.SINGLE_TURN_RANDOM)
    st.record_evict(rnd)
    check(st._total_evicted_tokens.get("random") == 32,
          f"random 未被单列: {st._total_evicted_tokens}")
    check(st._total_evicted_one_shot == 32,
          f"one_shot scalar={st._total_evicted_one_shot}, expected 32")


def test_multiturn_decode_is_not_one_shot():
    """多轮负载的 response(decode) 段不得进入 one-shot。

    UniCache 按会话类型归队：agentic/text 的 response 会成为下一轮 prompt 的
    前缀，是真实复用的一部分。原实现把 is_decode 无条件映射到 evict_first，
    导致「上一轮结束 → 下一轮开始」的窗口里被优先驱逐。
    只有单轮负载的生成段（以及随机单轮）才是一次性内容。
    """
    print("[22] multi-turn decode stays in its workload queue")
    strat = SimpleHybridStrategy()

    def node(tt, dec):
        return SimpleNamespace(
            value=[0] * 64, key=[0] * 64, task_type=tt, is_decode=dec,
            last_access_time=time.monotonic(), hit_count=0,
        )

    cases = [
        (SimpleHybridTaskType.MULTI_TURN_AGENTIC, True, "agentic", False),
        (SimpleHybridTaskType.MULTI_TURN_TEXT, True, "text", False),
        (SimpleHybridTaskType.SINGLE_TURN, True, "single", True),
        (SimpleHybridTaskType.SINGLE_TURN_RANDOM, False, "random", True),
    ]
    for tt, dec, want_q, want_os in cases:
        n = node(tt, dec)
        got_q, got_os = strat.workload_of(n), strat.is_one_shot(n)
        print(f"      {tt.value:<18} is_decode={dec!s:<5} -> {got_q:<12} one_shot={got_os}")
        check(got_q == want_q, f"{tt.value} decode={dec}: workload={got_q}, expected {want_q}")
        check(got_os == want_os, f"{tt.value} decode={dec}: one_shot={got_os}, expected {want_os}")


def test_decode_band_end_to_end():
    """端到端：同一段 response，多轮归本队列，单轮归 one-shot（三处口径一致）。"""
    print("[23] end-to-end: decode band attribution is consistent")

    def build(task_type):
        cache = SimpleHybridRadixCache(
            CacheInitParams(
                disable=False, req_to_token_pool=None,
                token_to_kv_pool_allocator=None, page_size=1,
                eviction_policy="simple_hybrid",
            )
        )
        # 先有 [0,50) 前缀，再以 decode_start=50 扩展到 100 -> 后段 is_decode=True
        cache.insert(InsertParams(
            key=RadixKey(array("q", list(range(0, 50)))),
            value=torch.tensor(list(range(0, 50)), dtype=torch.int64),
            task_type=task_type,
        ))
        cache.insert(InsertParams(
            key=RadixKey(array("q", list(range(0, 100)))),
            value=torch.tensor(list(range(0, 100)), dtype=torch.int64),
            task_type=task_type,
            decode_start=50,
        ))
        return cache

    for tt, label, want_os in (
        (SimpleHybridTaskType.MULTI_TURN_AGENTIC, "agentic", False),
        (SimpleHybridTaskType.SINGLE_TURN, "single", True),
    ):
        cache = build(tt)
        dec_nodes = [n for n in cache.evictable_leaves if getattr(n, "is_decode", False)]
        check(bool(dec_nodes), f"{label}: expected at least one is_decode node")
        if not dec_nodes:
            continue
        n = dec_nodes[0]
        got = cache._is_one_shot(n)
        print(f"      {label:<8} is_decode node -> _is_one_shot={got} (expect {want_os})")
        check(got == want_os, f"{label}: _is_one_shot={got}, expected {want_os}")
        check(
            cache.eviction_strategy.is_one_shot(n) == want_os,
            f"{label}: strategy/radix disagreement on one_shot",
        )
        counts = cache._per_workload_token_count()
        rec = counts.get(label, {"total": 0.0, "one_shot": 0.0})
        want_os_tokens = 50.0 if want_os else 0.0
        check(rec["total"] == 100.0, f"{label}: workload total={rec['total']}, expected 100")
        check(
            rec["one_shot"] == want_os_tokens,
            f"{label}: one_shot tokens={rec['one_shot']}, expected {want_os_tokens}",
        )


def main():
    tests = [
        test_fresh_insert_starts_at_zero,
        test_split_via_insert,
        test_split_via_match,
        test_deep_tree_offsets,
        test_reuse_score_uses_absolute_offset,
        test_no_abs_start_fallback,
        test_randomized_invariants,
        test_eviction_prefers_deep_single_turn_node,
        test_same_queue_picks_lowest_score,
        test_one_shot_still_drained_first,
        test_one_shot_order_by_tree_offset,
        test_one_shot_chain_stops_at_branch_and_lock,
        test_alpha_symmetry_stays_neutral,
        test_alpha_rewards_productive_queue,
        test_alpha_punishes_zero_hit_capacity,
        test_single_score_is_time_invariant,
        test_starved_queue_does_not_explode,
        test_alpha_step_is_rate_limited,
        test_alpha_converges_under_steady_load,
        test_workload_and_band_are_orthogonal,
        test_workload_occ_accounting_adds_up,
        test_record_hit_uses_workload_label,
        test_record_evict_one_shot_scalar,
        test_multiturn_decode_is_not_one_shot,
        test_decode_band_end_to_end,
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
