"""Proof tests: HeterCache "offset" must be the ABSOLUTE token offset.

HeterCache's heap key is (-num_hashed_tokens, ...), where num_hashed_tokens is a
block's end offset in the sequence. The sglang port used len(node.value)
(radix node segment length) instead, which is an artifact of sharing topology,
not of position in the sequence. These tests pin down the intended semantics:

  node.abs_start                     == absolute offset of the segment's first token
  node.abs_start + len(node.value)   == absolute end offset ("num_hashed_tokens")
  child.abs_start == parent.abs_start + len(parent.value)

以及节点标签 / 配额裁决不再依赖任何“全局价值分”。

Run:  /root/sl_old/bin/python test/manual/test_heter_cache_abs_offset.py
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
from sglang.srt.mem_cache.heter_cache_radix_cache import (  # noqa: E402
    HeterCacheRadixCache,
)
from sglang.srt.mem_cache.evict_policy import HeterCacheStrategy  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey  # noqa: E402
from sglang.srt.mem_cache.heter_utils import (  # noqa: E402
    SessionRegistry,
    SessionState,
    SessionType,
    HeterCacheNodeTag,
)

# 配额必填：HeterCache 只走配额仲裁，测试统一用一组均衡上限。
QUOTA_CFG = {"caps": {"struct": 0.3, "agentic": 0.4, "chat": 0.3}}

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


def _mk_node(
    n_tokens,
    abs_start,
    tag=HeterCacheNodeTag.QSTRUCT,
):
    return SimpleNamespace(
        value=[0] * n_tokens,
        key=[0] * n_tokens,
        abs_start=abs_start,
        queue_tag=tag,
        is_decode=False,
        last_access_time=time.monotonic(),
        hit_count=0,
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


def test_one_shot_still_drained_first():
    """#2 regression guard: one-shot content must still go before everything."""
    print("[10] one-shot band still drained first")

    class Alloc:
        device = torch.device("cpu")

        def free_segment(self, indices=None, start_pos=0):
            return None

    cache = HeterCacheRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Alloc(),
            page_size=1,
            eviction_policy="heter_cache",
            eviction_policy_config={
                "quota": {"caps": {"struct": 0.3, "agentic": 0.4, "chat": 0.3}}
            },
        )
    )

    def insert(tokens, tag):
        result = cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=torch.tensor(tokens, dtype=torch.int64),
            )
        )
        result.last_device_node.queue_tag = tag

    # random single-turn -> one-shot, low reuse value
    insert(list(range(0, 100)), HeterCacheNodeTag.QTRANS)
    # normal single-turn, shallow -> high reuse score
    insert(list(range(9000, 9032)), HeterCacheNodeTag.QSTRUCT)

    result = cache.evict(EvictParams(num_tokens=32))
    print(f"      evicted {result.num_tokens_evicted} tokens")
    check(
        result.num_tokens_evicted == 100,
        f"expected the 100-token one-shot node, got "
        f"{result.num_tokens_evicted} tokens",
    )
    survivors = [
        n
        for n in cache.evictable_leaves
        if n.queue_tag == HeterCacheNodeTag.QSTRUCT
    ]
    check(len(survivors) == 1, "the shallow single-turn node must survive")


def test_one_shot_order_by_tree_offset():
    """one-shot 带内按树上结束偏移降序（对齐 HeterCache -num_hashed_tokens）。

    两个单轮请求（独立分支）：response 段 A=end 160/len 60，B=end 540/len 40。
    按片段长度排序会先淘汰 A；按树上结束偏移应先淘汰 B（位置更深），
    且 B 上方的 prompt 死链（500 token，无其他子树）随 one-shot 一次删净。
    """
    print("[10b] one-shot ordering uses tree end offset, not segment length")

    class Alloc:
        device = torch.device("cpu")

        def free_segment(self, indices=None, start_pos=0):
            return None

    cache = HeterCacheRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Alloc(),
            page_size=1,
            eviction_policy="heter_cache",
            eviction_policy_config={
                "quota": {"caps": {"struct": 0.3, "agentic": 0.4, "chat": 0.3}}
            },
        )
    )

    def insert(tokens, decode_start=None):
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=torch.tensor(tokens, dtype=torch.int64),
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
        return HeterCacheRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=Alloc(),
                page_size=1,
                eviction_policy="heter_cache",
                eviction_policy_config={
                    "quota": {"caps": {"struct": 0.3, "agentic": 0.4, "chat": 0.3}}
                },
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

    ))
    cache.insert(InsertParams(
        key=RadixKey(array("q", list(range(0, 130)))),
        value=torch.tensor(list(range(0, 130)), dtype=torch.int64),
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





def _sh_cache():
    class Alloc:
        device = torch.device("cpu")

        def free_segment(self, indices=None, start_pos=0):
            return None

    return HeterCacheRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Alloc(),
            page_size=1,
            eviction_policy="heter_cache",
            eviction_policy_config={
                "quota": {"caps": {"struct": 0.3, "agentic": 0.4, "chat": 0.3}}
            },
        )
    )


def test_queue_and_band_are_orthogonal():
    """profile 口径：工作负载归属 与 淘汰带 是两个正交维度。

    两个维度：
      - queue：response 段属于 agentic（记账不能丢）；
      - one-shot：**多轮**的 response 不是一次性内容 —— 它会成为下一轮
                  prompt 的前缀（对齐 HeterCache：按会话类型归队，没有
                  「decode 就先进 one-shot」这条规则）。
    只有单轮负载的生成段才是一次性的。
    """
    print("[18] queue attribution vs eviction band must be orthogonal")
    st = HeterCacheStrategy(quota=QUOTA_CFG)
    # one-shot 判定看「会话类型 + 跨会话共享数」，因此节点需要会话证据。
    reg = SessionRegistry()
    st.session_registry = reg
    st.slru_evict.session_registry = reg
    for sid, stype in (
        ("ag", SessionType.AGENTIC),
        ("rnd", SessionType.TRANS),
        ("code", SessionType.STRUCT),
    ):
        reg._sessions[sid] = SessionState(session_id=sid, session_type=stype)

    prompt = _mk_node(32, 0, tag=HeterCacheNodeTag.QAGENT)
    prompt.session_id = "ag"
    resp = _mk_node(32, 32, tag=HeterCacheNodeTag.QAGENT)
    resp.is_decode = True
    resp.session_id = "ag"
    rnd = _mk_node(32, 0, tag=HeterCacheNodeTag.QTRANS)
    rnd.session_id = "rnd"
    code = _mk_node(32, 0, tag=HeterCacheNodeTag.QSTRUCT)
    code.session_id = "code"
    shared = _mk_node(32, 0, tag=HeterCacheNodeTag.QSTRUCT)
    shared.session_id = "code"
    # 被 >= share_min 个外部会话命中过：跨会话模板，退出 one-shot 带
    shared.share_sessions = st.slru_evict.share_min

    check(st._node_tag(prompt) is HeterCacheNodeTag.QAGENT, "agentic prompt tag")
    check(
        st._node_tag(resp) is HeterCacheNodeTag.QAGENT and not st.is_one_shot(resp),
        "多轮 agentic 的 response 应留在 agentic 队列，而不是 one-shot",
    )
    check(st.is_one_shot(rnd), "random request is one-shot")
    check(
        st.is_one_shot(code),
        "STRUCT 会话里未被跨会话复用的前缀仍是一次性内容",
    )
    check(
        not st.is_one_shot(shared),
        "STRUCT 前缀一旦被 >=share_min 个外部会话复用就不再是一次性内容",
    )
    check(st._node_tag(rnd) is HeterCacheNodeTag.QTRANS, "apps/mind2web must stay QTRANS")
    check(st._node_tag(code) is HeterCacheNodeTag.QSTRUCT, "codeparrot tag = QSTRUCT")


def test_queue_occ_accounting_adds_up():
    """占用统计：每个队列的 total/one_shot 必须与插入量对得上。"""
    print("[19] queue occupancy accounting must add up")
    cache = _sh_cache()

    # one-shot 判定看会话类型 + 共享数，所以插入时要带 session_id 并预先登记类型
    reg = cache.session_registry
    for sid, stype in (
        ("ag", SessionType.AGENTIC),
        ("rnd", SessionType.TRANS),
        ("code", SessionType.STRUCT),
    ):
        reg._sessions[sid] = SessionState(session_id=sid, session_type=stype)

    def ins(tokens, task, decode_start=None, session_id=None):
        result = cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=torch.tensor(tokens, dtype=torch.int64),
                decode_start=decode_start,
                session_id=session_id,
            )
        )
        result.last_device_node.queue_tag = task
        return result.last_device_node

    ins(list(range(0, 1000)), HeterCacheNodeTag.QAGENT,
        decode_start=1000, session_id="ag")
    ins(list(range(0, 2000)), HeterCacheNodeTag.QAGENT,
        decode_start=1000, session_id="ag")
    ins(list(range(5000, 5100)), HeterCacheNodeTag.QTRANS,
        session_id="rnd")
    shared = ins(list(range(9000, 9100)), HeterCacheNodeTag.QSTRUCT,
                 session_id="code")
    # 把 struct 前缀标记成「已被足够多外部会话复用」：它应退出 one-shot 带
    shared.share_sessions = cache.eviction_strategy.slru_evict.share_min

    wl = cache._per_queue_token_count()
    print(f"      queue={ {k:{kk:int(vv) for kk,vv in v.items()} for k,v in wl.items()} }")

    ag = wl.get("agentic", {})
    check(ag.get("total") == 2000, f"agentic total={ag.get('total')}, expected 2000")
    # 多轮 response 不算 one-shot -> agentic 的 2000 全部可调度
    check(ag.get("one_shot") == 0,
          f"agentic one_shot={ag.get('one_shot')}, expected 0 (多轮 response 不算一次性)")
    rn = wl.get("random", {})
    check(rn.get("total") == 100, f"random total={rn.get('total')}, expected 100")
    check(rn.get("one_shot") == 100, "random 全部是一次性内容")
    cd = wl.get("struct", {})
    check(cd.get("one_shot") == 0,
          f"struct one_shot={cd.get('one_shot')}, expected 0 (已跨会话复用)")

    for sid in ("ag", "rnd", "code"):
        reg._sessions.pop(sid, None)

    tot_total = sum(v["total"] for v in wl.values())
    tot_os = sum(v["one_shot"] for v in wl.values())
    check(tot_total == 2200, f"total={tot_total}, expected 2200")
    check(tot_os == 100, f"one_shot={tot_os}, expected 100 (只有 random 整行)")


def test_record_hit_uses_queue_name():
    """命中记账按队列归因到 HeterProfiler，且每请求只记一次。"""
    print("[20] record_hit: profile 按队列记账")
    st = HeterCacheStrategy(quota=QUOTA_CFG)
    req = SimpleNamespace(rid="r1", session_id="apps_123", extra_key=None)
    node = _mk_node(32, 0, tag=HeterCacheNodeTag.QTRANS)
    st.record_hit(req, node, 500)
    check(st.profiler._total_hit_tokens.get("random") == 500,
          f"profile 记到 {list(st.profiler._total_hit_tokens)}, expected random=500")
    st.record_hit(req, node, 500)
    check(st.profiler._total_hit_tokens.get("random") == 500,
          "同一 rid 重复上报不得重复记账")


def test_record_evict_one_shot_scalar():
    """淘汰记账：队列一本账 + one-shot 一个标量（decode 带可推导）。"""
    print("[21] record_evict: 队列账本 + one-shot 标量")
    st = HeterCacheStrategy(quota=QUOTA_CFG)
    resp = _mk_node(64, 32, tag=HeterCacheNodeTag.QAGENT)
    resp.is_decode = True
    st.record_evict(resp)
    check(st._total_evicted_tokens.get("agentic") == 64,
          f"队列记账={st._total_evicted_tokens}, expected agentic=64")
    check(st._total_evicted_one_shot == 0,
          f"多轮 response 不算 one-shot，实际={st._total_evicted_one_shot}")

    rnd = _mk_node(32, 0, tag=HeterCacheNodeTag.QTRANS)
    st.record_evict(rnd)
    check(st._total_evicted_tokens.get("random") == 32,
          f"random 未被单列: {st._total_evicted_tokens}")
    check(st._total_evicted_one_shot == 32,
          f"one_shot scalar={st._total_evicted_one_shot}, expected 32")


def test_multiturn_decode_is_not_one_shot():
    """多轮负载的 response(decode) 段不得进入 one-shot。

    HeterCache 按会话类型归队：agentic/text 的 response 会成为下一轮 prompt 的
    前缀，是真实复用的一部分。原实现把 is_decode 无条件映射到 evict_first，
    导致「上一轮结束 → 下一轮开始」的窗口里被优先驱逐。
    只有单轮负载的生成段（以及随机单轮）才是一次性内容。
    """
    print("[22] multi-turn decode stays in its queue")
    strat = HeterCacheStrategy(quota=QUOTA_CFG)

    def node(tt, dec):
        return SimpleNamespace(
            value=[0] * 64, key=[0] * 64, queue_tag=tt, is_decode=dec,
            last_access_time=time.monotonic(), hit_count=0,
        )

    cases = [
        (HeterCacheNodeTag.QAGENT, True, "agentic", False),
        (HeterCacheNodeTag.QCHAT, True, "chat", False),
        (HeterCacheNodeTag.QSTRUCT, True, "struct", True),
        (HeterCacheNodeTag.QTRANS, False, "random", True),
    ]
    for tt, dec, want_q, want_os in cases:
        n = node(tt, dec)
        got_q, got_os = strat._node_tag(n).queue_name, strat.is_one_shot(n)
        print(f"      {tt.value:<18} is_decode={dec!s:<5} -> {got_q:<12} one_shot={got_os}")
        check(got_q == want_q, f"{tt.value} decode={dec}: queue={got_q}, expected {want_q}")
        check(got_os == want_os, f"{tt.value} decode={dec}: one_shot={got_os}, expected {want_os}")


def test_decode_band_end_to_end():
    """端到端：同一段 response，多轮归本队列，单轮归 one-shot（三处口径一致）。"""
    print("[23] end-to-end: decode band attribution is consistent")

    def build(tag):
        cache = HeterCacheRadixCache(
            CacheInitParams(
                disable=False, req_to_token_pool=None,
                token_to_kv_pool_allocator=None, page_size=1,
                eviction_policy="heter_cache",
                eviction_policy_config={
                    "quota": {"caps": {"struct": 0.3, "agentic": 0.4, "chat": 0.3}}
                },
            )
        )
        # 先有 [0,50) 前缀，再以 decode_start=50 扩展到 100 -> 后段 is_decode=True
        result = cache.insert(InsertParams(
            key=RadixKey(array("q", list(range(0, 50)))),
            value=torch.tensor(list(range(0, 50)), dtype=torch.int64),
        ))
        result.last_device_node.queue_tag = tag
        result = cache.insert(InsertParams(
            key=RadixKey(array("q", list(range(0, 100)))),
            value=torch.tensor(list(range(0, 100)), dtype=torch.int64),
            decode_start=50,
        ))
        result.last_device_node.queue_tag = tag
        return cache

    for tt, label, want_os in (
        (HeterCacheNodeTag.QAGENT, "agentic", False),
        (HeterCacheNodeTag.QSTRUCT, "struct", True),
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
        counts = cache._per_queue_token_count()
        rec = counts.get(label, {"total": 0.0, "one_shot": 0.0})
        want_os_tokens = 50.0 if want_os else 0.0
        check(rec["total"] == 100.0, f"{label}: queue total={rec['total']}, expected 100")
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
        test_randomized_invariants,
        test_one_shot_still_drained_first,
        test_one_shot_order_by_tree_offset,
        test_one_shot_chain_stops_at_branch_and_lock,
        test_queue_and_band_are_orthogonal,
        test_queue_occ_accounting_adds_up,
        test_record_hit_uses_queue_name,
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
