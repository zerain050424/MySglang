"""回放后端：真实 RadixCache / 镜像 LRU 树。

两个后端实现同一份 ``Backend`` 协议，``replay.py`` 只依赖协议：

    acquire(req) -> AccessResult     # 一次请求：match → evict → alloc → insert
    stats() -> dict                  # 队内策略状态 + 淘汰来源
    close()

``RealBackend``
    在 CPU 上用真实 ``SimpleHybridRadixCache`` 跑（不加载模型），
    参照 ``test/registered/unit/mem_cache/test_radix_cache_slru_accuracy.py`` 的
    极小 ``MHATokenToKVPool`` 构造法。只做两处**运行时插桩**（不改框架代码）：

    * 子类覆写 ``_evict_one`` / ``_delete_leaf`` 记录淘汰事件；
    * 给 TreeNode 挂 ``_wtl_ph/_wtl_pd`` 记忆化「路径 token 哈希 / 绝对深度」，
      用于后悔率与保留度分析。

``FakeBackend``
    纯 Python 的镜像 radix 树（同 segment 粒度、同 LRU 语义），
    用于在没有 torch/sglang 的机器上验证调度、指标与输出 schema（smoke test），
    **不得**用于产出结论。``analyze.py`` 会拒绝把 fake 结果算进主表。

已知与真机的差异（两种后端都有）
--------------------------------
* 不建模 lock_ref：真机里运行中的请求会锁住自己的前缀；离线端淘汰只发生在
  下一次分配之前，因此更「干净」，可能高估命中率（同一个请求内部不会自淘汰）；
* 不建模并发批处理与 chunked prefill。
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
from typing import Callable, Dict, List, Optional, Sequence

from .metrics import fnv_init, fnv_update_tokens
from .trace import Request

logger = logging.getLogger("wtl.backends")

PAGE_SIZE = 1


class BackendUnavailable(RuntimeError):
    """sglang / torch 不可导入时抛出（本地无依赖机器上只能跑 FakeBackend）。"""


@dataclasses.dataclass
class AccessResult:
    ok: bool
    cached_len: int
    prompt_len: int
    total_len: int
    evicted_tokens: int
    oom: bool = False


# 淘汰事件回调：depth_end=该段结束的绝对位置，ph=token[0:depth_end] 的滚动哈希
EvictCb = Callable[[float, int, int, int, int, bool], None]
#            ts, depth_end, ph, tokens, region, is_victim


def _node_path_info(node, root) -> tuple:
    """返回 (绝对深度, 路径 token 滚动哈希)，记忆化到节点属性。

    记忆化安全性：``_split_node`` 只切分节点，不改任何节点「根到该节点结束」的
    token 序列；``_delete_leaf`` 只删不改。因此缓存值不会失效。
    """
    chain = []
    cur = node
    while cur is not None and cur is not root and getattr(cur, "_wtl_pd", None) is None:
        chain.append(cur)
        cur = cur.parent

    if cur is None or cur is root:
        depth, ph = 0, fnv_init()
    else:
        depth, ph = cur._wtl_pd, cur._wtl_ph

    for nd in reversed(chain):
        key = getattr(nd, "key", None)
        toks = key.token_ids if key is not None else ()
        ph = fnv_update_tokens(ph, toks)
        depth += len(toks)
        nd._wtl_pd, nd._wtl_ph = depth, ph
    return node._wtl_pd, node._wtl_ph


# ---------------------------------------------------------------------------
# Real backend
# ---------------------------------------------------------------------------
class RealBackend:
    """真实 ``SimpleHybridRadixCache`` 的离线回放后端（CPU，无需模型）。"""

    supports_struct = False

    def __init__(
        self,
        *,
        policy: str,
        config: Optional[dict],
        pool_tokens: int,
        on_evict: Optional[EvictCb] = None,
        device: str = "cpu",
        dtype: str = "float16",
    ) -> None:
        try:
            import torch  # noqa: F401
            radix = importlib.import_module("sglang.srt.mem_cache.radix_cache")
            cache_mod = importlib.import_module(
                "sglang.srt.mem_cache.simple_hybrid_radix_cache"
            )
            params_mod = importlib.import_module("sglang.srt.mem_cache.cache_init_params")
            base_mod = importlib.import_module("sglang.srt.mem_cache.base_prefix_cache")
            pool_mod = importlib.import_module("sglang.srt.mem_cache.memory_pool")
            alloc_mod = importlib.import_module("sglang.srt.mem_cache.allocator")
        except Exception as exc:  # pragma: no cover - 依赖缺失时的清晰报错
            raise BackendUnavailable(
                "无法导入 sglang/torch，无法使用 RealBackend；"
                "请在没有 GPU 的实验机上直接运行，或加 --backend fake 只验证链路。"
                f" 原始错误: {type(exc).__name__}: {exc}"
            ) from exc

        import torch

        self._torch = torch
        self._RadixKey = radix.RadixKey
        self._MatchPrefixParams = base_mod.MatchPrefixParams
        self._InsertParams = base_mod.InsertParams
        self._EvictParams = base_mod.EvictParams
        self._on_evict = on_evict

        tdtype = getattr(torch, dtype)

        # 极小 head 维度：每 token 只占 4 字节（k+v 各 2 字节），
        # 池容量只影响索引分配，不影响策略行为。
        kv = pool_mod.MHATokenToKVPool(
            size=int(pool_tokens),
            page_size=PAGE_SIZE,
            dtype=tdtype,
            head_num=1,
            head_dim=8,   # 只为满足形状；每 token 仅 32B，池容量不影响策略行为
            layer_num=1,
            device=device,
            enable_memory_saver=False,
        )
        allocator = alloc_mod.TokenToKVPoolAllocator(
            size=int(pool_tokens),
            dtype=tdtype,
            device=device,
            kvcache=kv,
            need_sort=False,
        )
        # req_to_token 在本回放路径里不被使用（只用 tree + allocator），
        # 但 CacheInitParams 要求非空；给小一点以免 12 路并发吃掉内存。
        req_pool = pool_mod.ReqToTokenPool(
            size=8, max_context_len=1 << 16, device=device, enable_memory_saver=False
        )
        init_params = params_mod.CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE_SIZE,
            eviction_policy=policy,
            eviction_policy_config=config,
            enable_metrics=False,
            enable_kv_cache_events=False,
        )

        base_cls = cache_mod.SimpleHybridRadixCache
        hooks = self

        class _Instrumented(base_cls):  # type: ignore[misc, valid-type]
            """只读插桩：记录淘汰事件，不改变任何淘汰/释放行为。

            ``_evict_one`` 只存在于 SimpleHybridRadixCache；当策略是 lru/slru 时
            （``--radix-eviction-policy lru`` 仍用同一个缓存类），淘汰走基类
            ``RadixCache.evict``，没有 "victim" 概念，此时把每次删除都记为 victim。
            """

            def evict(self, params):
                self._wtl_victim = None
                return super().evict(params)

            def _evict_one(self, node):  # 仅 SimpleHybridRadixCache 有
                self._wtl_victim = node
                return super()._evict_one(node)

            def _delete_leaf(self, node):
                try:
                    if hasattr(self.eviction_strategy, "is_one_shot"):
                        is_victim = node is getattr(self, "_wtl_victim", None)
                    else:
                        is_victim = True
                    hooks._record_delete(node, is_victim, self.root_node)
                except Exception:  # 观测失败绝不能影响回放
                    logger.exception("[wtl] delete hook failed")
                return super()._delete_leaf(node)

        self._clock = radix.time  # 已被 SimClock 替换；缓存引用省掉逐次 import
        self._slru_mode = False  # 建好 cache 后再判定（见下）
        self._cache_cls = _Instrumented
        self.cache = _Instrumented(init_params)
        # simple_hybrid 才带 SLRU 插件；lru/slru 臂没有，region 恒 0。
        self._slru_mode = (
            getattr(self.cache.eviction_strategy, "slru_evict", None) is not None
        )
        self._req_seq = 0
        self.pool_tokens = int(pool_tokens)

    # ------------------------------------------------------------------
    def acquire(self, req: Request) -> AccessResult:
        torch = self._torch
        tokens = req.prompt + req.output
        key = self._RadixKey(_array(tokens))
        duck = _DuckReq(req)
        res = self.cache.match_prefix(
            self._MatchPrefixParams(key=key, req=duck)
        )
        # 本 fork 的 MatchResult 没有 last_node_depth 字段，
        # device_indices 的长度就是命中的前缀长度（与 scheduler 同口径）。
        cached = int(res.device_indices.numel())
        need = len(tokens) - cached

        evicted = 0
        oom = False
        if need > 0:
            avail = self._available()
            if avail < need:
                before = self._available()
                self.cache.evict(self._EvictParams(num_tokens=need - avail))
                evicted = max(0, self._available() - before)
                if self._available() < need:
                    oom = True

        ok = not oom
        if ok and need > 0:
            new_idx = self.cache.token_to_kv_pool_allocator.alloc(need)
            if new_idx is None:
                ok = oom = True
            else:
                value = torch.cat([res.device_indices, new_idx])
                self.cache.insert(
                    self._InsertParams(
                        key=key,
                        value=value,
                        chunked=False,
                        session_id=req.sid,
                        decode_start=req.prompt_len if req.output else None,
                    )
                )
        elif ok and need == 0:
            # 全命中：仍要 insert 一次（对齐真机 cache_finished_req 的 _adopt 行为，
            # 它会让路径上的 hit_count 再 +1，从而影响 SLRU/LFU 一类策略）
            self.cache.insert(
                self._InsertParams(
                    key=key,
                    value=res.device_indices,
                    chunked=False,
                    session_id=req.sid,
                    decode_start=req.prompt_len if req.output else None,
                )
            )
        self._req_seq += 1
        return AccessResult(
            ok=ok,
            cached_len=cached,
            prompt_len=req.prompt_len,
            total_len=len(tokens),
            evicted_tokens=evicted,
            oom=oom,
        )

    # ------------------------------------------------------------------
    def _available(self) -> int:
        return int(self.cache.token_to_kv_pool_allocator.available_size())

    def _record_delete(self, node, is_victim: bool, root) -> None:
        if self._on_evict is None:
            return
        depth, ph = _node_path_info(node, root)
        tokens = int(len(node.value)) if node.value is not None else 0
        if self._slru_mode:
            # region 3 = 「受保护段」= 非一次性内容。判定与淘汰路径同源
            # （strategy.is_one_shot），而不是看 queue_tag —— 标签与共享数
            # 已经解耦，拿标签统计会把 M4 的 protected_waste_rate 算歪。
            strategy = self.cache.eviction_strategy
            region = 0 if strategy.is_one_shot(node) else 3
        else:
            region = 0
        self._on_evict(
            self._clock_now(), depth, ph, tokens, region, bool(is_victim)
        )

    def _clock_now(self) -> float:
        try:
            return float(self._clock.monotonic())
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def used_tokens(self) -> int:
        return int(self.pool_tokens - self._available())

    def stats(self) -> dict:
        strategy = self.cache.eviction_strategy
        snap = None
        victim_src = dict(getattr(strategy, "_victim_sources", {}) or {})
        alpha = dict(getattr(getattr(strategy, "alpha", None), "balance", {}) or {})
        slru = getattr(strategy, "slru_evict", None)
        return {
            "available": self._available(),
            "evictable_size": int(getattr(self.cache, "evictable_size_", 0) or 0),
            # 共享数账本 + 本改写器实际给出的候选数（机制证据）
            "slru": slru.stats() if slru is not None else None,
            "slru_share_min": getattr(slru, "share_min", None),
            "victim_src": victim_src,
            "alpha": alpha,
        }

    def close(self) -> None:
        self.cache = None


class _DuckReq:
    """``match_prefix`` 只读 ``req.rid``；用鸭子对象避免构造完整 Req。"""

    __slots__ = ("rid", "session_id")

    def __init__(self, req: Request) -> None:
        self.rid = req.rid
        self.session_id = req.sid


def _array(tokens: Sequence[int]):
    from array import array

    return array("q", tokens)


# ---------------------------------------------------------------------------
# Fake backend：镜像 radix 树（LRU），用于无依赖机器的链路 smoke test
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("key", "parent", "children", "last", "depth", "ph", "alive")

    def __init__(self, key: List[int], parent: Optional["_Node"], ts: float) -> None:
        self.key = key
        self.parent = parent
        self.children: Dict[int, "_Node"] = {}
        self.last = ts
        self.depth = (parent.depth if parent else 0) + len(key)
        self.ph = fnv_update_tokens(parent.ph if parent else fnv_init(), key)
        self.alive = True


class FakeBackend:
    """真 radix 语义的纯 Python 镜像（LRU 淘汰），仅用于链路验证。"""

    supports_struct = False

    def __init__(
        self,
        *,
        pool_tokens: int,
        on_evict: Optional[EvictCb] = None,
        **_ignored,
    ) -> None:
        self.pool = int(pool_tokens)
        self.used = 0
        self._on_evict = on_evict
        self.root = _Node([], None, 0.0)
        self.root.depth = 0
        self.clock = 0.0

    # ---- 基本操作 ----
    def _split(self, node: _Node, off: int) -> _Node:
        """在 node.key 的 off 处切分，返回前缀半段。"""
        parent = node.parent
        prefix = _Node(node.key[:off], parent, node.last)
        prefix.children = {node.key[off]: node}
        parent.children[prefix.key[0]] = prefix
        node.parent = prefix
        node.key = node.key[off:]
        node.depth = prefix.depth + len(node.key)
        node.ph = fnv_update_tokens(prefix.ph, node.key)
        return prefix

    def _match(self, tokens: List[int], ts: float) -> int:
        node = self.root
        i = 0
        n = len(tokens)
        while i < n:
            child = node.children.get(tokens[i])
            if child is None:
                break
            # 逐 token 比较
            j = 0
            klen = len(child.key)
            while j < klen and i + j < n and child.key[j] == tokens[i + j]:
                j += 1
            child.last = ts
            if j < klen:
                if j > 0:
                    split = self._split(child, j)
                    split.last = ts
                    i += j
                    node = split
                break
            i += j
            node = child
        return i

    def _insert(self, tokens: List[int], ts: float, spent: int) -> None:
        """沿已命中的前 ``spent`` 个 token 下潜，再把剩余部分作为新段挂上。

        注意：必须先走完命中前缀再创建节点。若直接从根开始用 ``i = spent``
        建节点，会把后缀挂到根上，树拓扑就坏了（曾经在这个实现上踩过：
        fake 与真 LRU 的 cached_tokens 只有 28% 一致，正是这个原因）。
        """
        node = self.root
        i = 0
        n = len(tokens)
        while i < spent:
            child = node.children.get(tokens[i])
            if child is None:  # 理论上不会发生：spent 由 _match 给出
                break
            node = child
            node.last = ts
            i += len(child.key)

        if i < n:
            nd = _Node(tokens[i:], node, ts)
            node.children[nd.key[0]] = nd
            self.used += len(nd.key)

    def _leaves(self) -> List[_Node]:
        out = []
        stack = [self.root]
        while stack:
            nd = stack.pop()
            if not nd.children:
                if nd is not self.root:
                    out.append(nd)
            else:
                stack.extend(nd.children.values())
        return out

    def _evict_one(self) -> int:
        leaves = self._leaves()
        if not leaves:
            return 0
        victim = min(leaves, key=lambda x: (x.last, -x.depth))
        freed = len(victim.key)
        self.used -= freed
        parent = victim.parent
        if parent is not None:
            parent.children.pop(victim.key[0], None)
        victim.alive = False
        if self._on_evict is not None:
            self._on_evict(self.clock, victim.depth, victim.ph, freed, 0, True)
        return freed

    # ---- 协议 ----
    def acquire(self, req: Request) -> AccessResult:
        tokens = req.prompt + req.output
        self.clock = req.ts
        cached = self._match(tokens, req.ts)
        need = len(tokens) - cached
        evicted = 0
        oom = False
        while need > 0 and self.pool - self.used < need:
            freed = self._evict_one()
            if freed == 0:
                oom = True
                break
            evicted += freed
        if not oom:
            self._insert(tokens, req.ts, cached)
        return AccessResult(
            ok=not oom,
            cached_len=cached,
            prompt_len=req.prompt_len,
            total_len=len(tokens),
            evicted_tokens=evicted,
            oom=oom,
        )

    def used_tokens(self) -> int:
        return int(self.used)

    def stats(self) -> dict:
        return {
            "available": self.pool - self.used,
            "evictable_size": self.used,
            "victim_src": {},
            "alpha": {},
        }

    def close(self) -> None:
        pass


def make_backend(name: str, **kwargs):
    name = (name or "auto").lower()
    if name == "fake":
        return FakeBackend(**kwargs)
    if name == "real":
        return RealBackend(**kwargs)
    if name == "auto":
        try:
            return RealBackend(**kwargs)
        except BackendUnavailable as exc:
            logger.warning("[wtl] RealBackend 不可用，退回 FakeBackend（结果仅可做链路验证）: %s", exc)
            kwargs.pop("policy", None)
            kwargs.pop("config", None)
            return FakeBackend(**kwargs)
    raise ValueError(f"未知 backend {name!r}（real/fake/auto）")
