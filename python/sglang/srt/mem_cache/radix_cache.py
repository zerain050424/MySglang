from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import logging
import os
import sys
import threading
import time
from array import array
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.uni_utils import (   
    SessionRegistry,
    SimpleHybridTaskType,
    classify_simple_hybrid_task_type,
    workload_label_of,
)
from sglang.srt.mem_cache.utils import (
    get_eviction_strategy,
    split_node_hash_value,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    """is_bigram=True: token_ids holds raw tokens (N+1 for N bigrams); slices share one boundary token."""

    __slots__ = ("token_ids", "extra_key", "cache_salt", "is_bigram", "limit")

    def __init__(
        self,
        token_ids: array[int], ## 当前这个 RadixKey 所描述的那一段 token 序列", list:  [ptr, ptr, ptr, ...]  →  每个 ptr 指向一个 PyLong 对象（约 28 字节）
                                # array: [int64, int64, int64, ...]  →  每个 8 字节，连续排列
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
        limit: Optional[int] = None,
        cache_salt: Optional[str] = None,
    ):
        # token ids sequence (raw ints in both modes)
        self.token_ids = token_ids
        # Namespaces the tree and storage; omitted from KV events.
        self.extra_key = extra_key
        # Namespaces the tree, storage and KV events.
        self.cache_salt = cache_salt or None
        # bigram view over token_ids: length = max(0, len(token_ids) - 1)
        self.is_bigram = is_bigram
        # Optional cap on raw tokens: behave as if token_ids were sliced to
        # token_ids[:limit], without the O(n) copy. None = use all tokens.
        self.limit = limit

    def _raw_len(self) -> int:
        n = len(self.token_ids)
        if self.limit is not None and self.limit < n:
            return self.limit
        return n

    def raw_token_ids(self) -> array:
        """token_ids honoring `limit` (copies only when capped)."""
        n = self._raw_len()
        t = self.token_ids
        return t if n == len(t) else t[:n]

    def __len__(self) -> int:
        n = self._raw_len()
        if self.is_bigram:
            return n - 1 if n > 0 else 0
        return n

    # TODO(Jialin): vectorize with numpy without PyLong boxing
    def __iter__(self) -> Iterator:
        t = self.token_ids
        n = self._raw_len()
        if self.is_bigram:
            for i in range(n - 1 if n > 0 else 0):
                yield (t[i], t[i + 1])
        elif n == len(t):
            yield from t
        else:
            for i in range(n):
                yield t[i]

    def __getitem__(self, idx: Union[int, slice]) -> RadixKey:
        # Normalize int -> 1-element slice so the rest handles one shape.
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"RadixKey index out of range: {idx}")
            idx = slice(idx, idx + 1)
        start, stop, step = idx.indices(len(self))
        if step != 1:
            raise ValueError("RadixKey slice step must be 1")

        if self.is_bigram:
            # bigrams [start, stop) span raw tokens [start, stop + 1);
            # empty slice -> empty raw tokens (not a dangling boundary token).
            raw = self.token_ids[start : stop + 1] if stop > start else array("q")
            return RadixKey(
                raw,
                self.extra_key,
                is_bigram=True,
                cache_salt=self.cache_salt,
            )
        return RadixKey(
            self.token_ids[start:stop],
            self.extra_key,
            cache_salt=self.cache_salt,
        )

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, cache_salt={self.cache_salt!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''}, is_bigram={self.is_bigram})"

    def page_aligned(self, page_size: int) -> RadixKey: ### 将RadixKey截断为page_size的整数倍长度
                                                        # 返回新的RadixKey,不足一页的token直接丢弃
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def maybe_to_bigram_view(
        self,
        is_eagle: bool,
        value: Optional[torch.Tensor] = None,
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        # O(1): flip the bigram flag instead of materializing a tuple list.
        # value is paired with raw tokens and gets truncated to the bigram count.
        if is_eagle and not self.is_bigram:
            self.is_bigram = True
            if value is not None:
                value = value[: len(self)]
        return self, value

    def _check_compatible(self, other: RadixKey) -> None:
        if self.extra_key != other.extra_key:
            raise ValueError(
                f"RadixKey operations require matching extra_key, but got "
                f"{self.extra_key=} != {other.extra_key=}"
            )
        if self.cache_salt != other.cache_salt:
            raise ValueError(
                f"RadixKey operations require matching cache_salt, but got "
                f"{self.cache_salt=} != {other.cache_salt=}"
            )

    def match(self, other: RadixKey, page_size: int = 1) -> int:
        """Logical-unit prefix length shared with ``other``. Result is rounded down to ``page_size``."""
        return self.match_at(other, offset=0, page_size=page_size)

    def match_at(self, other: RadixKey, offset: int, page_size: int = 1) -> int:
        """Match without slicing while preserving bigram boundaries and limit semantics."""
        self._check_compatible(other)
        if self.is_bigram != other.is_bigram:
            raise ValueError("RadixKey operations require matching bigram modes")
        if offset < 0 or offset > len(other):
            raise IndexError(f"RadixKey offset out of range: {offset}")
        t0, t1 = self.token_ids, other.token_ids
        assert type(t0) is type(t1), (type(t0), type(t1))
        n = min(self._raw_len(), other._raw_len() - offset) ###从offset开始匹配，匹配长度为两者的最小长度

        # Exponential search for the first diverging token: gallop in doubling
        # windows (one C-level slice compare each), then binary-search the window
        # holding the divergence -- no per-token Python loop on long shared prefixes.
        matched_tokens = n
        lo = 0
        step = 1
        while lo < n: ## 指数扩展二分查找,step每次翻倍，直到找到不匹配的窗口，然后在窗口内二分查找
            hi = lo + step if lo + step < n else n
            if t0[lo:hi] != t1[offset + lo : offset + hi]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if t0[lo:mid] == t1[offset + lo : offset + mid]: ## t0[a:b] != t1[c:d] 这种切片比较在 C 层完成，速度极快
                        lo = mid  ## 如果不相同，在 [lo, mid] 区间内继续二分查找 
                    else:
                        hi = mid
                matched_tokens = lo
                break
            lo = hi ## 如果相同，lo迅速移动到hi，继续下一轮指数扩展，[lo, lo + step]
            step *= 2

        if self.is_bigram:
            matched = max(0, min(matched_tokens - 1, len(self), len(other) - offset))
            return (matched // page_size) * page_size if page_size > 1 else matched

        matched_tokens = min(matched_tokens, len(self), len(other) - offset)
        if page_size == 1:
            return matched_tokens
        return (matched_tokens // page_size) * page_size ## 返回匹配到的token数量，向下取整为page_size的整数倍

    def child_key(self, page_size: int = 1): ## 对child_key_at的封装，默认从offset=0开始截取page_size个逻辑单位作为key
        """Hashable dict-key for the first ``page_size`` logical units, namespaced by ``extra_key``."""
        return self.child_key_at(offset=0, page_size=page_size)

    def child_key_at(self, offset: int, page_size: int = 1): ## 将序列从offset开始截取page_size个逻辑单位作为key,用来索引到它的子节点,用来构造键
        """Hashable child key at ``offset`` without slicing token storage."""
        if offset < 0 or offset + page_size > len(self):
            raise IndexError(
                f"RadixKey child range out of bounds: offset={offset}, "
                f"page_size={page_size}, len={len(self)}"
            )
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[offset], t[offset + 1])
            else:
                plain = tuple(
                    (t[j], t[j + 1]) for j in range(offset, offset + page_size)
                )
        else:
            plain = (
                t[offset] if page_size == 1 else tuple(t[offset : offset + page_size])
            )
        if self.cache_salt is not None:
            return ((self.extra_key, self.cache_salt), plain)
        return plain if self.extra_key is None else (self.extra_key, plain)


class TreeNode: ## TreeNode 是基数树（RadixTree）的节点，每个节点代表一条路径上的一个片段（RadixKey），
                # 并携带缓存值、访问统计、优先级等元信息，用于前缀缓存（prefix caching）场景。
    counter = 0

    def __init__(
        self, id: Optional[int] = None, priority: int = 0, abs_start: int = 0
    ):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None ##挂在在节点上的缓存值，通常是KV缓存的索引或数据片段
        ## SimpleHYBRID：本节点片段第一个 token 在整条序列中的绝对偏移（从根算起）。
        ## abs_start + len(value) 即 UniCache/vLLM 的 num_hashed_tokens（片段结束偏移）。
        ## 注意：radix 片段长度由「共享拓扑」决定，与序列位置无关，二者不可混用。
        self.abs_start = abs_start
        self.lock_ref = 0 ## 节点的锁引用计数，表示有多少请求正在使用该节点及其子树，防止被驱逐。
        self.last_access_time = time.monotonic() ## 节点的最后访问时间，用于LRU等驱逐策略。
        self.creation_time = time.monotonic() ## 节点的创建时间，用于统计和调试。

        self.hit_count = 0
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None 
        # Namespace-aware hashes used only for external KV events.
        self.event_hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        # SimpleHYBRID (UniCache) workload metadata.
        # task_type: multi-turn text/file/image/search/agentic vs single-turn.
        # is_decode: generated-output (decode) portion, never reused, evict first.
        self.priority = priority
        self.task_type = None
        self.is_decode = False
        # SimpleHYBRID：节点归属的会话（用于会话级缺口记账，见 uni_utils.SessionRegistry）
        self.session_id: Optional[str] = None
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self): ## 节点是否已经被驱逐，判断依据是节点的值是否为None。
        return self.value is None

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time


from sglang.srt.mem_cache.registry import (  # noqa: E402
    register_radix_cache_backend,
)


def _classic_factory(ctx):
    return RadixCache(ctx.params)


try:
    register_radix_cache_backend("classic", _classic_factory)
except ValueError:  # 重复导入时幂等
    pass


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.is_eagle = params.is_eagle
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_events = KVCacheEventRecorder(
            enabled=params.enable_kv_cache_events, page_size=self.page_size
        )

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            dev = self.token_to_kv_pool_allocator.device
            if isinstance(dev, (str, torch.device)):
                self.device = torch.device(dev)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.eviction_strategy = get_eviction_strategy(
            self.eviction_policy, params.eviction_policy_config
        )
        # Optional SimpleHYBRID hooks:present only on strategies
        # that implement them, so other policies pay nothing.
        ## 可选钩子
        self._hit_recorder = getattr(self.eviction_strategy, "record_hit", None)
        self._alpha_updater = getattr(
            self.eviction_strategy, "maybe_update_alpha", None
        )

        self.evictable_leaves = set()
        # SimpleHYBRID：会话级 KV 账本（每会话 d/L/gaps），供公平淘汰使用
        self.session_registry = SessionRegistry()
        self.reset()

        # ---- 策略无关的「按工作负载占用」采样器（env 门控，默认关）----
        # 用途：LRU 等不触发 [SimpleHybrid profile] 的策略，也能拿到每队列
        # 占用占比。__init__ 在 scheduler 进程内执行，这里起一个 daemon 线程，
        # 周期性调 _per_workload_token_count()（基类方法，树上节点带 task_type，
        # 任何策略都可用）。默认关闭：无 env 时这里只做一次 os.environ.get 判空。
        _occ_env = os.environ.get("SGLANG_WORKLOAD_OCC_LOG", "0")
        logger.info(
            "[occ-sampler] env=%s pid=%s policy=%s",
            _occ_env, os.getpid(), self.eviction_policy,
        )
        if _occ_env == "1":
            interval = int(os.environ.get("SGLANG_WORKLOAD_OCC_INTERVAL", "30"))
            t = threading.Thread(
                target=self._workload_occ_log_loop,
                args=(interval,),
                daemon=True,
                name="workload-occ-sampler",
            )
            t.start()
            logger.info("[occ-sampler] thread started, alive=%s", t.is_alive())

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=array("q"), extra_key=None)
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )
        self.kv_events.record_all_cleared()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0).
            ``last_device_node`` and ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)

        if self.disable or len(key) == 0:
            return self._empty_match_result

        key = key.page_aligned(self.page_size)

        if len(key) == 0:
            return self._empty_match_result

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = self._empty_match_result.device_indices
        # Phase 3: record the request's prefix-cache hit (once per request).
        if self._hit_recorder is not None and params.req is not None:
            self._hit_recorder(params.req, value.numel())
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked
        task_type = params.task_type
        decode_start = params.decode_start
        session_id = params.session_id
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            # Debug/test fallback: use token ids themselves as values.
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len, last_node = self._insert_helper(
            self.root_node, key, value, priority, chunked, task_type,
            decode_start, session_id,
        )
        return InsertResult(prefix_len=prefix_len, last_device_node=last_node)

    def _adopt(
        self,
        req: Req,
        token_ids,
        *,
        chunked: bool = False,
        key_limit: Optional[int] = None,
        split_prompt: bool = False,
    ) -> Tuple[RadixKey, torch.Tensor, int]:
        """Insert the page-aligned key of ``token_ids`` (cut at ``key_limit``)
        and free the duplicates it exposed; returns (key, row slice, matched)."""
        kv_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(token_ids)
        ]
        radix_key = RadixKey(
            token_ids,
            req.extra_key,
            is_bigram=self.is_eagle,
            cache_salt=req.cache_salt,
        ).page_aligned(self.page_size)
        if key_limit is not None:
            radix_key = radix_key[:key_limit]
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        priority = req.priority or 0
        # SimpleHYBRID: classify workload type and locate the prompt/output
        # boundary so decode-portion nodes can be flagged (evicted first).
        task_type = classify_simple_hybrid_task_type(req.session_id, req.extra_key)
        decode_start = (len(req.origin_input_ids) // self.page_size) * self.page_size
        # SimpleHYBRID：会话账本入口。请求写入后链完整驻留，缺口归零。
        self.session_registry.touch(
            req.session_id, getattr(req, "rid", ""), len(radix_key)
        )
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=priority,
                task_type=task_type,
                decode_start=decode_start,
                session_id=req.session_id,
               )
           )
        if split_prompt:
            # Split the leaf at the prompt boundary so eviction can drop the
            # output KV without the prompt; a prefix re-insert only changes topology.
            prompt_key = RadixKey(
                token_ids[: len(req.origin_input_ids)],
                req.extra_key,
                is_bigram=self.is_eagle,
                cache_salt=req.cache_salt,
            ).page_aligned(self.page_size)
            if 0 < len(prompt_key) < key_len:
                self.insert(
                    InsertParams(
                        key=prompt_key,
                        value=values[: len(prompt_key)],
                        priority=priority + 1,
                        # The request created these nodes moments ago; another
                        # hit_count bump would promote every prompt node.
                        chunked=True,
                        task_type=task_type,
                        session_id=req.session_id,
                        decode_start=decode_start
                    )
                )

        self.token_to_kv_pool_allocator.free_segment(
            kv_indices[req.kv.cache_protected_len : result.prefix_len],
            start_pos=req.kv.cache_protected_len,
        )
        return radix_key, kv_indices, result.prefix_len

    def cache_finished_req(self, req: Req, *, owned_kv_len: int):
        """Cache request when it finishes."""
        if not self.disable:
            token_ids = (req.origin_input_ids + req.output_ids)[:owned_kv_len]
            radix_key, _, _ = self._adopt(req, token_ids, split_prompt=True)
            req.kv.cache_protected_len = len(radix_key)
        # The protected prefix is not this req's to free.
        self.free_kv_row(req.kv, [(req.kv.cache_protected_len, owned_kv_len)])
        self.unpin(req)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        radix_key, kv_indices, _ = self._adopt(req, req.get_fill_ids(), chunked=chunked)

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(radix_key), (
            f"{len(new_indices)=}, {len(radix_key)=}"
        )

        self.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(req.kv.cache_protected_len, len(new_indices))),
            new_indices[req.kv.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.kv.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        # Phase 3: refresh per-type alphas from recent hit efficiency.
        # 同时传两份口径：alpha 用可调度容量，profile 用 workload 占用。
        if self._alpha_updater is not None:
            self._alpha_updater(
                *self._profile_token_counts(), self.session_registry.summary()
            )
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            # Tree values are page-aligned copies of a kv row: page-exact segment.
            self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self.kv_events.record_remove(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _is_one_shot(self, node) -> bool:
        """一次性内容判定：优先用策略口径（SimpleHybrid 有多轮/单轮之分）。

        非 SimpleHybrid 策略没有 is_one_shot，退回旧规则（is_decode 或随机单轮）。
        计数与淘汰必须用同一判定，否则 alpha 的分母口径与实际淘汰会互相打架。
        """
        fn = getattr(self.eviction_strategy, "is_one_shot", None)
        if fn is not None:
            return bool(fn(node))
        task_type = getattr(node, "task_type", None)
        return bool(getattr(node, "is_decode", False)) or (
            task_type == SimpleHybridTaskType.SINGLE_TURN_RANDOM
        )

    def _workload_occ_log_loop(self, interval: float) -> None:
        """后台循环：每 interval 秒打一行按工作负载的占用占比。
        输出格式：`[workload occ] single=.. random=.. text=.. agentic=.. ...`
        其中数值是 `_per_workload_token_count` 的 total（含 response 段）。
        并发说明：树在 scheduler 主线程上被增删，这里在 daemon 线程里只读遍历；
        GIL 保证不会崩，最坏读到一次瞬时不一致的计数（采样统计可接受）。
        """
        i = 0
        while True:
            time.sleep(interval)
            i += 1
            try:
                counts = self._per_workload_token_count()
                logger.info(
                    "[occ-sampler] tick=%s counts=%s", i,
                    {k: int(v['total']) for k, v in sorted(counts.items())} if counts else {},
                )
                if counts:
                    msg = " ".join(
                        f"{k}={int(v['total'])}" for k, v in sorted(counts.items())
                    )
                    logger.info("[workload occ] %s", msg)
            except Exception:  # 采样失败不能影响主流程
                logger.exception("workload occ sampler failed")

    def _per_workload_token_count(self) -> dict:
        """按工作负载统计驻留 token，**包含各自的 response(decode) 段**。
        「这个 workload 到底占了池子多少」——is_decode 只决定淘汰优先级，
        不改变归属（旧口径把归属和优先级压成一个 queue_name，多轮 agentic
        的 response 段被记进一次性带，真实占用被低估数倍）。
        返回 {label: {"total": t, "one_shot": o}}；
        one_shot = 该 workload 中一次性内容的 token（random 整行、
        single 的 response 段；多轮恒为 0）。
        Σ(total − one_shot) 即 alpha 的可调度分母，见 _schedulable_per_alpha_key。
        """
        out: dict = {}
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                if child.evicted:
                    continue
                task_type = getattr(child, "task_type", None)
                label = workload_label_of(task_type)
                rec = out.setdefault(label, {"total": 0.0, "one_shot": 0.0})
                n = len(child.key)
                rec["total"] += n
                if self._is_one_shot(child):
                    rec["one_shot"] += n
                stack.append(child)
        return out

    def _schedulable_per_alpha_key(self, counts: dict) -> dict:
        """alpha 分母：可调度容量 = total − one_shot，按 alpha_key 聚合。
        workload label 与 alpha_key 只在 random 上不同，而 random 整行
        one-shot（total − one_shot == 0）自然落空——分母纯度由构造保证。"""
        sched: dict = {}
        for label, rec in counts.items():
            s = rec["total"] - rec["one_shot"]
            if s > 0:
                sched[label] = sched.get(label, 0.0) + s
        return sched

    def _profile_token_counts(self):
        """一次遍历给出两份口径：(alpha 可调度容量, workload 占用)。"""
        counts = self._per_workload_token_count()
        return self._schedulable_per_alpha_key(counts), counts

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert node is self.root_node, (
                    "This request holds the node from another tree"
                )
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey): ## node：当前节点，key：要匹配的RadixKey
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = key.child_key(self.page_size) 
        ##获取当前key的前page_size个逻辑单位作为child_key，从offset=0开始
        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key] ## 获取匹配的子节点
            child.last_access_time = access_time
            prefix_len = child.key.match(key, page_size=self.page_size) ##计算当前节点的key与子节点的key的匹配长度，返回匹配的逻辑单位数量，向下取整为page_size的整数倍
            if prefix_len < len(child.key): ## 如果匹配长度小于子节点的key长度，说明匹配结束在子节点的中间，需要将子节点拆分为两个节点，一个表示匹配的前缀，一个表示剩余的后缀
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value) ## 追加KV张量
                node = new_node ## 将当前节点更新为新创建的节点
                break
            else:
                value.append(child.value) ## 完全匹配追加全量张量
                node = child ## 将当前节点更新为子节点
                key = key[prefix_len:] ## 将key截断为剩余未匹配的部分

                if len(key): ## 如果还有剩余的key,以page_size为单位获取child_key，用于下一轮匹配
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(
           self,
           key: RadixKey,
           child: TreeNode,
           split_len: int,
           child_abs_start: Optional[int] = None,
           decode_start: Optional[int] = None,
       ):
           new_node = TreeNode(priority=child.priority)
           # SimpleHYBRID：前缀一半继承原偏移，后缀一半整体后移 split_len。
           # 必须在覆盖 child.abs_start 之前把旧值读出来。
           new_node.abs_start = child.abs_start
           child.abs_start = child.abs_start + split_len
           new_node.hit_count = child.hit_count
           new_node.children = {key[split_len:].child_key(self.page_size): child}
           new_node.parent = child.parent
           new_node.lock_ref = child.lock_ref
           new_node.key = child.key[:split_len]
           new_node.value = child.value[:split_len].clone()
           # SimpleHYBRID metadata: prefix half inherits; suffix half is
           # reclassified as decode when the split lands at/after decode_start.
           new_node.task_type = child.task_type
           new_node.is_decode = child.is_decode
           new_node.session_id = child.session_id
           self.session_registry.attach(new_node.session_id, len(new_node.value))
           child.parent = new_node
           child.key = child.key[split_len:]
           child.value = child.value[split_len:].clone()
           if child_abs_start is not None and decode_start is not None:
               child.is_decode = child_abs_start >= decode_start
           new_node.parent.children[key.child_key(self.page_size)] = new_node

            # Split hash_value if it was already computed, otherwise leave as None
           new_node.hash_value, child.hash_value = split_node_hash_value(
             child.hash_value, split_len, self.page_size
            )
           new_node.event_hash_value, child.event_hash_value = split_node_hash_value(
            child.event_hash_value, split_len, self.page_size
           )

           return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # Skip the hit count update for chunked requests to avoid self-referencing
        # inflation where a chunked request increments hit_count on nodes it created
        # in previous chunks.
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
        task_type: Optional[SimpleHybridTaskType] = None,
        decode_start: Optional[int] = None,
        session_id: Optional[str] = None,
    ):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0, node

        child_key = key.child_key(self.page_size) ##获取子键

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key] ## 下沉到子节点
            node.last_access_time = access_time
            prefix_len = node.key.match(key, page_size=self.page_size) ##计算当前下沉到的子节点与key的匹配长度
            total_prefix_length += prefix_len ## 累加匹配长度
            key = key[prefix_len:] ## 将key截断为剩余未匹配的部分
            value = value[prefix_len:] ## 将KV截断为剩余未匹配的部分

            if prefix_len < len(node.key): ## 如果匹配长度小于key,证明在当前节点中不完全匹配,需要分裂节点
                new_node = self._split_node(node.key, node, prefix_len,child_abs_start=total_prefix_length,decode_start=decode_start)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node ## 分裂完后继续下沉
            else: ## 完全匹配,更新优先级和命中计数
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            if len(key): ## 如果还有剩余的key,以page_size为单位获取child_key，用于下一轮匹配
                child_key = key.child_key(self.page_size)

        if len(key): ## 如果还有剩余的key,说明没有匹配到子节点,需要创建新的子节点
            # SimpleHYBRID：新片段从「已匹配长度」处开始（绝对偏移）。
            new_node = TreeNode(priority=priority, abs_start=total_prefix_length)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            new_node.task_type = task_type
            new_node.is_decode = (
                decode_start is not None and total_prefix_length >= decode_start
            )
            new_node.session_id = session_id
            self.session_registry.attach(session_id, len(new_node.value))
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self.kv_events.record_store(new_node)
            node = new_node
        return total_prefix_length, node

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == child.key.child_key(self.page_size), (
                    f"{key=}, {child.key.child_key(self.page_size)=}"
                )

    def _delete_leaf(self, node):
        # SimpleHYBRID：节点消亡的唯一原语，会话缺口在此累加（覆盖淘汰与释放两条路径）
        self.session_registry.charge(node)
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None) ## 从父节点的children字典中删除当前节点，并返回删除的节点
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0: ## 节点已被驱逐且被锁
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node) ## 从可驱逐叶子节点集合中移除该节点
            return

        for child in node.children.values():
            if not child.evicted: ## 如果子节点没有被驱逐，说明当前节点不是叶子节点
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves: ## 节点没有被驱逐且没有被锁，且所有子节点都已被驱逐，说明当前节点是可驱逐叶子节点，将其添加到可驱逐叶子节点集合中
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5, 6, 7]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [8, 9, 10, 11, 12]))))
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=array("q", [1, 2, 3, 13, 14])))
        )
    )
