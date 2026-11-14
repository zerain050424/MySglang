from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_cache.components.base import (
        TreeComponent,
    )


@dataclasses.dataclass
class CacheInitParams:
    disable: bool
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int

    is_eagle: bool = False
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_cp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    pp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    eviction_policy: str = "lru"
    # Keyword arguments for the eviction policy's constructor; see the strategy
    # classes in evict_policy.py for what each policy accepts.
    eviction_policy_config: Optional[dict[str, Any]] = None

    enable_metrics: bool = False
    enable_kv_cache_events: bool = False
    enable_session_radix_cache: bool = False

    enable_mamba_extra_buffer: bool = False
    enable_mamba_extra_buffer_lazy: bool = False

    pp_rank: int = 0
    pp_size: int = 1

    attn_cp_rank: int = 0
    attn_cp_size: int = 1

    chunked_prefill_size: Optional[int] = None

    sliding_window_size: Optional[int] = None

    # Time-to-live for cache entries in seconds. If None, TTL is disabled.
    cache_ttl_seconds: Optional[float] = None

    # 时间源；None 表示 time.monotonic。
    #
    # 必须由缓存层单点持有：节点时间戳由缓存盖（TreeNode 创建 / match / insert），
    # 而 one-shot 豁免、老化 TTL、承压校准窗口、profile 时钟都拿它做差。任何一处
    # 仍读墙钟，注入假时钟后算出的就是「假 now − 真时间戳」，全部阈值失真。
    # 离线回放（仿真器 / 单测）注入步进时钟即可让 TTL 以虚拟秒推进。
    now: Optional[Callable[[], float]] = None

    tree_components: Optional[tuple[ComponentType, ...]] = None
    component_registry_override: Optional[dict[ComponentType, type[TreeComponent]]] = (
        None
    )

    mtp_draft_device_pools: tuple[object, ...] = ()
