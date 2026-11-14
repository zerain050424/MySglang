"""请求替身：喂给生产 ``match_prefix`` / ``_adopt`` 的最小字段集合。

生产里的 ``Req`` 带着采样参数、KV 行、输出缓冲、状态机等一整套东西；缓存路径
只读其中七项：

* ``rid`` / ``session_id``      —— 命中归因、会话账本；
* ``origin_input_ids``          —— 分类器的 prompt_len、decode_start；
* ``output_ids``                —— 收尾时 ``origin+output`` 一并写树；
* ``extra_key`` / ``cache_salt`` —— 命名空间隔离（仿真不用，但必须是 None）；
* ``priority``                  —— 树节点优先级；
* ``kv.req_pool_idx``           —— ``_adopt`` 从 req_to_token 取那一行。

为什么走 ``_adopt`` 而不是自己拼 ``insert``：``_adopt`` 里有三件容易漏的事——
``session_registry.touch``（重置 L/d、刷新活跃时钟）、``decode_start`` 页对齐、
``split_prompt`` 在 prompt 边界断开（让 response 段可被单独淘汰）。复刻这三行
等于把策略语义抄第二份，所以宁可依赖这个私有方法。
"""

from __future__ import annotations

from array import array
from typing import List, Optional, Sequence

import torch


class SimKv:
    """``_adopt`` 读 ``req.kv.req_pool_idx``，并在收尾时读 ``cache_protected_len``
    （已确定归属树的前缀长度；新请求从 0 起算）。"""

    __slots__ = ("req_pool_idx", "cache_protected_len")

    def __init__(self, req_pool_idx: int = 0, cache_protected_len: int = 0) -> None:
        self.req_pool_idx = req_pool_idx
        self.cache_protected_len = cache_protected_len


class FakeReqToTokenPool:
    """一行的 req_to_token 表。

    ``_adopt`` 取 ``req_to_token[row, :len(token_ids)]`` 再 ``.to(int64, copy=True)``
    当作节点 value。内容没有语义（缓存只关心长度与切片），但**宽度必须够**：
    请求的 prompt+output 有多长，池就要多宽。
    """

    __slots__ = ("req_to_token", "width")

    def __init__(self, width: int) -> None:
        if width <= 0:
            raise ValueError(f"req_to_token 宽度必须为正，got {width}")
        self.width = int(width)
        self.req_to_token = torch.arange(self.width, dtype=torch.int64).unsqueeze(0)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<FakeReqToTokenPool width={self.width}>"


class SimReq:
    """单轮请求。``prompt`` 只放 **prompt** token（不含 response）。"""

    __slots__ = (
        "rid",
        "session_id",
        "origin_input_ids",
        "output_ids",
        "extra_key",
        "cache_salt",
        "priority",
        "tool_call",
        "kv",
    )

    def __init__(
        self,
        rid: str,
        session_id: str,
        prompt: Sequence[int],
        output_len: int = 0,
        priority: int = 0,
        tool_call: Optional[bool] = None,
    ) -> None:
        self.rid = rid
        self.session_id = session_id
        self.origin_input_ids = array("q", prompt)
        self.output_ids: List[int] = [0] * int(output_len)
        self.extra_key = None
        self.cache_salt = None
        self.priority = priority
        self.tool_call = tool_call
        self.kv = SimKv()

    def fill_ids(self) -> array:
        """写入树的全序列：prompt + response（对齐 ``cache_finished_req``）。"""
        ids = array("q", self.origin_input_ids)
        ids.extend(self.output_ids)
        return ids

    @property
    def total_len(self) -> int:
        return len(self.origin_input_ids) + len(self.output_ids)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"<SimReq {self.rid} sid={self.session_id} "
            f"prompt={len(self.origin_input_ids)} out={len(self.output_ids)}>"
        )
