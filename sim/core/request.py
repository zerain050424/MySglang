"""供生产前缀缓存 match_prefix 与 _adopt 读取的最小请求替身。"""

from __future__ import annotations

from array import array
from typing import List, Optional, Sequence

import torch


class SimKv:
    """_adopt 读取 req.kv.req_pool_idx，并在结束时按 cache_protected_len 释放未重合前缀。"""

    __slots__ = ("req_pool_idx", "cache_protected_len")

    def __init__(self, req_pool_idx: int = 0, cache_protected_len: int = 0) -> None:
        self.req_pool_idx = req_pool_idx
        self.cache_protected_len = cache_protected_len


class FakeReqToTokenPool:
    """单行 req_to_token 张量池，宽度须不小于最长请求序列。"""

    __slots__ = ("req_to_token", "width")

    def __init__(self, width: int) -> None:
        if width <= 0:
            raise ValueError(f"req_to_token 宽度必须为正，got {width}")
        self.width = int(width)
        self.req_to_token = torch.arange(self.width, dtype=torch.int64).unsqueeze(0)

    def __repr__(self) -> str:
        return f"<FakeReqToTokenPool width={self.width}>"


# 无真实 response_ids 时使用的每请求唯一回复哨兵基址：
# 需高于真实词表，同时保证基址+请求数同时落在 C++ hash 的 uint32 与 opt_map array('i') int32 范围内。
OUTPUT_SENTINEL_BASE = 1 << 30
MAX_SENTINEL_REQUESTS = 1 << 20
assert OUTPUT_SENTINEL_BASE + MAX_SENTINEL_REQUESTS < 1 << 31, (
    "response 哨兵必须同时安全于 uint32 与有符号 int32"
)


class SimReq:
    """最小请求对象；origin_input_ids 仅放 prompt，避免污染分类器提示长度。"""

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
        output_token: Optional[int] = None,
        response_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.rid = rid
        self.session_id = session_id
        self.origin_input_ids = array("q", prompt)
        if response_ids is not None:
            if output_len and int(output_len) != len(response_ids):
                raise ValueError(
                    "response_ids length does not match output_len: "
                    f"{len(response_ids)} != {output_len}"
                )
            self.output_ids: List[int] = [int(token) for token in response_ids]
        else:
            token = (
                OUTPUT_SENTINEL_BASE + 1 if output_token is None else int(output_token)
            )
            self.output_ids = [token] * int(output_len)
        self.extra_key = None
        self.cache_salt = None
        self.priority = priority
        self.tool_call = tool_call
        self.kv = SimKv()

    def fill_ids(self) -> array:
        """返回落树完整序列（prompt + output）。"""
        ids = array("q", self.origin_input_ids)
        ids.extend(self.output_ids)
        return ids

    @property
    def total_len(self) -> int:
        return len(self.origin_input_ids) + len(self.output_ids)

    def __repr__(self) -> str:
        return (
            f"<SimReq {self.rid} sid={self.session_id} "
            f"prompt={len(self.origin_input_ids)} out={len(self.output_ids)}>"
        )
