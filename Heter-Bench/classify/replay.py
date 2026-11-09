"""把会话事件流重放进生产分类器，得到每条会话逐轮的判定。

这里只做两件事：
1. 构造 `SessionRegistry` 能看到的证据（请求开始时刻、prompt 长度、上一轮 tool_call、
   跨会话前缀共享）；
2. 按真实时间顺序喂进去，收集每轮的 `session_type`。

分类规则本身不在这里复制一份——直接调用
`python/sglang/srt/mem_cache/uni_utils.py` 的 `SessionRegistry`，
这样评测的就是生产代码，而不是它的替身。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SGLANG_PY = os.path.abspath(os.path.join(_HERE, "..", "..", "python"))
if _SGLANG_PY not in sys.path:
    sys.path.insert(0, _SGLANG_PY)

from sglang.srt.mem_cache.uni_utils import SessionRegistry  # noqa: E402

from session_data import PRED_LABEL, SessionRecord, Turn  # noqa: E402

_MASK = (1 << 64) - 1
_BLOCK_MULT = 1000003


@dataclass
class ReplayConfig:
    """分类器阈值 + 结构证据口径。

    前三个字段对应 `SessionRegistry` 的类属性，缺省即生产值；改动只发生在评测进程里。
    """

    multi_turn_min: int = SessionRegistry.MULTI_TURN_MIN
    gap_split_s: float = SessionRegistry.GAP_SPLIT_S
    growth_tokens: int = SessionRegistry.AGENTIC_CONTEXT_GROWTH_TOKENS
    structure: str = "prefix"  # "prefix" | "off"
    structure_turns: str = "first"  # "first" | "all"
    block: int = 16  # 前缀索引的块粒度（生产 page_size=1；块粒度只影响共享长度的量化）
    min_share_tokens: int = 1  # 共享前缀达到多少 token 才算跨会话结构证据
    tool_call_mode: str = "prior"  # "prior" | "same" | "off"

    def registry_class(self):
        return type(
            "EvalSessionRegistry",
            (SessionRegistry,),
            {
                "MULTI_TURN_MIN": self.multi_turn_min,
                "GAP_SPLIT_S": self.gap_split_s,
                "AGENTIC_CONTEXT_GROWTH_TOKENS": self.growth_tokens,
            },
        )


class PrefixIndex:
    """块粒度的前缀归属索引，模拟 radix 树里「这段前缀最早由哪个会话写入」。

    只登记尚未出现过的前缀块，因此内存 ≈ 不同前缀块的数量，而不是请求总量。
    `match` 返回 (被其他会话拥有的最长共享长度, 该 owner)，与
    `RadixCache._match_prefix_helper` 里逐节点 `_record_cross_session_hit` 的语义一致：
    只要命中的缓存节点属于别的会话，就算跨会话共享。
    """

    def __init__(self, block: int = 16):
        self.block = max(1, block)
        self._owner: Dict[int, str] = {}

    def _key(self, h: int) -> int:
        return h & _MASK

    def match(self, tokens: List[int], session_id: str) -> Tuple[int, Optional[str]]:
        shared_len, owner = 0, None
        h, matched = 0, 0
        for i in range(0, len(tokens), self.block):
            block = tuple(tokens[i : i + self.block])
            h = (h * _BLOCK_MULT + hash(block)) & _MASK
            key = self._key(h)
            other = self._owner.get(key)
            if other is None:
                break
            matched += len(block)
            if other != session_id:
                shared_len, owner = matched, other
        return shared_len, owner

    def insert(self, tokens: List[int], session_id: str) -> None:
        h = 0
        for i in range(0, len(tokens), self.block):
            block = tuple(tokens[i : i + self.block])
            h = (h * _BLOCK_MULT + hash(block)) & _MASK
            key = self._key(h)
            if key not in self._owner:
                self._owner[key] = session_id


@dataclass
class SessionResult:
    session_id: str
    gt: str
    n_turns: int
    preds: List[str] = field(default_factory=list)  # 每轮结束时的判定（PRED_LABEL 空间）
    shared_len: int = 0  # 观测到的跨会话共享前缀长度（结构证据）
    shared_owner: Optional[str] = None
    new_tokens: List[int] = field(default_factory=list)  # 每轮写入 KV 的 token 数

    @property
    def final(self) -> str:
        return self.preds[-1] if self.preds else "random"


def _tool_call_of(turn: Turn, mode: str) -> Optional[bool]:
    if mode == "off":
        return None
    if mode == "same":
        return turn.same_turn_tool_call
    return turn.prior_tool_call


def _events(sessions: Dict[str, SessionRecord]):
    events = []
    for sid, rec in sessions.items():
        for pos, turn in enumerate(rec.turns):
            events.append((turn.start, sid, pos))
    events.sort()
    return events


def replay(
    sessions: Dict[str, SessionRecord], cfg: ReplayConfig
) -> Dict[str, SessionResult]:
    """按时间顺序重放全部会话，返回每条会话逐轮判定。"""
    registry = cfg.registry_class()()
    results = {
        sid: SessionResult(session_id=sid, gt=rec.gt, n_turns=len(rec.turns))
        for sid, rec in sessions.items()
        if rec.turns
    }
    index = PrefixIndex(cfg.block) if cfg.structure == "prefix" else None

    for start, sid, pos in _events(sessions):
        rec = sessions[sid]
        turn = rec.turns[pos]

        # 1) 跨会话结构证据：当前请求命中「别的会话写入的前缀」。
        if index is not None and turn.tokens:
            if cfg.structure_turns == "all" or turn.index == 1:
                shared_len, owner = index.match(turn.tokens, sid)
                if owner is not None and shared_len >= cfg.min_share_tokens:
                    registry.record_cross_session_hit(sid, owner)
                    res = results[sid]
                    if shared_len >= res.shared_len:
                        res.shared_len, res.shared_owner = shared_len, owner
                index.insert(turn.tokens, sid)

        # 2) 请求级证据：轮间隔、prompt 长度、上一轮 tool_call。
        state = registry.observe_request(
            sid,
            rid=f"{sid}_turn_{turn.index}",
            prompt_len=turn.prompt_len,
            now=start,
            tool_call=_tool_call_of(turn, cfg.tool_call_mode),
        )
        results[sid].preds.append(PRED_LABEL[state.session_type.value])
        results[sid].new_tokens.append(turn.new_tokens)

    return results
