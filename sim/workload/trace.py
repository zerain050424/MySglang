"""离线会话 trace 数据结构与合成负载生成器。"""

from __future__ import annotations

import random
from array import array
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Turn:
    """单轮请求内容。

    Attributes:
        t: 到达时刻（虚拟秒）。
        prompt: 本轮提示词 token 序列（不含本轮回复）。
        out: 生成长度。
        tool_call: 上一轮是否发生工具调用（供分类器判定）。
        response: 可选真实回复 token；缺失时驱动层会生成独占哨兵，避免跨请求误命中。
    """

    t: float
    prompt: array
    out: int = 0
    tool_call: bool = False
    response: Optional[array] = None

    @property
    def response_len(self) -> int:
        return len(self.response) if self.response is not None else self.out


@dataclass(frozen=True)
class Session:
    """按时间升序排列的一条多轮或单轮会话。"""

    session_id: str
    turns: Tuple[Turn, ...]
    task_name: str = ""

    @property
    def n_turns(self) -> int:
        return len(self.turns)


def _tokens(values: Iterable[int]) -> array:
    return array("q", values)


def max_total_len(sessions: Iterable[Session]) -> int:
    """返回所有轮次中 prompt + response 的最大长度，用于确定 req_to_token 池宽度。"""
    return max(
        (len(t.prompt) + t.response_len for s in sessions for t in s.turns),
        default=0,
    )


def synthetic_sessions(
    *,
    seed: int = 0,
    n_struct: int = 4,
    n_random: int = 4,
    n_chat: int = 2,
    n_agentic: int = 2,
    horizon: float = 600.0,
    prefix_len: int = 128,
    base_len: int = 320,
    out_len: int = 48,
    max_prompt_len: int = 2048,
    chat_turns: int = 3,
    agentic_turns: int = 6,
    chat_gap_s: float = 30.0,
    agentic_gap_s: float = 3.0,
    growth: int = 200,
) -> List[Session]:
    """生成覆盖四类会话（struct/random/chat/agentic）的合成测试负载。"""

    rng = random.Random(seed)
    n_total = n_struct + n_random + n_chat + n_agentic
    arrivals = sorted(rng.uniform(0.0, horizon) for _ in range(n_total))

    prefix = _tokens(range(1, prefix_len + 1))
    sessions: List[Session] = []
    cursor = 0

    def unique_base(i: int) -> int:
        # 为每条会话分配互不重叠的 token 区间，避免偶然前缀共享。
        return 10_000 * (i + 1)

    def one_turn(sid: str, t: float, prompt: array) -> Session:
        return Session(sid, (Turn(t=t, prompt=prompt, out=out_len),))

    for i in range(n_struct):
        base = unique_base(i)
        tail = _tokens(range(base, base + max(1, base_len - prefix_len)))
        sessions.append(one_turn(f"Code-{i}", arrivals[cursor], prefix + tail))
        cursor += 1

    for i in range(n_random):
        base = unique_base(100 + i)
        sessions.append(
            one_turn(f"apps_{i}", arrivals[cursor], _tokens(range(base, base + base_len)))
        )
        cursor += 1

    for i in range(n_chat):
        base = unique_base(200 + i)
        head = _tokens(range(base, base + base_len))
        t = arrivals[cursor]
        turns = []
        for k in range(chat_turns):
            fill = min(growth * k, max(0, max_prompt_len - base_len))
            prompt = head + _tokens([k] * fill)
            turns.append(Turn(t=t, prompt=prompt, out=out_len))
            t += chat_gap_s * rng.uniform(0.85, 1.15)
        sessions.append(Session(f"ShareGPT-{i}", tuple(turns)))
        cursor += 1

    for i in range(n_agentic):
        base = unique_base(300 + i)
        head = _tokens(range(base, base + base_len))
        t = arrivals[cursor]
        turns = []
        for k in range(agentic_turns):
            fill = min(growth * 3 * k, max(0, max_prompt_len - base_len))
            prompt = head + _tokens([k] * fill)
            turns.append(Turn(t=t, prompt=prompt, out=out_len, tool_call=True))
            t += agentic_gap_s * rng.uniform(0.8, 1.2)
        sessions.append(Session(f"alfred_{i}", tuple(turns)))
        cursor += 1

    return sessions


def by_class(sessions: Sequence[Session]) -> Dict[str, int]:
    """按 session_id 前缀统计四类会话数。"""
    out = {"struct": 0, "random": 0, "chat": 0, "agentic": 0}
    for s in sessions:
        for prefix, label in (
            ("Code-", "struct"),
            ("apps_", "random"),
            ("ShareGPT", "chat"),
            ("alfred_", "agentic"),
        ):
            if s.session_id.startswith(prefix):
                out[label] += 1
                break
    return out
