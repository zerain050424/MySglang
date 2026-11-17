"""trace 结构 + 合成负载生成器。

输入只要求一件事：**每轮一个时刻 + 一段 prompt token + 一个 response**。
response 可以是 token 序列（回放数据集时），也可以只给长度（合成负载）。
不读任何数据集文件——真实 trace（ShareGPT / AgentBank / Codeparrot）由调用方
转成同样的结构喂进来即可。

合成负载只是为了让自检与冒烟能覆盖四类会话（结构复用 / 一次性 / 人速多轮 /
高频 agent），不追求与真实数据同分布。
"""

from __future__ import annotations

import random
from array import array
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Turn:
    """一轮请求。``prompt`` 只含 prompt token，``response`` 是本轮生成的回答。"""

    t: float
    prompt: array
    out: int = 0
    tool_call: bool = False
    # Dataset replay keeps the real response tokens so the next turn's prompt
    # (prev prompt + prev response + new) can hit the response segment.
    # None means the caller only knows the length; SimReq then uses a sentinel.
    response: Optional[array] = None

    @property
    def response_len(self) -> int:
        return len(self.response) if self.response is not None else self.out


@dataclass(frozen=True)
class Session:
    """一条会话的全部轮次（按 ``t`` 升序）。

    ``task_name`` 对齐 bench_heap 的 per-task 口径（``[heap] tasks=...``）：
    emitter 按 task 出命中率，仿真器按 session_type 出；两者同时保留。
    """

    session_id: str
    turns: Tuple[Turn, ...]
    task_name: str = ""

    @property
    def n_turns(self) -> int:
        return len(self.turns)


def _tokens(values: Iterable[int]) -> array:
    return array("q", values)


def max_total_len(sessions: Iterable[Session]) -> int:
    """最长的一轮 ``prompt + response``，决定 req_to_token 需要多宽。"""
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
    """造一批四类混合的会话。

    四类各自对应策略里的一条判定路径：

    * ``struct``：单轮、共享同一条长前缀 -> 跨会话共享数达标，进 struct 队；
    * ``random``：单轮、彼此不共享 -> 落 QTRANS，可被判 one-shot；
    * ``chat``：多轮、轮间隔 ≥ ``GAP_SPLIT_S``(15s) -> CHAT；
    * ``agentic``：多轮、轮间隔 < 15s、上下文暴涨 -> AGENTIC。

    ``max_prompt_len`` 把多轮会话的上下文增长截顶：顺序模型不模拟 retraction，
    单条请求超过池容量会直接报错，所以生成器默认保证每轮 prompt 不超过它。

    Returns:
        按会话给出，轮内已按时间升序；会话之间不排序（由驱动层按时刻归并）。
    """

    rng = random.Random(seed)
    n_total = n_struct + n_random + n_chat + n_agentic
    arrivals = sorted(rng.uniform(0.0, horizon) for _ in range(n_total))

    prefix = _tokens(range(1, prefix_len + 1))  # 跨会话共享的公共前缀
    sessions: List[Session] = []
    cursor = 0

    def unique_base(i: int) -> int:
        return 10_000 * (i + 1)  # 每会话独立的 token 区间，避免意外共享

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
            # 上下文暴涨：每轮追加的 tool 结果远大于普通人速会话
            fill = min(growth * 3 * k, max(0, max_prompt_len - base_len))
            prompt = head + _tokens([k] * fill)
            turns.append(Turn(t=t, prompt=prompt, out=out_len, tool_call=True))
            t += agentic_gap_s * rng.uniform(0.8, 1.2)
        sessions.append(Session(f"alfred_{i}", tuple(turns)))
        cursor += 1

    return sessions


def by_class(sessions: Sequence[Session]) -> Dict[str, int]:
    """按 session_id 前缀分桶计数（与 Heter-Bench 的 GT 口径一致）。"""
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
