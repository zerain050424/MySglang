"""``session_data.jsonl`` → ``List[Session]``（与 bench_heap 逐字段对齐）。

数据格式（每行一个 session，键即 session_id）::

    {"<sid>": {"id": "<sid>", "num_turns": N,
               "turn_1": {"input_text": ..., "input_ids": [...],
                          "response_text": ..., "response_ids": [...], "tool_call": ...},
               "turn_2": {...}, ...}}

对齐的三件事，逐条对应 `Heter-Bench/scripts/bench_heap.py`：

1. **prompt 用整段历史**：``turn_i.input_ids`` 本身就是完整历史（服务端直发
   input_ids），**不是**「上一轮 prompt + 上一轮 response + 本轮新增」自己拼的。
   emitter 也发这一份（``_resolve_prompt_ids`` 的返回值恒等于数据集）。
2. **response 保留 token**：emitter 发 ``max_new_tokens = len(response_ids)``；
   离线回放把 ``response_ids`` 一并带进 ``Turn.response``，因为数据集下一轮
   prompt 就是 ``上一轮 prompt + 上一轮 response + 新增内容``，只有写入同一串
   token，response 段才可能在下一轮命中。
3. **tool_call 取上一轮**：emitter 的 payload 是
   ``sess[f"turn_{idx-1}"].get("tool_call") if idx > 1 else None``。

没有 ``response_ids`` 的 trace（合成负载、只给长度的数据）仍回退到每请求唯一
哨兵：哨兵能避免“模型实际生成 ≠ 数据集回答”时的假命中，但会少算 response 复用。
"""

from __future__ import annotations

import json
import os
import random
from array import array
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sim.trace import Session, Turn


def _iter_records(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            # 每行只有一个键：{sid: {...}}；取 value（与 emitter 一致）
            yield list(payload.values())[0]


def _num_turns(session: Dict[str, Any]) -> int:
    declared = session.get("num_turns")
    if declared is not None:
        return int(declared)
    return sum(1 for k in session if k.startswith("turn_"))


def _turns_of(session: Dict[str, Any]) -> tuple:
    """按``turn_1..N``构造 Turn；``t`` 先留 0，由调度层（sim/schedule.py）赋。"""

    turns: List[Turn] = []
    n = _num_turns(session)
    for idx in range(1, n + 1):
        row = session.get(f"turn_{idx}")
        if row is None:
            raise ValueError(
                f"session {session.get('id')!r} 声明 num_turns={n}，但缺 turn_{idx}"
            )
        prompt = row.get("input_ids") or []
        response_ids = row.get("response_ids") or []
        prev = session.get(f"turn_{idx - 1}") if idx > 1 else None
        tool_call = prev.get("tool_call") if prev is not None else None
        turns.append(
            Turn(
                t=0.0,
                prompt=array("q", prompt),
                out=len(response_ids),
                tool_call=bool(tool_call),
                response=array("q", response_ids) if response_ids else None,
            )
        )
    return tuple(turns)


def load_task(
    path: str,
    *,
    task_name: str,
    max_record: int = -1,
    shuffle: bool = True,
    seed: Optional[int] = None,
    min_turns: Optional[int] = None,
) -> List[Session]:
    """读一个 task 的 session 文件，返回挂了 ``task_name`` 的会话列表。

    Args:
        path: ``*_session_data.jsonl``。
        task_name: 与 tasks_config 的 ``name`` 一致（per-task 指标的键）。
        max_record: >0 时截断（emitter 的 ``max_record``）。
        shuffle: 是否打乱；emitter 用 ``rng(seed + task_id)`` 打乱，故传
            ``shuffle=True, seed=seed+task_id`` 即等价。
        seed: 打乱用种子。``None`` 表示不打乱（保持文件序，如 mashqa）。
        min_turns: 只保留轮数 >= 它的会话（对应 ``drop_short_sessions.py`` 的
            结果校验；emitter 在 ``assert_turn_policy`` 里做同样的事）。

    Returns:
        ``List[Session]``，``turns[i].t`` 恒为 0（时间轴由调度层生成）。
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"session 数据不存在：{path}")

    sessions: List[Dict[str, Any]] = list(_iter_records(path))
    if shuffle:
        random.Random(seed).shuffle(sessions)
    if max_record and max_record > 0:
        sessions = sessions[:max_record]
    if min_turns:
        sessions = [s for s in sessions if _num_turns(s) >= int(min_turns)]

    out: List[Session] = []
    for session in sessions:
        sid = session.get("id")
        if not sid:
            raise ValueError(f"{path}: 有会话缺 id 字段")
        out.append(Session(sid, _turns_of(session), task_name=task_name))
    return out


def load_tasks(tasks: Sequence[Dict[str, Any]], *, base_dir: str = "") -> List[Session]:
    """按 tasks_config 的 ``tasks`` 列表批量加载（``data_path`` 可相对 ``base_dir``）。

    读取 / 打乱种子 / 截断 / 轮数校验的语义与
    ``bench_heap.build_hybrid_arrival_schedule`` 的前两步一致；到达时刻不在这里算，
    交给 ``sim/schedule.py`` 的 ``ScheduleSpec``。
    """

    out: List[Session] = []
    for task in tasks:
        data_path = task["data_path"]
        if base_dir and not os.path.isabs(data_path):
            data_path = os.path.join(base_dir, data_path)
        seed = task.get("seed", 42)
        out.extend(
            load_task(
                data_path,
                task_name=task["name"],
                max_record=int(task.get("max_record", -1)),
                shuffle=bool(task.get("shuffle", True)),
                seed=seed + int(task.get("task_id", 0)),
                min_turns=task.get("min_turns"),
            )
        )
    return out


__all__ = ["load_task", "load_tasks"]
