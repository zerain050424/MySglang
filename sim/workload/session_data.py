"""从 Heter-Bench session_data.jsonl 加载会话数据，字段规则与 bench_heap.py 一致。"""

from __future__ import annotations

import json
import os
import random
from array import array
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sim.workload.trace import Session, Turn


def _iter_records(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            yield list(payload.values())[0]


def _num_turns(session: Dict[str, Any]) -> int:
    declared = session.get("num_turns")
    if declared is not None:
        return int(declared)
    return sum(1 for k in session if k.startswith("turn_"))


def _turns_of(session: Dict[str, Any]) -> tuple:
    """提取单条会话的全部轮次；时刻 t 留 0 交由闭环调度器生成。"""
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
        # 与线上 emitter 一致：本轮是否属于 tool_call 后续取决于上一轮的返回状态。
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
    """读取单个 task 的 session 文件。

    Args:
        path: 数据集 jsonl 文件路径。
        task_name: 任务名称（用于按任务汇总指标）。
        max_record: 大于 0 时截断最大读取条数。
        shuffle: 是否按随机种子打乱（与 emitter 一致需传入 seed + task_id）。
        seed: 打乱用随机种子。
        min_turns: 仅保留总轮数不低于该阈值的会话。

    Returns:
        Turn.t 均为 0 的 Session 列表。
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
    """按 tasks_config 列表批量加载全部任务会话。"""
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
