"""会话级评测的数据口径。

评估单元从「节点/请求」换成「会话」：每条会话按真实时间轴重放出它的每一轮请求，
最后拿生产代码 `SessionRegistry.session_type` 的判定与该会话的语义真值比对。

真值（GT）来自会话语义（session_id 前缀 -> 负载家族），与
`evaluate_node_labels.py` 的节点级口径保持一致：

    Code-...              -> struct   (Codeparrot，单轮，跨会话共享同一代码前缀)
    apps_...              -> random   (AgentBank apps，单轮，彼此不共享内容)
    ShareGPT...           -> chat     (ShareGPT，人类节奏多轮)
    alfred_...            -> agentic  (AgentBank alfred，带 tool_call 的多轮)

预测（pred）由 `uni_utils.SessionType` 映射而来，映射见 `PRED_LABEL`：
`trans` 表示「未判定」——对单轮且无跨会话结构共享的会话，这就是 random 的正确桶
（`task_type_of_tag(QTRANS) == RANDOM_SINGLE_TURN`）。
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# 真值前缀表（与 Uni-Bench/evaluate_node_labels.py 一致）
GT_PREFIXES = (
    ("apps_", "random"),
    ("alfred_", "agentic"),
    ("ShareGPT", "chat"),
    ("Code-", "struct"),
)

GT_LABELS = ("struct", "random", "chat", "agentic")

# SessionType -> 会话级评测用的类别
PRED_LABEL = {
    "trans": "random",
    "struct": "struct",
    "chat": "chat",
    "agentic": "agentic",
}

# 各负载在 bench 里默认的轮间隔口径（见 phased_load_prediction.yaml）
DEFAULT_GAP_S = {
    "chat": 20.0,  # ShareGPT: gap_mode=fixed, gap_s=20
    "agentic": 2.0,  # alfred: gap_mode=fixed, gap_s=2
    "struct": 0.0,
    "random": 0.0,
}


def gt_of_session(session_id: Optional[str]) -> Optional[str]:
    if not session_id:
        return None
    for prefix, label in GT_PREFIXES:
        if session_id.startswith(prefix):
            return label
    return None


@dataclass
class Turn:
    """一轮请求在分类器视角下可见的证据。"""

    index: int  # 1-based
    start: float  # 请求开始时刻（秒，重放时间轴）
    prompt_len: int
    prior_tool_call: bool = False  # 上一轮已完成的 tool_call（生产口径：不泄漏当前轮）
    same_turn_tool_call: bool = False  # 当前轮自己的 tool_call（仅消融用）
    tokens: Optional[List[int]] = None  # 仅用于跨会话前缀共享的结构证据


@dataclass
class SessionRecord:
    session_id: str
    gt: str
    turns: List[Turn] = field(default_factory=list)
    source: str = ""

    @property
    def n_turns(self) -> int:
        return len(self.turns)


# ---------------------------------------------------------------------------
# 任务数据（TaskData/**.jsonl，一行一个 {"id": {"num_turns", "turn_k": {...}}}）
# ---------------------------------------------------------------------------


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _fast_key(line: str) -> Optional[str]:
    """不解析 JSON 取出该行唯一的 session id（大文件里只解析采样到的行）。"""
    i = line.find('"')
    if i < 0:
        return None
    j = line.find('"', i + 1)
    return line[i + 1 : j] if j > i else None


def _iter_lines(path: Path, max_sessions: Optional[int], seed: int):
    total = _count_lines(path)
    if not max_sessions or total <= max_sessions:
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield line
        return
    wanted = set(random.Random(seed).sample(range(total), max_sessions))
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in wanted:
                yield line


def _turns_of(sess: dict, keep_all_tokens: bool) -> List[Turn]:
    turns: List[Turn] = []
    prev_call = False
    for t in range(1, int(sess["num_turns"]) + 1):
        turn = sess.get(f"turn_{t}")
        if turn is None:
            break
        ids = turn.get("input_ids") or []
        call = bool(turn.get("tool_call"))
        tokens = ids if (t == 1 or keep_all_tokens) else None
        turns.append(
            Turn(
                index=t,
                start=0.0,
                prompt_len=len(ids),
                prior_tool_call=prev_call,
                same_turn_tool_call=call,
                tokens=tokens,
            )
        )
        prev_call = call
    return turns


def load_task_sessions(
    paths: Sequence[Path],
    max_sessions: Optional[int] = None,
    seed: int = 42,
    keep_all_tokens: bool = False,
    only_ids: Optional[Iterable[str]] = None,
) -> Dict[str, SessionRecord]:
    """从任务数据构造会话记录（时序留空，由 `assign_gaps` 或真实 trace 填）。

    `only_ids` 给定时按 id 精确加载（只解析命中的行），用于给真实 trace 回填
    prompt/token/tool_call 元数据。
    """
    wanted_ids = set(only_ids) if only_ids is not None else None
    out: Dict[str, SessionRecord] = {}
    for path in paths:
        path = Path(path)
        cap = None if wanted_ids else max_sessions
        with path.open(encoding="utf-8") as fh:
            for line in _iter_lines(path, cap, seed):
                if wanted_ids is not None and _fast_key(line) not in wanted_ids:
                    continue
                for sid, sess in json.loads(line).items():
                    gt = gt_of_session(sid)
                    if gt is None:
                        continue
                    rec = out.get(sid) or SessionRecord(session_id=sid, gt=gt, source=str(path))
                    rec.turns = _turns_of(sess, keep_all_tokens)
                    out[sid] = rec
    return out


def default_task_paths(sg_root: Path) -> List[Path]:
    """TaskData 下 Qwen2.5-7B-Instruct 的全部 *_session_data.jsonl。"""
    return sorted(sg_root.glob("TaskData/*/Qwen2.5-7B-Instruct/*_session_data.jsonl"))


# ---------------------------------------------------------------------------
# 真实 trace（bench_unicache.py 输出的 CSV）
# ---------------------------------------------------------------------------


def iter_trace_rows(csv_path: Path):
    with Path(csv_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield row


def trace_session_ids(csv_path: Path) -> List[str]:
    return sorted({row["session_id"] for row in iter_trace_rows(csv_path)})


def load_trace_sessions(
    csv_path: Path,
    task_index: Optional[Dict[str, SessionRecord]] = None,
    turns_limit: Optional[int] = None,
) -> Dict[str, SessionRecord]:
    """按 bench CSV 的真实时间轴重建会话。

    - 时间用 `ts`（请求开始时刻）的差分，正是分类器 `now - last_request_start` 的口径；
      不用 `gap_since_prev_end_s`（那是 end->start，语义不同）。
    - `prior_tool_call` 用任务数据里上一轮的 tool_call 回填：生产路径传的就是上一轮请求
      携带的那份元数据，不涉及当前轮结果。
    """
    task_index = task_index or {}
    grouped: Dict[str, List[dict]] = {}
    for row in iter_trace_rows(csv_path):
        grouped.setdefault(row["session_id"], []).append(row)

    out: Dict[str, SessionRecord] = {}
    for sid, rows in grouped.items():
        gt = gt_of_session(sid)
        if gt is None:
            continue
        rows.sort(key=lambda r: int(r["turn"]))
        if turns_limit:
            rows = rows[:turns_limit]
        ref = task_index.get(sid)
        turns: List[Turn] = []
        for i, row in enumerate(rows):
            turn = Turn(
                index=i + 1, start=float(row["ts"]), prompt_len=int(row["prompt_tokens"])
            )
            if ref is not None and i < len(ref.turns):
                src = ref.turns[i]
                turn.tokens = src.tokens
                turn.prior_tool_call = src.prior_tool_call
                turn.same_turn_tool_call = src.same_turn_tool_call
            turns.append(turn)
        out[sid] = SessionRecord(session_id=sid, gt=gt, turns=turns, source=str(csv_path))
    return out


# ---------------------------------------------------------------------------
# 时序：任务数据模式下的轮间隔注入
# ---------------------------------------------------------------------------


def assign_gaps(
    sessions: Dict[str, SessionRecord],
    gap_model: str = "per_workload",
    seed: int = 7,
) -> None:
    """给任务数据模式的会话补上轮间隔（start->start，秒）。

    gap_model:
        per_workload   按会话真值取 `DEFAULT_GAP_S`（chat=20s, agentic=2s, 单轮=0）
        fixed:<s>      所有轮间隔固定为 s
        lognormal:<mu>:<sigma>  按 bench 的 lognormal 取样（µ/σ 是 ln(秒) 的参数）
    """
    rng = random.Random(seed)
    for rec in sessions.values():
        if gap_model == "per_workload":
            gap = DEFAULT_GAP_S.get(rec.gt, 0.0)
        elif gap_model.startswith("fixed:"):
            gap = float(gap_model.split(":", 1)[1])
        elif gap_model.startswith("lognormal:"):
            _, mu, sigma = gap_model.split(":")
            mu, sigma = float(mu), float(sigma)
            rec.turns[0].start = 0.0
            for i in range(1, len(rec.turns)):
                rec.turns[i].start = rec.turns[i - 1].start + rng.lognormvariate(mu, sigma)
            continue
        else:
            raise ValueError(f"unknown gap model: {gap_model}")
        for i, turn in enumerate(rec.turns):
            turn.start = i * gap
