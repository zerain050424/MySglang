"""到达流规范与闭环多轮调度计划（对齐 bench_heap.py）。"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from sim.workload.trace import Session


def stable_seed_from_str(s: str, base_seed: int = 0) -> int:
    """将 session_id 映射为跨进程稳定的 32-bit 种子（对齐 bench_heap.py）。"""
    h = hashlib.sha256(s.encode()).hexdigest()
    return (int(h, 16) + base_seed) % (2**32 - 1)


def check_segments(segments: Sequence[Tuple[float, float]]) -> None:
    """校验分段到达强度表 ((until_s, lambda_rate), ...) 的合法性。"""
    if not segments:
        raise ValueError("arrival.mode='segments' 需要非空的 arrival.segments")
    prev = 0.0
    for idx, (until, rate) in enumerate(segments):
        if not (until > prev):
            raise ValueError(
                f"segments[{idx}].until_s={until} 必须严格大于前一段（{prev}）"
            )
        if rate < 0:
            raise ValueError(f"segments[{idx}].lambda_rate={rate} 不能为负")
        prev = until
    if all(rate == 0 for _until, rate in segments):
        raise ValueError("segments 的速率全为 0：不会有任何到达")


def _segment_offsets(spec: "TaskSpec", n: int) -> List[float]:
    """利用分段逆变换生成非齐次泊松到达偏移；末段静默时截断超出窗口的会话。"""
    check_segments(spec.segments)
    rng = random.Random(spec.rng_seed)
    segments = spec.segments
    out: List[float] = []
    t = 0.0
    for until, rate in segments:
        if rate <= 0.0:
            t = until
            continue
        while len(out) < n:
            nxt = t + rng.expovariate(rate)
            # 指数分布无记忆性允许直接丢弃跨出段尾的采样并从边界重抽。
            if nxt >= until:
                break
            out.append(nxt)
            t = nxt
        t = until
        if len(out) >= n:
            return out

    last_rate = segments[-1][1]
    # 末段若为静默窗，窗口外不再放行剩余会话，否则沿用末段速率填满。
    if last_rate <= 0.0:
        return out
    while len(out) < n:
        t += rng.expovariate(last_rate)
        out.append(t)
    return out


@dataclass(frozen=True)
class TaskSpec:
    """单个任务的到达与轮间等待配置。"""

    name: str
    lambda_rate: float = 1.0
    mu: float = 0.0
    sigma: float = 0.0
    seed: int = 42
    task_id: int = 0
    arrival_mode: str = "poisson"  # poisson | fixed | segments
    arrival_offset_s: float = 0.0
    arrival_step_s: float = 0.05
    segments: Tuple[Tuple[float, float], ...] = ()

    @classmethod
    def from_config(cls, task: Mapping) -> "TaskSpec":
        arrival = task.get("arrival") or {}
        mode = str(arrival.get("mode", "poisson"))
        segments: Tuple[Tuple[float, float], ...] = ()
        if mode == "segments":
            raw = arrival.get("segments") or []
            segments = tuple(
                (float(seg["until_s"]), float(seg.get("lambda_rate", 0.0)))
                for seg in raw
            )
            check_segments(segments)
        return cls(
            name=task["name"],
            lambda_rate=float(task.get("lambda_rate", 1)),
            mu=float(task.get("mu", 0.0)),
            sigma=float(task.get("sigma", 0.0)),
            seed=int(task.get("seed", 42)),
            task_id=int(task.get("task_id", 0)),
            arrival_mode=mode,
            arrival_offset_s=float(arrival.get("offset_s", 0.0)),
            arrival_step_s=float(arrival.get("step_s", 0.05)),
            segments=segments,
        )

    @property
    def rng_seed(self) -> int:
        return self.seed + self.task_id * 1000


def first_turn_offsets(spec: TaskSpec, n: int) -> List[float]:
    """按 TaskSpec 生成前 n 个会话的首轮到达时刻偏移。"""
    if spec.arrival_mode == "fixed":
        return [spec.arrival_offset_s + spec.arrival_step_s * i for i in range(n)]
    if spec.arrival_mode == "segments":
        return _segment_offsets(spec, n)
    if spec.arrival_mode != "poisson":
        raise ValueError(f"未知 arrival 模式：{spec.arrival_mode!r}")
    rng = random.Random(spec.rng_seed)
    out, acc = [], 0.0
    for _ in range(n):
        acc += rng.expovariate(spec.lambda_rate)
        out.append(acc)
    return out


class ArrivalPlan:
    """管理闭环多轮到达事件堆与各会话独立的轮间等待 RNG。"""

    def __init__(
        self, sessions: Sequence[Session], specs: Sequence[TaskSpec]
    ) -> None:
        self.sessions: List[Session] = list(sessions)
        self.specs: Dict[str, TaskSpec] = {spec.name: spec for spec in specs}
        self.by_sid: Dict[str, Session] = {}
        for session in self.sessions:
            if session.session_id in self.by_sid:
                raise ValueError(f"session_id 重复：{session.session_id}")
            if session.task_name not in self.specs:
                raise ValueError(
                    f"会话 {session.session_id} 的 task_name={session.task_name!r} "
                    f"没有对应 TaskSpec（有：{sorted(self.specs)}）"
                )
            self.by_sid[session.session_id] = session
        self._session_rngs: Dict[str, random.Random] = {}
        self.seq = itertools.count()

    def reset_run_state(self) -> None:
        """重置每次运行共用的入堆序号与每会话等待 RNG，保证重复运行可复现。"""
        self._session_rngs.clear()
        self.seq = itertools.count()

    def init_heap(self, arrival_scale: float = 1.0) -> List[Tuple[float, int, str, int]]:
        """生成所有任务的首轮到达事件堆，每项为 (due_ts, seq, sid, turn_idx_1based)。"""
        self.reset_run_state()
        heap: List[Tuple[float, int, str, int]] = []
        for spec in self.specs.values():
            group = [s for s in self.sessions if s.task_name == spec.name]
            for session, offset in zip(group, first_turn_offsets(spec, len(group))):
                heapq.heappush(
                    heap,
                    (offset * arrival_scale, next(self.seq), session.session_id, 1),
                )
        return heap

    def inter_turn_wait(
        self, spec: TaskSpec, sid: str, wait_scale: float
    ) -> float:
        """按每会话独立对数正态流采样轮间思考等待时长。"""
        if spec.mu == 0.0 and spec.sigma == 0.0:
            return 0.0
        rng = self._session_rngs.get(sid)
        if rng is None:
            rng = random.Random(
                stable_seed_from_str(sid, base_seed=spec.rng_seed)
            )
            self._session_rngs[sid] = rng
        return rng.lognormvariate(spec.mu, spec.sigma) * wait_scale

    def push_next_turn(
        self,
        heap: List[Tuple[float, int, str, int]],
        session_id: str,
        idx: int,
        finish: float,
        wait_scale: float,
    ) -> None:
        """若当前会话仍有后续轮次，按 finish + wait 将下一轮压入事件堆。"""
        session = self.by_sid[session_id]
        if idx >= session.n_turns:
            return
        spec = self.specs[session.task_name]
        wait = self.inter_turn_wait(spec, session_id, wait_scale)
        heapq.heappush(
            heap, (finish + wait, next(self.seq), session_id, idx + 1)
        )
