"""指标：逐请求记录 + 汇总，全部取自生产对象（不另算一套口径）。

两个来源：

* **请求侧**：``match_prefix`` 返回的命中长度（= 生产 ``meta_info.cached_tokens``
  的口径），以及驱动层算出的 shortfall / 淘汰量 / 水位；
* **策略侧**：``eviction_strategy.profiler`` 的累计量——按队列的 hit / evicted 与
  victim 来源（``one_shot`` / ``value:<queue>`` / ``value:fallback``）。
  ``CacheTelemetry.on_hit`` 会按 rid 去重，所以记录里的 rid 必须唯一。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RequestRecord:
    """一轮请求的落盘视图。"""

    t: float
    session_id: str
    rid: str
    prompt_len: int
    out_len: int
    hit: int
    need: int
    evicted: int
    used: int
    session_type: str
    task_name: str = ""
    skipped: bool = False
    # 轮次下标与 response 段用的唯一哨兵：OPT 的 use map 要用它们把「派发时刻」
    # 与 response 页的 token 还原到树上（见 ``SimResult.freeze_sessions`` /
    # ``response_tokens``）。``-1`` / ``0`` 表示未知（旧 jsonl 或非逐轮驱动）。
    # 不参与 ``line()``（指纹）与任何口径计算。
    turn: int = -1
    response_token: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hit / self.prompt_len if self.prompt_len else 0.0

    def line(self) -> str:
        """指纹用的紧凑串（浮点定点，避免末位抖动）。"""
        return (
            f"{self.t:.3f}|{self.session_id}|{self.rid}|{self.prompt_len}|"
            f"{self.out_len}|{self.hit}|{self.evicted}|{self.used}|"
            f"{self.session_type}|{int(self.skipped)}"
        )


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


@dataclass
class SimResult:
    """一次 run 的全部产出。"""

    records: List[RequestRecord] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    # 本次 run 每个 ``(session_id, 轮次)`` 的**派发时刻**（虚拟秒）。
    # 闭环调度只在运行中决定时刻、且不写回 ``Turn``（见 ``sim/schedule.py``），
    # 所以 OPT 的 use map 必须从这里取时间轴。
    dispatched: Dict[Tuple[str, int], float] = field(default_factory=dict)

    # ---- 常用聚合 ----
    @property
    def requests(self) -> int:
        return len(self.records)

    @property
    def hit_rate(self) -> float:
        return _ratio(
            sum(r.hit for r in self.records), sum(r.prompt_len for r in self.records)
        )

    def fingerprint(self) -> str:
        """逐请求指纹：同 seed 两次 run 必须相同（M4 的复现基线）。"""
        digest = hashlib.sha1()
        for record in self.records:
            digest.update(record.line().encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()[:16]

    # ---- 冻结时间轴（OPT 的 use map 需要决策时刻） ----
    def freeze_sessions(self, sessions: List) -> List:
        """把本次 run 的派发时刻回填进 ``Turn.t``，返回新的会话列表。

        为什么需要：数据集加载出来的 ``Turn.t`` 是 0，闭环调度
        （``run_scheduled`` / ``run_engine``）的时刻由发射器在运行中算出、
        **不写回** —— 直接 ``build_use_map(sessions)`` 会得到全零时间轴，
        ``OPTStrategy`` 于是把每个块都判成「不再使用」，静默退化成任意序。

        为什么合法：同一份 ``tasks.yaml`` / seed / arrival-scale / wait-scale 下，
        到达由 ``stable_seed`` 决定、服务时长只依赖 token 数，**与淘汰策略无关**，
        所以这一次 run 采到的派发时刻就是重放时会发生的那条时间轴。

        Args:
            sessions: 本次 run 用的会话（顺序、数量必须与 run 一致）。

        Returns:
            ``Turn.t`` 已替换为新时刻的会话列表。

        Raises:
            ValueError: 有轮次没有派发时刻（run 不是逐轮驱动的，或传错了 trace）。
        """
        from sim.trace import Session, Turn

        missing: List[Tuple[str, int]] = []
        frozen = []
        for session in sessions:
            turns = []
            for idx, turn in enumerate(session.turns):
                t = self.dispatched.get((session.session_id, idx))
                if t is None:
                    missing.append((session.session_id, idx))
                    turns.append(turn)
                else:
                    turns.append(
                        Turn(
                            t=float(t),
                            prompt=turn.prompt,
                            out=turn.out,
                            tool_call=turn.tool_call,
                            # 真实 response 必须带过去：树里要写它，下一轮 prompt
                            # 也靠它命中（丢掉就等于退回 fallback 哨兵）
                            response=getattr(turn, "response", None),
                        )
                    )
            frozen.append(
                Session(session.session_id, tuple(turns), session.task_name)
            )
        if missing:
            raise ValueError(
                f"{len(missing)} 轮没有派发时刻（例：{missing[:3]}）；"
                "只有逐轮驱动的 run 会记录 dispatched，且不能用旧 jsonl 反推轮次"
            )
        return frozen

    def response_tokens(self) -> Dict[Tuple[str, int], int]:
        """``(session_id, 轮次) -> response 段哨兵``。

        use map 要覆盖树里真正的 token：``_adopt(prompt + output)`` 里那 ``out``
        个 token 是每请求唯一的哨兵，不写进表就会让这批页永远查不到（未命中率
        被虚高、且「粒度错配」告警失去判别力）。
        """
        return {
            (r.session_id, r.turn): r.response_token
            for r in self.records
            if r.turn >= 0 and r.response_token
        }

    # ---- 输出 ----
    def to_jsonl(self, path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fp:
            for record in self.records:
                fp.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return target

    def render(self, limit: int = 12) -> str:
        """人读摘要：整体命中率 + 按会话类型 + 淘汰来源。"""
        s = self.summary
        lines = [
            f"  requests={s['requests']}  "
            f"prompt_tokens={s['prompt_tokens']}  hit_tokens={s['hit_tokens']}  "
            f"hit_rate={s['hit_rate']:.1%}",
            f"  capacity={s['capacity']}  used={s['final_used']}  "
            f"evicted={s['evicted_tokens']}  skipped={s['skipped_requests']}  "
            f"sessions={s['sessions']}",
        ]
        for name, rec in sorted(s["per_type"].items()):
            lines.append(
                f"  {name:8s} n={rec['requests']:3d} "
                f"hit_rate={rec['hit_rate']:.1%} "
                f"(hit {rec['hit_tokens']}/{rec['prompt_tokens']})"
            )
        if s.get("per_task"):
            lines.append("  per task:")
            for name, rec in sorted(s["per_task"].items()):
                lines.append(
                    f"    {name:20s} n={rec['requests']:4d} "
                    f"hit_rate={rec['hit_rate']:.1%} "
                    f"(hit {rec['hit_tokens']:.0f}/{rec['prompt_tokens']:.0f})"
                )
        if s["victim_sources"]:
            src = ", ".join(f"{k}={v:.0f}" for k, v in sorted(s["victim_sources"].items()))
            lines.append(f"  victim_sources: {src}")
        if s["queue_hit"]:
            hit = ", ".join(f"{k}={v:.0f}" for k, v in sorted(s["queue_hit"].items()))
            lines.append(f"  hit/queue     : {hit}")
        if s["sched_occ"]:
            occ = ", ".join(f"{k}={v:.0f}" for k, v in sorted(s["sched_occ"].items()))
            lines.append(f"  sched_occ     : {occ}")
        emit = s.get("emit")
        if emit:
            lines.append(
                f"  emit          : {emit.get('mode')} wall={emit.get('wall_s')}s "
                f"inflight_peak={emit.get('inflight_peak')} "
                f"dup_insert={emit.get('duplicate_insert_tokens')} "
                f"lag_p50={emit.get('send_lag_p50_ms')}ms "
                f"lag_p99={emit.get('send_lag_p99_ms')}ms"
            )
        if limit and self.records:
            lines.append(f"  first {min(limit, len(self.records))} requests:")
            for record in self.records[:limit]:
                lines.append(
                    f"    t={record.t:7.2f} {record.session_id:12s} "
                    f"prompt={record.prompt_len:5d} hit={record.hit:5d} "
                    f"({record.hit_rate:5.1%}) evicted={record.evicted:5d} "
                    f"used={record.used:6d} type={record.session_type}"
                )
        return "\n".join(lines)


def summarize(
    records: List[RequestRecord],
    *,
    capacity: int,
    final_used: int,
    sessions: int,
    profiler,
    occupancy,
) -> Dict:
    """把逐请求记录与策略累计量折成一份 dict。"""

    per_type: Dict[str, Dict[str, float]] = {}
    per_task: Dict[str, Dict[str, float]] = {}
    for record in records:
        for table, key in ((per_type, record.session_type), (per_task, record.task_name)):
            if not key:
                continue
            bucket = table.setdefault(
                key,
                {"requests": 0, "prompt_tokens": 0, "hit_tokens": 0, "hit_rate": 0.0},
            )
            bucket["requests"] += 1
            bucket["prompt_tokens"] += record.prompt_len
            bucket["hit_tokens"] += record.hit
    for table in (per_type, per_task):
        for bucket in table.values():
            bucket["hit_rate"] = _ratio(bucket["hit_tokens"], bucket["prompt_tokens"])

    prompt_tokens = sum(r.prompt_len for r in records)
    hit_tokens = sum(r.hit for r in records)
    return {
        "requests": len(records),
        "prompt_tokens": prompt_tokens,
        "hit_tokens": hit_tokens,
        "hit_rate": _ratio(hit_tokens, prompt_tokens),
        "per_type": per_type,
        "per_task": per_task,
        "capacity": capacity,
        "final_used": final_used,
        "evicted_tokens": sum(r.evicted for r in records),
        "skipped_requests": sum(1 for r in records if r.skipped),
        "sessions": sessions,
        # 以下三项直接读生产对象，口径不复制
        "victim_sources": dict(getattr(profiler, "_victim_sources", {})),
        "queue_hit": dict(getattr(profiler, "_total_hit_tokens", {})),
        "queue_evicted": dict(getattr(profiler, "_total_evicted_tokens", {})),
        "sched_occ": dict(occupancy.snapshot()) if occupancy is not None else {},
        "evicted_one_shot": getattr(profiler, "_total_evicted_one_shot", 0.0),
    }


__all__ = ["RequestRecord", "SimResult", "summarize"]
