"""逐请求记录、汇总指标、复现指纹与时间轴冻结（不反向依赖 core 或 drivers）。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RequestRecord:
    """单轮请求完成后的观测记录。"""

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
    turn: int = -1
    response_token: int = 0
    total_cost: float = 0.0
    saved_cost: float = 0.0
    # 到达调度时刻，不参与 line() 指纹计算，供排队延迟统计直接读取。
    arrival_t: Optional[float] = None

    @property
    def hit_rate(self) -> float:
        return self.hit / self.prompt_len if self.prompt_len else 0.0

    def line(self) -> str:
        """返回用于跨运行复现性对比的定长文本行。"""
        return (
            f"{self.t:.3f}|{self.session_id}|{self.rid}|{self.prompt_len}|"
            f"{self.out_len}|{self.hit}|{self.evicted}|{self.used}|"
            f"{self.session_type}|{int(self.skipped)}|"
            f"{self.total_cost:.1f}|{self.saved_cost:.1f}"
        )


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


@dataclass
class SimResult:
    """单次仿真运行的记录、摘要及真实派发时刻表。"""

    records: List[RequestRecord] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    dispatched: Dict[Tuple[str, int], float] = field(default_factory=dict)

    @property
    def served(self) -> List[RequestRecord]:
        return [r for r in self.records if not r.skipped]

    @property
    def requests(self) -> int:
        return len(self.served)

    @property
    def total_requests(self) -> int:
        return len(self.records)

    @property
    def skipped_requests(self) -> int:
        return len(self.records) - len(self.served)

    @property
    def hit_rate(self) -> float:
        served = self.served
        return _ratio(
            sum(r.hit for r in served), sum(r.prompt_len for r in served)
        )

    @property
    def compute_saving_ratio(self) -> float:
        served = self.served
        return _ratio(
            sum(r.saved_cost for r in served), sum(r.total_cost for r in served)
        )

    def fingerprint(self) -> str:
        """对全部请求记录生成稳定 SHA1 前缀，用于同配置复现性校验。"""
        digest = hashlib.sha1()
        for record in self.records:
            digest.update(record.line().encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()[:16]

    def freeze_sessions(self, sessions: List) -> List:
        """将运行期记录的真实派发时刻回填至 Turn.t，供 OPT 生成非零未来使用表。"""
        from sim.workload.trace import Session, Turn

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
                            # 保留真实 response，否则下一轮 prompt 无法在 OPT 树内命中上轮回复段。
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
        """返回各请求分配的独占 response 哨兵映射，供 OPT use map 覆盖生成段页。"""
        return {
            (r.session_id, r.turn): r.response_token
            for r in self.records
            if r.turn >= 0 and r.response_token
        }

    def to_jsonl(self, path) -> Path:
        """将逐请求明细按行写入 JSONL 文件。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fp:
            for record in self.records:
                fp.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return target

    def render(self, limit: int = 12) -> str:
        """生成面向人工检查的汇总摘要文本。"""
        s = self.summary
        lines = [
            f"  requests={s['requests']}  skipped={s['skipped_requests']}  "
            f"prompt_tokens={s['prompt_tokens']}  hit_tokens={s['hit_tokens']}  "
            f"hit_rate={s['hit_rate']:.1%}",
            f"  compute_saving={s['compute_saving_ratio']:.1%}  "
            f"(saved {s['saved_cost']:.0f}/{s['total_cost']:.0f}, "
            f"cost={s['cost_model']})",
            f"  capacity={s['capacity']}  used={s['final_used']}  "
            f"evicted={s['evicted_tokens']}  "
            f"skipped_prompt_tokens={s['skipped_prompt_tokens']}  "
            f"self_evict={s.get('self_evictions', 0)}  "
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
    cost_model: str = "uniform",
) -> Dict:
    """汇总请求侧命中与策略内部 profiler/occupancy 读数（跳过请求不计入分母）。"""
    served = [record for record in records if not record.skipped]
    skipped = [record for record in records if record.skipped]

    per_type: Dict[str, Dict[str, float]] = {}
    per_task: Dict[str, Dict[str, float]] = {}
    for record in served:
        for table, key in (
            (per_type, record.session_type),
            (per_task, record.task_name),
        ):
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

    prompt_tokens = sum(r.prompt_len for r in served)
    hit_tokens = sum(r.hit for r in served)
    total_cost = sum(r.total_cost for r in served)
    saved_cost = sum(r.saved_cost for r in served)
    return {
        "requests": len(served),
        "total_requests": len(records),
        "prompt_tokens": prompt_tokens,
        "hit_tokens": hit_tokens,
        "hit_rate": _ratio(hit_tokens, prompt_tokens),
        "total_cost": total_cost,
        "saved_cost": saved_cost,
        "compute_saving_ratio": _ratio(saved_cost, total_cost),
        "cost_model": cost_model,
        "per_type": per_type,
        "per_task": per_task,
        "capacity": capacity,
        "final_used": final_used,
        "evicted_tokens": sum(r.evicted for r in served),
        "skipped_requests": len(skipped),
        "skipped_prompt_tokens": sum(r.prompt_len for r in skipped),
        "sessions": sessions,
        "victim_sources": dict(getattr(profiler, "_victim_sources", {})),
        "queue_hit": dict(getattr(profiler, "_total_hit_tokens", {})),
        "queue_evicted": dict(getattr(profiler, "_total_evicted_tokens", {})),
        "sched_occ": dict(occupancy.snapshot()) if occupancy is not None else {},
        "evicted_one_shot": getattr(profiler, "_total_evicted_one_shot", 0.0),
    }


__all__ = ["RequestRecord", "SimResult", "summarize"]
