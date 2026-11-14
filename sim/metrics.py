"""指标：逐请求记录 + 汇总，全部取自生产对象（不另算一套口径）。

两个来源：

* **请求侧**：``match_prefix`` 返回的命中长度（= 生产 ``meta_info.cached_tokens``
  的口径），以及驱动层算出的 shortfall / 淘汰量 / 水位；
* **策略侧**：``eviction_strategy.profiler`` 的累计量——按队列的 hit / evicted 与
  victim 来源（``one_shot`` / ``weight:<queue>`` / ``weight:fallback``）。
  ``CacheTelemetry.on_hit`` 会按 rid 去重，所以记录里的 rid 必须唯一。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


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

    @property
    def hit_rate(self) -> float:
        return self.hit / self.prompt_len if self.prompt_len else 0.0

    def line(self) -> str:
        """指纹用的紧凑串（浮点定点，避免末位抖动）。"""
        return (
            f"{self.t:.3f}|{self.session_id}|{self.rid}|{self.prompt_len}|"
            f"{self.out_len}|{self.hit}|{self.evicted}|{self.used}|{self.session_type}"
        )


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


@dataclass
class SimResult:
    """一次 run 的全部产出。"""

    records: List[RequestRecord] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

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
            f"evicted={s['evicted_tokens']}  sessions={s['sessions']}",
        ]
        for name, rec in sorted(s["per_type"].items()):
            lines.append(
                f"  {name:8s} n={rec['requests']:3d} "
                f"hit_rate={rec['hit_rate']:.1%} "
                f"(hit {rec['hit_tokens']}/{rec['prompt_tokens']})"
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
    for record in records:
        bucket = per_type.setdefault(
            record.session_type,
            {"requests": 0, "prompt_tokens": 0, "hit_tokens": 0, "hit_rate": 0.0},
        )
        bucket["requests"] += 1
        bucket["prompt_tokens"] += record.prompt_len
        bucket["hit_tokens"] += record.hit
    for bucket in per_type.values():
        bucket["hit_rate"] = _ratio(bucket["hit_tokens"], bucket["prompt_tokens"])

    prompt_tokens = sum(r.prompt_len for r in records)
    hit_tokens = sum(r.hit for r in records)
    return {
        "requests": len(records),
        "prompt_tokens": prompt_tokens,
        "hit_tokens": hit_tokens,
        "hit_rate": _ratio(hit_tokens, prompt_tokens),
        "per_type": per_type,
        "capacity": capacity,
        "final_used": final_used,
        "evicted_tokens": sum(r.evicted for r in records),
        "sessions": sessions,
        # 以下三项直接读生产对象，口径不复制
        "victim_sources": dict(getattr(profiler, "_victim_sources", {})),
        "queue_hit": dict(getattr(profiler, "_total_hit_tokens", {})),
        "queue_evicted": dict(getattr(profiler, "_total_evicted_tokens", {})),
        "sched_occ": dict(occupancy.snapshot()) if occupancy is not None else {},
        "evicted_one_shot": getattr(profiler, "_total_evicted_one_shot", 0.0),
    }


__all__ = ["RequestRecord", "SimResult", "summarize"]
