"""各调度驱动共用的分位数与 emit 指标汇总工具。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def pct(values: Sequence[float], p: float) -> float:
    """线性插值计算百分位数（输入无需预先排序）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def build_emit_stats(
    *,
    mode: str,
    wall_s: float,
    inflight_peak: int,
    lifecycle,
    send_lags: Sequence[float],
    arrival_scale: float,
    wait_scale: float,
    service,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构建统一的 summary['emit'] 字段字典，复用延迟分位数与服务参数摘要。"""
    lags = sorted(send_lags)
    emit: Dict[str, Any] = {"mode": mode}
    if extra:
        emit.update(extra)
    emit.update(
        {
            "wall_s": round(wall_s, 3),
            "inflight_peak": int(inflight_peak),
            "inflight_token_peak": int(lifecycle.inflight_token_peak),
            "duplicate_insert_tokens": int(lifecycle.duplicate_insert_tokens),
            "send_lag_p50_ms": round(pct(lags, 50) * 1000, 3) if lags else 0.0,
            "send_lag_p99_ms": round(pct(lags, 99) * 1000, 3) if lags else 0.0,
            "send_lag_max_ms": round(max(lags) * 1000, 3) if lags else 0.0,
            "arrival_scale": arrival_scale,
            "wait_scale": wait_scale,
            "service": {
                "prefill_base_s": service.prefill_base_s,
                "prefill_tokens_per_s": service.prefill_tokens_per_s,
                "decode_tokens_per_s": service.decode_tokens_per_s,
            },
        }
    )
    return emit
