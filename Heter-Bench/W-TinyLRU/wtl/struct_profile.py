"""真机 struct 状态埋点（**运行时 monkeypatch，不改框架代码**）。

SLRU 的共享数账本（``SlruEvictPolicy.stats()``）、它实际给出的候选数
（``slru_picks``）与 ``victim_src`` 需要落盘，但框架里没有 dump 点，
而本实验约定「不改主体框架代码」。因此这里在 ``HeterCacheRadixCache.evict``
外面包一层：**只在 evict 返回之后再读状态**，不改变任何淘汰语义。

开关（env）
-----------
    SGLANG_STRUCT_PROFILE=/tmp/struct.jsonl   # 输出文件；空=不安装
    SGLANG_STRUCT_PROFILE_EVERY=30            # 落盘间隔（秒）
    SGLANG_STRUCT_PROFILE_TAG=A3_hybrid_100k  # 实验臂标签，便于分析时分组
    SGLANG_STRUCT_PROFILE_LINES=0             # 1=同时落 profile_summary 文本

每行一条 JSON：

    {"ts":..., "tag":..., "evict_calls":n, "evict_us_p50":..,"evict_us_p99":..,
     "available":.., "evictable_size":.., "protected_size":..,
     "slru":{"slru_picks":..,"share_counted":..,"share_dedup":..},
     "slru_share_min":2,
     "victim_src":{"one_shot":..,"qstruct":..,"global":..},
     "profile":"..."}

分析侧（``wtl.analyze``）用它画两件事：
  1. 受保护段（region 3）被淘汰的 token 序列 → 保护是否真的在起作用；
  2. one_shot / qstruct 淘汰占比（机制是否被行使——§8.5 的 gate）。
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("wtl.struct_profile")

_state: dict = {"installed": False}


def _pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _record(cache, tag: str, samples: list, calls: int) -> dict:
    strategy = getattr(cache, "eviction_strategy", None)
    slru = getattr(strategy, "slru_evict", None)
    snap = slru.stats() if slru is not None else None
    victim_src = dict(getattr(strategy, "_victim_sources", {}) or {})
    rec = {
        "ts": round(time.time(), 3),
        "tag": tag,
        "pid": os.getpid(),
        "evict_calls": calls,
        "evict_us_p50": round(_pct(samples, 50), 1),
        "evict_us_p99": round(_pct(samples, 99), 1),
        "available": int(cache.token_to_kv_pool_allocator.available_size()),
        "evictable_size": int(getattr(cache, "evictable_size_", 0) or 0),
        "protected_size": int(getattr(cache, "protected_size_", 0) or 0),
        "slru": snap,
        "slru_share_min": getattr(slru, "share_min", None),
        "victim_src": victim_src,
    }
    return rec


def install(
    path: Optional[str] = None,
    *,
    interval_s: float = 30.0,
    tag: str = "",
    with_profile_lines: bool = False,
) -> bool:
    """安装埋点；返回是否成功（导入失败则 False，不影响服务启动）。"""
    if _state.get("installed"):
        return True
    path = path or os.environ.get("SGLANG_STRUCT_PROFILE", "")
    if not path:
        return False
    interval_s = float(os.environ.get("SGLANG_STRUCT_PROFILE_EVERY", interval_s))
    tag = tag or os.environ.get("SGLANG_STRUCT_PROFILE_TAG", "")
    with_profile_lines = with_profile_lines or os.environ.get(
        "SGLANG_STRUCT_PROFILE_LINES", "0"
    ) in ("1", "true", "True")

    try:
        from sglang.srt.mem_cache.heter_cache_radix_cache import (
            HeterCacheRadixCache,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("[struct-profile] 无法导入 HeterCacheRadixCache: %s", exc)
        return False

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "a", encoding="utf-8", buffering=1)

    orig_evict = HeterCacheRadixCache.evict
    st = {
        "last": 0.0,
        "calls": 0,
        "samples": [],
        "cache": None,
        "evict_us_total": 0.0,
    }

    def _flush(force: bool = False) -> None:
        cache = st["cache"]
        if cache is None:
            return
        now = time.time()
        if not force and now - st["last"] < interval_s:
            return
        st["last"] = now
        try:
            rec = _record(cache, tag, st["samples"], st["calls"])
            if with_profile_lines:
                strategy = cache.eviction_strategy
                try:
                    rec["profile"] = strategy.profiler.summary(
                        cache._per_queue_token_count(),
                        cache.session_registry.summary(),
                        slru_policy=strategy.slru_evict,
                        mt_policy=strategy.mt_evict,
                    )
                except Exception:
                    rec["profile"] = None
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            st["samples"] = []
        except Exception:
            logger.exception("[struct-profile] flush 失败")

    def evict(self, params):
        t0 = time.perf_counter()
        res = orig_evict(self, params)
        dt_us = (time.perf_counter() - t0) * 1e6
        st["cache"] = self
        st["calls"] += 1
        st["samples"].append(dt_us)
        if len(st["samples"]) > 4096:
            st["samples"] = st["samples"][-4096:]
        _flush()
        return res

    HeterCacheRadixCache.evict = evict

    def _close() -> None:
        try:
            _flush(force=True)
        finally:
            try:
                fh.close()
            except Exception:
                pass

    atexit.register(_close)
    _state.update(installed=True, close=_close, path=str(out))
    logger.info("[struct-profile] 已安装 -> %s (interval=%.0fs tag=%s)",
                out, interval_s, tag)
    return True


def install_from_env() -> bool:
    return install()
