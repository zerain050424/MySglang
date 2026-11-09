"""指标：共享前缀保留度（M1）、后悔率 / 保护浪费率（M4）。

三个指标都在**请求粒度**上累计，可与在线 CSV 的逐请求明细逐条对齐。

M1 保留度
---------
对每个请求 r 与其 task 的分层前缀长度 L_k：

    I_k(r) = 1[cached_tokens_r >= L_k]          全保留率 R_k = mean I_k
    C_k(r) = min(cached_tokens_r, L_k) / L_k    覆盖度（长前缀更敏感）

后悔率（regret）
----------------
定义：**被淘汰的、且在 W 秒内又被请求的 token 数** ÷ 总 prompt token 数。
实现：对每个被删节点记录「根到该节点结束」的 token 序列滚动哈希；
请求到达时对 prompt 做滚动哈希，命中记录且位置 > 已命中长度 → 这些 token
本来能命中，是被淘汰掉的（同一前缀重算）。

保护浪费率（protected waste）
-----------------------------
protected 区被淘汰的 token 中，之后**再也没有被任何请求命中**的比例。
它衡量「保护错配」：protection 在保护没人再用模板。

与后悔率是一对孪生指标：好的策略压低后悔率、不抬高浪费率。
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK = (1 << 64) - 1
_TOKEN_SALT = 0x9E3779B97F4A7C15


def fnv_init() -> int:
    return _FNV_OFFSET


def fnv_update(h: int, token: int) -> int:
    h ^= (int(token) + _TOKEN_SALT) & _MASK
    h = (h * _FNV_PRIME) & _MASK
    return h


def fnv_update_tokens(h: int, tokens: Iterable[int]) -> int:
    for t in tokens:
        h ^= (int(t) + _TOKEN_SALT) & _MASK
        h = (h * _FNV_PRIME) & _MASK
    return h


def prefix_hashes(tokens: Sequence[int], top: Optional[int] = None) -> List[Tuple[int, int]]:
    """返回 [(depth, hash), ...]，depth 从 1 开始，覆盖 tokens[0:depth]。"""
    h = fnv_init()
    out = []
    n = len(tokens) if top is None else min(len(tokens), top)
    for i in range(n):
        h = fnv_update(h, tokens[i])
        out.append((i + 1, h))
    return out


# ---------------------------------------------------------------------------
# M1：保留度
# ---------------------------------------------------------------------------
class RetentionTracker:
    """按 task 的分层前缀长度统计全保留率与覆盖度。"""

    def __init__(self, layers_by_task: Dict[str, Sequence[int]]) -> None:
        self.layers_by_task = {k: sorted(int(x) for x in v) for k, v in layers_by_task.items()}
        self._acc: Dict[str, Dict[int, dict]] = {}
        self._hist: Dict[str, dict] = {}

    def add(self, task: str, cached: int) -> Dict[int, Tuple[bool, float]]:
        layers = self.layers_by_task.get(task) or []
        out: Dict[int, Tuple[bool, float]] = {}
        bucket = self._acc.setdefault(task, {})
        for L in layers:
            full = cached >= L
            cover = min(cached, L) / float(L)
            rec = bucket.setdefault(L, {"n": 0, "full": 0, "cover": 0.0})
            rec["n"] += 1
            rec["full"] += int(full)
            rec["cover"] += cover
            out[L] = (full, cover)
        return out

    def add_history(self, task: str, cached: int, prev_prompt_len: int) -> None:
        """会话内历史保留度：本轮是否还命中了「上一轮的完整 prompt」。"""
        if prev_prompt_len <= 0:
            return
        b = self._hist.setdefault(task, {"n": 0, "full": 0, "cover": 0.0})
        b["n"] += 1
        b["full"] += int(cached >= prev_prompt_len)
        b["cover"] += min(cached, prev_prompt_len) / float(prev_prompt_len)

    def history_summary(self) -> Dict[str, dict]:
        return {
            task: {
                "n": b["n"],
                "full_rate": round(b["full"] / b["n"], 4) if b["n"] else 0.0,
                "cover": round(b["cover"] / b["n"], 4) if b["n"] else 0.0,
            }
            for task, b in self._hist.items()
        }

    def summary(self) -> Dict[str, Dict[str, dict]]:
        res: Dict[str, Dict[str, dict]] = {}
        for task, bucket in self._acc.items():
            res[task] = {
                str(L): {
                    "n": r["n"],
                    "full_rate": round(r["full"] / r["n"], 4) if r["n"] else 0.0,
                    "cover": round(r["cover"] / r["n"], 4) if r["n"] else 0.0,
                }
                for L, r in sorted(bucket.items())
            }
        return res


# ---------------------------------------------------------------------------
# M4：后悔率 / 保护浪费率
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class RegretSummary:
    lost_tokens: int = 0
    prompt_tokens: int = 0
    evicted_tokens: int = 0
    evicted_nodes: int = 0
    protected_evicted_tokens: int = 0
    protected_reused_tokens: int = 0
    protected_cascade_deleted_tokens: int = 0
    cascade_deleted_tokens: int = 0

    @property
    def regret_rate(self) -> float:
        return self.lost_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def protected_waste_rate(self) -> float:
        if not self.protected_evicted_tokens:
            return 0.0
        return 1.0 - self.protected_reused_tokens / self.protected_evicted_tokens

    def as_dict(self) -> dict:
        return {
            "lost_tokens": self.lost_tokens,
            "prompt_tokens": self.prompt_tokens,
            "regret_rate": round(self.regret_rate, 5),
            "evicted_tokens": self.evicted_tokens,
            "evicted_nodes": self.evicted_nodes,
            "protected_evicted_tokens": self.protected_evicted_tokens,
            "protected_waste_rate": round(self.protected_waste_rate, 5),
            "protected_cascade_deleted_tokens": self.protected_cascade_deleted_tokens,
            "cascade_deleted_tokens": self.cascade_deleted_tokens,
        }


class RegretTracker:
    """后悔率 + 保护浪费率（块无关，节点级精确）。"""

    def __init__(self, *, window_s: float = 60.0, prune_every: int = 2048) -> None:
        self.window_s = float(window_s)
        self.prune_every = int(prune_every)
        self._evicted: Dict[int, Tuple[float, int]] = {}
        self._protected: Dict[int, dict] = {}
        self._req_count = 0
        self.sum = RegretSummary()

    # ---- 淘汰侧 ----
    def on_evict(
        self,
        ts: float,
        depth_end: int,
        ph: int,
        tokens: int,
        region: int,
        is_victim: bool,
    ) -> None:
        self.sum.evicted_tokens += tokens
        self.sum.evicted_nodes += 1
        if not is_victim:
            self.sum.cascade_deleted_tokens += tokens
        if depth_end <= 0:
            return
        self._evicted[ph] = (ts, depth_end)
        if region == 3:  # REGION_PROTECTED
            self.sum.protected_evicted_tokens += tokens
            if not is_victim:
                self.sum.protected_cascade_deleted_tokens += tokens
            self._protected[ph] = {"ts": ts, "tokens": tokens, "reused": False}

    # ---- 请求侧 ----
    def on_request(self, ts: float, prompt: Sequence[int], cached_len: int) -> int:
        """返回本请求的丢失命中 token（regret）。"""
        self.sum.prompt_tokens += len(prompt)
        self._req_count += 1
        lost = 0
        if not self._evicted and not self._protected:
            return 0
        h = fnv_init()
        for i, tok in enumerate(prompt):
            h = fnv_update(h, tok)
            d = i + 1
            rec = self._evicted.get(h)
            if rec is not None and d > cached_len:
                t_ev, _depth = rec
                if ts - t_ev <= self.window_s and d - cached_len > lost:
                    lost = d - cached_len
            pr = self._protected.get(h)
            if pr is not None and not pr["reused"] and d > 0:
                pr["reused"] = True
                self.sum.protected_reused_tokens += pr["tokens"]
        self.sum.lost_tokens += lost
        if self._req_count % self.prune_every == 0:
            self._prune(ts)
        return lost

    def _prune(self, ts: float) -> None:
        if not self._evicted:
            return
        cutoff = ts - self.window_s
        stale = [h for h, (t, _d) in self._evicted.items() if t < cutoff]
        for h in stale:
            self._evicted.pop(h, None)

    def summary(self) -> dict:
        return self.sum.as_dict()


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def merge_task_stats(rows: Iterable[dict]) -> Dict[str, dict]:
    """按 task 汇总 prompt/cached token（与在线端 per_task 同口径）。"""
    out: Dict[str, dict] = {}
    for r in rows:
        t = out.setdefault(
            r["task"], {"num_requests": 0, "prompt_tokens": 0, "cached_tokens": 0}
        )
        t["num_requests"] += 1
        t["prompt_tokens"] += r["prompt_tokens"]
        t["cached_tokens"] += r["cached_tokens"]
    for t in out.values():
        t["hit_pct"] = (
            round(100.0 * t["cached_tokens"] / t["prompt_tokens"], 3)
            if t["prompt_tokens"]
            else 0.0
        )
    return out
