"""淘汰后悔率（regret）账本：把「这次逐出亏了多少」量成可对账的两个数。

## 口径（三条，缺一条都会得出反向结论）

1. **一次淘汰决策 = 一个 stamp。** ``_pick_victim`` 选出的牺牲者连同
   ``_evict_one`` 级联清理掉的私有祖先链算同一次决策，只落一个 stamp，
   ``span`` = 本次腾出的全部 token。分开记会把一条链记成 N 次误判。
2. **键取链上最浅那一页，归因只给最浅的。** 请求的前缀匹配自根向下连续：
   第 k 页 miss 之后的内容反正都要重算，更深的被逐出页对这个请求没有额外
   损失。所以 stamp 的键 = 被腾出链**最浅节点**的首页内容键，更深的页即使
   被逐出也不会成为任何请求的断点（共享更深前缀必然共享更浅前缀）。
3. **损失按 stamp 的 span 计，不按请求的重算量计。**
   ``regret = span × 1{该内容在 H 内被再次请求}``，这正是价值模型要预测的
   那个量（``P_return × tokens``）的实现值，与 VD 的分子同量纲。请求侧的
   ``recompute = prompt - cached`` 另记全局税（``tax``），不参与归因——把它
   摊进 stamp 会把「本轮新增、本来就要 prefill」的 token 算成误判。

## 内容键

``(页号, 前 CONTEXT_TOKENS 个 token, 本页 token, 命名空间)`` 的 8 字节摘要。
不用缓存自己的累进 SHA256：那要求从根重量整条前缀（或把每页 hash 物化到
节点上，1 token/页时是 1.2M 个字符串），代价远超收益。页号 + 16 token 上下文
已足以把不同位置、不同前缀区分开；碰撞只会让某次归因记到另一条同位置同
上下文的页上，量级可忽略，且不会凭空制造样本。

## 三个必须显式报的量

* ``censored``：stamp 还不到 H 秒、run 就结束了。chat 的复用间隔 T50≈118s、
  p95≈965s，而统计期只有 600s——不报删失率就会系统性把 chat 的 regret 压低
  （详见 ``Heter-Bench/docs/reports`` 里的多轮实验）。
* ``join_rate``：断点页能在坟场里找到的比例。为 0 意味着两边的内容键没对齐
  （page_size / extra_key / 页对齐），此时读到的 ``regret=0`` 是采集失败，
  不是策略完美。因此它是渲染里的独立字段，低于 ``JOIN_WARN`` 会带 WARN。
* ``tax``：``(prompt - cached) / prompt``，与 profile 的 hit 口径互补。
"""

from __future__ import annotations

import struct
from collections import deque
from hashlib import blake2b
from typing import Dict, List, Optional, Sequence, Tuple

# 内容键取首页前的上下文 token 数：够区分同位置不同前缀，又不超过一次
# cache line 的量级（向上取祖先链不超过两跳）。
CONTEXT_TOKENS = 16

# regret 曲线的横轴（秒）。最短的一档服务 agentic（T50≈6s），最长的一档
# 服务 struct / 长间隔 chat。
DEFAULT_HORIZONS: Tuple[float, ...] = (30.0, 120.0, 300.0, 600.0)

# 坟场保留时长：必须覆盖最长横轴，否则长横轴上的样本是被保留策略删掉的，
# 不是没回访。超出即按 FIFO 过期，过期数单独计。
DEFAULT_RETAIN_S = 900.0

# 坟场容量上限（条）。稳态条目数 ≈ 淘汰速率 × 保留时长，正常是万级；
# 触顶说明负载远超预期，宁可丢最老的样本也不要吃光内存。
DEFAULT_MAX_STAMPS = 200_000

# 无复用间隔分布的队列（struct）在「按本队 T50」那一列回退到这个值：
# 它是政策视野，不是拍的常数。
DEFAULT_FALLBACK_HORIZON_S = 120.0

# join_rate 低于它就在渲染里显式告警：regret=0 的最常见原因是没对上号。
JOIN_WARN = 0.5

# 分组维度：渲染与 JSON 都按这几个轴切。
GROUP_FIELDS = ("source", "queue", "rule", "depth", "share")

# 断点深度分档（页）。「浅的贵、深的便宜」若成立，regret 应集中在浅档。
_DEPTH_BUCKETS = ((1, "depth<=1p"), (8, "depth 2-8p"), (64, "depth 9-64p"))
_DEPTH_OVERFLOW = "depth>64p"
# 共享会话数分档：share_min 是 2，所以 0/1 都算「未被跨会话复用」。
_SHARE_BUCKETS = ((0, "share=0"), (1, "share=1"))
_SHARE_OVERFLOW = "share>=2"
# 校准曲线的预测值分箱（左闭右开，最后一箱含 1）。
_CALIBRATION_EDGES = (0.02, 0.05, 0.2, 0.5, 0.8)

_AXIS_OWN = "own"


def _rule_of(source: str) -> str:
    """决策来源 → 队内规则名。来源字符串由 ``selector.pick_victim`` 定。"""
    if source == "one_shot":
        return "one_shot_key"
    if source == "value:struct":
        return "struct_tier"
    if source == "value:fallback":
        return "random"
    if source.startswith("priced:"):
        return "priced_fallback"
    return "reuse"


def _depth_bucket(page_index: int) -> str:
    for limit, label in _DEPTH_BUCKETS:
        if page_index < limit:
            return label
    return _DEPTH_OVERFLOW


def _share_bucket(share: int) -> str:
    for limit, label in _SHARE_BUCKETS:
        if share <= limit:
            return label
    return _SHARE_OVERFLOW


def _ns_int(namespace) -> int:
    """命名空间 → 32 位整数。

    只要求**同进程内两侧一致**，所以用 ``hash`` 就够；跨进程可复现是内容键
    不需要的性质（坟场不落盘）。
    """
    return hash(tuple(namespace or ())) & 0x7FFF_FFFF


def _digest(window: Sequence[int], page_index: int, ns: int) -> int:
    """``(页号, 命名空间, 窗口 token)`` 的 8 字节摘要。"""
    try:
        blob = struct.pack(f"<{len(window) + 2}i", page_index, ns, *window)
    except struct.error:
        # token id 超出 int32（或不是整数）：退回文本编码，不静默丢样本。
        blob = repr((page_index, ns, tuple(window))).encode()
    return int.from_bytes(blake2b(blob, digest_size=8).digest(), "little")


def content_key(
    tokens: Sequence[int],
    page_start: int,
    page_size: int,
    *,
    base: int = 0,
    context: int = CONTEXT_TOKENS,
    ns: int = 0,
) -> int:
    """取 ``tokens`` 中绝对位置 ``page_start`` 起那一页的内容键。

    Args:
        tokens: token 序列；``tokens[base]`` 对应绝对位置 ``base``。
            请求侧传全量 ids + ``base=0``，淘汰侧传祖先链切出的尾部窗口 +
            它自己的起点，两侧算出同一个值。
        page_start: 该页的绝对起始 token 位置（请求侧是已匹配长度）。
        page_size: 页内 token 数。
        base: ``tokens`` 的绝对起点。
        context: 页前上下文 token 数。
        ns: 命名空间摘要（``_ns_int``）。

    Returns:
        内容键；窗口不足（序列已到起点）时按实际长度算，与请求侧一致。
    """
    begin = max(0, page_start - context) - base
    if begin < 0:
        # 调用方给的窗口不够长：截到窗口起点，宁可少算上下文也不要越界。
        begin = 0
    end = page_start + page_size - base
    return _digest(tokens[begin:end], page_start // page_size, ns)


def queue_horizons_from_policy(
    fallback_s: float = DEFAULT_FALLBACK_HORIZON_S,
) -> Dict[str, float]:
    """各队列「按本队 T50」那一列的横轴。

    H 取该队列自己的复用间隔中位数：同一个 H 套在 chat（T50≈118s）和 agentic
    （T50≈6s）上，短的那个几乎必然判「没回访」，长的那个几乎必然判「回了」——
    regret 会退化成队列规模读出器。无分布数据的队列（struct）回退到政策视野。

    Args:
        fallback_s: 无复用分布时的回退横轴（秒）。

    Returns:
        ``{queue: 秒}``；sglang 不在场时返回空字典（该列回退同一个 fallback）。
    """
    try:
        from sglang.srt.mem_cache.heter_cache.utils import (
            VALUE_QUEUES,
            HeterCacheNodeTag,
            reuse_interval_median,
        )
    except Exception:  # noqa: BLE001 - 离线读账本时没有 sglang
        return {}
    out: Dict[str, float] = {}
    for queue in VALUE_QUEUES:
        tag = HeterCacheNodeTag.from_queue(queue)
        t50 = reuse_interval_median(tag) if tag is not None else None
        out[queue] = float(t50) if t50 else float(fallback_s)
    return out


def namespace_key(extra_key=None, cache_salt=None) -> int:
    """请求/节点的缓存命名空间 → 32 位摘要。

    两个命名空间下 token 序列完全相同，各自树里也是不同节点；不把命名空间
    折进内容键，跨命名空间的同位置同内容会被误 join。
    """
    return _ns_int((extra_key, cache_salt))


class RegretLedger:
    """坟场索引 + 请求侧 join + 分档聚合。

    只在 ``HETER_REGRET=1`` 时由 ``bench_bridge`` 装配；未装配时调用方
    （``heter_cache/radix_cache.py``）的所有钩子都被 ``is not None`` 短路。

    Args:
        now: 时间源，必须与缓存层同源（``RadixCache._now``）。
        horizons: 曲线的横轴（秒），按本队 T50 那一列另算。
        queue_horizon: ``queue -> 秒``；缺省回退 ``DEFAULT_FALLBACK_HORIZON_S``。
        retain_s: 坟场保留时长。
        max_stamps: 坟场条目上限。
        context: 内容键的上下文 token 数。
        out_path: 报告落盘路径（JSON，原子替换）；``None`` 表示只打日志。
    """

    __slots__ = (
        "context",
        "_now",
        "_horizons",
        "_queue_horizon",
        "_retain_s",
        "_max_stamps",
        "_out_path",
        "_dump_failures",
        "_skips",
        "_stamps",
        "_order",
        "_prompt_tokens",
        "_recompute_tokens",
        "_requests",
        "_joined",
        "_multi_hit",
        "_expired",
        "_dropped",
    )

    def __init__(
        self,
        *,
        now,
        horizons: Sequence[float] = DEFAULT_HORIZONS,
        queue_horizon: Optional[Dict[str, float]] = None,
        retain_s: float = DEFAULT_RETAIN_S,
        max_stamps: int = DEFAULT_MAX_STAMPS,
        context: int = CONTEXT_TOKENS,
        out_path: Optional[str] = None,
    ) -> None:
        import time

        self.context = int(context)
        self._now = now if now is not None else time.monotonic
        self._horizons = tuple(sorted({float(h) for h in horizons}))
        self._queue_horizon = dict(queue_horizon or {})
        self._retain_s = float(retain_s)
        self._max_stamps = int(max_stamps)
        self._out_path = out_path
        self._dump_failures = 0
        self._skips: Dict[str, int] = {}
        # 键 = 内容键；值 = [ts, span, source, queue, rule, share, chain,
        # depth, elapsed, hit_age]。用 list 而不是 dataclass：条目数是万级，
        # 每条约 100B 的 dataclass 开销在这里是白给的。
        self._stamps: Dict[int, list] = {}
        # (ts, key) 的 FIFO，用于按保留时长过期
        self._order: deque = deque()
        self._prompt_tokens = 0
        self._recompute_tokens = 0
        self._requests = 0
        self._joined = 0
        self._multi_hit = 0
        self._expired = 0
        self._dropped = 0

    # ------------------------------------------------------------------
    # 写入侧
    # ------------------------------------------------------------------
    def on_evict(
        self,
        *,
        key: int,
        span: int,
        source: str,
        queue: str,
        share: int,
        chain: int,
        depth: int,
        elapsed: float,
        ts: Optional[float] = None,
    ) -> None:
        """登记一次淘汰决策。

        Args:
            key: 被腾出链**最浅**节点首页的内容键（``content_key``）。
            span: 本次决策腾出的 token 总数（含级联清理的私有祖先）。
            source: ``one_shot`` / ``value:<queue>`` / ``value:fallback``。
            queue: 牺牲节点的队列名（决策归属，不是断点页的队列）。
            share: 牺牲节点的跨会话共享数。
            chain: 本次级联释放的节点数。
            depth: 断点页的绝对页号。
            elapsed: 决策时刻距今的「未访问时长」，供校准曲线复算预测值。
            ts: 决策时刻；``None`` 表示取账本时钟。
        """
        now = self._now() if ts is None else float(ts)
        self._expire(now)
        # 同一内容被再次逐出时覆盖旧 stamp：旧的已经不可能再被 join（内容
        # 已被重算并重新插入），留着只会抢走新决策的归因。
        if key in self._stamps:
            self._stamps.pop(key, None)
            self._dropped += 1
        self._stamps[key] = [
            now,
            int(span),
            str(source),
            str(queue),
            _rule_of(source),
            int(share),
            int(chain),
            int(depth),
            float(elapsed),
            None,
        ]
        self._order.append((now, key))
        while len(self._stamps) > self._max_stamps:
            self._evict_oldest()

    def on_request(
        self,
        *,
        key: Optional[int],
        recompute: int,
        prompt: int,
        ts: Optional[float] = None,
    ) -> bool:
        """登记一次 admission 匹配，并尝试 join 断点页。

        Args:
            key: 断点页（首个未命中页）的内容键；``None`` 表示这次匹配没有
                断点页（全命中 / 空 key），只计入税的分母。
            recompute: 本次请求需要重算的 token 数（``prompt - cached``）。
            prompt: 本次请求的 prompt token 数。
            ts: 请求时刻；``None`` 表示取账本时钟。

        Returns:
            是否 join 到了某个 stamp（即这次 miss 有明确的责任决策）。
        """
        now = self._now() if ts is None else float(ts)
        self._expire(now)
        self._requests += 1
        self._prompt_tokens += int(prompt)
        self._recompute_tokens += int(recompute)
        if key is None:
            return False
        stamp = self._stamps.get(key)
        if stamp is None:
            return False
        if stamp[9] is not None:
            # 同一内容在一个生成周期里被多个请求同时撞上（重插之前的并发
            # miss）。只归因给第一个，其余计入 multi_hit 供排查。
            self._multi_hit += 1
            return False
        self._joined += 1
        stamp[9] = max(0.0, now - stamp[0])
        return True

    def note_skip(self, reason: str) -> None:
        """记一次无法测量的请求/节点（bigram 视图等），不让它静默消失。"""
        self._skips[reason] = self._skips.get(reason, 0) + 1

    # ------------------------------------------------------------------
    # 坟场维护
    # ------------------------------------------------------------------
    def _expire(self, now: float) -> None:
        while self._order and now - self._order[0][0] > self._retain_s:
            _, key = self._order.popleft()
            if self._stamps.pop(key, None) is not None:
                self._expired += 1

    def _evict_oldest(self) -> None:
        while self._order:
            _, key = self._order.popleft()
            if self._stamps.pop(key, None) is not None:
                self._dropped += 1
                return

    # ------------------------------------------------------------------
    # 读侧
    # ------------------------------------------------------------------
    def horizon_of(self, queue: str) -> float:
        """该队列「按本队 T50」那一列的横轴。"""
        h = self._queue_horizon.get(queue)
        return float(h) if h else DEFAULT_FALLBACK_HORIZON_S

    def axes(self) -> List[object]:
        """曲线横轴：显式档位 + ``own``（按各队自己的 T50）。"""
        return [*self._horizons, _AXIS_OWN]

    def report(self) -> dict:
        """聚合出一份可渲染/可落盘的报告（单遍扫描坟场）。"""
        now = self._now()
        axes = self.axes()
        # groups[field][axis][bucket] = [evicted, regret, censored, stamps]
        groups: Dict[str, Dict[object, Dict[str, list]]] = {
            f: {a: {} for a in axes} for f in GROUP_FIELDS
        }
        calib: Dict[object, Dict[int, list]] = {a: {} for a in axes}
        evicted_total = 0
        regret_total = 0
        chain_total = 0
        for ts, span, source, queue, rule, share, chain, depth, elapsed, hit_age in (
            self._stamps.values()
        ):
            own = self.horizon_of(queue)
            fresh = now - ts
            evicted_total += span
            chain_total += chain
            if hit_age is not None:
                regret_total += span
            stamp_fields = (
                ("source", source),
                ("queue", queue),
                ("rule", rule),
                ("depth", _depth_bucket(depth)),
                ("share", _share_bucket(share)),
            )
            for axis in axes:
                horizon = own if axis == _AXIS_OWN else axis
                hit = hit_age is not None and hit_age <= horizon
                censored = hit_age is None and fresh < horizon
                for field, bucket in stamp_fields:
                    cell = groups[field][axis].setdefault(bucket, [0, 0, 0, 0])
                    cell[0] += span
                    cell[1] += span if hit else 0
                    cell[2] += span if censored else 0
                    cell[3] += 1
                predicted = self._predicted(queue, elapsed, horizon)
                if predicted is not None:
                    idx = _bucket_of(predicted)
                    cell = calib[axis].setdefault(idx, [0.0, 0, 0])
                    cell[0] += predicted * span
                    cell[1] += span
                    cell[2] += span if hit else 0
        return {
            "tax": {
                "requests": self._requests,
                "prompt_tokens": self._prompt_tokens,
                "recompute_tokens": self._recompute_tokens,
                "recompute_pct": _pct(self._recompute_tokens, self._prompt_tokens),
                "regret_tokens": regret_total,
                "regret_per_prompt": _pct(regret_total, self._prompt_tokens),
                "stamps": len(self._stamps),
                "evicted_tokens": evicted_total,
                "nodes_per_stamp": round(chain_total / len(self._stamps), 2)
                if self._stamps
                else 0.0,
                "joined": self._joined,
                "multi_hit": self._multi_hit,
                "join_rate": _pct(self._joined, self._requests),
                "expired": self._expired,
                "dropped": self._dropped,
                "skips": dict(self._skips),
                "dump_failures": self._dump_failures,
            },
            "queue_horizon": {
                q: self.horizon_of(q)
                for q in sorted({s[3] for s in self._stamps.values()})
            },
            "axes": axes,
            "groups": {
                field: {
                    axis: {
                        bucket: _cell(cell) for bucket, cell in cells.items()
                    }
                    for axis, cells in axes_map.items()
                }
                for field, axes_map in groups.items()
            },
            "calibration": {
                axis: {
                    bucket: {
                        "predicted": cell[0] / cell[1] if cell[1] else 0.0,
                        "realized": cell[2] / cell[1] if cell[1] else 0.0,
                        "tokens": cell[1],
                    }
                    for bucket, cell in cells.items()
                }
                for axis, cells in calib.items()
            },
        }

    def emit(self) -> str:
        """生成报告：落盘（若配了路径）并返回可打进日志的文本。"""
        report = self.report()
        if self._out_path:
            self._dump(report)
        return self.render(report)

    def _dump(self, report: dict) -> None:
        """原子写 JSON。失败只累加计数，不打断正在跑的实验。"""
        import json
        import os

        tmp = self._out_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=1)
            os.replace(tmp, self._out_path)
        except OSError:
            self._dump_failures += 1

    def render(self, report: Optional[dict] = None) -> str:
        """把报告渲染成可放进 ``[HeterCache profile]`` 旁边的多行文本。"""
        rep = self.report() if report is None else report
        tax = rep["tax"]
        lines = [
            (
                f"[HeterCache regret] H={_AXIS_OWN}(T50)"
                f" requests={tax['requests']}"
                f" tax={tax['recompute_pct']}%"
                f" stamps={tax['stamps']}"
                f" evicted={tax['evicted_tokens']:.0f}"
                f" regret={tax['regret_tokens']:.0f}"
                f" ({tax['regret_per_prompt']}% of prompt)"
                f" join={tax['join_rate']}%"
            )
        ]
        own = rep["groups"]
        for field in ("source", "queue", "rule", "depth", "share"):
            cells = own[field].get(_AXIS_OWN, {})
            if not cells:
                continue
            lines.append(f"  by {field:<10}{'evicted':>12}{'regret':>12}"
                         f"{'regret%':>9}{'censored%':>11}")
            for bucket in sorted(cells, key=lambda b: -cells[b]["evicted"]):
                c = cells[bucket]
                lines.append(
                    f"  {bucket:<13}{c['evicted']:>12.0f}{c['regret']:>12.0f}"
                    f"{c['regret_pct']:>8.2f}%{c['censored_pct']:>10.2f}%"
                )
        calib = rep["calibration"].get(_AXIS_OWN, {})
        if calib:
            points = " ".join(
                f"{calib[i]['predicted']:.2f}->{calib[i]['realized']:.2f}"
                for i in sorted(calib)
            )
            lines.append(f"  calibration pred->real: {points}")
        if tax["join_rate"] < JOIN_WARN * 100.0:
            lines.append(
                f"  WARN join_rate={tax['join_rate']}% < {JOIN_WARN * 100:.0f}%："
                "两侧内容键没对上（page_size / extra_key / 页对齐），"
                "此处的 regret 不可读"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 校准
    # ------------------------------------------------------------------
    @staticmethod
    def _predicted(queue: str, elapsed: float, horizon: float) -> Optional[float]:
        """用队列的复用分布复算「决策时刻应该给的 P_return」。

        惰性导入 sglang：本模块要在无 torch 环境里被单测加载。
        """
        try:
            from sglang.srt.mem_cache.heter_cache.utils import (
                HeterCacheNodeTag,
                reuse_window_probability,
            )
        except Exception:  # noqa: BLE001 - 离线跑账本时不带校准
            return None
        tag = HeterCacheNodeTag.from_queue(queue)
        cfg = None if tag is None else tag.reuse_config
        if cfg is None:
            return None
        mu, sigma = cfg
        return reuse_window_probability(max(0.0, elapsed), horizon, mu, sigma)

    def reset(self) -> None:
        """清空坟场与累计量（缓存 flush / 跨 run 复用同一进程）。"""
        self._stamps.clear()
        self._order.clear()
        self._prompt_tokens = 0
        self._recompute_tokens = 0
        self._requests = 0
        self._joined = 0
        self._multi_hit = 0
        self._expired = 0
        self._dropped = 0
        self._skips.clear()


def _pct(num, den) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def _cell(cell: list) -> dict:
    evicted, regret, censored, stamps = cell
    return {
        "evicted": evicted,
        "regret": regret,
        "censored": censored,
        "stamps": stamps,
        "regret_pct": _pct(regret, evicted),
        "censored_pct": _pct(censored, evicted),
    }


def _bucket_of(predicted: float) -> int:
    for idx, edge in enumerate(_CALIBRATION_EDGES):
        if predicted < edge:
            return idx
    return len(_CALIBRATION_EDGES)
