"""跨队列仲裁的后悔弃权门：按滑窗后悔率 R_q = regret_sum / evicted_sum 过滤高后悔队列。"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional

from sglang.srt.mem_cache.heter_cache.utils import RegretGateParams, median


class RegretGate:
    """每队列滑窗后悔密度 + 弃权裁决。排序键无状态，可重复调用。"""

    def __init__(self, params: RegretGateParams) -> None:
        self.params = params
        self._now = time.monotonic
        # key -> [ts, queue, span]；与账本同构（list 省万级条目的对象开销）
        self._stamps: Dict[int, list] = {}
        self._order: deque = deque()
        # 每队列滑窗：deque[(ts, tokens)] + 增量和
        self._ev: Dict[str, deque] = {}
        self._rg: Dict[str, deque] = {}
        self._ev_sum: Dict[str, int] = {}
        self._rg_sum: Dict[str, int] = {}
        # ---- 观测 ----
        self.joins = 0
        self.abstains: Dict[str, int] = {}
        self.cheapest_pick = 0
        self._skips: Dict[str, int] = {}

    # ------------------------------------------------------------ 时钟/复位
    def use_clock(self, now) -> None:
        """接管时间源（与 HeterCacheStrategy.use_clock 同一约定）。"""
        self._now = now

    def reset(self) -> None:
        self._stamps.clear()
        self._order.clear()
        for table in (self._ev, self._rg, self._ev_sum, self._rg_sum):
            table.clear()
        self.joins = 0
        self.abstains.clear()
        self.cheapest_pick = 0
        self._skips.clear()

    # ------------------------------------------------------------ 写入侧
    def on_evict(self, *, key: Optional[int], queue: str, span: int,
                 ts: Optional[float] = None) -> None:
        """登记一次淘汰决策的 stamp。

        Args:
            key: 断点页内容键；``None`` 表示该视角无法测（bigram/eagle），
                计入 skips 而不是静默消失。
            queue: 决策归属队列（``strategy.node_tag(victim).queue_name``）。
            span: 本次决策腾出的 token 数。
        """
        now = self._now() if ts is None else float(ts)
        self._prune(now)
        if key is None:
            self._skips["evict:no_key"] = self._skips.get("evict:no_key", 0) + 1
            return
        # 同键覆盖：内容被重算重插后再被逐出，旧 stamp 已不可能被 join
        # （与账本同语义，覆盖即丢弃）。
        self._stamps.pop(key, None)
        self._stamps[key] = [now, str(queue), int(span)]
        self._order.append((now, key))
        while len(self._stamps) > self.params.max_stamps:
            _, old = self._order.popleft()
            self._stamps.pop(old, None)
        self._push(self._ev, self._ev_sum, queue, now, span)

    def on_request(
        self,
        *,
        key: Optional[int],
        recompute: Optional[int] = None,
        ts: Optional[float] = None,
    ) -> None:
        """断点页 join：命中 stamp 时按尾重算代价（recompute >= tail_recompute_min）记后悔。

        Args:
            key: 断点页内容键；``None``（全命中）不 join。
            recompute: 本次请求实际重算 token 数（prompt - matched）；
                省略时退回单节点 ``span``（兼容旧单测）。
        """
        if key is None:
            return
        now = self._now() if ts is None else float(ts)
        self._prune(now)
        stamp = self._stamps.pop(key, None)
        if stamp is None:
            # 无 stamp 的 miss（命中早被过期/覆盖的内容键）不归因——
            # 与账本语义一致：join 不上就没有责任决策。
            return
        _, queue, span = stamp
        # 摘除后再记后悔：一个 stamp 只归因一次（账本用 hit_age 达成同一语义）
        self.joins += 1
        if recompute is None:
            regret_tokens = span
        else:
            rec = int(recompute)
            regret_tokens = rec if rec >= self.params.tail_recompute_min else 0
        if regret_tokens > 0:
            self._push(self._rg, self._rg_sum, queue, now, regret_tokens)

    def note_skip(self, reason: str) -> None:
        """无法测量的请求/节点（bigram 视图等），不让它静默消失。"""
        self._skips[reason] = self._skips.get(reason, 0) + 1

    # ------------------------------------------------------------ 读侧
    def sampled(self, queue: str) -> bool:
        """窗口内削掉量是否达到样本下限。不足样本的队列不参与弃权判定。"""
        self._prune(self._now())
        return self._ev_sum.get(queue, 0) >= self.params.min_evicted

    def r(self, queue: str) -> Optional[float]:
        """R_q = 后悔 / 削掉；样本不足返回 None。"""
        self._prune(self._now())
        evicted = self._ev_sum.get(queue, 0)
        if evicted < self.params.min_evicted:
            return None
        return self._rg_sum.get(queue, 0) / evicted

    def abstain(self, queue: str, order: List[str]) -> bool:
        """弃权判定：R_q >= r_min 且 R_q > k * 同侪中位数（排除自身）；无同侪采样时不弃权。"""
        r_q = self.r(queue)
        if r_q is None or r_q < self.params.r_min:
            return False
        peers = [self.r(q) for q in order if q != queue and self.sampled(q)]
        if not peers:
            return False
        if r_q > self.params.k * median(peers):
            self.abstains[queue] = self.abstains.get(queue, 0) + 1
            return True
        return False

    def cheapest(self, order: List[str]) -> Optional[str]:
        """全弃权时的跨队兜底：已采样队列里 R 最小者；无人采样返回 None。"""
        sampled = [q for q in order if self.sampled(q)]
        if not sampled:
            return None
        self.cheapest_pick += 1
        return min(sampled, key=self.r)

    def stats(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "joins": self.joins,
            "cheapest_pick": self.cheapest_pick,
        }
        for queue in sorted(set(self._ev) | set(self._rg)):
            r_q = self.r(queue)
            if r_q is not None:
                out[f"r_{queue}"] = round(r_q, 4)
        for queue, count in self.abstains.items():
            out[f"abstain_{queue}"] = count
        return out

    # ------------------------------------------------------------ 内部
    def _push(self, table: Dict[str, deque], sums: Dict[str, int],
              queue: str, ts: float, tokens: int) -> None:
        table.setdefault(queue, deque()).append((ts, tokens))
        sums[queue] = sums.get(queue, 0) + tokens

    @staticmethod
    def _prune_window(
        table: Dict[str, deque], sums: Dict[str, int], window: float, now: float
    ) -> None:
        for queue in list(table):
            dq = table[queue]
            while dq and now - dq[0][0] > window:
                _, tokens = dq.popleft()
                sums[queue] -= tokens
            if not dq:
                table.pop(queue, None)
                sums.pop(queue, None)

    def _prune(self, now: float) -> None:
        """按 retain_s 清理过期 stamp，按 window_s 清理滑窗队列计数。"""
        retain = self.params.retain_s
        while self._order and now - self._order[0][0] > retain:
            _, key = self._order.popleft()
            self._stamps.pop(key, None)
        window = self.params.window_s
        self._prune_window(self._ev, self._ev_sum, window, now)
        self._prune_window(self._rg, self._rg_sum, window, now)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<RegretGate window_s={self.params.window_s} joins={self.joins}>"
