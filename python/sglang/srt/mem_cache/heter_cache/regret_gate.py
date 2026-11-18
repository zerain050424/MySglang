"""跨队列仲裁的后悔密度门（控制路径，档 A）。

信号与决策规则见 ``RegretGateParams``（heter_cache/utils.py）。本模块只做
仲裁需要的最小记账：

* stamp：淘汰决策 → ``(内容键 → 队列归属, span)``。内容键由缓存层统一计算
  传入（``HeterCacheRadixCache._breakpoint_key``），与观测账本
  （Heter-Bench/profile/regret_ledger.py）同源——键不同源就没有 join。
* join：admission miss 的断点页键查 stamp，命中则把该 stamp 的 span 记到
  队列的后悔窗口，并把 stamp 摘除（一个 stamp 只归因一次，与账本的
  ``hit_age`` 语义一致）。
* R_q：窗口内 ``regret_sum / evicted_sum``；样本不足返回 ``None``。

层位约束（AGENTS）：观测实现全部在 Heter-Bench，sglang 侧控制路径不依赖它。
账本继续做观测报告；本门是控制路径的独立轻量计数器，两者定义一致、互不依赖
（观测账本在场时 content_key 的 context 以账本为准，由缓存层统一传参保证）。

无锁安全：所有入口只在调度主线程（match / evict）执行，与账本同一约束。
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from sglang.srt.mem_cache.heter_cache.utils import RegretGateParams

if TYPE_CHECKING:  # pragma: no cover
    pass


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

    def on_request(self, *, key: Optional[int], ts: Optional[float] = None) -> None:
        """断点页 join：命中 stamp 则按其队列归属记一次后悔（span 全额，一次）。

        Args:
            key: 断点页内容键；``None``（全命中）不 join。
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
        self._push(self._rg, self._rg_sum, queue, now, span)

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
        """档 A 弃权判定：R_q ≥ max(r_min, k × 已采样队列 R 的中位数)。

        已采样队列少于 2（没有对照）时不弃权——单队列的 R 没有参照系，
        回退现状比瞎跳队安全。
        """
        decision = False
        r_q = self.r(queue)
        if r_q is not None and r_q >= self.params.r_min:
            sampled = [self.r(q) for q in order if self.sampled(q)]
            if len(sampled) >= 2:
                ordered = sorted(sampled)
                mid = len(ordered) // 2
                median = (
                    ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
                )
                decision = r_q > self.params.k * median
        if decision:
            self.abstains[queue] = self.abstains.get(queue, 0) + 1
        return decision

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

    def _prune(self, now: float) -> None:
        """过期维护：stamp 按 retain_s、窗口按 window_s。O(过期满的条数)。"""
        retain = self.params.retain_s
        while self._order and now - self._order[0][0] > retain:
            _, key = self._order.popleft()
            if self._stamps.pop(key, None) is not None:
                pass  # 过期即消失；未 join 的 stamp 不产生后悔，无需回补
        window = self.params.window_s
        for queue in list(self._ev):
            dq = self._ev[queue]
            while dq and now - dq[0][0] > window:
                _, tokens = dq.popleft()
                self._ev_sum[queue] -= tokens
            if not dq:
                self._ev.pop(queue, None)
                self._ev_sum.pop(queue, None)
        for queue in list(self._rg):
            dq = self._rg[queue]
            while dq and now - dq[0][0] > window:
                _, tokens = dq.popleft()
                self._rg_sum[queue] -= tokens
            if not dq:
                self._rg.pop(queue, None)
                self._rg_sum.pop(queue, None)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<RegretGate window_s={self.params.window_s} joins={self.joins}>"
