"""Q_struct 队列内部的 W-TinyLRU **状态机**（成员 / 三区 / 准入 / aging）。

职责边界：
  * 本模块只回答「谁在队列里、在哪个区、区头是谁」——不回答「该淘汰谁」。
    淘汰面由 ``victim_override.StructTinyLruPolicy`` 薄转发（applies_to / pick）。
  * 缓存层直接持有本状态机（钩子 + ``owns``，死链上溯闸要用）；策略层把它
    包成插件进 ``victim_overrides``。
  * 本类不实现 ``VictimOverride``，也不 import ``victim_override``，只依赖 uni_utils。

不释放任何 KV、不删节点：
  1. 某个节点是否还在 struct 队列里（成员判定 M3 + 退队）；
  2. 若在，队内下一个该出的是谁（window → probation → protected 的区头）。

三条需求的落点：
  * 按频数驻留      -> share_count（不同 rid 数）+ 三区分档；
  * 单次文档询问被删 -> 准入失败 -> _reject() 降级为 QTRANS，交给现有 one-shot 带处决；
  * 高频共享前缀驻留 -> 成员判定 M3 + 降级继承 + 退队只在 share<2 时发生。
"""

from __future__ import annotations

import time
import zlib
from collections import OrderedDict
from typing import Dict, Optional

from sglang.srt.mem_cache.uni_utils import (
    SimpleHybridQueueTag,
    session_bloom_bits,
)

# struct_region 取值（写在 TreeNode 上，命中判定路径需要一个 O(1) 的所有权查询）
REGION_NONE = 0
REGION_WINDOW = 1
REGION_PROBATION = 2
REGION_PROTECTED = 3
REGION_PARKED = 4

_REGION_ORDER = (REGION_WINDOW, REGION_PROBATION, REGION_PROTECTED)
_REGION_NAMES = {
    REGION_NONE: "none",
    REGION_WINDOW: "window",
    REGION_PROBATION: "probation",
    REGION_PROTECTED: "protected",
    REGION_PARKED: "parked",
}

# share_count / share_sessions 是 u8，饱和即可：K 只有个位数，超出的部分没有信息量
_MAX_SHARE = 255

# 低于这个值的节点只能靠分叉成为成员：share>=2 才谈得上被共享
_MIN_SHARE = 2


class StructQueueState:
    """Q_struct 队列内部的 W-TinyLRU 队内状态（window + SLRU + 频数准入 + aging）。

    只维护队列状态：成员判定、三区索引、准入、aging、归属查询 ``owns``。
    淘汰候选通过 ``eviction_candidate()`` 暴露给 ``StructTinyLruPolicy``，
    由它在 ``victim_overrides`` 链里作为「候选修正」参与全局淘汰。

    设计文档：python/sglang/srt/mem_cache/STRUCT_QUEUE_WTINYLRU.md
    """

    def __init__(
        self,
        enabled: bool = True,
        k: int = 3,
        window_entries: int = 64, ## window队列条目上限
        epoch_events: int = 8192, ## 到达后进行一次Aging
        admission_tie: str = "reject",
        admission_metric: str = "share",
        inherit_share: bool = True,
        share_metric: str = "hybrid",
        protected_budget_frac: float = 0.0,
        stale_grace_s: float = 0.0,
    ):
        self.enabled = enabled
        self.k = max(2, int(k))
        self.window_entries = max(1, int(window_entries))
        self.epoch_events = max(1, int(epoch_events))
        self.admission_tie = admission_tie
        self.admission_metric = admission_metric
        self.inherit_share = bool(inherit_share)
        # A：频数口径。
        #   hybrid  = 成员/准入沿用 request 口径，但 **protected 只留给跨会话共享**
        #             （sessions>=k 才 protected；会话内热点最多 probation）——默认
        #   session = 只数不同会话（消融：会话内热点完全不受保护）
        #   rid     = 只数不同请求（旧口径：单会话多轮也会进 protected）
        self.share_metric = (
            share_metric if share_metric in ("hybrid", "session", "rid") else "hybrid"
        )
        # B：protected 的 token 预算（占"活跃区 token"的比例），0 = 关闭
        self.protected_budget_frac = max(0.0, min(1.0, float(protected_budget_frac or 0.0)))
        # C：叶子降级时的新鲜度闸（秒），0 = 关闭
        self.stale_grace_s = max(0.0, float(stale_grace_s or 0.0))

        # 三个活跃区 + 挂起区。OrderedDict 提供 O(1) 头部取用 / 尾部插入 / 尾部移动。
        self.regions: Dict[int, "OrderedDict[int, object]"] = {
            REGION_WINDOW: OrderedDict(),
            REGION_PROBATION: OrderedDict(),
            REGION_PROTECTED: OrderedDict(),
        }
        self.parked: "OrderedDict[int, object]" = OrderedDict()

        # aging：事件驱动（入区 / 淘汰），与 wall-clock 无关，保证回放可复现
        self._events = 0
        self._aging = False

        # 同一个请求会连续命中一整条路径：rid -> bloom bits 做一次局部缓存
        self._last_rid: Optional[str] = None
        self._last_bits = 0
        # 会话 bloom 的局部缓存（A 的口径）
        self._last_sid: Optional[str] = None
        self._last_sbits = 0

        # B：区/挂起区的 token 计数。push/remove 增量维护；
        # 分裂与 aging 会改变对象长度，那里整体重算。
        self._tokens = {REGION_WINDOW: 0, REGION_PROBATION: 0, REGION_PROTECTED: 0}
        self._parked_tokens = 0

        self.stats = {
            "admit": 0,
            "reject": 0,
            "retire": 0,
            "park": 0,
            "ages": 0,
            "dedup_hits": 0,
            "forced": 0,
            # A/B/C 的观测量
            "dedup_sessions": 0,
            "budget_demote": 0,
            "stale_deflect": 0,
        }

    # ------------------------------------------------------------------ 判定

    # ------------------------------------------------------- 频数口径（A）

    def _sessions_of(self, node) -> int:
        """不同会话数；老节点/测试桩没有该字段时退回 rid 口径。"""
        sessions = getattr(node, "share_sessions", None)
        if sessions is None:
            return int(getattr(node, "share_count", 0) or 0)
        return int(sessions)

    def _freq_of(self, node) -> int:
        """成员判定与准入用的频数。

        * session：不同会话数
        * rid / hybrid：不同请求数（保持「会话内多轮复用」也被计为热点）
        """
        if self.share_metric == "session":
            return self._sessions_of(node)
        return int(getattr(node, "share_count", 0) or 0)

    def _session_bits(self, session_id) -> int:
        """session_id -> 64 bit bloom（与标签版共用 uni_utils.session_bloom_bits）。"""
        if session_id != self._last_sid:
            self._last_sid = session_id
            self._last_sbits = session_bloom_bits(session_id)
        return self._last_sbits

    @staticmethod
    def _n_tokens(node) -> int:
        value = getattr(node, "value", None)
        return len(value) if value is not None else 0

    def _recount_tokens(self) -> None:
        """整体重算 token 计数（分裂/aging 之后调用）。"""
        for region in self._tokens:
            self._tokens[region] = 0
        for region, table in self.regions.items():
            self._tokens[region] = sum(self._n_tokens(n) for n in table.values())
        self._parked_tokens = sum(self._n_tokens(n) for n in self.parked.values())

    def _zone_tokens(self) -> int:
        return sum(self._tokens.values())

    @staticmethod
    def _is_branch(node) -> bool:
        return len(node.children) >= 2

    @staticmethod
    def _is_evictable_leaf(node) -> bool:
        if node.value is None or node.lock_ref > 0:
            return False
        for child in node.children.values():
            if not child.evicted:
                return False
        return True

    def _is_member(self, node) -> bool:
        """
        如何判断属于struct：分叉点，或被共享的主干（share>=2 且父链最终落到分叉点/根）。
        迭代实现：沿着 parent 向上走，任一节点分叉即成立；任一节点 share<2 即不成立；
        走到根说明这条主干一直共享到序列开头，同样成立。
        """
        if node is None or node.parent is None or node.evicted:
            return False
        if node.struct_ever_branch:  # 本节点曾经分叉,则保留
            return True
        cur = node
        while cur is not None:
            if cur.parent is None:  # 到达根：整条前缀一直共享
                return True
            if self._is_branch(cur) or cur.struct_ever_branch:
                return True
            if self._freq_of(cur) < _MIN_SHARE:
                return False
            cur = cur.parent
        return False

    # ------------------------------------------------------------ 索引维护

    def _index_remove(self, node) -> None:
        region = node.struct_region
        tokens = self._n_tokens(node)
        if region in self.regions:
            if self.regions[region].pop(node.id, None) is not None:
                self._tokens[region] = max(0, self._tokens[region] - tokens)
        elif region == REGION_PARKED:
            if self.parked.pop(node.id, None) is not None:
                self._parked_tokens = max(0, self._parked_tokens - tokens)
        node.struct_region = REGION_NONE

    def _push(self, region: int, node, at_head: bool = False) -> None:
        node.struct_region = region
        tokens = self._n_tokens(node)
        if region == REGION_PARKED:
            self.parked[node.id] = node
            self._parked_tokens += tokens
            return
        table = self.regions[region]
        table[node.id] = node
        self._tokens[region] += tokens
        if at_head:
            table.move_to_end(node.id, last=False)

    def _touch(self, node) -> None:
        """命中后把节点移到区尾（window 是 FIFO，不移动）。"""
        region = node.struct_region
        if region in (REGION_PROBATION, REGION_PROTECTED):
            table = self.regions[region]
            if node.id in table:
                table.move_to_end(node.id, last=True)

    # ------------------------------------------------------------ 状态迁移

    def _leave_struct(self, node) -> None:
        """不是成员：移出索引，但**保留** share_count / rid_bloom。

        保留是必要的：频数在节点成为成员之前就要开始累积（一个前缀被 3 个请求
        用过之后才分叉，它一入队就该直接进 protected），清掉会让计数永远在
        0/1 之间震荡。
        """
        self._index_remove(node)

    def _retire(self, node) -> None:
        """退队（aging / 降级用）：移出索引、清计数、标签降回 QTRANS。"""
        was_member = node.struct_region != REGION_NONE
        self._index_remove(node)
        node.share_count = 0
        node.rid_bloom = 0
        node.share_sessions = 0
        node.session_bloom = 0
        node.struct_ever_branch = False
        if getattr(node, "queue_tag", None) == SimpleHybridQueueTag.QSTRUCT:
            node.queue_tag = SimpleHybridQueueTag.QTRANS
        if was_member:
            self.stats["retire"] += 1

    def _park(self, node) -> None:
        if node.struct_region == REGION_PARKED:
            return
        self._index_remove(node)
        self._push(REGION_PARKED, node)
        self.stats["park"] += 1

    def _band_of(self, node) -> int:
        """分档。hybrid 模式下 protected 是**跨会话**专属：

        | 证据 | 落区 |
        |---|---|
        | 不同会话数 >= k | protected（真·共享模板） |
        | 不同会话数 == 2 | probation |
        | 只有 1 个会话、但请求数 >= 2（会话内热点） | probation（不升级） |
        | 其余 | window |

        ``rid`` 模式保留旧行为（请求数 >= k 直接 protected），``session`` 模式则
        完全不认会话内复用。三档可做单变量消融。
        """
        sessions = self._sessions_of(node)
        rids = int(getattr(node, "share_count", 0) or 0)
        if self.share_metric == "rid":
            if rids >= self.k:
                return REGION_PROTECTED
            return REGION_PROBATION if rids == 2 else REGION_WINDOW
        if self.share_metric == "session":
            if sessions >= self.k:
                return REGION_PROTECTED
            return REGION_PROBATION if sessions == 2 else REGION_WINDOW
        # hybrid（默认）
        if sessions >= self.k:
            return REGION_PROTECTED
        if sessions >= 2:
            return REGION_PROBATION
        if rids >= 2:
            return REGION_PROBATION   # 会话内热点：给一级保护，但不进 protected
        return REGION_WINDOW

    def _place(self, node, origin: str = "promote", force_band: Optional[int] = None) -> None:
        """把成员放进三区；不可淘汰则挂起。

        ``origin`` 决定区头/区尾与继承策略；``force_band`` 供 C（新鲜度闸）把
        陈旧叶子直接送进 window 重新观察。
        """
        if not (self.enabled and self._is_member(node)):
            self._leave_struct(node)
            return
        if not self._is_evictable_leaf(node):
            self._park(node)
            return

        self._index_remove(node)
        band = self._band_of(node) if force_band is None else force_band
        # §3.4 继承表：首次入区的 probation 成员进区头（老成员优先被比较），
        # 其余一律进区尾。
        at_head = origin == "inherit" and band == REGION_PROBATION
        self._push(band, node, at_head=at_head)
        self._note_event()
        if band == REGION_WINDOW:
            self._drain_window()
        elif band == REGION_PROTECTED:
            self._enforce_protected_budget()

    def _evaluate(self, node, origin: str = "promote") -> None:
        if not self.enabled or node is None:
            return
        self._place(node, origin)

    # ---------------------------------------------------------------- 准入

    def _share_of(self, node):
        freq = self._freq_of(node)
        if self.admission_metric == "share_per_token":
            return (freq, freq / max(1, self._n_tokens(node)))
        return (freq,)

    def _head(self, region: int):
        table = self.regions[region]
        while table:
            node_id = next(iter(table))
            node = table[node_id]
            if self._is_evictable_leaf(node):
                return node
            # 惰性清理：索引与可淘汰集合应该始终一致（INV-2），这里只是兜底
            table.pop(node_id, None)
            self._tokens[region] = max(0, self._tokens[region] - self._n_tokens(node))
            if node.struct_region == region:
                node.struct_region = REGION_NONE
        return None

    def _drain_window(self) -> None:
        """window 超配额时做频数准入：赢了进 main，输了降级交回上层。"""
        while len(self.regions[REGION_WINDOW]) > self.window_entries:
            cand = self._head(REGION_WINDOW)
            if cand is None:
                return
            victim = self._head(REGION_PROBATION) or self._head(REGION_PROTECTED)
            if victim is None:
                self._promote(cand)
                continue
            cand_key = self._share_of(cand)
            victim_key = self._share_of(victim)
            if cand_key > victim_key:
                self._promote(cand)
            elif cand_key == victim_key and self.admission_tie == "admit":
                self._promote(cand)
            elif self._freq_of(cand) >= _MIN_SHARE:
                # 已经是「被两个请求共享」的证据，不能再退化成一次性内容，
                # 否则会破坏 INV-2（成员不在任何索引里）。
                self._promote(cand)
            else:
                self._reject(cand)

    def _promote(self, node) -> None:
        """晋升 = 「观察通过」：必须离开 window（文档 §5.2）。

        原实现在 band 仍为 window 时递归 ``_drain_window()``，当 probation/protected
        都为空（没有 victim 可比较）时无界递归 → RecursionError，会打挂服务。
        这里把 band 抬到 probation，并把「继续 drain」交给外层 while。
        """
        band = self._band_of(node)
        if band == REGION_WINDOW:
            band = REGION_PROBATION
        self._index_remove(node)
        self._push(band, node, at_head=False)
        self.stats["admit"] += 1
        self._note_event()
        if band == REGION_PROTECTED:
            self._enforce_protected_budget()

    # ------------------------------------------------------------ 保护预算（B）

    def _protected_budget(self) -> Optional[int]:
        """protected 允许占用的 token 上限（相对活跃区 token）；None = 不限。"""
        if self.protected_budget_frac <= 0.0:
            return None
        total = self._zone_tokens()
        if total <= 0:
            return None
        return max(1, int(total * self.protected_budget_frac))

    def _enforce_protected_budget(self) -> None:
        """protected 超预算：把**队头**降级到 probation 尾（教科书 SLRU 的 demotion）。

        只改区归属，不释放 KV、不删节点（INV-2/INV-3）。被降级的节点若再次被命中，
        会按它当时的频数重新分档 —— 因此这是「让出保护」而不是「判死刑」。
        """
        budget = self._protected_budget()
        if budget is None or self._tokens[REGION_PROTECTED] <= budget:
            return
        guard = len(self.regions[REGION_PROTECTED]) + 1
        while self._tokens[REGION_PROTECTED] > budget and guard > 0:
            guard -= 1
            if len(self.regions[REGION_PROTECTED]) <= 1:
                break
            head = self._head(REGION_PROTECTED)
            if head is None:
                break
            before = self._tokens[REGION_PROTECTED]
            self._index_remove(head)
            self._push(REGION_PROBATION, head, at_head=False)
            self.stats["budget_demote"] += 1
            if self._tokens[REGION_PROTECTED] >= before:
                break  # token 数为 0 的节点无法收敛，放弃

    def _reject(self, node) -> None:
        """准入拒绝：移出索引、清结构证据、标签降为 QTRANS、记账。

        不释放 KV、不删节点（INV-3）。降级后 ``is_one_shot()`` 为真，
        现有的 one-shot 带会在下一轮把它收割掉，父节点（可能是模板前缀）
        随后经 ``on_became_leaf`` 进入 protected。
        """
        self._index_remove(node)
        node.struct_ever_branch = False
        node.queue_tag = SimpleHybridQueueTag.QTRANS
        self.stats["reject"] += 1

    # ---------------------------------------------------------------- 事件

    def _rid_bits(self, rid) -> int:
        if rid != self._last_rid:
            text = rid if isinstance(rid, str) else str(rid)
            h = zlib.crc32(text.encode("utf-8", "replace")) & 0xFFFFFFFF
            h = (h * 0x9E3779B1) & 0xFFFFFFFF
            bits = 0
            for i in range(4):
                bits |= 1 << ((h >> (i * 6)) & 63)
            self._last_rid = rid
            self._last_bits = bits
        return self._last_bits

    def on_hit(self, node, rid=None, session_id=None) -> None:
        """命中路径。维护两套频数口径：

        * ``share_count`` / ``rid_bloom``      —— 不同**请求**数（诊断口径，INV-6 保留）
        * ``share_sessions`` / ``session_bloom`` —— 不同**会话**数（A：结构复用口径）

        判档/成员/准入一律用后者（``share_metric="rid"`` 可切回旧口径做消融）。
        ``rid is None`` 表示无法去重的访问（chunked 复查）：只刷新顺序，不计数。
        没有 ``session_id`` 的老调用方退化为「rid 即会话」，行为与旧版一致。
        """
        if not self.enabled or node is None or node.parent is None:
            return

        if rid is None and session_id is None:
            self._touch(node)
            return

        # aging 时钟按「不同请求」计一次（不是每个节点一次）：请求的首个命中推进纪元，
        # 否则一条 50 段的路径会一次吃掉 50 个事件，aging 频率与路径长度耦合。
        new_request = rid is not None and rid != self._last_rid
        if new_request:
            self._note_event()

        changed = False
        # 口径一：不同 rid（诊断）
        if rid is not None:
            bits = self._rid_bits(rid)
            if (getattr(node, "rid_bloom", 0) & bits) == bits:
                self.stats["dedup_hits"] += 1
            else:
                node.rid_bloom = getattr(node, "rid_bloom", 0) | bits
                node.share_count = min(_MAX_SHARE, getattr(node, "share_count", 0) + 1)
                changed = True

        # 口径二：不同 session（A）
        sid = session_id if session_id is not None else rid
        if sid is not None:
            sbits = self._session_bits(sid)
            if (getattr(node, "session_bloom", 0) & sbits) != sbits:
                node.session_bloom = getattr(node, "session_bloom", 0) | sbits
                node.share_sessions = min(
                    _MAX_SHARE, getattr(node, "share_sessions", 0) + 1
                )
                changed = True
            else:
                self.stats["dedup_sessions"] += 1

        if not changed:
            self._touch(node)
            return

        region = node.struct_region
        if region == REGION_WINDOW:
            # window 是 FIFO：没拿到新证据就不动顺序
            if self._freq_of(node) >= _MIN_SHARE:
                self._place(node, origin="promote")
        elif region in (REGION_PROBATION, REGION_PROTECTED):
            self._place(node, origin="promote")  # 可能升区，同时刷新区尾
        elif region == REGION_NONE and self._freq_of(node) >= _MIN_SHARE:
            # 刚拿到共享证据：可能因此成为主干成员
            self._evaluate(node, origin="promote")

    def on_children_added(self, node) -> None:
        """插入使某节点分叉：打上黏性结构标记，并按成员规则重新入队。"""
        if not self.enabled or node is None:
            return
        if self._is_branch(node):
            node.struct_ever_branch = True
        self._evaluate(node, origin="promote")

    def on_became_leaf(self, node) -> None:
        """节点刚成为可淘汰叶子（子链被删光/被上锁解除）：按继承表落区。

        C：若该叶子已经空闲超过 ``stale_grace_s``，不再「继承式保护」，
        而是先落 window 重新证明自己（否则一个早就没人用的模板会靠历史频数
        直接进 protected，最后被 forced 抽干 —— 实测 protected_waste≈1.0）。
        """
        if not self.enabled or node is None or node.parent is None:
            return
        if not self.inherit_share and self._freq_of(node) < self.k:
            # 消融开关：不继承频数时重新观察
            node.share_count = 0
            node.rid_bloom = 0
            node.share_sessions = 0
            node.session_bloom = 0
        force_band = None
        if self.stale_grace_s > 0.0:
            idle = time.monotonic() - float(getattr(node, "last_access_time", 0.0) or 0.0)
            if idle > self.stale_grace_s:
                force_band = REGION_WINDOW
                self.stats["stale_deflect"] += 1
        self._place(node, origin="inherit", force_band=force_band)

    def on_no_longer_leaf(self, node) -> None:
        """节点长出子节点或被上锁：不可淘汰，挂起但不丢频数。"""
        if not self.enabled or node is None:
            return
        if node.struct_region in _REGION_ORDER:
            self._park(node)

    def on_delete(self, node) -> None:
        if not self.enabled or node is None:
            return
        self._index_remove(node)
        self._note_event()

    def on_split(self, prefix_half, suffix_half) -> None:
        """分裂：前缀半段按 M3 重新判定；后缀半段是同一个对象，索引自然有效。

        分裂只改两个对象的 ``value`` 长度（总和不变），所以这里整体重算 token 计数。
        """
        if not self.enabled:
            return
        self._recount_tokens()
        self._evaluate(prefix_half, origin="inherit")

    # ---------------------------------------------------------------- aging

    def _note_event(self) -> None:
        if self._aging:
            return
        self._events += 1
        if self._events >= self.epoch_events:
            self._events = 0
            self._age()

    def _age(self) -> None:
        """频数减半 + bloom 清空 + 按 band 重排。

        清 bloom 是必需的：否则「很久以前访问过这个模板」的 rid 会被永久判为已计数，
        新模板再也涨不上去。``hit_count`` 不动（其他策略与指标依赖它）。
        """
        self._aging = True
        try:
            ordered = list(self.parked.values())
            for region in _REGION_ORDER:
                ordered.extend(self.regions[region].values())
            self.parked.clear()
            for table in self.regions.values():
                table.clear()
            self._parked_tokens = 0
            for region in self._tokens:
                self._tokens[region] = 0
            for node in ordered:
                node.struct_region = REGION_NONE
                node.share_count >>= 1
                node.rid_bloom = 0
                node.share_sessions >>= 1
                node.session_bloom = 0
                if not self._is_branch(node) and self._freq_of(node) < _MIN_SHARE:
                    # 结构证据与共享证据都没了：退队（清黏性标记）
                    self._retire(node)
                    continue
                self._place(node, origin="inherit")
            self._recount_tokens()
            self.stats["ages"] += 1
        finally:
            self._aging = False

    # ---------------------------------------------------------------- 淘汰

    def owns(self, node) -> bool:
        """O(1)：该节点当前是否归 struct 队列管。"""
        return (
            self.enabled
            and node is not None
            and getattr(node, "struct_region", REGION_NONE) != REGION_NONE
        )

    def eviction_candidate(self):
        """队内下一个该出的成员；None 表示队内暂时没有可淘汰成员（上层必须回退）。

        只给 ``StructTinyLruPolicy.pick()`` 用：本方法不做跨队列比较，
        返回的是纯队内顺序（window → probation → protected 的区头）。
        """
        if not self.enabled:
            return None
        for region in _REGION_ORDER:
            node = self._head(region)
            if node is not None:
                if region == REGION_PROTECTED:
                    self.stats["forced"] += 1
                return node
        return None

    # ------------------------------------------------------------ 生命周期

    def reset(self) -> None:
        """清空三区与挂起区、复位 aging 时钟（缓存 reset / 进程复用）。"""
        for table in self.regions.values():
            for node in table.values():
                node.struct_region = REGION_NONE
            table.clear()
        for node in self.parked.values():
            node.struct_region = REGION_NONE
        self.parked.clear()
        self._events = 0
        self._last_rid = None
        self._last_bits = 0

    # ---------------------------------------------------------------- 观测

    def snapshot(self) -> Dict[str, object]:
        return {
            "window": len(self.regions[REGION_WINDOW]),
            "probation": len(self.regions[REGION_PROBATION]),
            "protected": len(self.regions[REGION_PROTECTED]),
            "parked": len(self.parked),
            "window_tokens": self._tokens[REGION_WINDOW],
            "probation_tokens": self._tokens[REGION_PROBATION],
            "protected_tokens": self._tokens[REGION_PROTECTED],
            "parked_tokens": self._parked_tokens,
            "protected_budget": self._protected_budget(),
            "share_metric": self.share_metric,
            "protected_budget_frac": self.protected_budget_frac,
            "stale_grace_s": self.stale_grace_s,
            "epoch_events": self.epoch_events,
            **self.stats,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        snap = self.snapshot()
        return (
            "[struct wtinylru] w={window} p={probation} r={protected} "
            "parked={parked} admit={admit} reject={reject} ages={ages}".format(**snap)
        )
