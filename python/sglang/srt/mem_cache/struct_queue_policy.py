"""Q_struct 队列内部的 W-TinyLRU 队内淘汰策略。

设计文档：python/sglang/srt/mem_cache/STRUCT_QUEUE_WTINYLRU.md
不释放任何 KV：
  1. 某个节点是否还在 struct 队列里（成员判定 M3 + 退队）；
  2. 若在，队内下一个该出的是谁（window → probation → protected）。

三条需求的落点：
  * 按频数驻留      -> share_count（不同 rid 数）+ 三区分档；
  * 单次文档询问被删 -> 准入失败 -> _reject() 降级为 QTRANS，交给现有 one-shot 带处决；
  * 高频共享前缀驻留 -> 成员判定 M3 + 降级继承 + 退队只在 share<2 时发生。
"""

from __future__ import annotations

import zlib
from collections import OrderedDict
from typing import Dict, Optional

from sglang.srt.mem_cache.uni_utils import SimpleHybridQueueTag

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

# share_count 是 u8，饱和即可：K 只有个位数，超出的部分没有信息量
_MAX_SHARE = 255

# 低于这个值的节点只能靠分叉成为成员：share>=2 才谈得上被共享
_MIN_SHARE = 2


class StructQueuePolicy:
    """仅 Q_struct 队列内部的 W-TinyLRU（window + SLRU + 频数准入）。"""

    def __init__(
        self,
        enabled: bool = True,
        k: int = 3,
        window_entries: int = 64, ## window队列条目上限
        epoch_events: int = 8192, ## 到达后进行一次Aging
        admission_tie: str = "reject",
        admission_metric: str = "share",
        inherit_share: bool = True,
    ):
        self.enabled = enabled
        self.k = max(2, int(k))
        self.window_entries = max(1, int(window_entries))
        self.epoch_events = max(1, int(epoch_events))
        self.admission_tie = admission_tie
        self.admission_metric = admission_metric
        self.inherit_share = bool(inherit_share)

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

        self.stats = {
            "admit": 0,
            "reject": 0,
            "retire": 0,
            "park": 0,
            "ages": 0,
            "dedup_hits": 0,
            "forced": 0,
        }

    # ------------------------------------------------------------------ 判定

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
            if cur.share_count < _MIN_SHARE:
                return False
            cur = cur.parent
        return False

    # ------------------------------------------------------------ 索引维护

    def _index_remove(self, node) -> None:
        region = node.struct_region
        if region in self.regions:
            self.regions[region].pop(node.id, None)
        elif region == REGION_PARKED:
            self.parked.pop(node.id, None)
        node.struct_region = REGION_NONE

    def _push(self, region: int, node, at_head: bool = False) -> None:
        node.struct_region = region
        if region == REGION_PARKED:
            self.parked[node.id] = node
            return
        table = self.regions[region]
        table[node.id] = node
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
        share = node.share_count
        if share >= self.k:
            return REGION_PROTECTED
        if share == 2:
            return REGION_PROBATION
        return REGION_WINDOW

    def _place(self, node, origin: str = "promote") -> None:
        """把成员放进三区；不可淘汰则挂起。``origin`` 决定区头/区尾与继承策略。"""
        if not (self.enabled and self._is_member(node)):
            self._leave_struct(node)
            return
        if not self._is_evictable_leaf(node):
            self._park(node)
            return

        self._index_remove(node)
        band = self._band_of(node)
        # §3.4 继承表：首次入区的 probation 成员进区头（老成员优先被比较），
        # 其余一律进区尾。
        at_head = origin == "inherit" and band == REGION_PROBATION
        self._push(band, node, at_head=at_head)
        self._note_event()
        if band == REGION_WINDOW:
            self._drain_window()

    def _evaluate(self, node, origin: str = "promote") -> None:
        if not self.enabled or node is None:
            return
        self._place(node, origin)

    # ---------------------------------------------------------------- 准入

    def _share_of(self, node):
        if self.admission_metric == "share_per_token":
            tokens = len(node.value) if node.value is not None else 0
            return (node.share_count, node.share_count / max(1, tokens))
        return (node.share_count,)

    def _head(self, region: int):
        table = self.regions[region]
        while table:
            node_id = next(iter(table))
            node = table[node_id]
            if self._is_evictable_leaf(node):
                return node
            # 惰性清理：索引与可淘汰集合应该始终一致（INV-2），这里只是兜底
            table.pop(node_id, None)
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
            elif cand.share_count >= _MIN_SHARE:
                # 已经是「被两个请求共享」的证据，不能再退化成一次性内容，
                # 否则会破坏 INV-2（成员不在任何索引里）。
                self._promote(cand)
            else:
                self._reject(cand)

    def _promote(self, node) -> None:
        band = self._band_of(node)
        self._index_remove(node)
        self._push(band, node, at_head=False)
        self.stats["admit"] += 1
        self._note_event()
        if band == REGION_WINDOW:
            self._drain_window()

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

    def on_hit(self, node, rid=None) -> None:
        """命中路径。``rid is None`` 表示无法去重的访问（如 chunked 复查），只刷新顺序。

        频数口径是**不同请求数**：同一 rid 重复命中只算一次。chunked 分块的自引用
        命中不会拿到 rid，因此不会被计入——这正是「单次文档询问」被误判为高频的原因。
        """
        if not self.enabled or node is None or node.parent is None:
            return

        if rid is None:
            self._touch(node)
            return

        # aging 时钟按「不同请求」计一次（不是每个节点一次）：请求的首个命中推进纪元，
        # 否则一条 50 段的路径会一次吃掉 50 个事件，aging 频率与路径长度耦合。
        new_request = rid != self._last_rid
        bits = self._rid_bits(rid)
        if new_request:
            self._note_event()
        if (node.rid_bloom & bits) == bits:
            self.stats["dedup_hits"] += 1
            self._touch(node)
            return

        node.rid_bloom |= bits
        node.share_count = min(_MAX_SHARE, node.share_count + 1)

        region = node.struct_region
        if region == REGION_WINDOW:
            # window 是 FIFO：没拿到新证据就不动顺序
            if node.share_count >= _MIN_SHARE:
                self._place(node, origin="promote")
        elif region in (REGION_PROBATION, REGION_PROTECTED):
            self._place(node, origin="promote")  # 可能升区，同时刷新区尾
        elif region == REGION_NONE and node.share_count >= _MIN_SHARE:
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
        """节点刚成为可淘汰叶子（子链被删光/被上锁解除）：按继承表落区。"""
        if not self.enabled or node is None or node.parent is None:
            return
        if not self.inherit_share and node.share_count < self.k:
            # 消融开关：不继承频数时重新观察
            node.share_count = 0
            node.rid_bloom = 0
        self._place(node, origin="inherit")

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
        """分裂：前缀半段按 M3 重新判定；后缀半段是同一个对象，索引自然有效。"""
        if not self.enabled:
            return
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
            for node in ordered:
                node.struct_region = REGION_NONE
                node.share_count >>= 1
                node.rid_bloom = 0
                if not self._is_branch(node) and node.share_count < _MIN_SHARE:
                    # 结构证据与共享证据都没了：退队（清黏性标记）
                    self._retire(node)
                    continue
                self._place(node, origin="inherit")
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

    def victim(self):
        """下一个该出的成员；None 表示队内暂时没有可淘汰成员（上层必须回退）。"""
        if not self.enabled:
            return None
        for region in _REGION_ORDER:
            node = self._head(region)
            if node is not None:
                if region == REGION_PROTECTED:
                    self.stats["forced"] += 1
                return node
        return None

    def reset(self) -> None:
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
            "window_tokens": sum(
                len(n.value) for n in self.regions[REGION_WINDOW].values() if n.value is not None
            ),
            "protected_tokens": sum(
                len(n.value) for n in self.regions[REGION_PROTECTED].values() if n.value is not None
            ),
            "epoch_events": self.epoch_events,
            **self.stats,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        snap = self.snapshot()
        return (
            "[struct wtinylru] w={window} p={probation} r={protected} "
            "parked={parked} admit={admit} reject={reject} ages={ages}".format(**snap)
        )
