"""请求驱动：把 trace 按时刻喂进生产缓存。

单轮请求的六步，顺序对齐线上（``schedule_batch`` 的 match → 挂锁 → 分配/写入 →
release），差别只在「顺序模型没有并发」：

1. ``clock.advance_to(t)``          —— 时间只由 trace 推进；
2. ``match_prefix``                 —— 内含 ``observe_request``（分类证据）；
3. ``inc_lock_ref(匹配节点)``        —— 在飞请求保护已匹配前缀不被自己淘汰；
4. 短缺即 ``evict`` → ``alloc``      —— 对齐 allocator 的 shortfall 回调；
5. ``_adopt(prompt+output)``        —— 内含 ``touch`` + ``decode_start`` + 分裂；
6. ``dec_lock_ref``                 —— 请求结束。

为什么第 3 步必须有：没有锁，第 4 步可能把刚命中的前缀淘汰掉，再在插入时重建，
命中率与淘汰来源都会偏离线上（线上在飞请求的节点不在 ``evictable_leaves`` 里）。
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from sim import bootstrap
from sim.allocator import OutOfTokens, TokenAllocator, tree_tokens
from sim.clock import VirtualClock
from sim.metrics import RequestRecord, SimResult, summarize
from sim.req import FakeReqToTokenPool, SimReq
from sim.trace import Session, Turn, max_total_len

DEFAULT_WEIGHTS = {"struct": 0.2, "agentic": 0.5, "chat": 0.3}


@dataclass
class SimConfig:
    """一次 run 的全部可调项。权重比例等价（只影响仲裁顺序，不参与队内排序）。"""

    capacity: int = 4096
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    seed: int = 0
    page_size: int = 1
    eviction_policy: str = "heter_cache"
    idle_ttl: float = 3600.0


class Simulator:
    """按 trace 时刻步进的仿真驱动。

    用法::

        sim = Simulator(SimConfig(capacity=4096))
        result = sim.run(sessions)
        print(result.render())

    需要手工编排时钟（例如复刻某个并发时序）时，先 ``prepare(width)``
    再逐条 ``request(turn, sid)``。
    """

    def __init__(self, config: Optional[SimConfig] = None) -> None:
        self.config = config or SimConfig()
        self.sym = bootstrap.load()
        self.clock: Optional[VirtualClock] = None
        self.alloc: Optional[TokenAllocator] = None
        self.req_pool: Optional[FakeReqToTokenPool] = None
        self.cache = None
        self.records: List[RequestRecord] = []
        self.tasks: Dict[str, str] = {}
        self._rid_seq = 0

    # ------------------------------------------------------------------ 装配
    def prepare(self, width: int) -> None:
        """按 req_to_token 宽度装配一次干净的 run（时钟/分配器/缓存全部重建）。"""

        cfg = self.config
        if width <= 0:
            raise ValueError(f"req_to_token 宽度必须为正，got {width}")

        # 会话账本是进程级单例：先销毁再建缓存，避免上一轮的分类证据串味。
        self.sym["SessionRegistry"].reset_global()

        self.clock = VirtualClock(0.0)
        self.alloc = TokenAllocator(cfg.capacity)
        self.req_pool = FakeReqToTokenPool(width)
        self.cache = self.sym["HeterCacheRadixCache"].create_simulated(
            mock_allocator=self.alloc,
            mock_req_pool=self.req_pool,
            page_size=cfg.page_size,
            eviction_policy=cfg.eviction_policy,
            eviction_policy_config={
                "weight": {"weights": dict(cfg.weights), "seed": cfg.seed}
            },
            now=self.clock,
        )
        # 会话账本的空闲回收阈值（生产默认 3600s，虚拟时间下同样按秒算）
        self.cache.session_registry.idle_ttl = cfg.idle_ttl
        self.records = []
        self.tasks = {}
        self._rid_seq = 0

    # ------------------------------------------------------------------ 单请求
    def request(
        self,
        turn: Turn,
        session_id: str,
        rid: Optional[str] = None,
        task_name: str = "",
    ) -> RequestRecord:
        """重放一轮请求并返回它的记录。

        Args:
            turn: 本轮内容（``turn.t`` 是绝对时刻）。
            session_id: 会话 id（同时用于分类与跨会话共享判定）。
            rid: 请求 id；缺省用 ``session_id#序号``。必须每请求唯一
                （``CacheTelemetry.on_hit`` 按 rid 去重）。
            task_name: 负载名，仅用于 per-task 指标（对齐 emitter 的 ``per_task``）。
        """

        if self.cache is None:
            raise RuntimeError("先 prepare(width) 或直接用 run(sessions)")

        sym = self.sym
        self.clock.advance_to(turn.t)
        self._rid_seq += 1
        rid = rid or f"{session_id}#{self._rid_seq}"

        req = SimReq(rid, session_id, turn.prompt, turn.out, tool_call=turn.tool_call)
        # 仿真不建模请求自己的 kv 行（顺序模型里它只是树的引用）：声明整条已由树
        # 持有，于是 _adopt 收尾的「释放重复槽位」段为空（len=0），池账只跟树走。
        req.kv.cache_protected_len = len(req.fill_ids())

        # 2) 命中（observe_request 在 match_prefix 内部）
        key = sym["RadixKey"](array("q", turn.prompt)).page_aligned(self.cache.page_size)
        matched = self.cache.match_prefix(
            sym["MatchPrefixParams"](key=key, req=req)
        )
        hit = int(matched.device_indices.numel())

        # 本次会写入树的完整 key（prompt + response），与 _adopt 的计算口径一致
        full = sym["RadixKey"](array("q", req.fill_ids())).page_aligned(
            self.cache.page_size
        )
        total = len(full)
        need = total - hit

        # 3) 挂锁 -> 4) 短缺即淘汰 -> 预留
        #
        # 预留量取 need（= 页对齐后的 prompt+response - 命中），因为命中段的槽位
        # 本来就在树里、被同一个请求引用，不需要再占一份。
        node = matched.last_device_node
        self.cache.inc_lock_ref(node)
        evicted = 0
        try:
            shortfall = need - self.alloc.available_size()
            if shortfall > 0:
                evicted = self.cache.evict(
                    sym["EvictParams"](num_tokens=int(shortfall))
                ).num_tokens_evicted
            try:
                self.alloc.alloc(need)
            except OutOfTokens as exc:
                raise OutOfTokens(
                    f"{exc}（rid={rid} total={total} hit={hit} need={need}）——"
                    "单请求超出池容量；顺序模型不模拟 retraction"
                    "（生产会把请求拆 chunk 或重算），请调大 capacity 或截断 trace"
                ) from exc
        finally:
            self.cache.dec_lock_ref(node)

        # 5) 写入树（含 touch / decode_start / prompt 边界分裂）
        tree_before = tree_tokens(self.cache)
        self.cache._adopt(req, req.fill_ids(), split_prompt=True)
        grew = tree_tokens(self.cache) - tree_before
        # 预留量是「全序列 - prompt 命中」，但 response 段也可能已经在树里
        # （同一 prompt 的重放会命中整条），差额就是 _adopt 本该释放的重复槽位；
        # 因为仿真不建模请求行，这里显式归还，保持池账 == 树内 token。
        if grew < need:
            self.alloc.release(need - grew)
        elif grew > need:
            raise AssertionError(
                f"树增长超过预留：grew={grew} > need={need} (rid={rid})"
            )

        # 6) 不变量：槽位账 == 树内 token，任何漂移都是记账 bug
        tree = tree_tokens(self.cache)
        if self.alloc.used != tree:
            raise AssertionError(
                f"池账不平：alloc.used={self.alloc.used} tree={tree} "
                f"(rid={rid}, need={need}, hit={hit}, evicted={evicted})"
            )

        state = self.cache.session_registry.get(session_id)
        record = RequestRecord(
            t=turn.t,
            session_id=session_id,
            rid=rid,
            prompt_len=len(req.origin_input_ids),
            out_len=len(req.output_ids),
            hit=hit,
            need=need,
            evicted=int(evicted),
            used=self.alloc.used,
            session_type=state.session_type.value if state is not None else "none",
            task_name=task_name,
        )
        self.records.append(record)
        return record

    # ------------------------------------------------------------------ 整条 trace
    def run(self, sessions: Iterable[Session]) -> SimResult:
        """按时刻归并全部会话的轮次，逐条重放。"""

        sessions = list(sessions)
        seen = set()
        for session in sessions:
            if session.session_id in seen:
                raise ValueError(f"session_id 重复：{session.session_id}")
            seen.add(session.session_id)

        width = max_total_len(sessions)
        if width == 0:
            return SimResult(records=[], summary={})
        self.prepare(width)
        self.tasks = {session.session_id: session.task_name for session in sessions}

        # (时刻, session_id, 轮次下标) 全局归并；同刻按 session_id 稳定排序
        events = sorted(
            (turn.t, session.session_id, idx)
            for session in sessions
            for idx, turn in enumerate(session.turns)
        )
        turns = {session.session_id: session.turns for session in sessions}
        for _t, sid, idx in events:
            self.request(turns[sid][idx], sid, task_name=self.tasks[sid])

        return SimResult(records=list(self.records), summary=self.summary())

    # ------------------------------------------------------------------ 汇总
    def summary(self) -> Dict:
        strategy = self.cache.eviction_strategy
        return summarize(
            self.records,
            capacity=self.config.capacity,
            final_used=self.alloc.used,
            sessions=len({r.session_id for r in self.records}),
            profiler=getattr(strategy, "profiler", None),
            occupancy=getattr(self.cache, "occupancy", None),
        )


def run_synthetic(config: Optional[SimConfig] = None, **trace_kwargs) -> SimResult:
    """冒烟用：造一批合成会话并跑一遍。``trace_kwargs`` 透传给 generator。"""

    from sim.trace import synthetic_sessions

    sim = Simulator(config)
    return sim.run(synthetic_sessions(**trace_kwargs))
