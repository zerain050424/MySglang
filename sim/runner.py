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
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from sim import bootstrap
from sim.allocator import OutOfTokens, TokenAllocator, tree_tokens
from sim.clock import VirtualClock
from sim.metrics import RequestRecord, SimResult, summarize
from sim.req import OUTPUT_SENTINEL_BASE, FakeReqToTokenPool, SimReq
from sim.trace import Session, Turn, max_total_len


@dataclass
class Pending:
    """已到点、已占槽、已挂锁，还没落树的一轮请求（见 ``Simulator.begin``）。

    记录在 :meth:`Simulator.end` 里生成，因此 ``Simulator.records`` 是**完成序**
    ——与 ``bench_heap`` 的 ``stats.rows``（响应读完后追加）一致。
    """

    req: SimReq
    node: object
    need: int
    t: float
    session_id: str
    rid: str
    prompt_len: int
    out_len: int
    hit: int
    evicted: int
    session_type: str
    task_name: str
    skipped: bool = False


@dataclass
class SimConfig:
    """一次 run 的全部可调项。VD 参数只影响跨队列顺序，不参与队内排序。"""

    capacity: int = 4096
    value_horizon_s: float = 120.0
    value_smoothing_k: float = 4.0
    value_floor_tokens: float = 0.0
    seed: int = 0
    page_size: int = 1
    eviction_policy: str = "heter_cache"
    # OPT 上界专用：离线 trace 生成的 `{block_hash: [使用时刻]}`。
    # 只在 eviction_policy="opt" 时读；缺失即报错，不做静默退化。
    opt_use_map: Optional[str] = None
    idle_ttl: float = 3600.0
    # 单请求无法在池里放下时的行为：
    #   "error" —— 抛 OutOfTokens（默认；逼调用方直面 trace/容量不匹配）；
    #   "skip"  —— 跳过落树、计数后继续（生产用 chunked prefill / retraction 伺候）。
    oversize: str = "error"


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
        self.inflight_tokens = 0
        self.inflight_token_peak = 0
        self.duplicate_insert_tokens = 0
        self.skipped_requests = 0

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
        # 策略参数按族分派：HeterCache 读 VD 参数，OPT 只读 use map。
        if cfg.eviction_policy == "heter_cache":
            policy_config = {
                "value": {
                    "horizon_s": cfg.value_horizon_s,
                    "smoothing_k": cfg.value_smoothing_k,
                    "floor_tokens": cfg.value_floor_tokens,
                    "seed": cfg.seed,
                }
            }
        elif cfg.eviction_policy == "opt":
            if not cfg.opt_use_map:
                raise ValueError(
                    "eviction_policy='opt' requires SimConfig.opt_use_map "
                    "(see sim/opt_map.build_use_map)"
                )
            policy_config = {
                "use_map_path": cfg.opt_use_map,
                "block_size": cfg.page_size,
            }
        else:
            policy_config = None

        self.cache = self.sym["HeterCacheRadixCache"].create_simulated(
            mock_allocator=self.alloc,
            mock_req_pool=self.req_pool,
            page_size=cfg.page_size,
            eviction_policy=cfg.eviction_policy,
            eviction_policy_config=policy_config,
            now=self.clock,
        )
        # 会话账本的空闲回收阈值（生产默认 3600s，虚拟时间下同样按秒算）。
        # 非 HeterCache 策略不装账本，跳过。
        if getattr(self.cache, "session_registry", None) is not None:
            self.cache.session_registry.idle_ttl = cfg.idle_ttl
        self.records = []
        self.tasks = {}
        self._rid_seq = 0
        self.inflight_tokens = 0
        self.inflight_token_peak = 0
        self.duplicate_insert_tokens = 0
        self.skipped_requests = 0

    # ------------------------------------------------------------------ 单请求
    def begin(
        self,
        turn: Turn,
        session_id: str,
        rid: Optional[str] = None,
        task_name: str = "",
    ) -> "Pending":
        """请求到点：match → 挂锁 → 短缺即淘汰 → 预留；写树留到 :meth:`end`。

        并发窗口里这一步和 ``end()`` 之间可能隔着别的请求，所以两件事必须分开：

        * **写树在 end()**：生产里 KV 是请求结束时 ``cache_finished_req`` 才落树
          （chunked prefill 例外）。若在 begin 写，同一时刻到点的后一条请求会命中
          前一条刚写的前缀——竞态（racing）就消失了，命中率会虚高。
        * **锁持到 end()**：生产里 ``inc_lock_ref`` 从 match 一直持到请求结束，
          期间命中的前缀不可淘汰；提前放锁会让并发场景的占用与淘汰偏松。

        Args:
            turn: 本轮内容（``turn.t`` 是绝对时刻）。
            session_id: 会话 id（同时用于分类与跨会话共享判定）。
            rid: 请求 id；缺省 ``session_id#序号``，必须每请求唯一。
            task_name: 负载名，仅用于 per-task 指标。

        Returns:
            :class:`Pending`；必须交给 :meth:`end` 收尾（否则锁与槽位都漏）。
        """

        if self.cache is None:
            raise RuntimeError("先 prepare(width) 或直接用 run(sessions)")

        sym = self.sym
        self.clock.advance_to(turn.t)
        self._rid_seq += 1
        rid = rid or f"{session_id}#{self._rid_seq}"

        req = SimReq(
            rid,
            session_id,
            turn.prompt,
            turn.out,
            tool_call=turn.tool_call,
            output_token=OUTPUT_SENTINEL_BASE + self._rid_seq,
        )
        # 仿真不建模请求自己的 kv 行（它只是树的引用）：声明整条已由树持有，
        # 于是 _adopt 收尾的「释放重复槽位」段为空，池账只跟树 + 在飞走。
        req.kv.cache_protected_len = len(req.fill_ids())

        # 1) 命中（observe_request 在 match_prefix 内部）
        key = sym["RadixKey"](array("q", turn.prompt)).page_aligned(self.cache.page_size)
        matched = self.cache.match_prefix(sym["MatchPrefixParams"](key=key, req=req))
        hit = int(matched.device_indices.numel())

        # 本次会写入树的完整 key（prompt + response），与 _adopt 的计算口径一致
        full = sym["RadixKey"](array("q", req.fill_ids())).page_aligned(
            self.cache.page_size
        )
        total = len(full)
        need = total - hit

        # 2) 挂锁 -> 3) 短缺即淘汰 -> 预留
        node = matched.last_device_node
        state = self.cache.session_registry.get(session_id)
        session_type = state.session_type.value if state is not None else "none"

        def _pending(skipped: bool, evicted: int) -> Pending:
            return Pending(
                req=req,
                node=node,
                need=need,
                t=turn.t,
                session_id=session_id,
                rid=rid,
                prompt_len=len(req.origin_input_ids),
                out_len=len(req.output_ids),
                hit=hit,
                evicted=int(evicted),
                session_type=session_type,
                task_name=task_name,
                skipped=skipped,
            )

        if need > self.alloc.size:
            # 整条请求比池子还大：生产会拆 chunk / retraction，仿真装不下。
            if self.config.oversize != "skip":
                raise OutOfTokens(
                    f"单请求 {total} token 超过池容量 {self.alloc.size}"
                    f"（rid={rid} hit={hit} need={need}）；"
                    "生产会拆 chunk 重算，仿真请显式选择："
                    'SimConfig(oversize="skip") 跳过并计数，或调大 capacity'
                )
            self.skipped_requests += 1
            self._assert_balanced(where=f"skip({rid})")
            return _pending(True, 0)

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
                # 淘汰也腾不出来（例如槽位全被别的在飞请求锁着）
                if self.config.oversize != "skip":
                    raise OutOfTokens(
                        f"{exc}（rid={rid} total={total} hit={hit} need={need}）——"
                        "仿真不模拟 retraction（生产会拆分/重算），"
                        "请调大 capacity 或 SimConfig(oversize=\"skip\")"
                    ) from exc
                self.cache.dec_lock_ref(node)
                self.skipped_requests += 1
                self._assert_balanced(where=f"skip({rid})")
                return _pending(True, evicted)
        except BaseException:
            self.cache.dec_lock_ref(node)  # 失败也要放锁，否则后续 evict 全被挡住
            raise

        # 槽位已占、树里还没有：这段就是「在飞但未落树」的 KV
        self.inflight_tokens += need
        self.inflight_token_peak = max(self.inflight_token_peak, self.inflight_tokens)

        self._assert_balanced(where=f"begin({rid})")
        return _pending(False, evicted)

    def end(self, pending: "Pending") -> RequestRecord:
        """请求完成：写树 → 归还重复槽位 → 放锁（跳过的请求只出记录）。"""

        if not pending.skipped:
            req = pending.req
            need = pending.need

            # 4) 写入树（含 touch / decode_start / prompt 边界分裂）
            tree_before = tree_tokens(self.cache)
            self.cache._adopt(req, req.fill_ids(), split_prompt=True)
            grew = tree_tokens(self.cache) - tree_before

            self.inflight_tokens -= need
            if grew < need:
                # 预留量是「全序列 - prompt 命中」，但等我们落树期间可能已有别的
                # 请求先写入了同一段前缀（竞态），差额在这里归还并记为重复写入。
                self.alloc.release(need - grew)
                self.duplicate_insert_tokens += need - grew
            elif grew > need:
                raise AssertionError(
                    f"树增长超过预留：grew={grew} > need={need} (rid={pending.rid})"
                )

            # 5) 放锁（生产：cache_finished_req 之后 unpin）
            self.cache.dec_lock_ref(pending.node)
            self._assert_balanced(where=f"end({pending.rid})")

        record = RequestRecord(
            t=pending.t,
            session_id=pending.session_id,
            rid=pending.rid,
            prompt_len=pending.prompt_len,
            out_len=pending.out_len,
            hit=pending.hit,
            need=pending.need,
            evicted=pending.evicted,
            used=self.alloc.used,
            session_type=pending.session_type,
            task_name=pending.task_name,
            skipped=pending.skipped,
        )
        self.records.append(record)  # 完成序，对齐 emitter 的 stats.rows
        return record

    def request(
        self,
        turn: Turn,
        session_id: str,
        rid: Optional[str] = None,
        task_name: str = "",
    ) -> RequestRecord:
        """begin + 立即 end（顺序模型走这条；并发由 Emitter 分开调）。"""

        return self.end(self.begin(turn, session_id, rid, task_name))

    def _assert_balanced(self, where: str = "") -> None:
        """池账 == 树内 token + 在飞未落树的 token。"""

        tree = tree_tokens(self.cache)
        expect = tree + self.inflight_tokens
        if self.alloc.used != expect:
            raise AssertionError(
                f"池账不平（{where}）：alloc.used={self.alloc.used} "
                f"tree={tree} inflight={self.inflight_tokens}"
            )

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


    # ------------------------------------------------------------------ 发射器口径
    def run_scheduled(
        self,
        sessions: Iterable[Session],
        specs: Iterable,
        *,
        service=None,
        arrival_scale: float = 1.0,
        wait_scale: float = 1.0,
        max_inflight: int = 0,
    ) -> SimResult:
        """按 ``bench_heap`` 的发射规则跑（闭环轮次 + 服务时长模型）。

        与 ``run()`` 的区别：``run()`` 吃的是「已有绝对时刻」的 trace（开环）；
        本方法自己根据「完成时刻 + lognormal」生成时刻（闭环），并推进时钟。

        Args:
            max_inflight: 同时在飞上限；0 = 不限（emitter 默认），1 = 单通道。
        """

        from sim.schedule import Emitter

        return Emitter(self, sessions, specs, service).run(
            arrival_scale=arrival_scale,
            wait_scale=wait_scale,
            max_inflight=max_inflight,
        )


def run_synthetic(config: Optional[SimConfig] = None, **trace_kwargs) -> SimResult:
    """冒烟用：造一批合成会话并跑一遍。``trace_kwargs`` 透传给 generator。"""

    from sim.trace import synthetic_sessions

    sim = Simulator(config)
    return sim.run(synthetic_sessions(**trace_kwargs))
