"""单请求缓存生命周期原语与池账守恒维护。"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import msgspec

from sim.core import bootstrap
from sim.core.allocator import OutOfTokens, TokenAllocator, tree_tokens
from sim.core.clock import VirtualClock
from sim.core.config import SimConfig, build_policy_config
from sim.core.pricing import PromptPricer
from sim.core.request import OUTPUT_SENTINEL_BASE, FakeReqToTokenPool, SimReq
from sim.cost import CostModel
from sim.metrics import RequestRecord, SimResult, summarize
from sim.workload.trace import Turn


class Matched(msgspec.Struct, frozen=True, kw_only=True):
    """前缀匹配结果（仅完成命中判定与观测打点，尚未上锁与占槽）。"""

    req: SimReq
    node: object
    need: int
    t: float
    session_id: str
    rid: str
    prompt_len: int
    out_len: int
    hit: int
    session_type: str
    task_name: str
    turn: int = -1
    arrival_t: Optional[float] = None


@dataclass
class Pending:
    """已准入并发运行的单轮请求上下文，在 finish 时生成 RequestRecord。"""

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
    turn: int = -1
    response_token: int = 0
    allocated_len: int = 0
    committed_len: int = 0
    arrival_t: Optional[float] = None


class RequestLifecycle:
    """管理单条仿真 run 的时钟、分配器、生产缓存实例及请求状态机原语。"""

    def __init__(self, config: Optional[SimConfig] = None) -> None:
        self.config = config or SimConfig()
        self.sym = bootstrap.load()
        self.clock: Optional[VirtualClock] = None
        self.alloc: Optional[TokenAllocator] = None
        self.req_pool: Optional[FakeReqToTokenPool] = None
        self.cache = None
        self.records: list[RequestRecord] = []
        self.tasks: Dict[str, str] = {}
        self._dispatched: Dict[Tuple[str, int], float] = {}
        self._rid_seq = 0
        self.inflight_tokens = 0
        self.inflight_token_peak = 0
        self.duplicate_insert_tokens = 0
        self.skipped_requests = 0
        self.self_evictions = 0
        self.unexpected_evictions = 0
        self.cost_model = CostModel.parse(self.config.cost_model)
        self.pricer = PromptPricer(self.cost_model, self.config.page_size)

    def _require_prepared(self) -> None:
        if self.cache is None:
            raise RuntimeError("先调用 prepare(width) 再执行请求原语")

    def prepare(self, width: int) -> None:
        """按给定槽位宽度重建时钟、分配器、Req 池与生产缓存实例。"""
        cfg = self.config
        if width <= 0:
            raise ValueError(f"req_to_token 宽度必须为正，got {width}")

        # SessionRegistry 是进程级单例，必须在每次 run 前清零以防上轮分类证据串味。
        self.sym["SessionRegistry"].reset_global()

        self.clock = VirtualClock(0.0)
        self.alloc = TokenAllocator(cfg.capacity)
        self.req_pool = FakeReqToTokenPool(width)
        policy_config = build_policy_config(cfg)

        self.cache = self.sym["HeterCacheRadixCache"].create_simulated(
            mock_allocator=self.alloc,
            mock_req_pool=self.req_pool,
            page_size=cfg.page_size,
            eviction_policy=cfg.eviction_policy,
            eviction_policy_config=policy_config,
            now=self.clock,
        )
        if getattr(self.cache, "session_registry", None) is not None:
            self.cache.session_registry.idle_ttl = cfg.idle_ttl

        self.records = []
        self.tasks = {}
        self._dispatched = {}
        self._rid_seq = 0
        self.inflight_tokens = 0
        self.inflight_token_peak = 0
        self.duplicate_insert_tokens = 0
        self.skipped_requests = 0
        self.self_evictions = 0
        self.unexpected_evictions = 0
        self.cost_model = CostModel.parse(cfg.cost_model)
        self.pricer = PromptPricer(self.cost_model, cfg.page_size, width)

    def match(
        self,
        turn: Turn,
        session_id: str,
        rid: Optional[str] = None,
        task_name: str = "",
        turn_idx: int = -1,
        arrival_t: Optional[float] = None,
    ) -> Matched:
        """推进虚拟时钟并执行前缀匹配，不加锁、不触发淘汰或占槽。"""
        self._require_prepared()
        sym = self.sym
        self.clock.advance_to(turn.t)
        if turn_idx >= 0:
            self._dispatched[(session_id, int(turn_idx))] = float(turn.t)
        self._rid_seq += 1
        rid = rid or f"{session_id}#{self._rid_seq}"

        req = SimReq(
            rid,
            session_id,
            turn.prompt,
            turn.out,
            tool_call=turn.tool_call,
            output_token=OUTPUT_SENTINEL_BASE + self._rid_seq,
            response_ids=turn.response,
        )
        # 仿真只把树当作唯一占用主体；将 protected_len 设为全长以禁用 _adopt 对请求行的重复释放。
        req.kv.cache_protected_len = len(req.fill_ids())

        key = sym["RadixKey"](array("q", turn.prompt)).page_aligned(self.cache.page_size)
        matched = self.cache.match_prefix(sym["MatchPrefixParams"](key=key, req=req))
        hit = int(matched.device_indices.numel())

        full = sym["RadixKey"](array("q", req.fill_ids())).page_aligned(
            self.cache.page_size
        )
        need = len(full) - hit

        state = self.cache.session_registry.get(session_id)
        session_type = state.session_type.value if state is not None else "none"
        return Matched(
            req=req,
            node=matched.last_device_node,
            need=need,
            t=turn.t,
            session_id=session_id,
            rid=rid,
            prompt_len=len(req.origin_input_ids),
            out_len=len(req.output_ids),
            hit=hit,
            session_type=session_type,
            task_name=task_name,
            turn=int(turn_idx),
            arrival_t=arrival_t,
        )

    def new_pending(
        self, matched: Matched, *, skipped: bool, evicted: int
    ) -> Pending:
        """从匹配结果构造初始 Pending 上下文。"""
        return Pending(
            req=matched.req,
            node=matched.node,
            need=matched.need,
            t=matched.t,
            session_id=matched.session_id,
            rid=matched.rid,
            prompt_len=matched.prompt_len,
            out_len=matched.out_len,
            hit=matched.hit,
            evicted=int(evicted),
            session_type=matched.session_type,
            task_name=matched.task_name,
            skipped=skipped,
            turn=matched.turn,
            response_token=(
                int(matched.req.output_ids[0]) if matched.req.output_ids else 0
            ),
            allocated_len=matched.hit,
            committed_len=matched.hit,
            arrival_t=matched.arrival_t,
        )

    def lock(self, matched: Matched) -> None:
        """锁定已命中节点链路，防止后续淘汰逐出正被当前批次使用的前缀。"""
        self._require_prepared()
        self.cache.inc_lock_ref(matched.node)

    def reserve_upto(self, pending: Pending, seq_len: int) -> int:
        """将请求的物理占槽水位推进至 seq_len 覆盖的页边界，不足时先淘汰再分配。"""
        self._require_prepared()
        if pending.skipped:
            return 0

        page = self.cache.page_size
        total_aligned = pending.hit + pending.need
        target = min(((int(seq_len) + page - 1) // page) * page, total_aligned)
        delta = target - pending.allocated_len
        if delta <= 0:
            return 0

        shortfall = delta - self.alloc.available_size()
        evicted = 0
        if shortfall > 0:
            evicted = self.cache.evict(
                self.sym["EvictParams"](num_tokens=int(shortfall))
            ).num_tokens_evicted
            if pending.node.evicted:
                self.self_evictions += 1
                raise AssertionError(
                    "evict removed a locked matched prefix: "
                    f"rid={pending.rid} hit={pending.hit} shortfall={shortfall}"
                )
        self.alloc.alloc(delta)
        pending.allocated_len = target
        pending.evicted += int(evicted)
        self.inflight_tokens += delta
        self.inflight_token_peak = max(self.inflight_token_peak, self.inflight_tokens)
        self.assert_balanced(where=f"reserve_upto({pending.rid}, {seq_len})")
        return int(evicted)

    def commit_upto(
        self, pending: Pending, seq_len: int, *, chunked: bool = False
    ) -> None:
        """将已算完的满页前缀增量写入前缀树，并将锁平稳下移到新叶节点。"""
        self._require_prepared()
        if pending.skipped:
            return

        page = self.cache.page_size
        commit_len = min((int(seq_len) // page) * page, pending.allocated_len)
        delta = commit_len - pending.committed_len
        if delta <= 0:
            return

        req = pending.req
        prefix_ids = req.fill_ids()[:commit_len]
        tree_before = tree_tokens(self.cache)
        radix_key, _, _ = self.cache._adopt(
            req, prefix_ids, chunked=chunked, split_prompt=False
        )
        grew = tree_tokens(self.cache) - tree_before

        self.inflight_tokens -= delta
        if grew < delta:
            self.alloc.release(delta - grew)
            self.duplicate_insert_tokens += delta - grew
        elif grew > delta:
            raise AssertionError(
                f"树增长超过本步提交量：grew={grew} > delta={delta} (rid={pending.rid})"
            )

        # 用 count_reuse=False 查询新叶以免提前虚增复用计数；先锁新叶再解旧锁避免出现未锁窗口。
        matched = self.cache.match_prefix(
            self.sym["MatchPrefixParams"](key=radix_key, count_reuse=False)
        )
        new_node = matched.last_device_node
        self.cache.inc_lock_ref(new_node)
        self.cache.dec_lock_ref(pending.node)
        pending.node = new_node
        pending.committed_len = commit_len
        self.assert_balanced(where=f"commit_upto({pending.rid}, {commit_len})")

    def admit_locked(self, matched: Matched) -> Pending:
        """为已完成上锁的请求一次性淘汰短缺并预留全部 prompt + output 槽位。"""
        self._require_prepared()
        pending = self.new_pending(matched, skipped=False, evicted=0)
        self.reserve_upto(pending, matched.hit + matched.need)
        return pending

    def admit(self, matched: Matched) -> Pending:
        """准入单条已匹配请求（含 oversize 检查、上锁与整段槽位预留）。"""
        self._require_prepared()
        need = matched.need
        total = need + matched.hit
        if need > self.alloc.size:
            if self.config.oversize != "skip":
                raise OutOfTokens(
                    f"单请求 {total} token 超过池容量 {self.alloc.size}"
                    f"（rid={matched.rid} hit={matched.hit} need={need}）；"
                    '请调大 capacity 或设置 SimConfig(oversize="skip")'
                )
            self.skipped_requests += 1
            self.assert_balanced(where=f"skip({matched.rid})")
            return self.new_pending(matched, skipped=True, evicted=0)

        self.lock(matched)
        try:
            return self.admit_locked(matched)
        except OutOfTokens as exc:
            self.cache.dec_lock_ref(matched.node)
            if self.config.oversize != "skip":
                raise OutOfTokens(
                    f"{exc}（rid={matched.rid} total={total} hit={matched.hit} "
                    f"need={need}）——请调大 capacity 或设置 SimConfig(oversize=\"skip\")"
                ) from exc
            self.skipped_requests += 1
            self.assert_balanced(where=f"skip({matched.rid})")
            return self.new_pending(matched, skipped=True, evicted=0)
        except BaseException:
            self.cache.dec_lock_ref(matched.node)
            raise

    def start(
        self,
        turn: Turn,
        session_id: str,
        rid: Optional[str] = None,
        task_name: str = "",
        turn_idx: int = -1,
        arrival_t: Optional[float] = None,
    ) -> Pending:
        """执行匹配与全量预留；落树推迟到 finish()，保留在飞期间的竞态与锁保护窗口。"""
        return self.admit(
            self.match(
                turn,
                session_id,
                rid,
                task_name,
                turn_idx=turn_idx,
                arrival_t=arrival_t,
            )
        )

    def finish(self, pending: Pending) -> RequestRecord:
        """补齐剩余槽位、将完整序列写入树（按 prompt 边界切分）、释放锁并记账。"""
        if not pending.skipped:
            req = pending.req
            total_aligned = pending.hit + pending.need
            if pending.allocated_len < total_aligned:
                self.reserve_upto(pending, total_aligned)
            if self.config.verify and pending.node.evicted:
                self.self_evictions += 1
                raise AssertionError(
                    f"matched prefix evicted before end: rid={pending.rid}"
                )

            rem = pending.allocated_len - pending.committed_len
            tree_before = tree_tokens(self.cache)
            self.cache._adopt(
                req,
                req.fill_ids(),
                chunked=(pending.committed_len > pending.hit),
                split_prompt=True,
            )
            grew = tree_tokens(self.cache) - tree_before

            self.inflight_tokens -= rem
            if grew < rem:
                self.alloc.release(rem - grew)
                self.duplicate_insert_tokens += rem - grew
            elif grew > rem:
                raise AssertionError(
                    f"树增长超过预留：grew={grew} > need={rem} (rid={pending.rid})"
                )
            pending.committed_len = pending.allocated_len

            self.cache.dec_lock_ref(pending.node)
            self.assert_balanced(where=f"end({pending.rid})")

        total_cost, saved_cost = self.pricer.price(pending.prompt_len, pending.hit)
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
            turn=pending.turn,
            response_token=pending.response_token,
            total_cost=total_cost,
            saved_cost=saved_cost,
            arrival_t=pending.arrival_t,
        )
        self.records.append(record)
        return record

    def serve(
        self,
        turn: Turn,
        session_id: str,
        rid: Optional[str] = None,
        task_name: str = "",
        turn_idx: int = -1,
        arrival_t: Optional[float] = None,
    ) -> RequestRecord:
        """立即执行并结束单轮请求，用于无并发开环回放。"""
        return self.finish(
            self.start(
                turn,
                session_id,
                rid,
                task_name,
                turn_idx=turn_idx,
                arrival_t=arrival_t,
            )
        )

    def assert_balanced(self, where: str = "") -> None:
        """校验证实：池已用槽位 == 树内 token + 在飞未提交 token。"""
        tree = tree_tokens(self.cache)
        expect = tree + self.inflight_tokens
        if self.alloc.used != expect:
            raise AssertionError(
                f"池账不平（{where}）：alloc.used={self.alloc.used} "
                f"tree={tree} inflight={self.inflight_tokens}"
            )

    def result(self) -> SimResult:
        """封装当前记录、汇总及派发时间戳表。"""
        out = SimResult(records=list(self.records), summary=self.summary())
        out.dispatched = dict(self._dispatched)
        return out

    def summary(self) -> Dict:
        strategy = self.cache.eviction_strategy
        served = [record for record in self.records if not record.skipped]
        summary = summarize(
            self.records,
            capacity=self.config.capacity,
            final_used=self.alloc.used,
            sessions=len({record.session_id for record in served}),
            profiler=getattr(strategy, "profiler", None),
            occupancy=getattr(self.cache, "occupancy", None),
            cost_model=str(self.cost_model),
        )
        summary["self_evictions"] = self.self_evictions
        summary["unexpected_evictions"] = self.unexpected_evictions
        summary["verify"] = self.config.verify
        return summary
