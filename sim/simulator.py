"""仿真器：组合 RequestLifecycle 与三类执行驱动。"""

from __future__ import annotations

from typing import Iterable, Optional

from sim.core.config import SimConfig
from sim.core.lifecycle import RequestLifecycle
from sim.drivers.emitter import ClosedLoopEmitter
from sim.drivers.engine import EngineConfig, EngineScheduler
from sim.drivers.replay import ReplayDriver
from sim.analysis.metrics import SimResult
from sim.workload.trace import Session, synthetic_sessions


class Simulator:
    """离线前缀缓存仿真器对外统一入口。"""

    def __init__(self, config: Optional[SimConfig] = None) -> None:
        self.lifecycle = RequestLifecycle(config)

    @property
    def config(self) -> SimConfig:
        return self.lifecycle.config

    @property
    def cache(self):
        return self.lifecycle.cache

    @property
    def alloc(self):
        return self.lifecycle.alloc

    @property
    def clock(self):
        return self.lifecycle.clock

    @property
    def req_pool(self):
        return self.lifecycle.req_pool

    @property
    def records(self):
        return self.lifecycle.records

    @property
    def sym(self):
        return self.lifecycle.sym

    @property
    def inflight_tokens(self) -> int:
        return self.lifecycle.inflight_tokens

    def run_replay(self, sessions: Iterable[Session]) -> SimResult:
        """按静态 Turn.t 时间戳执行无并发开环回放。"""
        return ReplayDriver(self.lifecycle).run(sessions)

    def run_closed_loop(
        self,
        sessions: Iterable[Session],
        specs: Iterable,
        *,
        service=None,
        arrival_scale: float = 1.0,
        wait_scale: float = 1.0,
        max_inflight: int = 0,
    ) -> SimResult:
        """按闭环轮间思考与服务时长模型执行并发仿真。"""
        return ClosedLoopEmitter(
            self.lifecycle, list(sessions), list(specs), service=service
        ).run(
            arrival_scale=arrival_scale,
            wait_scale=wait_scale,
            max_inflight=max_inflight,
        )

    def run_engine(
        self,
        sessions: Iterable[Session],
        specs: Iterable,
        *,
        scheduler: Optional[EngineConfig] = None,
        service=None,
        arrival_scale: float = 1.0,
        wait_scale: float = 1.0,
    ) -> SimResult:
        """按 SGLang 引擎步批预算与 chunked/decode 调度执行仿真。"""
        return EngineScheduler(
            self.lifecycle,
            sessions,
            specs,
            scheduler=scheduler or EngineConfig(),
            service=service,
        ).run(arrival_scale=arrival_scale, wait_scale=wait_scale)

    def result(self) -> SimResult:
        """返回当前生命周期的运行结果快照。"""
        return self.lifecycle.result()


def run_synthetic(config: Optional[SimConfig] = None, **trace_kwargs) -> SimResult:
    """生成四类混合合成负载并执行开环回放。"""
    return Simulator(config).run_replay(synthetic_sessions(**trace_kwargs))
