"""RegretGate 与生产缓存的接线回归（需要 sim 环境；缺 torch 自动跳过）。

验证三件事（tail-stress 之外的最小闭环，不构成实验结果）：

1. 淘汰决策的 stamp 记进了弃权门（``on_evict``），队列归属/ span 正确；
2. 会话回来 miss 时，断点页键 join 上当初削它的 stamp（``on_request``），
   R 记到该队列——观测账本与弃权门同键（``_breakpoint_key`` 同源）；
3. 一个 stamp 只归因一次。

负载：1 个会话先落 300 token，再用 25 个一次性会话把 4096 的池挤穿
（淘汰经 fallback 落在这些 untagged TRANS 内容上），随后原会话回来 hit=0，
断点页应 join。容量/请求粒度刻意避开 ``oversize`` 路径（见 sim/runner）。
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

sim = pytest.importorskip("sim.simulator", reason="sim 需要完整 torch 导入链")


def test_regret_gate_stamps_and_joins_through_real_cache():
    import array
    import random

    from sim.core.config import SimConfig
    from sim.simulator import Simulator
    from sim.workload.trace import Turn

    cfg = SimConfig(
        capacity=4096,
        page_size=8,
        eviction_policy="heter_cache",
        seed=0,
        oversize="error",
        regret_gate={"enabled": True, "window_s": 120.0, "k": 1.0, "r_min": 0.5},
    )
    sim = Simulator(cfg)
    sim.lifecycle.prepare(width=512)

    def mk(seed, n):
        rng = random.Random(seed)
        return array.array("q", [rng.randrange(1000, 60000) for _ in range(n)])

    def admit(rid, sid, seed, n, t):
        matched = sim.lifecycle.match(Turn(t=t, prompt=mk(seed, n), out=8), sid, rid=rid)
        pending = sim.lifecycle.admit(matched)
        sim.lifecycle.finish(pending)

    admit("a0", "s1", 1, 300, 0.0)
    for i in range(25):
        admit(f"x{i}", f"x{i}", 100 + i, 300, 1.0 + i)

    gate = sim.cache.eviction_strategy.regret_gate
    assert gate is not None, "SimConfig.regret_gate 透传断裂"
    evicted = sum(gate._ev_sum.values())
    assert evicted > 0, "淘汰量没有进弃权门的 evict 窗口"
    assert gate.joins == 0

    # 原会话回来：内容已被挤掉 → hit=0，断点页 join 回当初削它的队列
    rematch = sim.lifecycle.match(Turn(t=50.0, prompt=mk(1, 300), out=8), "s1", rid="a1")
    assert rematch.hit == 0, f"挤占后应整链丢失，got hit={rematch.hit}"
    assert gate.joins == 1, f"断点页应 join 恰好一次，got {gate.joins}"
    attributed = {q for q in gate._rg if gate._rg_sum.get(q, 0) > 0}
    assert len(attributed) == 1, f"应归因到唯一队列，got {attributed}"
    # 再来一次同样的 miss：stamp 已摘除，不重复归因
    sim.lifecycle.match(Turn(t=51.0, prompt=mk(1, 300), out=8), "s1", rid="a2")
    assert gate.joins == 1, "一个 stamp 只归因一次"
