"""树拓扑不变量、账本校准误差与确定性保真检查工具。"""

from __future__ import annotations

import random
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from sim.analysis.metrics import SimResult
from sim.workload.trace import Turn

if TYPE_CHECKING:
    from sim.core.config import SimConfig


def check_tree(cache, where: str = "") -> List[str]:
    """全树遍历校验 abs_start 链、key/value 等长及 evictable_leaves 双向一致性。"""
    problems: List[str] = []
    nodes = set()
    stack: List[Tuple[object, int]] = [(cache.root_node, 0)]

    while stack:
        node, expect_start = stack.pop()
        nodes.add(id(node))
        got = getattr(node, "abs_start", None)
        if node is not cache.root_node and got != expect_start:
            problems.append(
                f"{where}: abs_start={got}, 期望 {expect_start}（父偏移+父段长）"
            )
        key_len = len(node.key) if node.key is not None else 0
        if node is not cache.root_node:
            value_len = len(node.value) if node.value is not None else 0
            if value_len != key_len:
                problems.append(
                    f"{where}: nid={getattr(node, 'id', '?')} "
                    f"key_len={key_len} != value_len={value_len}"
                )
        for child in node.children.values():
            if child.evicted:
                continue
            stack.append((child, (got or 0) + key_len))

    for node in list(cache.evictable_leaves):
        if id(node) not in nodes:
            problems.append(
                f"{where}: evictable_leaves 含已不在树上的节点 "
                f"nid={getattr(node, 'id', '?')}"
            )
            continue
        if node.children:
            problems.append(
                f"{where}: evictable_leaves 含非叶子 "
                f"nid={getattr(node, 'id', '?')}"
            )
        if node.lock_ref:
            problems.append(
                f"{where}: evictable_leaves 含已上锁节点 "
                f"nid={getattr(node, 'id', '?')}"
            )

    for node_id in nodes:
        node = _lookup(cache.root_node, node_id)
        if node is None or node is cache.root_node:
            continue
        if not node.children and not node.lock_ref and node not in cache.evictable_leaves:
            problems.append(
                f"{where}: 叶子 nid={getattr(node, 'id', '?')} 未进 evictable_leaves"
            )
    return problems


def _lookup(root, node_id: int):
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) == node_id:
            return node
        stack.extend(node.children.values())
    return None


def ledger_report(cache) -> Dict:
    """比较增量占用账本与全树快照在 reconcile 前后的差异。"""
    tracker = cache.occupancy
    snapshot = cache._per_queue_token_count()
    expected = {
        q: float((snapshot.get(q) or {}).get("total", 0.0))
        - float((snapshot.get(q) or {}).get("one_shot", 0.0))
        for q in tracker.occ
    }
    before = {q: tracker.occ[q] - expected[q] for q in tracker.occ}

    tracker.reconcile(snapshot, tracker.capacity)
    after = {q: tracker.occ[q] - expected[q] for q in tracker.occ}

    return {
        "drift_before_reconcile": before,
        "drift_after_reconcile": after,
        "exact_after_reconcile": all(abs(v) <= 1e-9 for v in after.values()),
        "occ": tracker.snapshot(),
        "snapshot_total": {
            q: (snapshot.get(q) or {}).get("total", 0.0) for q in tracker.occ
        },
    }


def stress_events(
    *,
    seed: int = 11,
    n: int = 80,
    prefix_len: int = 96,
    max_prompt_len: int = 768,
) -> List[Tuple[float, str, Turn]]:
    """生成覆盖节点分裂、跨会话共享、一次性内容与超时老化的压力事件序列。"""
    rng = random.Random(seed)
    shared = array("q", range(1, prefix_len + 1))
    events: List[Tuple[float, str, Turn]] = []
    t = 0.0

    for i in range(n):
        mode = i % 4
        if mode == 0:
            sid = f"ShareGPT-{i % 3}"
            length = min(prefix_len + 64 * (1 + i % 6), max_prompt_len)
            prompt = shared + array("q", [i % 7] * (length - prefix_len))
        elif mode == 1:
            sid = f"Code-{i % 2}"
            prompt = shared + array("q", [100 + i % 2] * 64)
        elif mode == 2:
            sid = f"apps_{i}"
            base = 10_000 * (i + 1)
            prompt = array("q", range(base, base + 256))
        else:
            sid = f"ShareGPT-{i % 3}"
            prompt = shared + array("q", [0] * 64)
            t += 120.0
        events.append((t, sid, Turn(t=t, prompt=prompt, out=16)))
        t += rng.uniform(1.0, 8.0)

    events.sort(key=lambda item: (item[0], item[1]))
    return events


@dataclass
class FidelityReport:
    requests: int
    violations: List[str]
    ledger: Dict
    fingerprint: str
    matched: bool


def run_fidelity(
    config: Optional["SimConfig"] = None, *, seed: int = 11, n: int = 80
) -> FidelityReport:
    """运行压力事件序列并逐请求检查树拓扑、账本对拍及双次复现指纹。"""
    from sim.core.config import SimConfig
    from sim.simulator import Simulator

    config = config or SimConfig(capacity=2048, seed=seed)
    events = stress_events(seed=seed, n=n)
    width = max(len(turn.prompt) + turn.response_len for _t, _sid, turn in events)

    def one_pass() -> Tuple[List[str], Simulator]:
        sim = Simulator(config)
        sim.lifecycle.prepare(width)
        problems: List[str] = []
        for t, sid, turn in events:
            sim.lifecycle.serve(turn, sid)
            problems.extend(check_tree(sim.cache, where=f"{sid}@{t:.0f}"))
            if problems:
                break
        return problems, sim

    violations, sim = one_pass()
    ledger = ledger_report(sim.cache) if not violations else {}
    fingerprint = SimResult(records=list(sim.records), summary={}).fingerprint()

    _, sim2 = one_pass()
    matched = (
        fingerprint == SimResult(records=list(sim2.records), summary={}).fingerprint()
    )

    return FidelityReport(
        requests=len(events),
        violations=violations,
        ledger=ledger,
        fingerprint=fingerprint,
        matched=matched,
    )


__all__ = [
    "check_tree",
    "ledger_report",
    "stress_events",
    "run_fidelity",
    "FidelityReport",
]
