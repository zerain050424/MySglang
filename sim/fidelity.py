"""保真对拍：把「还原得对不对」变成可执行的断言。

三组检查，对应三个可能悄悄跑偏的口径：

1. **拓扑不变量**（每次请求后全树校验）：``abs_start`` 链、``len(key) == len(value)``、
   ``evictable_leaves`` 与树形一致。分裂是这里最容易出错的路径——prefix 半继承
   原偏移、suffix 半整体后移，任何一边算错都不会报错，只会让命中位置漂移。
2. **账本对拍**：``occupancy.occ``（增量维护）与全树快照
   （``total - one_shot``）的差。增量记账**有意**允许两次校准之间漂移，所以这里
   只在 reconcile 之后要求逐键相等；reconcile 之前的差作为观测量报出来，不判失败。
3. **复现性**：同 seed 两次 run 的逐请求指纹必须相同（含淘汰量与水位）。

不含：与真 server 的命中率对拍——那需要 GPU 起服务，属于搬运回仓库后的实验。
"""

from __future__ import annotations

import random
from array import array
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sim.core.config import SimConfig
from sim.metrics import SimResult
from sim.simulator import Simulator
from sim.workload.trace import Turn, max_total_len


# ---------------------------------------------------------------------------
# 拓扑不变量
# ---------------------------------------------------------------------------
def check_tree(cache, where: str = "") -> List[str]:
    """全树校验，返回问题描述列表（空 = 通过）。"""

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

    # evictable_leaves 与树形双向一致：无子节点且未上锁 <=> 在可淘汰集合里
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


# ---------------------------------------------------------------------------
# 账本对拍
# ---------------------------------------------------------------------------
def ledger_report(cache) -> Dict:
    """比较增量账本与全树快照；reconcile 之后必须逐键相等。"""

    tracker = cache.occupancy
    snapshot = cache._per_queue_token_count()
    expected = {
        q: float((snapshot.get(q) or {}).get("total", 0.0))
        - float((snapshot.get(q) or {}).get("one_shot", 0.0))
        for q in tracker.occ
    }
    before = {q: tracker.occ[q] - expected[q] for q in tracker.occ}

    # 全量校准（吸收标签变更 / one-shot 晋升带来的漂移）
    tracker.reconcile(snapshot, tracker.capacity)
    after = {q: tracker.occ[q] - expected[q] for q in tracker.occ}

    return {
        "drift_before_reconcile": before,
        "drift_after_reconcile": after,
        "exact_after_reconcile": all(abs(v) <= 1e-9 for v in after.values()),
        "occ": tracker.snapshot(),
        "snapshot_total": {q: (snapshot.get(q) or {}).get("total", 0.0) for q in tracker.occ},
    }


# ---------------------------------------------------------------------------
# 压力 trace（专挑分裂 / 复用 / 老化 / 承压）
# ---------------------------------------------------------------------------
def stress_events(
    *,
    seed: int = 11,
    n: int = 80,
    prefix_len: int = 96,
    max_prompt_len: int = 768,
) -> List[Tuple[float, str, Turn]]:
    """按时刻排序的 (t, session_id, Turn)，四类模式轮流出现。

    模式：``grow``（同一会话不断加长 -> 连续分裂）、``shared``（跨会话同一前缀 ->
    共享数账本 + struct 晋升）、``unique``（一次性内容）、``idle``（长时间不访问 ->
    老化 TTL）。
    """

    rng = random.Random(seed)
    shared = array("q", range(1, prefix_len + 1))
    events: List[Tuple[float, str, Turn]] = []
    t = 0.0

    for i in range(n):
        mode = i % 4
        if mode == 0:  # grow
            sid = f"ShareGPT-{i % 3}"
            length = min(prefix_len + 64 * (1 + i % 6), max_prompt_len)
            prompt = shared + array("q", [i % 7] * (length - prefix_len))
        elif mode == 1:  # shared
            sid = f"Code-{i % 2}"
            prompt = shared + array("q", [100 + i % 2] * 64)
        elif mode == 2:  # unique
            sid = f"apps_{i}"
            base = 10_000 * (i + 1)
            prompt = array("q", range(base, base + 256))
        else:  # idle: 时间跳一大步后回到老会话
            sid = f"ShareGPT-{i % 3}"
            prompt = shared + array("q", [0] * 64)
            t += 120.0
        events.append((t, sid, Turn(t=t, prompt=prompt, out=16)))
        t += rng.uniform(1.0, 8.0)

    events.sort(key=lambda item: (item[0], item[1]))
    return events


# ---------------------------------------------------------------------------
# 对拍主流程
# ---------------------------------------------------------------------------
@dataclass
class FidelityReport:
    requests: int
    violations: List[str]
    ledger: Dict
    fingerprint: str
    matched: bool


def run_fidelity(
    config: Optional[SimConfig] = None, *, seed: int = 11, n: int = 80
) -> FidelityReport:
    """跑一遍压力 trace，逐请求校验拓扑，最后对账 + 取指纹。"""

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
            if problems:  # 首个问题就停：后面的树已经不可信
                break
        return problems, sim

    violations, sim = one_pass()
    ledger = ledger_report(sim.cache) if not violations else {}
    result = SimResult(records=list(sim.records), summary={})
    fingerprint = result.fingerprint()

    _, sim2 = one_pass()
    matched = fingerprint == SimResult(records=list(sim2.records), summary={}).fingerprint()

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
    "max_total_len",
]
