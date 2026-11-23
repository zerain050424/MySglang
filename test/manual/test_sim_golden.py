"""离线模拟器行为黄金快照回归测试。

覆盖开环回放、闭环并发发射、SGLang 步级引擎调度（FCFS/LPM 及
lazy/progressive 四组合）、heter 变体、oversize/cost 口径与 OPT 上界。

录制黄金快照::

    PYTHONPATH=$PWD:$PWD/python python test/manual/test_sim_golden.py --record

运行回归校验::

    PYTHONPATH=$PWD:$PWD/python python -m pytest -q test/manual/test_sim_golden.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from array import array
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

GOLDEN_PATH = HERE / "data" / "sim_golden.json"


# ---------------------------------------------------------------------------
# 确定性测试负载（合成多类 + 手工带 response 多轮）
# ---------------------------------------------------------------------------
def _build_fixture():
    from sim.schedule import ServiceModel, TaskSpec
    from sim.trace import Session, Turn, synthetic_sessions

    raw_sessions = synthetic_sessions(
        seed=7,
        n_struct=4,
        n_random=6,
        n_chat=0,
        n_agentic=0,
        horizon=250.0,
        prefix_len=128,
        base_len=320,
        out_len=48,
        max_prompt_len=1200,
    )

    sessions: List[Session] = []
    for s in raw_sessions:
        tname = "struct" if s.session_id.startswith("Code-") else "random"
        sessions.append(Session(s.session_id, s.turns, task_name=tname))

    # 让 chat / agentic / multi_resp 的每段节点与 out_len 都有独特页对齐长度，
    # 避免生产 ReuseScoreEvict (缺 node.id 平票键) 因同长叶子在同整秒下触发 set 遍历序依赖。
    prime_lens = iter(
        k * 8
        for k in range(3, 300)
        if k not in (8, 16, 24, 32, 40, 48, 64)
    )
    for i in range(3):
        base = 200_000 + i * 10_000
        head_len = next(prime_lens) + 160
        head = array("q", range(base, base + head_len))
        t = 20.0 + i * 7.0
        turns = []
        for k in range(3):
            extra = next(prime_lens) * k if k > 0 else 0
            prompt = head + array("q", [k + 1] * extra)
            out_len = next(prime_lens)
            turns.append(Turn(t=t, prompt=prompt, out=out_len))
            t += 23.0 + (i + k) * 2.1
        sessions.append(Session(f"ShareGPT-{i}", tuple(turns), task_name="chat"))

    for i in range(3):
        base = 300_000 + i * 10_000
        head_len = next(prime_lens) + 160
        head = array("q", range(base, base + head_len))
        t = 12.0 + i * 5.0
        turns = []
        for k in range(4):
            extra = (next(prime_lens) + 64) if k > 0 else 0
            if k > 0:
                prompt = turns[-1].prompt + array("q", [10 + k] * extra)
            else:
                prompt = head
            out_len = next(prime_lens)
            turns.append(Turn(t=t, prompt=prompt, out=out_len, tool_call=True))
            t += 1.7 + i * 0.4 + k * 0.2
        sessions.append(Session(f"alfred_{i}", tuple(turns), task_name="agentic"))

    # 额外加入带真实 response token 的多轮会话，覆盖渐进写树与 response 段命中
    for i in range(3):
        base = 900_000 + i * 10_000
        p1 = array("q", range(base, base + 184 + i * 16))
        r1 = array("q", range(base + 1000, base + 1048 + i * 8))
        p2 = p1 + r1 + array("q", range(base + 2000, base + 2136 + i * 16))
        r2 = array("q", range(base + 3000, base + 3056 + i * 8))
        p3 = p2 + r2 + array("q", range(base + 4000, base + 4152 + i * 24))
        turns = (
            Turn(t=15.0 + i * 5.0, prompt=p1, out=len(r1), tool_call=False, response=r1),
            Turn(t=45.0 + i * 5.0, prompt=p2, out=len(r2), tool_call=True, response=r2),
            Turn(t=75.0 + i * 5.0, prompt=p3, out=24 + i * 8, tool_call=True, response=None),
        )
        sessions.append(Session(f"alfred_resp_{i}", turns, task_name="multi_resp"))

    specs = [
        TaskSpec(name="struct", lambda_rate=0.05, mu=0.0, sigma=0.0, seed=42, task_id=0),
        TaskSpec(name="random", lambda_rate=0.08, mu=0.0, sigma=0.0, seed=42, task_id=1),
        TaskSpec(name="chat", lambda_rate=0.04, mu=2.8, sigma=0.4, seed=42, task_id=2),
        TaskSpec(name="agentic", lambda_rate=0.06, mu=0.8, sigma=0.5, seed=42, task_id=3),
        TaskSpec(
            name="multi_resp",
            arrival_mode="segments",
            segments=((20.0, 0.2), (120.0, 0.1)),
            mu=1.2,
            sigma=0.3,
            seed=42,
            task_id=4,
        ),
    ]
    service = ServiceModel(
        prefill_base_s=0.02,
        prefill_tokens_per_s=2000.0,
        decode_tokens_per_s=40.0,
    )
    return sessions, specs, service


SCENARIOS: List[Dict[str, Any]] = [
    {"name": "replay_heter_p1", "mode": "replay", "capacity": 4096, "page_size": 1, "policy": "heter_cache"},
    {"name": "replay_lru_p8", "mode": "replay", "capacity": 4096, "page_size": 8, "policy": "lru"},
    {"name": "closed_inflight0_heter", "mode": "closed", "capacity": 4096, "page_size": 8, "policy": "heter_cache", "max_inflight": 0},
    {"name": "closed_inflight1_heter", "mode": "closed", "capacity": 4096, "page_size": 8, "policy": "heter_cache", "max_inflight": 1},
    {
        "name": "closed_inflight4_chain_lru",
        "mode": "closed",
        "capacity": 4096,
        "page_size": 8,
        "policy": "chain_lru",
        "max_inflight": 4,
        "policy_config": {"floor_tokens": 256, "weight": 0.02},
    },
    {"name": "engine_fcfs_lazy_prog", "mode": "engine", "capacity": 4096, "page_size": 8, "policy": "heter_cache", "schedule_policy": "fcfs", "chunk": 256, "lazy": True, "progressive": True},
    {"name": "engine_fcfs_lazy_noprog", "mode": "engine", "capacity": 4096, "page_size": 8, "policy": "heter_cache", "schedule_policy": "fcfs", "chunk": 256, "lazy": True, "progressive": False},
    {"name": "engine_fcfs_eager_prog", "mode": "engine", "capacity": 4096, "page_size": 8, "policy": "heter_cache", "schedule_policy": "fcfs", "chunk": 256, "lazy": False, "progressive": True},
    {"name": "engine_fcfs_eager_noprog", "mode": "engine", "capacity": 4096, "page_size": 8, "policy": "heter_cache", "schedule_policy": "fcfs", "chunk": 256, "lazy": False, "progressive": False},
    {"name": "engine_lpm_chunk256", "mode": "engine", "capacity": 4096, "page_size": 8, "policy": "heter_cache", "schedule_policy": "lpm", "chunk": 256, "lazy": True, "progressive": True},
    {
        "name": "heter_regret_gate",
        "mode": "engine",
        "capacity": 4096,
        "page_size": 8,
        "policy": "heter_cache",
        "schedule_policy": "fcfs",
        "chunk": 256,
        "lazy": True,
        "progressive": True,
        "regret_gate": {
            "enabled": True,
            "k": 0.6,
            "r_min": 0.1,
            "min_evicted": 256,
            "priced_fallback": True,
        },
    },
    {
        "name": "heter_chain_policy",
        "mode": "engine",
        "capacity": 4096,
        "page_size": 8,
        "policy": "heter_cache",
        "schedule_policy": "fcfs",
        "chunk": 256,
        "lazy": True,
        "progressive": True,
        "chain_policy": {"enabled": True, "floor_tokens": 256, "weight": 0.02},
    },
    {"name": "heter_session_life", "mode": "replay", "capacity": 4096, "page_size": 8, "policy": "heter_cache", "ageing_mode": "session_life"},
    {"name": "oversize_skip", "mode": "replay", "capacity": 700, "page_size": 8, "policy": "heter_cache", "oversize": "skip"},
    {"name": "cost_position", "mode": "replay", "capacity": 4096, "page_size": 8, "policy": "heter_cache", "cost_model": "position"},
    {"name": "opt_frozen_chain", "mode": "opt", "capacity": 4096, "page_size": 8},
]


def _native_hash_available() -> bool:
    try:
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.utils import get_storage_hash_str

        get_storage_hash_str(RadixKey(array("i", range(8))), None)
        return True
    except Exception:
        return False


def _canonical(obj: Any) -> Any:
    """递归规范化数值与字典键序，保证 JSON 逐字稳定。"""
    if isinstance(obj, float):
        return round(obj, 12)
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# 适配层：后续重构改 API 时，只改这里，不改场景清单与快照
# ---------------------------------------------------------------------------
def _run_scenario(case: Dict[str, Any]) -> Dict[str, Any]:
    from sim.engine import EngineConfig
    from sim.opt_map import use_map_from_run, write_use_map
    from sim.runner import SimConfig, Simulator

    sessions, specs, service = _build_fixture()
    mode = case["mode"]

    if mode == "opt":
        if not _native_hash_available():
            return {"skipped": "native_hash unavailable under CPU shim"}
        base_cfg = SimConfig(capacity=case["capacity"], page_size=case["page_size"], eviction_policy="lru", seed=7)
        base_res = Simulator(base_cfg).run_scheduled(sessions, specs, service=service, max_inflight=0)
        with tempfile.TemporaryDirectory() as tmp:
            umap_path = os.path.join(tmp, "use_map.json")
            write_use_map(umap_path, use_map_from_run(base_res, sessions, page_size=case["page_size"]))
            opt_cfg = SimConfig(
                capacity=case["capacity"],
                page_size=case["page_size"],
                eviction_policy="opt",
                opt_use_map=umap_path,
                seed=7,
            )
            res = Simulator(opt_cfg).run(base_res.freeze_sessions(sessions))
        return _canonical({"fingerprint": res.fingerprint(), "summary": res.summary})

    cfg = SimConfig(
        capacity=case["capacity"],
        page_size=case["page_size"],
        eviction_policy=case["policy"],
        policy_config=case.get("policy_config"),
        chain_policy=case.get("chain_policy"),
        regret_gate=case.get("regret_gate"),
        ageing_mode=case.get("ageing_mode", "queue_ttl"),
        oversize=case.get("oversize", "error"),
        cost_model=case.get("cost_model", "uniform"),
        seed=7,
    )
    sim = Simulator(cfg)

    if mode == "replay":
        res = sim.run(sessions)
    elif mode == "closed":
        res = sim.run_scheduled(
            sessions,
            specs,
            service=service,
            max_inflight=case["max_inflight"],
        )
    elif mode == "engine":
        engine_cfg = EngineConfig(
            max_prefill_tokens=1024,
            chunked_prefill_size=case["chunk"],
            max_running_requests=8,
            schedule_policy=case["schedule_policy"],
            progressive_cache=case["progressive"],
            lazy_decode_alloc=case["lazy"],
        )
        res = sim.run_engine(sessions, specs, scheduler=engine_cfg, service=service)
    else:
        raise ValueError(f"unknown mode: {mode}")

    return _canonical({"fingerprint": res.fingerprint(), "summary": res.summary})


def record_golden() -> Dict[str, Any]:
    data = {case["name"]: _run_scenario(case) for case in SCENARIOS}
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


@pytest.mark.parametrize("case", SCENARIOS, ids=[c["name"] for c in SCENARIOS])
def test_sim_golden_snapshot(case: Dict[str, Any]):
    assert GOLDEN_PATH.exists(), f"缺少黄金快照文件：{GOLDEN_PATH}（先用 --record 生成）"
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    expected = golden[case["name"]]
    if isinstance(expected, dict) and expected.get("skipped") and not _native_hash_available():
        pytest.skip(expected["skipped"])
    actual = _run_scenario(case)
    if isinstance(actual, dict) and actual.get("skipped"):
        pytest.skip(actual["skipped"])
    assert actual == expected, f"场景 {case['name']} 黄金对拍不一致"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="录制黄金快照")
    args = ap.parse_args(argv)
    if args.record:
        os.environ.setdefault("STRAT_LOG_LEVEL", "quiet")
        data = record_golden()
        for name, row in data.items():
            if row.get("skipped"):
                print(f"  [SKIP] {name:28s} ({row['skipped']})")
            else:
                s = row["summary"]
                print(
                    f"  [REC ] {name:28s} fp={row['fingerprint']} "
                    f"reqs={s['requests']} hit={s['hit_rate']:.4f} evict={s['evicted_tokens']}"
                )
        print(f"Wrote {GOLDEN_PATH} ({len(data)} scenarios)")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
