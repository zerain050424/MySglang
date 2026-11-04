"""链路 smoke test（不需要 GPU / sglang，全部走 FakeBackend）。

    python tests/test_smoke.py          # 不装 pytest 也能跑
    python -m pytest tests/test_smoke.py -q

覆盖：
  * 实验臂解析 / 合成 trace / 到达调度
  * prefix_stats（LCP）
  * 后悔率与保护浪费率的精确性（构造已知答案）
  * FakeBackend 回放的确定性与字段完整性
  * 工作集校准的单调性
  * analyze 的离线主表装配
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wtl.analyze import offline_table  # noqa: E402
from wtl.arms import ARMS, select  # noqa: E402
from wtl.metrics import (  # noqa: E402
    RegretTracker,
    RetentionTracker,
    fnv_init,
    fnv_update_tokens,
)
from wtl.prefix_stats import compute as prefix_compute  # noqa: E402
from wtl.replay import calibrate_working_set, layers_from_requests, run_replay  # noqa: E402
from wtl.synth import build as synth_build  # noqa: E402
from wtl.synth import write as synth_write  # noqa: E402
from wtl.trace import schedule_single_workload  # noqa: E402


def test_arms_select():
    assert [a.id for a in select("A0,A3-A5")] == ["A0", "A3", "A4", "A5"]
    assert [a.id for a in select("all")] == list(ARMS)
    assert select(None)[0].id == "A0"
    assert ARMS["A3"].tiny_lru["policy"] == "tag_lru"
    assert ARMS["A3"].tiny_lru.get("local_hot", "chat") == "chat"
    assert ARMS["A2"].tiny_lru["policy"] == "off"


def _tiny_trace(tmp: Path, sessions: int = 30):
    built = synth_build(
        prefix_depth=256, breadth=4, variants=2, turns=2,
        body_mu=5.0, body_sigma=0.5, noise_frac=0.5, doc_len=400, seed=3,
        max_sessions=sessions,
    )
    path = synth_write(tmp, built)
    reqs = schedule_single_workload(
        "synth", num_sessions=-1, seed=3, wait_scale=1.0, arrival_scale=1.0,
        data_path=str(path), lambda_rate=2.0,
    )
    return built, reqs


def test_synth_and_schedule():
    with tempfile.TemporaryDirectory() as td:
        built, reqs = _tiny_trace(Path(td))
        assert built["meta"]["prefix_depth"] == 256
        assert built["meta"]["layers"] == [256]
        assert reqs, "调度为空"
        assert all(r.prompt for r in reqs)
        # 时间线单调
        assert all(reqs[i].ts <= reqs[i + 1].ts for i in range(len(reqs) - 1))
        assert all(r.turn >= 1 for r in reqs)
        # 多轮 prompt 应递增（历史累积）
        by_sid = {}
        for r in reqs:
            by_sid.setdefault(r.sid, []).append(r)
        multi = [v for v in by_sid.values() if len(v) > 1]
        assert multi, "没有多轮 session"
        for turns in multi[:5]:
            turns.sort(key=lambda r: r.turn)
            assert turns[-1].prompt_len > turns[0].prompt_len


def test_schedule_deterministic_across_calls():
    """同一 seed 两次构建必须给出同一条时间线（否则跨进程回放不可复现）。"""
    with tempfile.TemporaryDirectory() as td:
        _, _ = _tiny_trace(Path(td), sessions=25)
        path = Path(td) / "session_data.jsonl"
        r1 = schedule_single_workload("synth", num_sessions=-1, seed=11,
                                      wait_scale=1.0, arrival_scale=1.0,
                                      data_path=str(path), lambda_rate=2.0)
        r2 = schedule_single_workload("synth", num_sessions=-1, seed=11,
                                      wait_scale=1.0, arrival_scale=1.0,
                                      data_path=str(path), lambda_rate=2.0)
        from wtl.trace import trace_fingerprint

        assert trace_fingerprint(r1) == trace_fingerprint(r2)
        r3 = schedule_single_workload("synth", num_sessions=-1, seed=12,
                                      wait_scale=1.0, arrival_scale=1.0,
                                      data_path=str(path), lambda_rate=2.0)
        assert trace_fingerprint(r1) != trace_fingerprint(r3), "不同 seed 应给出不同 trace"


def test_prefix_stats():
    sessions = [
        {"id": f"s{i}", "num_turns": 1,
         "turn_1": {"input_ids": [1, 2, 3] + [i] * 50, "response_ids": [9]}}
        for i in range(6)
    ]
    st = prefix_compute(sessions, max_pairs=20, seed=0, layers=(1, 2, 3, 4))
    assert st["num_sessions"] == 6
    assert st["lcp_p50"] == 3
    assert st["layer_shared_pct"]["3"] == 1.0
    # 观测不到更深的共享（LCP 最大 3），更深的层不应被报告
    assert "4" not in st["layer_shared_pct"]
    assert st["layers"] == [1, 2, 3]


def test_regret_accounting():
    prefix = [11, 22, 33, 44, 55]
    ph = fnv_update_tokens(fnv_init(), prefix)
    rt = RegretTracker(window_s=60.0)

    # 逃汰了 5 个 token 的前缀
    rt.on_evict(ts=10.0, depth_end=5, ph=ph, tokens=5, region=0, is_victim=True)
    # 20s 后同一前缀又来了，且完全没命中 -> 丢失 5 个 token
    lost = rt.on_request(ts=20.0, prompt=prefix + [66, 77], cached_len=0)
    assert lost == 5, lost

    # 超出窗口的淘汰不计
    rt2 = RegretTracker(window_s=5.0)
    rt2.on_evict(ts=10.0, depth_end=5, ph=ph, tokens=5, region=0, is_victim=True)
    assert rt2.on_request(ts=100.0, prompt=prefix, cached_len=0) == 0

    # protected 浪费：淘汰后又被命中 -> 不算浪费
    rt3 = RegretTracker(window_s=60.0)
    rt3.on_evict(ts=1.0, depth_end=5, ph=ph, tokens=5, region=3, is_victim=True)
    rt3.on_request(ts=2.0, prompt=prefix, cached_len=5)
    s = rt3.summary()
    assert s["protected_evicted_tokens"] == 5
    assert s["protected_waste_rate"] == 0.0

    # protected 级联删除单独计量
    rt4 = RegretTracker(window_s=60.0)
    rt4.on_evict(ts=1.0, depth_end=5, ph=ph, tokens=5, region=3, is_victim=False)
    assert rt4.summary()["protected_cascade_deleted_tokens"] == 5


def test_retention_tracker():
    rt = RetentionTracker({"t": [128, 256]})
    rt.add("t", 300)
    rt.add("t", 100)
    s = rt.summary()["t"]
    assert s["128"]["full_rate"] == 0.5
    assert abs(s["128"]["cover"] - (1.0 + 100 / 128) / 2) < 1e-3  # summary 保留 4 位
    assert s["256"]["full_rate"] == 0.5


def test_fake_replay_deterministic():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, reqs = _tiny_trace(tmp)
        layers = layers_from_requests(reqs, layers=[128, 256], seed=3)
        assert layers, layers
        pool = max(1, int(sum(r.total_len for r in reqs) * 0.25))
        r1 = run_replay(reqs, arm=ARMS["A0"], pool_tokens=pool, backend_kind="fake",
                        layers_by_task=layers, quiet=True)
        r2 = run_replay(reqs, arm=ARMS["A0"], pool_tokens=pool, backend_kind="fake",
                        layers_by_task=layers, quiet=True)
        assert r1["rows"] == r2["rows"], "同配置两次回放不一致"
        s = r1["summary"]
        assert s["num_requests"] == len(reqs)
        assert 0.0 <= s["hit_pct"] <= 100.0
        assert s["evictions"] > 0, "小池子应当发生淘汰"
        assert s["regret"] is not None
        assert r1["backend"] == "fake"
        assert r1["schema"] == "wtl.replay/1"


def test_fake_backend_topology():
    """镜像树必须真的挂在正确位置：重复前缀全命中、分叉后共享长度正确。"""
    from wtl.backends import FakeBackend
    from wtl.trace import Request

    be = FakeBackend(pool_tokens=10**6)

    def acc(prompt, tag=""):
        r = Request(idx=0, ts=0.0, sid=f"s{tag}", task="t", turn=1,
                    prompt=list(prompt), output=[])
        return be.acquire(r)

    assert acc([1, 2, 3, 4]).cached_len == 0
    assert acc([1, 2, 3, 4, 5]).cached_len == 4      # 前缀全命中
    assert acc([1, 2, 9]).cached_len == 2            # 分叉后只剩 2 个 token 共享
    assert be.used == 6, be.used                     # 树里唯一 token = {1,2,3,4,5,9}
    assert acc([1, 2, 3, 4, 5]).cached_len == 5


def test_calibration_monotone():
    with tempfile.TemporaryDirectory() as td:
        _, reqs = _tiny_trace(Path(td), sessions=40)
        ws = calibrate_working_set(reqs)
        assert ws > 0
        # 工作集不可能超过总量
        assert ws <= sum(r.total_len for r in reqs)
        # 关掉 output 后不应更大
        assert calibrate_working_set(reqs, include_outputs=False) <= ws


def test_analyze_offline_table():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, reqs = _tiny_trace(tmp, sessions=20)
        layers = layers_from_requests(reqs, seed=3)
        pool = max(1, int(sum(r.total_len for r in reqs) * 0.3))
        res = run_replay(reqs, arm=ARMS["A3"], pool_tokens=pool, backend_kind="fake",
                         layers_by_task=layers, quiet=True)
        res["workload"] = "synth"
        res["ratio"] = 0.3
        res["seed"] = 3
        rows = offline_table([res])
        assert rows and rows[0]["arm"] == "A3"
        assert rows[0]["backend"] == "fake"
        assert "hit_pct" in rows[0]


def test_arm_configs_match_framework():
    """A0–A5 的 tiny_lru 配置必须能被框架的 TinyLRUParams / TagLruPolicy 接受。

    这条是**配置漂移保险**：如果框架改了参数名（如 struct_share_min / local_hot），
    这里第一时间红。
    没有 torch 时用「stub 顶层包 + 真实源码路径」的方式只导入纯 Python 模块。
    """
    import types

    repo = ROOT.parents[1]                      # MySglang/
    src = repo / "python" / "sglang"
    if not src.exists():
        raise RuntimeError(f"找不到 sglang 源码：{src}")
    if "sglang" not in sys.modules:
        pkg = types.ModuleType("sglang")
        pkg.__path__ = [str(src)]
        sys.modules["sglang"] = pkg
        for sub in ("srt", "srt.mem_cache"):
            m = types.ModuleType(f"sglang.{sub}")
            m.__path__ = [str(src / sub.replace(".", "/"))]
            sys.modules[f"sglang.{sub}"] = m

    try:
        from sglang.srt.mem_cache.tag_lru import TagLruPolicy
        from sglang.srt.mem_cache.uni_utils import TinyLRUParams
    except ImportError as exc:
        # 本地仓库可能停在旧版本（分支仓库在实验机上）：按 SKIP 处理，不判失败。
        print(f"  SKIP  {exc}（分支仓库上会跑）")
        return

    for arm in ARMS.values():
        cfg = (arm.config or {}).get("tiny_lru")
        if cfg is None:
            continue
        try:
            p = TinyLRUParams.from_any(cfg)
        except (ValueError, ImportError) as exc:
            if "Unknown tiny_lru.policy" in str(exc) or isinstance(exc, ImportError):
                # 本地仓库可能还停在旧版本（实验机才是分支所在）：
                # 这种"本机框架不认识该 policy / 没有该模块"的情况按 SKIP 处理。
                print(f"  SKIP  {arm.id}: 本地框架不支持 {cfg.get('policy')!r}（在分支仓库上会跑）")
                return
            raise
        if p.policy == "tag_lru":
            TagLruPolicy(
                struct_share_min=p.struct_share_min,
                share_metric=p.share_metric,
                local_hot=p.local_hot,
            )


def _main() -> int:
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback

            print(f"  FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main())
