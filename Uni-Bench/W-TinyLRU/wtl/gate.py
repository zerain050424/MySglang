"""Phase-0 前置门控（设计文档 §10）。

不过门控就不该跑正式矩阵：没有机制生效证据的实验结果，只能说明「测了个寂寞」。

检查项
------
G1 可复现         同配置跑两遍，逐请求明细必须完全一致（含 SimClock，无 wall-clock 依赖）
G2 机制被行使     共享数被记账过（slru.share_counted>0）且 SLRU 给出过候选（slru_picks>0）
G3 无压力不变性   pool/工作集 = 2.0 时 A0/A1/A2/A3 的 hit% 差 < 0.5pp（对照必须是平的）
G4 后端忠实度     真 LRU 与镜像 LRU 在同一小 trace 上的 cached_len 一致率 ≥ 99.5%
G5 上溯闸         A3 下受保护段（region 3）被「级联删除」的 token == 0（最大正确性风险）

    python -m wtl.gate --backend auto
    python -m wtl.gate --backend fake        # 无 sglang 的机器上只验链路（G2/G4 记 N/A）
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from .arms import ARMS
from .backends import BackendUnavailable
from .replay import calibrate_working_set, layers_from_requests, run_replay
from .synth import build as build_synth
from .synth import write as write_synth
from .trace import schedule_single_workload

# 门控用的合成 trace：宽度要够大，才能让「成员降级为可淘汰叶子」的数量
# 超过 window_entries（默认 64），从而真正触发频数准入。
SYNTH = dict(
    # 关键：**不要把一次性文档（noise）放进来**。
    # 一次性内容走 one-shot 硬优先带，会吃掉全部淘汰压力，
    # one-shot 硬优先带一旦被一次性内容占满，队内策略就没有决策机会（见 README §7）。
    prefix_depth=2048, breadth=20, variants=2, turns=4,
    body_mu=6.0, body_sigma=0.8, noise_frac=0.0, doc_len=1200, seed=7,
    phased=True,
)
# G2 用更小的池子逼出 global 带的淘汰（一次测试约 160 请求）
G2_POOL_RATIO = 0.10


def _requests(tmp: Path, num_sessions: int = 250):
    built = build_synth(max_sessions=num_sessions, **SYNTH)
    path = write_synth(tmp, built)
    return schedule_single_workload(
        "synth", num_sessions=-1, seed=7, wait_scale=1.0, arrival_scale=1.0,
        data_path=str(path), lambda_rate=2.0,
    )


def _rows_hash(result: dict) -> str:
    import hashlib

    h = hashlib.sha256()
    for r in result.get("rows") or []:
        h.update(
            f"{r['ts']}|{r['sid']}|{r['turn']}|{r['prompt_tokens']}|"
            f"{r['cached_tokens']}|{r['evicted']}|{r['oom']}\n".encode()
        )
    return h.hexdigest()[:16]


def _run(requests, arm_id: str, pool_tokens: int, backend: str, layers) -> dict:
    return run_replay(
        requests, arm=ARMS[arm_id], pool_tokens=pool_tokens, backend_kind=backend,
        layers_by_task=layers, regret_window=60.0, collect_rows=True, quiet=True,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase-0 门控")
    p.add_argument("--backend", default="auto", choices=["auto", "real", "fake"])
    p.add_argument("--arm", default="A3", help="门控主对象（默认 A3 = 标签版 TinyLRU）")
    p.add_argument("--out", default="result/gate.json")
    a = p.parse_args(argv)

    results: List[Tuple[str, str, str, str]] = []  # (id, name, status, detail)
    payload: Dict[str, object] = {"backend": a.backend, "checks": []}

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        arm = a.arm
        requests = _requests(tmp)
        layers = layers_from_requests(requests, seed=7)
        # 压力点必须相对「工作集」而不是「请求总量」：
        # 工作集 = 不淘汰时的树占用（用镜像树校准），否则共享前缀会让
        # 「请求总量」远大于真正需要驻留的 token，压力点被稀释成无淘汰。
        wk = calibrate_working_set(requests)
        demand = sum(r.total_len for r in requests)
        pool = max(1, int(wk * 0.25))
        g2_pool = max(1, int(wk * G2_POOL_RATIO))
        print(f"[gate] trace: {len(requests)} reqs, demand={demand}, "
              f"working_set={wk}, pool={pool} ({pool / max(wk,1):.2f}x), "
              f"layers={ {k: v for k, v in layers.items()} }")

        # ---------------- G1 ----------------
        try:
            r1 = _run(requests, arm, g2_pool, a.backend, layers)
            r2 = _run(requests, arm, g2_pool, a.backend, layers)
            h1, h2 = _rows_hash(r1), _rows_hash(r2)
            ok = h1 == h2
            results.append(("G1", "可复现", "PASS" if ok else "FAIL", f"{h1} vs {h2}"))
            payload["checks"].append({"id": "G1", "pass": ok, "hash": [h1, h2],
                                      "backend": r1["backend"]})
        except BackendUnavailable as exc:
            results.append(("G1", "可复现", "N/A", str(exc)[:80]))
            r1 = None

        # ---------------- G2 / G5 ----------------
        if r1 is not None and r1["backend"] == "real":
            s = r1["summary"]
            slru = s.get("slru") or {}
            vsrc = s.get("victim_src") or {}
            counted = int(slru.get("share_counted") or 0)
            picks = int(slru.get("slru_picks") or 0)
            q_evicted = float(vsrc.get("qstruct", 0.0) or 0.0)
            os_evicted = float(vsrc.get("one_shot", 0.0) or 0.0)
            # 机制被行使 = 共享数真的被记账过 + SLRU 真的给出过候选
            exercised = counted > 0 and picks > 0
            results.append(
                ("G2", "机制被行使", "PASS" if exercised else "FAIL",
                 f"share_counted={counted} share_dedup={slru.get('share_dedup')} "
                 f"slru_picks={picks} share_min={s.get('slru_share_min')} "
                 f"one_shot_evicted={os_evicted:.0f} qstruct_evicted={q_evicted:.0f}")
            )
            cascade = (s.get("regret") or {}).get("protected_cascade_deleted_tokens", 0)
            results.append(
                ("G5", "上溯闸", "PASS" if cascade == 0 else "FAIL",
                 f"受保护前缀被级联删除 {cascade} tokens")
            )
            payload["checks"].append({"id": "G2", "pass": exercised, "slru": slru,
                                      "victim_src": vsrc})
            payload["checks"].append({"id": "G5", "pass": cascade == 0,
                                      "protected_cascade_tokens": cascade})

        else:
            results.append(("G2", "机制被行使", "N/A", "需要 RealBackend"))
            results.append(("G5", "上溯闸", "N/A", "需要 RealBackend"))
            payload["checks"] += [{"id": "G2", "pass": None}, {"id": "G5", "pass": None}]

        # ---------------- G3 ----------------
        try:
            hits = {}
            wk_pool = int(wk * 2.0)
            for arm in ("A0", "A1", "A2", "A3"):
                rr = _run(requests, arm, wk_pool, a.backend, layers)
                if rr["backend"] != "real":
                    raise BackendUnavailable("FakeBackend 不体现策略差异")
                hits[arm] = rr["summary"]["hit_pct"]
            spread = max(hits.values()) - min(hits.values())
            ok = spread < 0.5
            results.append(("G3", "无压力不变性", "PASS" if ok else "FAIL",
                            " ".join(f"{k}={v:.2f}" for k, v in hits.items()) + f" spread={spread:.3f}pp"))
            payload["checks"].append({"id": "G3", "pass": ok, "hits": hits, "spread": spread})
        except BackendUnavailable as exc:
            results.append(("G3", "无压力不变性", "N/A", str(exc)[:80]))
            payload["checks"].append({"id": "G3", "pass": None})

        # ---------------- G4 ----------------
        if r1 is not None and r1["backend"] == "real":
            # 语义忠实度：在「无淘汰」的无限池下，真 LRU 与镜像 LRU 的
            # cached_tokens 必须逐请求一致（有淘汰时两者的 LRU 打破平局顺序
            # 会不同——分段粒度和时间戳刷新不同，那属于实现细节，由 G3 兜底）。
            inf_pool = int(wk * 2.0) + 1
            fake = _run(requests, "A0", inf_pool, "fake", layers)
            real = _run(requests, "A0", inf_pool, "real", layers)
            kr = {(r["task"], r["sid"], r["turn"]): r for r in real["rows"]}
            kf = {(r["task"], r["sid"], r["turn"]): r for r in fake["rows"]}
            keys = sorted(set(kr) & set(kf))
            same = sum(1 for k in keys if kr[k]["cached_tokens"] == kf[k]["cached_tokens"])
            ratio = same / max(len(keys), 1)
            ok = ratio >= 0.995
            results.append(("G4", "后端忠实度", "PASS" if ok else "FAIL",
                            f"无限池下 {ratio * 100:.2f}% 请求的 cached_tokens 与镜像 LRU 逐请求一致 (n={len(keys)})"))
            payload["checks"].append({"id": "G4", "pass": ok, "match_ratio": ratio})
        else:
            results.append(("G4", "后端忠实度", "N/A", "需要 RealBackend"))
            payload["checks"].append({"id": "G4", "pass": None})

    # ---------------- 输出 ----------------
    print("\n=== W-TinyLRU Phase-0 Gate ===")
    for gid, name, status, detail in results:
        print(f"  {gid:3s} {name:14s} {status:4s}  {detail}")
    passed = all(s != "FAIL" for _, _, s, _ in results)
    print(f"=== {'PASS' if passed else 'FAIL'} ===")
    payload["passed"] = passed
    payload["results"] = [
        {"id": i, "name": n, "status": s, "detail": d} for i, n, s, d in results
    ]
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gate] -> {outp}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
