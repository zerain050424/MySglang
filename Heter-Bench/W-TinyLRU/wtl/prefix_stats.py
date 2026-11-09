"""共享前缀统计（M1 指标的基准）。

输入是 session_data.jsonl，输出 ``prefix_stats.json``：

    {"tasks": {"<task>": {
        "num_sessions": 500,
        "lcp_p50": 1893, "lcp_p05": 1024, "lcp_p95": 2140,
        "layers": [256, 512, 1024],
        "layer_shared_pct": {"256": 0.99, "512": 0.99, "1024": 0.97},
        "turn_prompt_median": [1893, 2100, 2300]
    }}}

**为什么这是「共享前缀保留度」的基准**：判据是客户端可算的
``cached_tokens >= L_k``（全保留）与 ``min(cached, L_k)/L_k``（覆盖度），
不需要任何服务端埋点。这里给出的 ``L_k`` 就是分层深度。

口径说明
--------
* 跨会话共享前缀：用 turn_1 的 ``input_ids``（多轮数据里 turn_1 已是完整 system+首轮）；
* 成对 LCP 抽样 ``--max-pairs`` 次（默认 200，seed 固定）——O(n²) 全量在 5k session 上无必要；
* 会话内共享前缀（ShareGPT 的 H5 护栏）：按「第 t 轮 prompt 长度」的中位数刻画，
  命中判据改成「本轮命中的 token 数 ≥ 上一轮 prompt 长度」。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

from .trace import load_sessions

DEFAULT_LAYERS = (256, 512, 1024, 2048, 4096, 8192)


def lcp_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _percentile(xs: List[int], p: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return int(round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo)))


def compute(
    sessions: Sequence[dict],
    *,
    max_pairs: int = 200,
    seed: int = 42,
    layers: Sequence[int] = DEFAULT_LAYERS,
) -> dict:
    prefixes = []
    turn_lengths: List[List[int]] = []
    for sess in sessions:
        t1 = sess.get("turn_1") or {}
        ids = t1.get("input_ids") or []
        prefixes.append(ids)
        lengths = []
        for t in range(1, int(sess.get("num_turns", 0)) + 1):
            turn = sess.get(f"turn_{t}") or {}
            if turn.get("input_ids"):
                lengths.append(len(turn["input_ids"]))
        turn_lengths.append(lengths)

    n = len(prefixes)
    rng = random.Random(seed)
    pairs: List[int] = []
    if n >= 2:
        for _ in range(min(max_pairs, n * (n - 1) // 2)):
            i, j = rng.sample(range(n), 2)
            pairs.append(lcp_len(prefixes[i], prefixes[j]))

    # 固定层（256/512/...）是给「深共享前缀」用的；真实数据里跨会话 LCP 可能只有
    # 几十个 token（实测 AgentBank/ShareGPT 的 turn_1 LCP 中位数 = 24），
    # 这时按数据自身分位给层，否则保留度永远是 N/A。
    used_layers = [L for L in layers if pairs and L <= max(pairs)]
    if not used_layers and pairs and max(pairs) > 0:
        used_layers = sorted({max(1, _percentile(pairs, 5)), max(1, _percentile(pairs, 50))})
    layer_pct = {
        str(L): round(sum(1 for v in pairs if v >= L) / max(len(pairs), 1), 4)
        for L in used_layers
    }

    max_turns = max((len(x) for x in turn_lengths), default=0)
    turn_median = [
        _percentile([x[t] for x in turn_lengths if len(x) > t], 50)
        for t in range(max_turns)
    ]

    return {
        "num_sessions": n,
        "num_pairs": len(pairs),
        "lcp_p50": _percentile(pairs, 50),
        "lcp_p05": _percentile(pairs, 5),
        "lcp_p95": _percentile(pairs, 95),
        "layers": used_layers,
        "layer_shared_pct": layer_pct,
        "turn_prompt_median": turn_median,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="共享前缀 LCP 统计")
    p.add_argument("--data-path", action="append", required=True,
                   help="session_data.jsonl（可重复，每个文件一个 task）")
    p.add_argument("--task", action="append", default=None,
                   help="与 --data-path 一一对应的 task 名；缺省用文件名推断")
    p.add_argument("--num-sessions", type=int, default=-1)
    p.add_argument("--read-limit", type=int, default=0)
    p.add_argument("--shuffle", action="store_true", help="先按 seed 打乱再截取")
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", default="256,512,1024,2048,4096,8192")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    tasks = a.task or []
    out: Dict[str, dict] = {}
    for i, path in enumerate(a.data_path):
        name = tasks[i] if i < len(tasks) else _infer_task(path)
        sessions = load_sessions(
            path, a.num_sessions, seed=a.seed if a.shuffle else None,
            shuffle=a.shuffle, read_limit=a.read_limit,
        )
        stats = compute(sessions, max_pairs=a.max_pairs, seed=a.seed, layers=layers)
        stats["source"] = str(Path(path).resolve())
        out[name] = stats
        print(
            f"[prefix_stats] {name}: n={stats['num_sessions']} "
            f"lcp_p50={stats['lcp_p50']} p05={stats['lcp_p05']} layers={stats['layers']}"
        )

    payload = {"tasks": out, "max_pairs": a.max_pairs, "seed": a.seed}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[prefix_stats] -> {a.out}")


def _infer_task(path: str) -> str:
    stem = Path(path).stem
    for key in ("AgentBank_alfred", "AgentBank_apps", "Codeparrot", "ShareGPT", "ShareGPT_cleaned"):
        if key.lower() in stem.lower():
            return key
    return stem[:32]


if __name__ == "__main__":
    main()
