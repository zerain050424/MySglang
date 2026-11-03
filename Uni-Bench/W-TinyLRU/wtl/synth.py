"""合成「共享前缀强度阶梯」trace（设计文档 §5.1，场景 A）。

生成的是与 ``Uni-Bench/TaskData`` 同格式的 session_data.jsonl，
因此可以直接喂给 ``wtl.replay`` / ``bench_unicache.py``：

    {"<id>": {"id": ..., "num_turns": N, "turn_1": {...}, ...}}

三个杠杆（各自单调，便于把机制和噪声分开）：

    --prefix-depth   共享前缀深度（512 / 2048 / 8192）
    --breadth        每个变体被多少个 session 共享（3 / 10 / 50）
    --variants       变体数（模板层数，决定分叉点数量）
    --noise-frac     一次性文档 session 占比（扫描压力，设计文档的「单次询问」靶子）

另外附一条 ``meta.json``，记录共享前缀的实际长度与变体表，
使 M1 指标不必再跑 prefix_stats。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

_VOCAB = 30_000
_TOKEN_BASE = 1_000
_VARIANT_LEN = 256
_OUTPUT_LEN = 64


def _rand_tokens(rng: random.Random, n: int) -> List[int]:
    return [rng.randrange(_TOKEN_BASE, _TOKEN_BASE + _VOCAB) for _ in range(n)]


def build(
    *,
    prefix_depth: int,
    breadth: int,
    variants: int,
    turns: int,
    body_mu: float,
    body_sigma: float,
    noise_frac: float,
    doc_len: int,
    seed: int,
    max_sessions: int = -1,
    phased: bool = False,
) -> Dict[str, object]:
    rng = random.Random(seed)
    shared = _rand_tokens(rng, prefix_depth)
    variant_tokens = [_rand_tokens(random.Random(seed * 7919 + i), _VARIANT_LEN) for i in range(variants)]

    sessions_by_variant: Dict[int, List[Dict[str, object]]] = {}
    meta_variants = []

    n_template_sessions = breadth * variants
    for v in range(variants):
        for b in range(breadth):
            sid = f"synth-v{v}-s{b}-seed{seed}"
            s_rng = random.Random(hash((seed, v, b)) & 0xFFFFFFFF)
            body_len = max(64, int(s_rng.lognormvariate(body_mu, body_sigma)))
            body = _rand_tokens(s_rng, body_len)
            prompt = shared + variant_tokens[v]
            turn_prompt_len = len(prompt)
            session: Dict[str, object] = {"id": sid, "num_turns": turns}
            acc_out: List[int] = []
            for t in range(1, turns + 1):
                # 每轮：历史（前缀+本体+上轮输出）+ 本轮新增用户 token
                user_tokens = _rand_tokens(s_rng, max(32, body_len // max(turns, 1)))
                prompt = prompt + acc_out + user_tokens
                out = _rand_tokens(s_rng, _OUTPUT_LEN)
                session[f"turn_{t}"] = {
                    "input_ids": list(prompt),
                    "response_ids": list(out),
                    "prompt_tokens": len(prompt),
                }
                acc_out = out
            sessions_by_variant.setdefault(v, []).append(session)
        meta_variants.append({"variant": v, "sessions": breadth})

    # 一次性文档：唯一 token 前缀，单轮，制造扫描压力
    n_noise = int(round(n_template_sessions * noise_frac))
    noise_sessions: List[Dict[str, object]] = []
    for i in range(n_noise):
        sid = f"synth-noise-{i}-seed{seed}"
        rng_doc = random.Random(hash((seed, "noise", i)) & 0xFFFFFFFF)
        doc = _rand_tokens(rng_doc, doc_len)
        noise_sessions.append(
            {
                "id": sid,
                "num_turns": 1,
                "turn_1": {
                    "input_ids": doc,
                    "response_ids": _rand_tokens(rng_doc, _OUTPUT_LEN),
                    "prompt_tokens": len(doc),
                },
            }
        )

    if phased:
        # 分阶段：变体块之间插一段一次性文档，作为「冲刷」。
        # 目的：让某个变体的子链被删光 → 共享前缀段变成可淘汰叶子 →
        # 成员真正进入 window/probation/protected，从而触发频数准入与 aging。
        # 随机混流下共享前缀几乎总是有多个子节点，永远是 parked，策略无从行使。
        sessions: List[Dict[str, object]] = []
        per_block = max(1, len(noise_sessions) // max(variants, 1))
        for v in range(variants):
            sessions.extend(sessions_by_variant.get(v, []))
            start = v * per_block
            sessions.extend(noise_sessions[start : start + per_block])
        sessions.extend(noise_sessions[variants * per_block :])
    else:
        sessions = [s for v in range(variants) for s in sessions_by_variant.get(v, [])]
        sessions.extend(noise_sessions)
        rng.shuffle(sessions)
    if max_sessions and max_sessions > 0:
        sessions = sessions[:max_sessions]

    meta = {
        "kind": "synth-prefix-ladder",
        "seed": seed,
        "prefix_depth": prefix_depth,
        "shared_prefix_tokens": prefix_depth,
        "variant_len": _VARIANT_LEN,
        "variants": meta_variants,
        "breadth": breadth,
        "turns": turns,
        "noise_sessions": n_noise,
        "phased": phased,
        "noise_frac": noise_frac,
        "doc_len": doc_len,
        "body_mu": body_mu,
        "body_sigma": body_sigma,
        "num_sessions": len(sessions),
        "layers": [x for x in (256, 512, 1024, 2048, 4096, 8192) if x <= prefix_depth],
    }
    return {"sessions": sessions, "meta": meta}


def write(out_dir: str | Path, built: Dict[str, object]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "session_data.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for sess in built["sessions"]:  # type: ignore[index]
            f.write(json.dumps({sess["id"]: sess}, ensure_ascii=False) + "\n")
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(built["meta"], f, ensure_ascii=False, indent=2)
    return data


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="生成合成共享前缀阶梯 trace")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix-depth", type=int, default=2048)
    p.add_argument("--breadth", type=int, default=10)
    p.add_argument("--variants", type=int, default=4)
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--body-mu", type=float, default=6.5)   # ~665 tokens
    p.add_argument("--body-sigma", type=float, default=0.8)
    p.add_argument("--noise-frac", type=float, default=0.0)
    p.add_argument("--doc-len", type=int, default=2000)
    p.add_argument("--max-sessions", type=int, default=-1)
    p.add_argument("--phased", action="store_true",
                   help="按变体分块并在块间插入冲刷文档（门控 G2 需要）")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    built = build(
        prefix_depth=a.prefix_depth, breadth=a.breadth, variants=a.variants,
        turns=a.turns, body_mu=a.body_mu, body_sigma=a.body_sigma,
        noise_frac=a.noise_frac, doc_len=a.doc_len, seed=a.seed,
        max_sessions=a.max_sessions, phased=a.phased,
    )
    path = write(a.out_dir, built)
    meta = built["meta"]
    print(f"[synth] {path}  sessions={meta['num_sessions']} "
          f"prefix={meta['prefix_depth']} breadth={meta['breadth']} "
          f"variants={a.variants} noise={meta['noise_sessions']}")


if __name__ == "__main__":
    main()
