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

``--scenario dslru``（D-SLRU，本轮"两带 SLRU vs LRU vs 上游 SLRU"的主负载）
三条成分各自对准一条机制弱点：

    模板层   跨块长间隔复发      → LRU 按 recency 会把模板挤掉
    私有层   只有本会话反复读    → 上游 SLRU 按 hit_count 把它晋升进 protected
    burst    单轮唯一 token      → 制造真实容量压力

调度是决定性的：模板 v 在 ``v % blocks`` 块出生（放两个会话，先把
``share_sessions`` 顶到 >= share_min），之后每 ``+2`` 块复发一个会话，
因此相邻两次使用之间总隔着一整个 burst 块。
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


def build_dslru(
    *,
    prefix_depth: int = 2048,
    variants: int = 16,
    sessions_per_template: int = 5,
    turns: int = 3,
    private_mu: float = 7.6,
    private_sigma: float = 0.4,
    user_len: int = 128,
    output_len: int = 256,
    blocks: int = 8,
    burst_sessions: int = 384,
    doc_len: int = 1024,
    seed: int = 42,
    max_sessions: int = -1,
) -> Dict[str, object]:
    """生成 D-SLRU trace：跨块复发的模板 × 会话内私有热点 × 单轮 burst。

    单请求结构（顺序不能变）::

        [模板 prefix_depth + variant][私有上下文][历史输出][本轮用户 token]
        → 生成 output_len

    * 私有上下文是**唯一** token，且每轮都完整出现在 prompt 里 —— 它会被
      同一条会话反复命中（``hit_count`` 涨到 >=2，上游 SLRU 会晋升它），
      但 ``share_sessions`` 停在 1（bloom 对同一会话去重），所以两带 SLRU
      永远把它留在 one-shot 带里先清。
    * ``turns`` 必须 < 5（分类器 ``MULTI_TURN_MIN``），否则会话被判成
      AGENTIC/CHAT，节点就不再进一次性带，本负载的靶子失效。
    """
    rng = random.Random(seed)
    shared = _rand_tokens(rng, prefix_depth)
    variant_tokens = [
        _rand_tokens(random.Random(seed * 7919 + i), _VARIANT_LEN)
        for i in range(variants)
    ]
    turns = max(1, int(turns))
    blocks = max(1, int(blocks))
    sessions_per_template = max(3, int(sessions_per_template))

    # 模板 v 的会话落在哪些块：出生块 2 个 + 之后每 +2 块 1 个
    per_block: Dict[int, List[Dict[str, object]]] = {b: [] for b in range(blocks)}
    private_lens: List[int] = []
    template_uses: Dict[int, int] = {}
    for v in range(variants):
        birth = v % blocks
        schedule = [birth, birth] + [
            (birth + 2 * k) % blocks for k in range(1, sessions_per_template - 1)
        ]
        for j, blk in enumerate(schedule):
            sid = f"dslru-t{v}-s{j}-seed{seed}"
            s_rng = random.Random(hash((seed, "dslru", v, j)) & 0xFFFFFFFF)
            body_len = max(64, int(s_rng.lognormvariate(private_mu, private_sigma)))
            private_lens.append(body_len)
            body = _rand_tokens(s_rng, body_len)
            prompt = shared + variant_tokens[v] + body
            acc_out: List[int] = []
            session: Dict[str, object] = {"id": sid, "num_turns": turns}
            for t in range(1, turns + 1):
                prompt = prompt + acc_out + _rand_tokens(s_rng, user_len)
                out = _rand_tokens(s_rng, output_len)
                session[f"turn_{t}"] = {
                    "input_ids": list(prompt),
                    "response_ids": list(out),
                    "prompt_tokens": len(prompt),
                }
                acc_out = out
            per_block[blk].append(session)
            template_uses[v] = template_uses.get(v, 0) + turns

    # burst：单轮唯一文档，按块均分，紧跟在同块的模板会话之后
    burst_per_block = max(0, int(burst_sessions) // blocks)
    noise_n = 0
    for b in range(blocks):
        for k in range(burst_per_block):
            idx = b * burst_per_block + k
            n_rng = random.Random(hash((seed, "dslru-noise", idx)) & 0xFFFFFFFF)
            doc = _rand_tokens(n_rng, doc_len)
            per_block[b].append(
                {
                    "id": f"dslru-noise-{idx}-seed{seed}",
                    "num_turns": 1,
                    "turn_1": {
                        "input_ids": doc,
                        "response_ids": _rand_tokens(n_rng, max(32, output_len // 2)),
                        "prompt_tokens": len(doc),
                    },
                }
            )
            noise_n += 1

    sessions = [sess for b in range(blocks) for sess in per_block[b]]
    if max_sessions and max_sessions > 0:
        sessions = sessions[:max_sessions]

    template_len = prefix_depth + _VARIANT_LEN
    layers = sorted({
        x for x in (256, 512, 1024, 2048, 4096, 8192, template_len)
        if x <= template_len
    })
    meta = {
        "kind": "synth-dslru",
        "seed": seed,
        "prefix_depth": prefix_depth,
        "shared_prefix_tokens": prefix_depth,
        "variant_len": _VARIANT_LEN,
        "template_len": template_len,
        # 注意：per_block 以「块」为键，与变体下标无关（variants 常常 > blocks）。
        "variants": [
            {"variant": v, "sessions": sessions_per_template} for v in range(variants)
        ],
        "breadth": sessions_per_template,
        "turns": turns,
        "blocks": blocks,
        "burst_per_block": burst_per_block,
        "noise_sessions": noise_n,
        "noise_frac": round(noise_n / max(len(sessions) - noise_n, 1), 3),
        "doc_len": doc_len,
        "private_mu": private_mu,
        "private_sigma": private_sigma,
        "private_tokens_mean": round(sum(private_lens) / max(len(private_lens), 1)),
        "user_len": user_len,
        "output_len": output_len,
        "template_uses": template_uses,
        "num_sessions": len(sessions),
        # M1 分层：模板整段（prefix_depth + variant）也在里面
        "layers": layers,
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
                   help="[ladder] 按变体分块并在块间插入冲刷文档（门控 G2 需要）")
    p.add_argument("--scenario", default="ladder", choices=["ladder", "dslru"],
                   help="ladder=共享前缀阶梯（旧场景）；dslru=两带 SLRU 主负载")
    # ---- dslru 专用 ----
    p.add_argument("--blocks", type=int, default=8,
                   help="[dslru] 调度块数：块=模板会话 -> burst -> 下一块")
    p.add_argument("--sessions-per-template", type=int, default=5,
                   help="[dslru] 每模板会话数（>=3：出生块 2 个 + 复发）")
    p.add_argument("--private-mu", type=float, default=7.6,
                   help="[dslru] 会话私有上下文长度 lognormal mu（7.6≈2000 token）")
    p.add_argument("--private-sigma", type=float, default=0.4)
    p.add_argument("--user-len", type=int, default=128, help="[dslru] 每轮新增用户 token")
    p.add_argument("--output-len", type=int, default=256, help="[dslru] 每轮生成 token")
    p.add_argument("--burst-sessions", type=int, default=384,
                   help="[dslru] burst 会话总数（按块均分）")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    if a.scenario == "dslru":
        built = build_dslru(
            prefix_depth=a.prefix_depth, variants=a.variants,
            sessions_per_template=a.sessions_per_template, turns=a.turns,
            private_mu=a.private_mu, private_sigma=a.private_sigma,
            user_len=a.user_len, output_len=a.output_len,
            blocks=a.blocks, burst_sessions=a.burst_sessions,
            doc_len=a.doc_len, seed=a.seed, max_sessions=a.max_sessions,
        )
    else:
        built = build(
            prefix_depth=a.prefix_depth, breadth=a.breadth, variants=a.variants,
            turns=a.turns, body_mu=a.body_mu, body_sigma=a.body_sigma,
            noise_frac=a.noise_frac, doc_len=a.doc_len, seed=a.seed,
            max_sessions=a.max_sessions, phased=a.phased,
        )
    path = write(a.out_dir, built)
    meta = built["meta"]
    print(f"[synth] {path}  kind={meta['kind']} sessions={meta['num_sessions']} "
          f"prefix={meta['prefix_depth']} breadth={meta['breadth']} "
          f"variants={a.variants} noise={meta['noise_sessions']}"
          + (f" blocks={meta['blocks']}" if a.scenario == "dslru" else ""))


if __name__ == "__main__":
    main()
