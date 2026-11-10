#!/usr/bin/env python3
"""构造 exp3 的五类混合负载数据。

五类负载彼此独立，各自的 token 占比只由自身规模决定：

  chat     ShareGPT 多轮（6–8 轮为主，20% 取 10–15 轮）
  agentic  AgentBank alfred（30 条 60+ 轮长尾，其余 6–10 轮）
  apps     AgentBank_apps 单轮（真实会话 + 合成冲刷会话，二者都无复用）
  struct   独立构造：K 条长前缀链，每条链上的会话逐轮共用同一段前缀

struct 是唯一带「跨会话共享前缀」的负载：其他四类保持原始文本，不做任何前缀拼接，
免得把 struct 的 token 摊到别的类别上。想让 struct 占比更高，就调
``--struct-prefix-tokens`` / ``--struct-sessions`` 这两个旋钮，别动别的类。

输入是已经剥过 Qwen system 段的 ``*.nosys.jsonl``（由 ``mixexp/prepare_data.sh``
生成）。输出写到 ``TaskData/exp3/``，每行一个 ``{session_id: session}`` 对象。

用法
----
    # 全量构造
    python mixexp/tools/build_exp3_data.py --tokenizer /root/autodl-tmp/model/Qwen2.5-7B-Instruct

    # 只造 struct，并把前缀拉到 4000 token
    python mixexp/tools/build_exp3_data.py --only struct --struct-prefix-tokens 4000

    # 只看计划
    python mixexp/tools/build_exp3_data.py --dry-run

字段约定（与 bench 读取方式一致）：
  session = {"id": str, "num_turns": int, "turn_1": {"input_ids": [...],
             "response_ids": [...], ...}, ...}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

BENCH_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_TASKDATA = BENCH_DIR / "TaskData"
DEFAULT_OUT = DEFAULT_TASKDATA / "exp3"
DEFAULT_TOKENIZER = "/root/autodl-tmp/model/Qwen2.5-7B-Instruct"

# struct 负载的前缀文本：一份较长的技术文档，按变体切片后铺到目标长度。
# 只有这段文本会在会话之间共享，其他四类不含公共前缀。
STRUCT_DOC = (
    "Service architecture notes. The gateway terminates client connections, validates the "
    "authentication token, and forwards the request to a scheduler. The scheduler maintains "
    "a queue per tenant, admits requests according to the configured concurrency, and records "
    "an admission timestamp used later for latency accounting. Each worker owns a fixed share "
    "of the key-value cache and reports usage counters every two hundred milliseconds. "
    "When the cache is under pressure the eviction policy walks the tree from the least "
    "recently used leaf upward, releasing whole subtrees before individual nodes, because a "
    "partial subtree cannot be reused by any later request. Structural nodes, meaning nodes "
    "shared by two or more sessions, are handled by a separate accounting path so that a burst "
    "of single-use requests cannot displace them. The accounting path keeps a per-queue quota, "
    "a reservation floor, and a loss cap that bounds how much of a single session may be "
    "dropped in one eviction pass. Metrics are exported as counters: prompt tokens, cached "
    "tokens, evicted tokens, and the number of requests that had to wait for capacity. "
    "Retention is measured by replaying the same arrival trace offline and comparing, for each "
    "request, the depth of the matching prefix at admission time against the depth it had when "
    "it was last seen. Every offline replay is checked against the online counters; a replay "
    "that diverges by more than a few percentage points is treated as a modelling bug rather "
    "than a policy difference. The operations runbook lists the following invariants: a request "
    "never observes a partially released prefix, capacity accounting never exceeds the pool, "
    "and the eviction order is stable across runs that share the same seed. "
)


@dataclass
class TaskSpec:
    """一类负载的构造规则。"""

    name: str
    source: str                       # 相对 TaskData 的源文件；struct 为空
    target: int                       # 目标会话数
    buckets: List[tuple] = field(default_factory=list)  # [(轮数下限, 轮数上限, 占比)]
    shuffle_seed: int = 42

    def bucket_of(self, num_turns: int) -> Optional[int]:
        for i, (lo, hi, _) in enumerate(self.buckets):
            if lo <= num_turns <= hi:
                return i
        return None


SPECS: Dict[str, TaskSpec] = {
    # chat：8 轮均值，其中 20% 取 10–15 轮
    "chat": TaskSpec(
        name="chat",
        source="SharedGPT-multi-turn/Qwen2.5-7B-Instruct/"
               "ShareGPT_cleaned_valid_deduplicated_gt5_session_data.nosys.jsonl",
        target=800,
        buckets=[(6, 8, 0.80), (10, 15, 0.20)],
    ),
    # agentic：30 条 60+ 轮长尾，其余 6–10 轮
    "agentic": TaskSpec(
        name="agentic",
        source="AgentBankData/Qwen2.5-7B-Instruct/"
               "AgentBank_alfred_scale_5000_session_data.nosys.jsonl",
        target=96,
        buckets=[(60, 10 ** 6, 30 / 96), (6, 10, 66 / 96)],
    ),
    # apps：单轮，真实会话用尽后用同长度分布的合成会话补足（无复用，纯冲刷）
    "apps": TaskSpec(
        name="apps",
        source="AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.nosys.jsonl",
        target=15000,
        buckets=[(1, 1, 1.0)],
    ),
    # struct：独立构造，单轮，同一变体内的会话共用一段长前缀
    "struct": TaskSpec(
        name="struct",
        source="",
        target=2600,
        buckets=[(1, 1, 1.0)],
    ),
}

# struct 每条会话的私有正文长度（token），用来把共享前缀和私有部分区分开
STRUCT_BODY_TOKENS = 64


# ---------------------------------------------------------------------------
# 读取与筛选
# ---------------------------------------------------------------------------


def iter_sessions(path: Path) -> Iterable[Dict[str, Any]]:
    """逐行产出 {session_id: session} 中的 session。"""
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} 不是合法 JSON：{exc}")
            for sid, session in obj.items():
                yield {"session_id": sid, **session}
                break  # 每行只含一条会话


def select_by_bucket(
    sessions: Sequence[Dict[str, Any]], spec: TaskSpec, target: Optional[int] = None
) -> List[Dict[str, Any]]:
    """按桶占比挑会话；某一桶不够时用剩余会话补足。"""
    target = spec.target if target is None else target
    rng = random.Random(spec.shuffle_seed)
    groups: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(spec.buckets))}
    leftover: List[Dict[str, Any]] = []
    for session in sessions:
        idx = spec.bucket_of(int(session.get("num_turns") or 0))
        if idx is None:
            continue
        groups[idx].append(session)
    for idx in groups:
        rng.shuffle(groups[idx])

    picked: List[Dict[str, Any]] = []
    for idx, (_lo, _hi, share) in enumerate(spec.buckets):
        want = int(round(target * share))
        picked.extend(groups[idx][:want])
        leftover.extend(groups[idx][want:])

    if len(picked) < target:
        rng.shuffle(leftover)
        picked.extend(leftover[: target - len(picked)])
    rng.shuffle(picked)
    return picked[:target]


# ---------------------------------------------------------------------------
# apps：真实会话 + 合成冲刷会话
# ---------------------------------------------------------------------------


def synth_apps_sessions(
    real: Sequence[Dict[str, Any]], count: int, seed: int = 42
) -> List[Dict[str, Any]]:
    """按真实 apps 的 prompt 长度分布生成 count 条无复用的单轮会话。

    这些会话充当纯冲刷流量：每条 body 都是独立随机 token，长度从真实分布里采样，
    与其他会话没有任何共享前缀，也不会被误判成结构性节点。
    """
    lengths: List[int] = []
    responses: List[int] = []
    for session in real:
        turn = session.get("turn_1") or {}
        ids = turn.get("input_ids") or []
        if ids:
            lengths.append(len(ids))
        rid = turn.get("response_ids") or []
        if rid:
            responses.append(len(rid))
    if not lengths:
        raise SystemExit("apps 源数据为空，无法采样长度分布")

    rng = random.Random(seed)
    default_resp = sorted(responses)[len(responses) // 2] if responses else 3
    sessions: List[Dict[str, Any]] = []
    for index in range(count):
        sid = f"apps-synth-{index:05d}"
        body_len = rng.choice(lengths)
        body = [rng.randrange(1000, 30000) for _ in range(body_len)]
        sessions.append({
            "session_id": sid,
            "id": sid,
            "num_turns": 1,
            "turn_1": {
                "input_text": f"[apps synth request {index}]",
                "input_ids": body,
                "response_text": "ok",
                "response_ids": [13, 271, 488][:default_resp] or [13],
            },
        })
    return sessions


# ---------------------------------------------------------------------------
# struct：K 条长前缀链
# ---------------------------------------------------------------------------


def build_struct_sessions(
    spec: TaskSpec, tokenizer, prefix_tokens: int, variants: int, target: int
) -> List[Dict[str, Any]]:
    """生成 struct 会话：变体 v 内的所有会话共用同一段 prefix_tokens 长的前缀。

    变体之间前缀互不重叠（把文档按变体切块后各自铺满），因此不会额外长出
    跨变体的共享链；私有正文每会话唯一，保证除前缀外没有别的共享节点。
    """
    base_ids = tokenizer(STRUCT_DOC, add_special_tokens=False).input_ids
    if not base_ids:
        raise SystemExit("tokenizer 把 STRUCT_DOC 编码成了空序列")

    rng = random.Random(spec.shuffle_seed)
    per_variant = [target // variants] * variants
    for i in range(target - sum(per_variant)):
        per_variant[i] += 1

    sessions: List[Dict[str, Any]] = []
    for variant, count in enumerate(per_variant):
        # 每个变体取文档的一个切片，再自我平铺到目标长度
        chunk = base_ids[variant::variants] or base_ids
        prefix: List[int] = []
        while len(prefix) < prefix_tokens:
            prefix.extend(chunk)
        prefix = prefix[:prefix_tokens]

        for index in range(count):
            sid = f"struct-v{variant:02d}-{index:04d}"
            body = [rng.randrange(1000, 30000) for _ in range(STRUCT_BODY_TOKENS)]
            sessions.append({
                "session_id": sid,
                "id": sid,
                "num_turns": 1,
                "turn_1": {
                    "input_text": f"[struct v{variant} prefix + body {index}]",
                    "input_ids": list(prefix) + body,
                    "response_text": "ok",
                    "response_ids": [13, 271, 488],
                },
            })
    rng.shuffle(sessions)
    return sessions


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def load_tokenizer(path: str):
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            "缺少 transformers，无法编码 struct 前缀。请在装有该依赖的环境里运行，"
            "或用 --tokenizer 指向可加载的模型目录。"
        )
    return AutoTokenizer.from_pretrained(path)


def write_sessions(out_path: Path, sessions: Sequence[Dict[str, Any]]) -> int:
    """写 jsonl，返回 prompt token 总数。"""
    total_tokens = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            num_turns = int(session.get("num_turns") or 0)
            for i in range(1, num_turns + 1):
                turn = session.get(f"turn_{i}")
                if isinstance(turn, dict):
                    total_tokens += len(turn.get("input_ids") or [])
            handle.write(json.dumps({session["session_id"]: session}, ensure_ascii=False) + "\n")
    return total_tokens


def build_one(spec: TaskSpec, args, tokenizer) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}_exp3.nosys.jsonl"

    if spec.name == "struct":
        count = args.struct_sessions or spec.target
        sessions = build_struct_sessions(
            spec, tokenizer, args.struct_prefix_tokens, args.struct_variants, count
        )
    else:
        src = Path(args.taskdata) / spec.source
        if not src.is_file():
            raise SystemExit(f"缺少输入数据：{src}\n先运行：bash mixexp/prepare_data.sh")
        target = spec.target
        if spec.name == "apps" and args.apps_sessions:
            target = args.apps_sessions
        sessions = select_by_bucket(list(iter_sessions(src)), spec, target)
        if spec.name == "apps" and len(sessions) < target:
            extra = synth_apps_sessions(sessions, target - len(sessions), spec.shuffle_seed)
            sessions = list(sessions) + extra
        if args.limit:
            sessions = sessions[: args.limit]

    total_tokens = write_sessions(out_path, sessions)
    turns = [int(s.get("num_turns") or 0) for s in sessions]
    info = {
        "task": spec.name,
        "output": str(out_path),
        "sessions": len(sessions),
        "requests": sum(turns),
        "prompt_tokens": total_tokens,
        "turns_min": min(turns) if turns else 0,
        "turns_max": max(turns) if turns else 0,
        "turns_mean": round(sum(turns) / len(turns), 2) if turns else 0,
    }
    if spec.name == "apps":
        info["synthetic"] = sum(1 for s in sessions if str(s["session_id"]).startswith("apps-synth"))
        info["real"] = len(sessions) - info["synthetic"]
    if spec.name == "struct":
        info["prefix_tokens"] = args.struct_prefix_tokens
        info["variants"] = args.struct_variants
        extra = f"  前缀 {args.struct_prefix_tokens} tok × {args.struct_variants} 变体"
    elif spec.name == "apps":
        extra = f"  真实 {info['real']} + 合成冲刷 {info['synthetic']}"
    else:
        extra = ""
    print(
        f"  {spec.name:8s} 会话 {info['sessions']:>5}  请求 {info['requests']:>6}"
        f"  轮数 {info['turns_min']}–{info['turns_max']}（均值 {info['turns_mean']}）"
        f"  prompt {info['prompt_tokens']/1e6:5.2f}M{extra}"
    )
    return info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="构造 exp3 五类混合负载数据（struct 独立一类）"
    )
    parser.add_argument("--taskdata", default=str(DEFAULT_TASKDATA), help="TaskData 根目录")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="输出目录")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="tokenizer 路径")
    parser.add_argument("--only", default="", help="只构造某一类：chat/agentic/apps/struct")
    parser.add_argument("--limit", type=int, default=0, help="每类最多取多少条（调试用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不读数据")
    # struct 的两个旋钮：占比只由它们决定
    parser.add_argument("--apps-sessions", type=int, default=0,
                        help="apps 会话数（默认 15000；超出真实数据部分用合成冲刷会话补足）")
    parser.add_argument("--struct-sessions", type=int, default=0,
                        help="struct 会话数（默认 2600）")
    parser.add_argument("--struct-prefix-tokens", type=int, default=2000,
                        help="struct 每条会话的共享前缀长度（默认 2000）")
    parser.add_argument("--struct-variants", type=int, default=8,
                        help="struct 前缀链条数（默认 8）")
    args = parser.parse_args()

    names = list(SPECS) if not args.only else [args.only]
    for name in names:
        if name not in SPECS:
            raise SystemExit(f"未知类别：{name}（可选 {list(SPECS)}）")

    if args.dry_run:
        print("计划：")
        for name in names:
            spec = SPECS[name]
            if name == "struct":
                count = args.struct_sessions or spec.target
                tokens = count * (args.struct_prefix_tokens + STRUCT_BODY_TOKENS)
                print(f"  struct   会话 {count:>5}  前缀 {args.struct_prefix_tokens} tok"
                      f" × {args.struct_variants} 变体  ≈{tokens/1e6:.2f}M token")
                continue
            target = args.apps_sessions or spec.target if name == "apps" else spec.target
            note = "（真实数据不足部分用合成冲刷会话补足）" if name == "apps" else ""
            print(f"  {name:8s} 目标 {target:>5} 条  源 {spec.source}{note}")
            for lo, hi, share in spec.buckets:
                hi_s = "∞" if hi >= 10 ** 6 else hi
                print(f"           轮数 {lo}–{hi_s} 占 {share*100:.0f}%")
        return 0

    tokenizer = load_tokenizer(args.tokenizer)
    print("构造 exp3 数据：")
    infos = [build_one(SPECS[name], args, tokenizer) for name in names]

    total = sum(i["prompt_tokens"] for i in infos) or 1
    print("\nprompt token 占比：")
    for info in infos:
        print(f"  {info['task']:8s} {info['prompt_tokens']/1e6:5.2f}M  "
              f"{info['prompt_tokens']/total*100:4.1f}%")
    total_req = sum(i["requests"] for i in infos)
    print(f"  合计 {total/1e6:.2f}M token / {total_req:,} 请求")

    if len(infos) == len(SPECS):
        manifest = Path(args.out_dir) / "manifest.json"
        manifest.write_text(
            json.dumps({"tasks": infos, "prompt_tokens": total,
                        "requests": total_req}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n清单已写入 {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
