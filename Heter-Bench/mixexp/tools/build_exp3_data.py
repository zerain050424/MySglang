#!/usr/bin/env python3
"""构造 exp3 的混合负载数据：四类负载的「单类主导」cell。

四类负载彼此独立（各自的 token 占比只由自身规模决定）：

  apps     AgentBank_apps 单轮（真实会话 + 合成冲刷会话，二者都无复用）
  struct   独立构造：K 条长前缀链，每条链上的会话逐轮共用同一段前缀
  chat     ShareGPT 多轮（6–8 轮为主，20% 取 10–15 轮）
  agentic  AgentBank alfred（30 条 60+ 轮长尾，其余 6–10 轮）

三种产物，两条路径：

1. **单组成（legacy）**：``--only`` 不指定 ``--mix``，按 ``SPECS`` 里的固定 target
   写四份文件到 ``--out-dir``（exp3 原有的那份 22/35/22/22 组成）。
2. **单类主导 cell**：``--mix scan|struct|chat|agentic``（可重复，或 ``all``）。
   每个 cell 让一类负载占 **≥ ``--dominant-share``（默认 60%）的 prompt token**，
   其余三类均分剩下的部分。主导形态由 token 数量决定，与会话数无关——
   这条口径必须守住：401 条单轮冲刷会话的 token 可能比 96 条 agentic 会话还少。

口径
----
- 主导类 token 份额 = 该类 prompt token / 四类 prompt token 总和；
- 配角均分 (1-share)/3；
- 先由配角实际 token 反解主导类需要多少 token：
  ``dom_tokens = share/(1-share) · others_tokens``，与总预算无关；
  ``--total-tokens`` 只用来定配角的初始规模；
- 主导类池不够时**反向缩配角**（而不是接受一低于阈值的份额），仍达不到则报错；
- 每轮的会话数由字节数实测反解，而不是先拍一个会话数——
  长会是（agentic 均值 ~89k token/条）与短会话（apps ~0.4k token/条）差两个数量级。

struct 是唯一带「跨会话共享前缀」的负载：其他三类保持原始文本，不做任何前缀拼接。

输入是已经剥过 Qwen system 段的 ``*.nosys.jsonl``（由 ``mixexp/prepare_data.sh``
生成，多轮源已剔掉轮数 < MULTI_TURN_MIN 的会话）。输出写到
``TaskData/exp3/<cell>/``：四份 ``*_exp3.nosys.jsonl`` + ``manifest.json``
+ 可直接跑的 ``tasks.yaml``（已含 max_record 与 lambda_rate）。

用法
----
    # 四个主导 cell 全造（推荐）
    python mixexp/tools/build_exp3_data.py --mix all \\
        --tokenizer /root/autodl-tmp/model/Qwen2.5-7B-Instruct

    # 只造 agentic 主导，要求 ≥70% token
    python mixexp/tools/build_exp3_data.py --mix agentic --dominant-share 0.70

    # 只看每个 cell 的 token 预算与会话数，不落盘
    python mixexp/tools/build_exp3_data.py --mix all --dry-run

    # legacy：原来的单组成（22/35/22/22）
    python mixexp/tools/build_exp3_data.py --only struct --struct-prefix-tokens 4000

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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    # chat：8 轮均值，其中 20% 取 10–15 轮。
    # 源用 prepare_data.sh 产出的 mt5（已剔除轮数 < MULTI_TURN_MIN 的会话）——
    # 否则那些 1–4 轮会话终身判 TRANS，会挤占 one-shot 带、把 chat 的命中率压下去。
    "chat": TaskSpec(
        name="chat",
        source="SharedGPT-multi-turn/Qwen2.5-7B-Instruct/"
               "ShareGPT_cleaned_valid_deduplicated_session_data.nosys.mt5.jsonl",
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
# 候选会话流：真实会话 + 合成补足（apps / struct 可无限补）
# ---------------------------------------------------------------------------


def _apps_length_profile(real: Sequence[Dict[str, Any]]) -> tuple:
    """真实 apps 的 prompt 长度分布与常见 response 长度（供合成会话采样）。"""
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
    median_resp = sorted(responses)[len(responses) // 2] if responses else 3
    return lengths, [13, 271, 488][:median_resp] or [13]


def iter_apps_candidates(
    real: Sequence[Dict[str, Any]], seed: int = 42
) -> Iterator[Dict[str, Any]]:
    """先吐真实 apps 会话，用尽后**无限**吐合成冲刷会话。

    合成会话的 body 是独立随机 token、长度从真实分布采样，与其他会话没有任何
    共享前缀，也不会被误判成结构性节点——纯冲刷流量。
    """
    yield from real
    lengths, response_ids = _apps_length_profile(real)
    rng = random.Random(seed)
    index = 0
    while True:
        sid = f"apps-synth-{index:05d}"
        yield {
            "session_id": sid,
            "id": sid,
            "num_turns": 1,
            "turn_1": {
                "input_text": f"[apps synth request {index}]",
                "input_ids": [rng.randrange(1000, 30000) for _ in range(rng.choice(lengths))],
                "response_text": "ok",
                "response_ids": list(response_ids),
            },
        }
        index += 1


def iter_struct_candidates(
    spec: TaskSpec, tokenizer, prefix_tokens: int, variants: int
) -> Iterator[Dict[str, Any]]:
    """按变体轮转**无限**吐 struct 会话。

    变体 v 内的所有会话共用同一段 prefix_tokens 长的前缀；变体之间前缀互不重叠
    （把文档按变体切块后各自铺满），因此不会额外长出跨变体的共享链；
    私有正文每会话唯一，保证除前缀外没有别的共享节点。
    """
    base_ids = tokenizer(STRUCT_DOC, add_special_tokens=False).input_ids
    if not base_ids:
        raise SystemExit("tokenizer 把 STRUCT_DOC 编码成了空序列")
    variants = max(1, int(variants))
    prefix_by_variant: List[List[int]] = []
    for variant in range(variants):
        chunk = base_ids[variant::variants] or base_ids
        prefix: List[int] = []
        while len(prefix) < prefix_tokens:
            prefix.extend(chunk)
        prefix_by_variant.append(prefix[:prefix_tokens])

    rng = random.Random(spec.shuffle_seed)
    index = 0
    while True:
        variant = index % variants
        sid = f"struct-v{variant:02d}-{index:04d}"
        yield {
            "session_id": sid,
            "id": sid,
            "num_turns": 1,
            "turn_1": {
                "input_text": f"[struct v{variant} prefix + body {index}]",
                "input_ids": list(prefix_by_variant[variant])
                + [rng.randrange(1000, 30000) for _ in range(STRUCT_BODY_TOKENS)],
                "response_text": "ok",
                "response_ids": [13, 271, 488],
            },
        }
        index += 1


def synth_apps_sessions(
    real: Sequence[Dict[str, Any]], count: int, seed: int = 42
) -> List[Dict[str, Any]]:
    """取 count 条合成 apps 会话（legacy 路径用；语义同 ``iter_apps_candidates``）。"""
    stream = iter_apps_candidates(list(real), seed)
    for _ in range(len(real)):  # 跳过真实会话，只取合成段
        next(stream)
    return [next(stream) for _ in range(max(0, int(count)))]


def build_struct_sessions(
    spec: TaskSpec, tokenizer, prefix_tokens: int, variants: int, target: int
) -> List[Dict[str, Any]]:
    """取 target 条 struct 会话（legacy 路径用）。"""
    stream = iter_struct_candidates(spec, tokenizer, prefix_tokens, variants)
    return [next(stream) for _ in range(max(0, int(target)))]


# ---------------------------------------------------------------------------
# 四类「按 token 主导」的 cell：预算求解
# ---------------------------------------------------------------------------

CLASSES = ("apps", "struct", "chat", "agentic")

# cell 名 → 主导类。注意 ``struct`` 既是类名也是 cell 名，两回事：
# cell struct = 「结构复用类占 token 主导」。
CELLS = {"scan": "apps", "struct": "struct", "chat": "chat", "agentic": "agentic"}

# 类名 → 落到 tasks.yaml 里的 task 名（与 arms/analyze 的 per_task 键一致）
TASK_NAMES = {
    "apps": "AgentBank_apps",
    "struct": "Struct",
    "chat": "ShareGPT",
    "agentic": "AgentBank_alfred",
}
TASK_IDS = {"apps": 0, "struct": 4, "chat": 2, "agentic": 3}
# 轮间 lognormal（与 mixexp/configs/mix_full.yaml 保持一致）
TASK_ARRIVAL = {
    "apps": (0.0, 0.0),
    "struct": (0.0, 0.0),
    "chat": (4.15, 0.971),
    "agentic": (1.81, 1.092),
}
# 多轮类必须跑前声明轮数下限（bench 的 assert_turn_policy 会校）
TASK_MIN_TURNS = {"chat": 5, "agentic": 5}

SHARE_TOL = 0.005          # 主导份额容差 0.5pp

_POOL_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def session_prompt_tokens(session: Dict[str, Any]) -> int:
    """一条会话的 prompt token 总数（逐轮 input_ids 累加）。"""
    total = 0
    for i in range(1, int(session.get("num_turns") or 0) + 1):
        turn = session.get(f"turn_{i}")
        if isinstance(turn, dict):
            total += len(turn.get("input_ids") or [])
    return total


def class_pool(spec: TaskSpec, args) -> Optional[List[Dict[str, Any]]]:
    """文件型负载的候选会话（按源路径缓存，一个进程只读一遍）。"""
    if not spec.source:
        return None
    src = Path(args.taskdata) / spec.source
    if not src.is_file():
        raise SystemExit(f"缺少输入数据：{src}\n先运行：bash mixexp/prepare_data.sh")
    key = str(src)
    if key not in _POOL_CACHE:
        _POOL_CACHE[key] = list(iter_sessions(src))
    return _POOL_CACHE[key]


def shuffled_pool(
    spec: TaskSpec, pool: Optional[Sequence[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """把一类候选会话按桶占比打散。

    ``select_by_bucket`` 先按桶配比取，再整体 shuffle，所以取它的**前缀**
    就是该类的无偏抽样（轮数结构不会被短前缀带偏）。
    """
    pool = list(pool or [])
    if not pool:
        return []
    return select_by_bucket(pool, spec, len(pool))


def class_candidates(
    name: str, args, tokenizer, pool: Optional[Sequence[Dict[str, Any]]]
) -> Iterator[Dict[str, Any]]:
    """一类的候选会话流：apps / struct 可无限补，chat / agentic 取尽即止。"""
    spec = SPECS[name]
    if name == "struct":
        return iter_struct_candidates(
            spec, tokenizer, args.struct_prefix_tokens, args.struct_variants
        )
    if name == "apps":
        return iter_apps_candidates(shuffled_pool(spec, pool), spec.shuffle_seed)
    return iter(shuffled_pool(spec, pool))


def take_token_budget(
    candidates: Iterator[Dict[str, Any]], budget: float, allow_overshoot: bool
) -> Tuple[List[Dict[str, Any]], int, bool]:
    """顺序取会话，让 prompt token 累计贴近 budget。

    Args:
        allow_overshoot:
            True —— 主导类用：取到**累计 ≥ budget** 为止，份额下限由构造保证；
            False —— 配角用：一旦加下一条会超预算就停，累计**不超过** budget。
            两个方向分别保证「主导 ≥ 目标份额」与「配角 ≤ 预算」，不需要迭代修正。

    Returns:
        (会话列表, 实测 token, 是否取尽候选流)。至少返回一条（除非流是空的），
        所以「按 token 定规模」不会把一类取成 0 条。

    为什么不按均值反解会话数：会话长度是重尾的（agentic 单条可达 60 万 token，
    apps 单条约 400 token），均值法在四舍五入上反复震荡，主导份额无法保证。
    """
    picked: List[Dict[str, Any]] = []
    total = 0
    for session in candidates:
        tokens = session_prompt_tokens(session)
        if picked and not allow_overshoot and total + tokens > budget:
            return picked, total, False
        picked.append(session)
        total += tokens
        if allow_overshoot and total >= budget:
            return picked, total, False
    return picked, total, True


def plan_cell(
    cell: str, args, tokenizer, log=print
) -> Dict[str, Dict[str, Any]]:
    """求解一个 cell 的四类会话，使主导类 token 份额 ≥ ``--dominant-share``。

    份额口径：主导类 prompt token / 四类 prompt token 之和，与会话数无关。

    求解顺序（闭式，无迭代）：
      1. 配角各自取 ``(1-share)/3 · total_tokens`` 的 token 预算；
      2. 主导类预算 = ``share/(1-share) · 配角实测 token``——与总预算无关，
         所以份额由构造保证 ≥ share，越出的部分最多是最后一条会话；
      3. 主导类池子取尽仍达不到预算时反向缩配角（而不是接受低于阈值的份额）；
         缩到每类只剩 1 条仍不够就报错，不静默出一条无效数据。
    """
    dominant = CELLS[cell]
    share = float(args.dominant_share)
    if not 0.25 < share < 1.0:
        raise SystemExit(f"--dominant-share 必须在 (0.25, 1) 之间，拿到 {share}")
    budget = float(args.total_tokens)
    if budget <= 0:
        raise SystemExit(f"--total-tokens 必须为正，拿到 {budget}")
    others = [c for c in CLASSES if c != dominant]

    pools = {name: class_pool(SPECS[name], args) for name in CLASSES}
    for name, pool in pools.items():
        if pool is not None and not pool:
            raise SystemExit(f"{name} 源数据为空：{SPECS[name].source}")

    def take(name: str, token_budget: float, dominant_class: bool = False):
        return take_token_budget(
            class_candidates(name, args, tokenizer, pools[name]),
            token_budget,
            allow_overshoot=dominant_class,
        )

    picked: Dict[str, List[Dict[str, Any]]] = {}
    tokens: Dict[str, int] = {}
    exhausted: Dict[str, bool] = {}

    per_other = budget * (1.0 - share) / len(others)
    for name in others:
        picked[name], tokens[name], exhausted[name] = take(name, per_other)

    dom_budget = share / (1.0 - share) * sum(tokens[n] for n in others)
    picked[dominant], tokens[dominant], exhausted[dominant] = take(
        dominant, dom_budget, dominant_class=True
    )

    if exhausted[dominant]:
        # 主导类池子取尽（数据源上限）：把配角按主导类实测 token 反向缩放。
        # 配角按「不超预算」取，所以缩完份额一定 ≥ share，不需要迭代。
        shrunk = tokens[dominant] * (1.0 - share) / share / len(others)
        for name in others:
            picked[name], tokens[name], exhausted[name] = take(name, shrunk)

    total = sum(tokens.values())
    now = tokens[dominant] / total if total else 0.0
    log(
        f"  [{cell}] 主导 {dominant}: {now:.1%} token"
        f"（目标 ≥{share:.0%}，{len(picked[dominant])} 条会话）"
    )
    if now < share - SHARE_TOL:
        raise SystemExit(
            f"[{cell}] 主导类 {dominant} 只能做到 {now:.1%}，低于 {share:.0%}："
            f"该类数据池只有 {len(picked[dominant])} 条可用会话"
            f"（{tokens[dominant] / 1e6:.2f}M token）。\n"
            "请降低 --dominant-share 或 --total-tokens，或补数据源。"
        )

    plans: Dict[str, Dict[str, Any]] = {}
    for name in CLASSES:
        plans[name] = {
            "sessions": picked[name],
            "count": len(picked[name]),
            "tokens": tokens[name],
            "mean": tokens[name] / max(1, len(picked[name])),
            "exhausted": exhausted[name],
            "share": tokens[name] / total,
        }
    return plans


def render_tasks_yaml(cell: str, plans: Dict[str, Dict[str, Any]], args) -> str:
    """生成可直接跑的 tasks.yaml（含 max_record 与 lambda_rate，都不需手改）。"""
    duration = float(args.duration_s)
    lines = [
        f"# exp3 · {cell} 主导 cell —— 由 mixexp/tools/build_exp3_data.py 生成，勿手改",
        f"# 主导类 {CELLS[cell]}：{plans[CELLS[cell]]['share']:.1%} prompt token",
        f"# lambda_rate = 会话数 / {duration:.0f}s，四类同时收尾，token 组成不随时间漂移",
        "tasks:",
    ]
    for name in CLASSES:
        plan = plans[name]
        if plan["count"] <= 0:
            continue
        out_path = (Path(args.out_dir) / cell / f"{name}_exp3.nosys.jsonl").resolve()
        try:
            data_path = out_path.relative_to(BENCH_DIR.resolve())
        except ValueError:  # 输出目录在仓库外时用绝对路径
            data_path = out_path
        mu, sigma = TASK_ARRIVAL[name]
        lines += [
            f"  - name: {TASK_NAMES[name]}",
            f"    task_id: {TASK_IDS[name]}",
            f"    data_path: {data_path}",
        ]
        if name in TASK_MIN_TURNS:
            lines.append(f"    min_turns: {TASK_MIN_TURNS[name]}")
        lines += [
            f"    lambda_rate: {plan['count'] / duration:.4f}",
            f"    mu: {mu}",
            f"    sigma: {sigma}",
            "    seed: 42",
            f"    max_record: {plan['count']}",
            "    shuffle: true",
        ]
    return "\n".join(lines) + "\n"


def build_cell(cell: str, args, tokenizer) -> Dict[str, Any]:
    """求解 + 落地一个 cell，写 jsonl 与 manifest.json。"""
    plans = plan_cell(cell, args, tokenizer)
    out_dir = Path(args.out_dir) / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    total_tokens = total_requests = 0
    info = {
        "cell": cell,
        "dominant": CELLS[cell],
        "dominant_share": args.dominant_share,
        "duration_s": args.duration_s,
        "tasks": {},
    }
    print(f"\n[{cell}] 主导类 {CELLS[cell]}（按 token）")
    for name in CLASSES:
        plan = plans[name]
        if not plan["sessions"]:
            continue
        out_path = out_dir / f"{name}_exp3.nosys.jsonl"
        written = write_sessions(out_path, plan["sessions"])
        turns = [int(s.get("num_turns") or 0) for s in plan["sessions"]]
        total_tokens += written
        total_requests += sum(turns)
        info["tasks"][name] = {
            "task": TASK_NAMES[name],
            "output": str(out_path),
            "sessions": len(plan["sessions"]),
            "requests": sum(turns),
            "prompt_tokens": written,
            "share": round(plan["share"], 4),
            "mean_tokens": round(plan["mean"], 1),
            "turns_min": min(turns),
            "turns_max": max(turns),
        }
        print(
            f"  {name:8s} 会话 {len(plan['sessions']):>6}  请求 {sum(turns):>7}"
            f"  prompt {written / 1e6:5.2f}M  份额 {plan['share']:5.1%}"
            f"  轮数 {min(turns)}–{max(turns)}"
        )
    info["prompt_tokens"] = total_tokens
    info["requests"] = total_requests

    tasks_yaml = out_dir / "tasks.yaml"
    tasks_yaml.write_text(render_tasks_yaml(cell, plans, args), encoding="utf-8")
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"  合计 prompt {total_tokens / 1e6:.2f}M / 请求 {total_requests:,}"
        f"  →  {tasks_yaml}，{manifest}"
    )
    return info


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
        description="构造 exp3 混合负载数据：四类「单类 token 主导」的 cell"
    )
    parser.add_argument("--taskdata", default=str(DEFAULT_TASKDATA), help="TaskData 根目录")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="输出目录")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="tokenizer 路径")
    parser.add_argument("--only", default="", help="legacy：只构造某一类：chat/agentic/apps/struct")
    parser.add_argument("--limit", type=int, default=0, help="每类最多取多少条（调试用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不读数据")
    # ---- 四类主导 cell ----
    parser.add_argument(
        "--mix",
        action="append",
        default=None,
        metavar="{" + "|".join(CELLS) + "|all}",
        help="构造哪个（可重复）主导 cell；all = 四个都造；不给则走 legacy 单组成",
    )
    parser.add_argument(
        "--dominant-share",
        type=float,
        default=0.60,
        help="主导类至少要占的 prompt token 份额（默认 0.60）",
    )
    parser.add_argument(
        "--total-tokens",
        type=float,
        default=25e6,
        help="单个 cell 的 prompt token 预算（只用于定配角规模，默认 25M）",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=12000.0,
        help="四类同时收尾的时长，lambda_rate = 会话数 / 该值（默认 12000s）",
    )
    # struct 的两个旋钮
    parser.add_argument("--apps-sessions", type=int, default=0,
                        help="legacy：apps 会话数（默认 15000；超出真实数据部分用合成冲刷会话补足）")
    parser.add_argument("--struct-sessions", type=int, default=0,
                        help="legacy：struct 会话数（默认 2600）")
    parser.add_argument("--struct-prefix-tokens", type=int, default=2000,
                        help="struct 每条会话的共享前缀长度（默认 2000）")
    parser.add_argument("--struct-variants", type=int, default=8,
                        help="struct 前缀链条数（默认 8）")
    args = parser.parse_args()

    mixes = args.mix or []
    cells: List[str] = []
    for item in mixes:
        if item == "all":
            cells.extend(CELLS)
        elif item in CELLS:
            cells.append(item)
        else:
            raise SystemExit(f"未知 cell：{item}（可选 {list(CELLS)} 或 all）")
    cells = list(dict.fromkeys(cells))  # 去重保序

    if cells:
        if args.only:
            raise SystemExit("--mix 与 --only 互斥（--only 是 legacy 单组成路径）")
        if args.dry_run:
            print("计划（主导类 = 按 prompt token 计，与会话数无关）：")
            budget = args.total_tokens
            support = (1.0 - args.dominant_share) * budget / 3
            for cell in cells:
                dom = CELLS[cell]
                spec = SPECS[dom]
                print(
                    f"  {cell:8s} 主导 {dom:8s} ≥{args.dominant_share:.0%} token"
                    f"（≈{args.dominant_share * budget / 1e6:.1f}M），"
                    f"其余三类各 ≈{support / 1e6:.1f}M"
                    f"   源 {spec.source or '独立构造（struct）'}"
                )
                print(f"           输出 {Path(args.out_dir) / cell}/ + tasks.yaml")
            return 0

        tokenizer = load_tokenizer(args.tokenizer)
        infos = [build_cell(cell, args, tokenizer) for cell in cells]
        print("\n汇总（每 cell 一行；份额按 prompt token 计）：")
        for info in infos:
            shares = "  ".join(
                f"{name} {task['share']:.0%}" for name, task in info["tasks"].items()
            )
            print(
                f"  {info['cell']:8s} 主导 {info['dominant']:8s}"
                f"  {info['prompt_tokens']/1e6:5.2f}M token  [{shares}]"
            )
        return 0

    # ---- legacy：原来的单组成（22/35/22/22）----
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
