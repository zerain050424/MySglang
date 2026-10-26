#!/usr/bin/env python3
"""Uni-Bench：对 sglang SimpleHybridRadixCache 做前缀缓存 benchmark。

支持两种负载模式：
  1) 单负载模式（--data-path）：单个 session_data.jsonl 压测（向后兼容）。
  2) 混合负载模式（--tasks-config）：多个异构负载按 UniCache 的「比例」混成一条
     到达流交错压测，见 hybrid_tasks_config.yaml。

核心设计（方案 A）：
  - 复用 UniCache 的 session_data.jsonl（已 tokenize 的多轮会话）。
  - 直发 input_ids（与 UniCache 逐 token 一致，无 chat template 差异）。
  - session_id 打分类标签：同一 session 的多轮用同一个 session_id，
    且形如 "ShareGPT-multi-turn-*" → 触发 classify_simple_hybrid_task_type
    的 MULTI_TURN_TEXT 分类（进 text 队列）。
    注意：不能用 extra_key 做 session 标签——extra_key 会隔离 radix tree
    命名空间，导致跨 session 的前缀（31-token system prompt）无法共享。
  - 并发模型：会话间并发（Semaphore），会话内逐轮串行（下一轮依赖上一轮 KV）。
  - 指标：命中率 = Σcached_tokens / Σprompt_tokens，来自每次响应的 meta_info。

混合负载设计（逐函数对齐 UniCache vllm/simulator/simulator.py）：
  - init_heap                    → build_hybrid_arrival_schedule()
      每 task 用 rng(seed+task_id) 打乱自身 session，Poisson(lambda_rate) 生成到达
      时刻，全部 task 的 (arrival_time, counter) 合并进同一个全局到达流。
  - generate_poisson_requests()  → 指数间隔 + 累加，得到 Poisson 到达时刻。
  - _maybe_schedule_next_turn()  → run_session() 内 lognormal(mu,sigma) 轮间等待。
  - create_session_id_2_task_name_mapping / _init_session_rngs
                                  → 每 session 用 stable_seed_from_str 固定自身 RNG。
  - 「比例」= 各 task 的 lambda_rate 之比（session 到达速率之比）。

用法（单负载）：
  python bench_unicache.py \
      --data-path /path/ShareGPT_..._session_data.jsonl \
      --base-url http://127.0.0.1:30000 \
      --concurrency 32 --output result/bench.json

用法（混合负载，模仿 UniCache Hybrid/tasks_config.yaml）：
  python bench_unicache.py \
      --tasks-config hybrid_tasks_config.yaml \
      --base-url http://127.0.0.1:30000 \
      --concurrency 32 --arrival-scale 1 --wait-scale 1 --output result/hybrid.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import yaml


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_sessions(path: str, num_sessions: int, seed: Optional[int] = None,
                  shuffle: bool = True) -> List[Dict[str, Any]]:
    """读取 session_data.jsonl（每行 {"<id>": {id, num_turns, turn_1..N}}）。

    shuffle=False 时保持文件原始顺序：用于 mashqa 这类"同一文档的多个问题
    在文件中相邻"的数据，避免打乱后同文档兄弟问题被拆散、共享长前缀失效。
    混合负载模式下 shuffle 由 build_hybrid_arrival_schedule 用 (seed+task_id)
    的 RNG 完成（对齐 UniCache init_heap），故此处传 shuffle=False。
    """
    sessions: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 每行只有一个键，取该 session 的 value
            sessions.append(list(json.loads(line).values())[0])
            # 提前终止：仅当不需要「先打乱全集再截取」（shuffle=False）时，
            # 读满 num_sessions 即停，避免读入 GB 级大文件。
            # 对齐 UniCache data_parser.read_simulator_data 的 max_records 提前 break。
            if (not shuffle and num_sessions and num_sessions > 0
                    and len(sessions) >= num_sessions):
                break
    # 固定种子打乱，保证可复现（shuffle=True 时需先读全集再打乱再截取）
    if shuffle:
        random.Random(seed).shuffle(sessions)
    if num_sessions and num_sessions > 0:
        sessions = sessions[:num_sessions]
    return sessions


# --------------------------------------------------------------------------
# 到达间隔配置（模仿 UniCache TaskData/*/tasks_config.yaml + vllm/simulator/simulator.py）
# --------------------------------------------------------------------------
def load_arrival_config(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """读取到达间隔 YAML，返回 {task_name: {task_id, lambda_rate, mu, sigma, seed}}。

    结构对齐 UniCache 的 tasks_config.yaml（见 Uni-Bench/arrival_config.yaml）。
    仅单负载模式使用：根据 session_id 前缀查该负载类型的轮间 mu/sigma。
    """
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {t["name"]: t for t in cfg.get("tasks", [])}


def classify_session(session_id: str) -> str:
    """session_id -> workload task name。

    与 sglang mem_cache/uni_utils.py 的 classify_simple_hybrid_task_type 一致；
    对应 UniCache simulator.py 的 create_session_id_2_task_name_mapping 语义。
    """
    if session_id.startswith(("ShareGPT-multi-turn", "text_ShareGPT-multi-turn")):
        return "text"
    if session_id.startswith("image_ShareGPT-multi-turn"):
        return "image"
    if session_id.startswith("search_ShareGPT-multi-turn"):
        return "search"
    if session_id.startswith("file_ShareGPT-multi-turn"):
        return "file"
    if session_id.startswith("alfred_"):
        return "agentic"
    if session_id.startswith(("apps_", "mind2web_")):
        return "random"
    return "single"


def stable_seed_from_str(s: str, base_seed: int = 0) -> int:
    """字符串 -> 稳定 32-bit seed（模仿 UniCache simulator.py stable_seed_from_str）。"""
    h = hashlib.sha256(s.encode()).hexdigest()
    return (int(h, 16) + base_seed) % (2**32 - 1)


def pick_arrival_params(task_cfg: Dict[str, Any], args) -> tuple:
    """按负载取 (mu, sigma)；显式指定 --mu/--sigma 时全局 override。

    对应 UniCache simulator.py 的：
        task_para = task_paras[task_name]
        wait_time = rng.lognormal(mean=task_para["mu"], sigma=task_para["sigma"])
    """
    if args.mu is not None and args.sigma is not None:
        return args.mu, args.sigma
    return task_cfg.get("mu", 0.0), task_cfg.get("sigma", 0.0)


class BurstGate:
    """victim burst 计数屏障。

    搅拌等到全部 victim 越过 pause_after_turn 才放行，使淘汰压力只落在 pause 窗口内。
    用进度计数而不是墙钟：单轮时延随 prompt 增长，20 轮累计漂移可达数秒。
    """

    def __init__(self, expect: int):
        self.expect = expect
        self.count = 0
        self.event = asyncio.Event()

    def done_one(self) -> None:
        self.count += 1
        if self.count >= self.expect:
            self.event.set()


# --------------------------------------------------------------------------
# 混合负载：Poisson 会话到达调度（模仿 UniCache simulator.py init_heap）
# --------------------------------------------------------------------------
@dataclass
class SessionArrival:
    """混合负载的一个「会话首轮到达事件」。

    对齐 UniCache future_request_heap 的元素：
      (timestamp, counter, request) —— timestamp 即 arrival_offset，counter 作 tiebreaker。
    """
    arrival_offset: float         # 相对起始时刻的到达时间（秒，Poisson 采样）
    counter: int                  # 全局递增计数器，时间戳相同时按入堆顺序稳定排序
    task_name: str                # 所属负载（task name）
    task_cfg: Dict[str, Any]      # 该负载完整配置（lambda_rate/mu/sigma/seed/task_id）
    session: Dict[str, Any]       # session 数据（含 id/num_turns/turn_*）


def generate_poisson_requests(lambda_rate: float, num_sessions: int,
                              seed: int) -> List[float]:
    """模仿 UniCache simulator/utils.py::generate_poisson_requests。
    Poisson 过程：session 到达间隔服从 Exp(lambda_rate)（均值 1/lambda_rate），
    累加得到每个 session 的到达时刻。
    numpy 的 default_rng.exponential(scale=1/λ) 与 stdlib random.expovariate(λ)
    同分布（均值 1/λ），这里用 stdlib 以避免引入 numpy。
    """
    rng = random.Random(seed)
    arrival_times: List[float] = []
    acc = 0.0
    for _ in range(num_sessions):
        acc += rng.expovariate(lambda_rate)   # 指数分布到达间隔
        arrival_times.append(acc)
    return arrival_times


def build_hybrid_arrival_schedule(tasks_config: Dict[str, Any],
                                  base_dir: Optional[str] = None
                                  ) -> List[SessionArrival]:
    """把多个异构负载按 lambda_rate「比例」混成一条全局到达流。

    逐行对齐 UniCache simulator.py::init_heap：
      1. 对每个 task：读其 session 数据（max_record 限制数量）
      2. 用 rng(seed + task_id) 打乱该 task 自身 session 顺序
      3. 用 Poisson(lambda_rate) 生成该 task 的 session 到达时刻
      4. 所有 task 的 (arrival_time, counter, session) 合并进同一个列表
      5. 按 (arrival_time, counter) 升序排序 → 异构负载在时间轴上交错混合

    返回按到达时刻升序的 SessionArrival 列表。「比例」即各 task 的 lambda_rate
    之比（例如 task A lambda_rate=2、task B lambda_rate=1 → 2:1 混合）。
    """
    events: List[SessionArrival] = []
    counter = 0  # 全局计数器（模仿 itertools.count）：时间戳相同按入堆顺序稳定排序

    for task in tasks_config.get("tasks", []):
        name = task["name"]
        data_path = task["data_path"]
        # data_path 相对 tasks_config 所在目录解析（而非 CWD），更稳健
        if base_dir and not os.path.isabs(data_path):
            data_path = os.path.join(base_dir, data_path)

        max_record = task.get("max_record", -1)
        lambda_rate = task.get("lambda_rate", 1)
        seed = task.get("seed", 42)
        task_id = task.get("task_id", 0)
        shuffle = task.get("shuffle", True)  # mashqa 等同文档问题相邻的数据可设 false

        # 读 session（保持文件原序），再由该 task 独立 RNG 打乱（对齐 UniCache rng.shuffle）
        sessions = load_sessions(data_path, max_record, seed=None, shuffle=False)
        rng = random.Random(seed + task_id)
        if shuffle:
            rng.shuffle(sessions)

        # arrival.mode=fixed：确定性到达（victim 需同相位，Poisson 会把 pause 打散）
        arrival = task.get("arrival") or {}
        if arrival.get("mode") == "fixed":
            base = float(arrival.get("offset_s", 0.0))
            step = float(arrival.get("step_s", 0.05))
            arrival_times = [base + step * i for i in range(len(sessions))]
        else:
            # Poisson 到达时刻（seed+task_id 保证各 task 到达流相互独立且可复现）
            arrival_times = generate_poisson_requests(
                lambda_rate, len(sessions), seed + task_id)

        for sess, at in zip(sessions, arrival_times):
            events.append(SessionArrival(at, counter, name, task, sess))
            counter += 1

    events.sort(key=lambda e: (e.arrival_offset, e.counter))
    return events


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
class Stats:
    def __init__(self) -> None:
        self.num_requests = 0
        self.num_sessions = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.ttfts: List[float] = []      # 首 token 时延（秒）
        self.e2es: List[float] = []       # 端到端时延（秒）
        self.errors = 0
        # 混合负载：按 task 细分统计（异构负载各自的命中率）
        self.per_task: Dict[str, Dict[str, int]] = {}
        # 逐请求明细（供 CSV 输出：TTFT vs 重算量逐点分析）
        self.rows: List[Dict[str, Any]] = []

    def record_request(self, task_name: str, prompt_tokens: int,
                       cached_tokens: int, completion_tokens: int) -> None:
        """累计一次请求的全局与 per-task 计数。"""
        self.num_requests += 1
        self.prompt_tokens += prompt_tokens
        self.cached_tokens += cached_tokens
        self.completion_tokens += completion_tokens
        t = self.per_task.setdefault(task_name, {
            "num_sessions": 0, "num_requests": 0,
            "prompt_tokens": 0, "cached_tokens": 0,
        })
        t["num_requests"] += 1
        t["prompt_tokens"] += prompt_tokens
        t["cached_tokens"] += cached_tokens

    def record_session(self, task_name: str) -> None:
        """累计一个 session 的全局与 per-task 计数。"""
        self.num_sessions += 1
        t = self.per_task.setdefault(task_name, {
            "num_sessions": 0, "num_requests": 0,
            "prompt_tokens": 0, "cached_tokens": 0,
        })
        t["num_sessions"] += 1

    def summary(self) -> Dict[str, Any]:
        def pct(xs: List[float], p: float) -> Optional[float]:
            if not xs:
                return None
            xs = sorted(xs)
            # 线性插值分位数
            k = (len(xs) - 1) * p / 100.0
            lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
            return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

        hit_rate = (
            self.cached_tokens / self.prompt_tokens
            if self.prompt_tokens > 0
            else 0.0
        )

        # 混合负载：输出每个 task 的命中率细分
        per_task = {}
        for name, t in self.per_task.items():
            per_task[name] = {
                "num_sessions": t["num_sessions"],
                "num_requests": t["num_requests"],
                "prompt_tokens": t["prompt_tokens"],
                "cached_tokens": t["cached_tokens"],
                "cache_hit_rate_pct": round(
                    t["cached_tokens"] / t["prompt_tokens"] * 100, 2)
                if t["prompt_tokens"] else 0.0,
            }

        return {
            "num_sessions": self.num_sessions,
            "num_requests": self.num_requests,
            "num_errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hit_rate": round(hit_rate, 4),
            "cache_hit_rate_pct": round(hit_rate * 100, 2),
            "ttft_p50_ms": round(pct(self.ttfts, 50) * 1000, 2) if self.ttfts else None,
            "ttft_p90_ms": round(pct(self.ttfts, 90) * 1000, 2) if self.ttfts else None,
            "ttft_p99_ms": round(pct(self.ttfts, 99) * 1000, 2) if self.ttfts else None,
            "e2e_mean_ms": round(statistics.mean(self.e2es) * 1000, 2) if self.e2es else None,
            "e2e_p99_ms": round(pct(self.e2es, 99) * 1000, 2) if self.e2es else None,
            "per_task": per_task,
        }


# --------------------------------------------------------------------------
# 单次请求（流式：可测 TTFT）
# --------------------------------------------------------------------------
# data: {
#   "text": "To",
#   "output_ids": [1249],
#   "meta_info": {
#     "id": "ShareGPT-multi-turn-wNBG8Gp_0_turn_8",
#     "finish_reason": null,
#     "prompt_tokens": 1829,
#     "weight_version": "default",
#     "num_retractions": 0,
#     "forward_entry_time": 1792316271.9735017,
#     "prefill_finished_time": 1792316271.9923403,
#     "queue_time": 0.000496807973831892,
#     "reasoning_tokens": 0,
#     "completion_tokens": 1,
#     "cached_tokens": 1828,
#     "cached_tokens_details": {
#       "device": 1828,
#       "host": 0
#     },
#     "dp_rank": null,
#     "response_sent_to_client_ts": 1792316271.9970071
#   }
# }
async def one_request(
    http: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    """发一次 /generate（流式），返回 ttft/e2e/prompt_tokens/cached_tokens。"""
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    last_meta: Dict[str, Any] = {}

    body = dict(payload)
    body["stream"] = True  # 流式：第一个 chunk 到达即 TTFT

    async with http.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue
            data = line[len(b"data:"):].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            meta = obj.get("meta_info")
            if meta:
                last_meta = meta

    e2e = time.perf_counter() - t0
    # TTFT 分解：forward_entry_time → prefill_finished_time 是纯 prefill，
    # 其余为 tokenizer 入队与调度排队。用于把尾部归因到重算而非排队。
    fe = last_meta.get("forward_entry_time")
    pf = last_meta.get("prefill_finished_time")
    return {
        "ttft": ttft,
        "e2e": e2e,
        "prompt_tokens": last_meta.get("prompt_tokens", 0),
        "cached_tokens": last_meta.get("cached_tokens", 0),
        "completion_tokens": last_meta.get("completion_tokens", 0),
        "queue_time_ms": round((last_meta.get("queue_time") or 0.0) * 1000, 3),
        "prefill_ms": round((pf - fe) * 1000, 3) if fe and pf else None,
    }


# --------------------------------------------------------------------------
# 单个 session：逐轮串行
# --------------------------------------------------------------------------
async def run_session(
    http: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    sess: Dict[str, Any],
    args: argparse.Namespace,
    stats: Stats,
    url: str,
    task_cfg: Optional[Dict[str, Any]] = None,
    arrival_offset: float = 0.0,
    gate: Optional[BurstGate] = None,
) -> None:
    """跑一个 session 的全部轮次（会话内逐轮串行）。

    混合负载模式：先按 Poisson 到达时刻 sleep（真实墙钟时间）再竞争并发槽；
    单负载模式：arrival_offset=0，立即竞争并发槽。

    task_cfg 传入时（混合负载）直接用其 mu/sigma/seed/task_id；
    否则（单负载）按 session_id 前缀从 arrival_config 查。
    """
    sid = sess["id"]  # 形如 "ShareGPT-multi-turn-<x>"，作为分类标签
    if task_cfg is None:
        task_cfg = args.arrival_config.get(classify_session(sid), {})
    task_name = task_cfg.get("name") or classify_session(sid)

    # 每 session 独立 RNG，seed 由 (sid, task.seed, task.task_id) 确定性生成
    # （模仿 UniCache simulator.py 的 stable_seed_from_str + _init_session_rngs）
    base_seed = task_cfg.get("seed", args.seed) + task_cfg.get("task_id", 0) * 1000
    rng = random.Random(stable_seed_from_str(sid, base_seed=base_seed))
    num_turns = sess["num_turns"]

    # 带 gate_after 的流（搅拌）等 victim burst 屏障释放后才开始计到达
    if gate is not None and task_cfg.get("gate_after"):
        await gate.event.wait()

    # 混合负载：Poisson 到达调度（真实墙钟）。arrival_scale 可整体缩放到达节奏。
    if arrival_offset > 0:
        await asyncio.sleep(arrival_offset)

    prev_end: Optional[float] = None  # 上一轮请求完成时刻（算真实轮间隔）
    pause_done = False
    async with sem:  ##控制并发粒度
        for t in range(1, num_turns + 1):
            turn = sess[f"turn_{t}"]
            payload = {
                # 直发 token ids：与 UniCache 数据逐 token 一致
                "input_ids": turn["input_ids"],
                # 固定输出长度，复现 UniCache 的 response_ids 长度
                "max_new_tokens": len(turn["response_ids"]),
                # 分类标签：同一 session 恒定；不影响 radix tree 命名空间
                "session_id": sid,
                "rid": f"{sid}_turn_{t}",
            }
            try:
                r = await one_request(http, url, payload, args.timeout_s)
                stats.record_request(
                    task_name,
                    r["prompt_tokens"] or len(turn["input_ids"]),
                    r["cached_tokens"],
                    r["completion_tokens"],
                )
                if r["ttft"] is not None:
                    stats.ttfts.append(r["ttft"])
                stats.e2es.append(r["e2e"])
                now = time.perf_counter()
                ptok = r["prompt_tokens"] or len(turn["input_ids"])
                ctok = r["cached_tokens"] or 0
                stats.rows.append({
                    "task": task_name,
                    "session_id": sid,
                    "turn": t,
                    "prompt_tokens": ptok,
                    "cached_tokens": ctok,
                    "recompute_tokens": ptok - ctok,
                    "ttft_ms": round(r["ttft"] * 1000, 2) if r["ttft"] is not None else None,
                    "e2e_ms": round(r["e2e"] * 1000, 2),
                    "gap_since_prev_end_s": (
                        round(now - prev_end, 3) if prev_end is not None else None
                    ),
                    # 跨会话对齐时间轴（SimultaneityIndex 用）；e2e 分解用后两列
                    "queue_time_ms": r["queue_time_ms"],
                    "prefill_ms": r["prefill_ms"],
                    "ts": round(time.time(), 3),
                })
                prev_end = now
            except Exception as exc:  # noqa: BLE001
                stats.errors += 1
                print(f"[ERR] {sid} turn {t}: {type(exc).__name__}: {exc}")

            # burst 段结束：计一次屏障，然后进入 pause（tool 往返/人工等待）。
            # pause 不乘 wait_scale：它是自变量，必须精确。
            pause_after = task_cfg.get("pause_after_turn")
            if pause_after and t == pause_after:
                pause_done = True
                if gate is not None:
                    gate.done_one()
                await asyncio.sleep(task_cfg.get("pause_s", 0.0))

            # 轮间等待（inter-turn 到达间隔，模仿 UniCache simulator.py
            #   _maybe_schedule_next_turn）：
            # gap_mode=fixed 用固定间隔；否则按负载类型取 lognormal(mu, sigma) 采样。
            # wait_scale=0 不等（快速跑）；仅当还有下一轮时等待（对齐 UniCache）。
            if args.wait_scale > 0 and t < num_turns:
                if task_cfg.get("gap_mode") == "fixed":
                    wait = task_cfg.get("gap_s", 0.0)
                else:
                    mu, sigma = pick_arrival_params(task_cfg, args)
                    if mu == 0.0 and sigma == 0.0:
                        wait = 0.0
                    else:
                        wait = rng.lognormvariate(mu, sigma)
                await asyncio.sleep(wait * args.wait_scale)

    # 轮数不足 pause_after_turn 的 session 也要放行屏障，否则搅拌永久等待
    if gate is not None and task_cfg.get("pause_after_turn") and not pause_done:
        gate.done_one()

    stats.record_session(task_name)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
async def main_async(args: argparse.Namespace) -> Dict[str, Any]:
    stats = Stats()
    sem = asyncio.Semaphore(args.concurrency)  ## 建立并发信号量
    url = args.base_url.rstrip("/") + "/generate"

    t0 = time.perf_counter()
    async with aiohttp.ClientSession(  ## 复用TCP连接
        read_bufsize=10 * 1024**2,  # 高并发下放大读缓冲
    ) as http:
        if args.tasks_config:
            # ============================================================
            # 混合负载模式：多负载按 lambda_rate 比例混成一条到达流
            # ============================================================
            with open(args.tasks_config, encoding="utf-8") as f:
                tasks_config = yaml.safe_load(f)
            base_dir = str(Path(args.tasks_config).resolve().parent)
            schedule = build_hybrid_arrival_schedule(tasks_config, base_dir)

            print(f"[Uni-Bench] 混合负载模式：{len(tasks_config['tasks'])} 个 task，"
                  f"共 {len(schedule)} 个 session（按 lambda_rate 比例混合）")
            for t in tasks_config["tasks"]:
                print(f"    - {t['name']:16s} lambda_rate={t.get('lambda_rate', 1):>4} "
                      f"max_record={t.get('max_record', -1):>5} "
                      f"mu={t.get('mu', 0)} sigma={t.get('sigma', 0)} "
                      f"data={t['data_path']}")
            # 屏障参与方 = 带 pause_after_turn 的 victim 会话数
            gate = BurstGate(
                sum(1 for ev in schedule if ev.task_cfg.get("pause_after_turn")))
            print(f"[Uni-Bench] base_url={args.base_url} concurrency={args.concurrency} "
                  f"arrival_scale={args.arrival_scale} wait_scale={args.wait_scale} "
                  f"burst_gate={gate.expect}")

            # 每个 SessionArrival 一个协程：先按到达时刻 sleep，再竞争并发槽。
            # 会话间并发（sem）、会话内串行（run_session 内逐轮）。
            tasks = [
                asyncio.create_task(run_session(
                    http, sem, ev.session, args, stats, url,
                    task_cfg=ev.task_cfg,
                    arrival_offset=ev.arrival_offset * args.arrival_scale,
                    gate=gate,
                ))
                for ev in schedule
            ]
            await asyncio.gather(*tasks)
        else:
            # ============================================================
            # 单负载模式（向后兼容）：所有 session 立即并发
            # ============================================================
            args.arrival_config = load_arrival_config(args.arrival_config)
            sessions = load_sessions(args.data_path, args.num_sessions, args.seed,
                                     shuffle=not args.no_shuffle)
            print(f"[Uni-Bench] Loaded {len(sessions)} sessions from {args.data_path}")
            print(f"[Uni-Bench] base_url={args.base_url} concurrency={args.concurrency} "
                  f"wait_scale={args.wait_scale}")
            print(f"[Uni-Bench] arrival tasks: {sorted(args.arrival_config)}")

            tasks = [
                asyncio.create_task(run_session(http, sem, s, args, stats, url))
                for s in sessions
            ]
            await asyncio.gather(*tasks)

    wall = time.perf_counter() - t0

    summary = stats.summary()
    summary["wall_time_s"] = round(wall, 2)
    summary["request_throughput"] = (
        round(stats.num_requests / wall, 2) if wall > 0 else None
    )
    summary["base_url"] = args.base_url
    summary["concurrency"] = args.concurrency
    summary["wait_scale"] = args.wait_scale
    if args.tasks_config:
        summary["mode"] = "hybrid"
        summary["tasks_config"] = args.tasks_config
        summary["arrival_scale"] = args.arrival_scale
    else:
        summary["mode"] = "single"
        summary["data_path"] = args.data_path
    return summary, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Uni-Bench: 前缀缓存 benchmark（单负载 / 混合负载）")
    # ---- 单负载模式 ----
    p.add_argument(
        "--data-path",
        default=None,
        help="session_data.jsonl 路径（单负载模式；与 --tasks-config 二选一）",
    )
    # ---- 混合负载模式 ----
    p.add_argument(
        "--tasks-config",
        default=None,
        help="混合负载 tasks_config.yaml 路径（模仿 UniCache tasks_config.yaml，"
             "含多个 task 的 data_path/lambda_rate/mu/sigma/seed/max_record）",
    )
    p.add_argument(
        "--arrival-scale",
        type=float,
        default=1.0,
        help="混合负载的 session 到达时刻整体缩放：1=UniCache 真实 Poisson 节奏；"
             "0=所有 session 立即到达（等同单负载并发）；<1 加速压测",
    )

    p.add_argument("--base-url", default="http://127.0.0.1:30000", help="sglang server 地址")
    p.add_argument("--concurrency", type=int, default=32, help="并发会话数")
    p.add_argument("--num-sessions", type=int, default=-1, help="使用的 session 数，-1 表示全部（单负载模式）")
    p.add_argument("--seed", type=int, default=42, help="session 打乱与轮间等待的随机种子")
    p.add_argument("--no-shuffle", action="store_true", help="保持 session 文件原始顺序（mashqa 同文档问题相邻）")
    p.add_argument(
        "--arrival-config",
        default=str(Path(__file__).resolve().parent / "arrival_config.yaml"),
        help="到达间隔 YAML 路径（单负载模式按 session_id 前缀查 mu/sigma）",
    )
    p.add_argument("--mu", type=float, default=None,
                   help="轮间等待 lognormal mean 全局覆盖；默认 None=按负载类型取")
    p.add_argument("--sigma", type=float, default=None,
                   help="轮间等待 lognormal sigma 全局覆盖；默认 None=按负载类型取")
    p.add_argument("--wait-scale", type=float, default=0.0, help="轮间等待缩放；0=不等（快速跑）")
    p.add_argument("--timeout-s", type=float, default=1800.0, help="单请求超时（秒）")
    p.add_argument("--output", default=None, help="结果 JSON 输出路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # 单负载 / 混合负载必须二选一
    if not args.tasks_config and not args.data_path:
        raise SystemExit("请指定 --data-path（单负载）或 --tasks-config（混合负载）")

    summary, stats = asyncio.run(main_async(args))
    print("\n================ Uni-Bench 结果 ================")
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print("================================================")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[Uni-Bench] 结果已写入 {args.output}")
        csv_path = str(args.output).rsplit(".json", 1)[0] + ".csv"
        if stats.rows:
            import csv as _csv
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=list(stats.rows[0].keys()))
                w.writeheader()
                w.writerows(stats.rows)
            print(f"[Uni-Bench] 逐请求明细已写入 {csv_path} ({len(stats.rows)} rows)")


if __name__ == "__main__":
    main()
