#!/usr/bin/env python3
"""发送端队列命中率统计组件（零探针：只消费后端仿真/推理已产出的逐请求 metric）。

输入是各运行器 dump 的逐请求 records.jsonl，每行一个 JSON 对象。兼容两种方言：

- sim 逐请求记录（sim_exp3_cells / drift3_run dump）：
  {task_name, prompt_len, hit, skipped, arrival_t, dispatch_t, ...}
- tail-stress rows（build_tail_stress dump）：
  {task, plen, hit, ttft, whole, ...}

统计逻辑：每条请求按其数据集（task_name / task）映射到一个队列桶
（struct / chat / agentic / trans），按策略臂聚合：
请求数、prompt tokens、hit tokens、token 加权命中率、TTFT p50/p90/p99。

TTFT 口径与设计文档一致：
    TTFT = (派发时刻 − 到达时刻) + (prompt_len − hit) / prefill_tokens_per_s
tail-stress 方言直接用行内 ttft；sim 方言缺 dispatch_t 的行不计 TTFT（仍计 hit）。

数据集→队列映射（--mapping 覆盖为 JSON 文件 {"数据集子串": 队列}）：

    struct → struct       chat → chat      agentic → agentic
    apps / code / 单轮源 → trans          其他 → 原样保留为独立桶

用法::

    # exp3：整个 records 目录递归扫描（目录结构 <cell>/<arm>.records.jsonl）
    python mixexp/tools/queue_hit_report.py \\
        --dir Heter-Bench/mixexp/result/sim_exp3_records \\
        --rate 8319.6 --out-md Heter-Bench/result/queue_hit_exp3.md

    # tail-stress / drift3：显式列文件，文件名（去后缀）即臂名
    python mixexp/tools/queue_hit_report.py \\
        --dir Heter-Bench/mixexp/result/sim_tail_stress_records --rate 8319.6
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple

AUTO_MAP: List[Tuple[str, str]] = [
    ("struct", "struct"),
    ("chat", "chat"),
    ("agentic", "agentic"),
    ("alfred", "agentic"),
    ("apps", "trans"),
    ("code", "trans"),
]


def default_queue(dataset: str, mapping: Dict[str, str]) -> str:
    """数据集名 → 队列桶；优先用户映射（子串匹配），其次自动规则，否则原样保留。"""
    low = dataset.lower()
    for key, queue in mapping.items():
        if key and key.lower() in low:
            return queue
    for key, queue in AUTO_MAP:
        if key in low:
            return queue
    return dataset


def _row_fields(row: dict) -> Tuple[str, int, int, bool]:
    """兼容两种方言：返回 (dataset, prompt_len, hit, skipped)。"""
    dataset = str(row.get("task_name") or row.get("task") or "")
    plen = int(row.get("prompt_len", row.get("plen", 0)) or 0)
    hit = int(row.get("hit", 0) or 0)
    skipped = bool(row.get("skipped", False))
    return dataset, plen, hit, skipped


def _ttft_of(row: dict, rate: float) -> float | None:
    """单请求 TTFT（秒）；无法计算时返回 None（不计入分位数，仍计入命中）。"""
    if "ttft" in row:
        return float(row["ttft"])
    dispatch, arrival = row.get("dispatch_t"), row.get("arrival_t")
    if dispatch is None or arrival is None:
        return None
    plen, hit = int(row.get("prompt_len", 0) or 0), int(row.get("hit", 0) or 0)
    return max(0.0, float(dispatch) - float(arrival)) + max(0, plen - hit) / rate


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))]


def load_records(path: pathlib.Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_arm(path: pathlib.Path, mapping: Dict[str, str], rate: float) -> dict:
    """一个臂的 records.jsonl → {队列桶: {requests, prompt_tokens, hit_tokens, hit_rate, ttft}}。"""
    buckets: Dict[str, dict] = {}

    def bucket(q: str) -> dict:
        return buckets.setdefault(q, {
            "requests": 0, "prompt_tokens": 0, "hit_tokens": 0, "ttfts": [],
        })

    for row in load_records(path):
        dataset, plen, hit, skipped = _row_fields(row)
        if skipped:
            continue
        b = bucket(default_queue(dataset, mapping))
        b["requests"] += 1
        b["prompt_tokens"] += plen
        b["hit_tokens"] += hit
        tt = _ttft_of(row, rate)
        if tt is not None:
            b["ttfts"].append(tt * 1000.0)

    out: Dict[str, dict] = {}
    for q, b in sorted(buckets.items()):
        total = b["prompt_tokens"]
        ttfts = b.pop("ttfts")
        out[q] = {
            "requests": b["requests"],
            "prompt_tokens": total,
            "hit_tokens": b["hit_tokens"],
            "hit_rate_pct": round(100.0 * b["hit_tokens"] / total, 2) if total else 0.0,
            "ttft_ms": {
                "p50": round(_pct(ttfts, 50), 1),
                "p90": round(_pct(ttfts, 90), 1),
                "p99": round(_pct(ttfts, 99), 1),
            } if ttfts else None,
        }
    return out


def render_md(report: dict) -> str:
    lines = ["# 策略 × 队列命中率明细（发送端统计，零探针）", ""]
    for arm, queues in report.items():
        lines.append(f"## {arm}")
        lines.append("")
        lines.append("| 队列桶 | 请求数 | prompt tok | hit tok | hit rate | TTFT p50/p90/p99 (ms) |")
        lines.append("|---|---|---|---|---|---|")
        for q, s in queues.items():
            tt = s["ttft_ms"] or {}
            ttft_s = "/".join(str(tt.get(k, "-")) for k in ("p50", "p90", "p99")) if tt else "-"
            lines.append(
                f"| {q} | {s['requests']} | {s['prompt_tokens']} | {s['hit_tokens']} "
                f"| {s['hit_rate_pct']:.2f}% | {ttft_s} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default="", help="递归扫描目录（<任意>/<臂名>.records.jsonl）")
    ap.add_argument("--records", nargs="*", default=[], help="显式 records.jsonl 路径列表")
    ap.add_argument("--mapping", default="", help="数据集→队列映射 JSON（{子串: 队列}），覆盖自动规则")
    ap.add_argument("--rate", type=float, default=8319.6, help="prefill 吞吐（tok/s），TTFT 用")
    ap.add_argument("--out-md", default="", help="Markdown 输出路径")
    ap.add_argument("--out-json", default="", help="JSON 输出路径")
    a = ap.parse_args(argv)

    files: List[Tuple[str, pathlib.Path]] = []
    if a.dir:
        root = pathlib.Path(a.dir)
        if not root.is_dir():
            print(f"❌ 目录不存在: {root}", file=sys.stderr)
            return 1
        for f in sorted(root.rglob("*.records.jsonl")):
            rel_parent = f.parent.relative_to(root)
            base = f.stem.replace(".records", "")
            arm = f"{rel_parent.as_posix()}/{base}" if str(rel_parent) != "." else base
            files.append((arm, f))
    for p in a.records:
        fp = pathlib.Path(p)
        files.append((fp.stem.replace(".records", ""), fp))
    if not files:
        print("❌ 没有任何 records.jsonl 输入（--dir 或 --records）", file=sys.stderr)
        return 2

    mapping = {}
    if a.mapping:
        mapping = json.loads(pathlib.Path(a.mapping).read_text(encoding="utf-8"))

    report: Dict[str, dict] = {}
    for arm, f in files:
        report[arm] = summarize_arm(f, mapping, a.rate)

    md = render_md(report)
    print(md)
    if a.out_md:
        pathlib.Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out_md).write_text(md, encoding="utf-8")
    if a.out_json:
        pathlib.Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[queue_hit_report] arms={len(report)} files={len(files)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
