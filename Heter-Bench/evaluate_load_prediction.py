#!/usr/bin/env python3
"""Evaluate window-level workload prediction from Heter-Bench outputs.

数据源：server 日志里的 `[QueueOcc]` 行（每 30s 一行，由 radix_cache 的队列占用
采样器打印）。**该采样器已移除**——现在跑这个脚本会解析不到任何样本，主流程会
直接报错而不是输出空表。要恢复，先把采样器加回 radix_cache，或把解析口径改成
`[HeterCache profile]` 的多行表格。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LABELS = ("struct", "random", "chat", "agentic")
DEFAULT_TASK_LABELS = {
    "Codeparrot": "struct",
    "AgentBank_apps": "random",
    "ShareGPT": "chat",
    "AgentBank_alfred": "agentic",
}


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _task_labels(value: Optional[str]) -> Dict[str, str]:
    if not value:
        return DEFAULT_TASK_LABELS.copy()
    with open(value, encoding="utf-8") as file:
        data = json.load(file)
    labels = DEFAULT_TASK_LABELS.copy()
    labels.update(data)
    unknown = set(labels.values()) - set(LABELS)
    if unknown:
        raise ValueError(f"unsupported labels in {value}: {sorted(unknown)}")
    return labels


def _request_observations(
    rows: Iterable[Dict[str, str]], labels: Dict[str, str]
) -> List[Dict[str, Any]]:
    observations = []
    for row in rows:
        if not row.get("ts") or row.get("task") not in labels:
            continue
        observations.append({
            "ts": float(row["ts"]),
            "label": labels[row["task"]],
            "tokens": max(float(row.get("prompt_tokens", 0)), 0.0),
            "phase": row.get("phase") or "",
        })
    return sorted(observations, key=lambda item: item["ts"])


def _parse_occ_line(line: str) -> Optional[Tuple[float, Dict[str, float]]]:
    match = re.search(
        r"^\s*(?P<ts>\d+(?:\.\d+)?)\s+\[QueueOcc\]\s+(?P<body>.*)", line
    )
    if not match:
        match = re.search(
            r"(?P<date>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d+)?).*"
            r"\[QueueOcc\]\s+(?P<body>.*)",
            line,
        )
    if not match:
        return None
    counts = {}
    for item in match.group("body").split():
        name, _, value = item.partition("=")
        if name in LABELS and value:
            counts[name] = float(value)
    match_groups = match.groupdict()
    if match_groups.get("ts"):
        timestamp = float(match_groups["ts"])
    else:
        date_text = match_groups["date"].replace(",", ".")
        timestamp = datetime.fromisoformat(date_text).timestamp()
    return timestamp, counts


def _occ_observations(path: str) -> List[Dict[str, Any]]:
    observations = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            parsed = _parse_occ_line(line)
            if parsed is not None:
                ts, counts = parsed
                observations.append({"ts": ts, "counts": counts})
    return observations


def _prediction_at(
    observations: List[Dict[str, Any]], timestamp: float
) -> Optional[str]:
    candidates = [item for item in observations if item["ts"] <= timestamp]
    if not candidates:
        return None
    counts = candidates[-1]["counts"]
    if not counts:
        return None
    return max(LABELS, key=lambda label: counts.get(label, 0.0))


def _majority_label(
    observations: List[Dict[str, Any]], margin: float
) -> Tuple[Optional[str], Dict[str, float]]:
    totals = {label: 0.0 for label in LABELS}
    for item in observations:
        totals[item["label"]] += item["tokens"]
    ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    total = sum(totals.values())
    if not total or (len(ordered) > 1 and
                     ordered[0][1] - ordered[1][1] < total * margin):
        return None, totals
    return ordered[0][0], totals


def evaluate(
    request_rows: List[Dict[str, str]],
    occ_rows: List[Dict[str, Any]],
    labels: Dict[str, str],
    window_s: float,
    step_s: float,
    margin: float,
) -> Dict[str, Any]:
    requests = _request_observations(request_rows, labels)
    if not requests:
        raise ValueError("request CSV has no usable timestamped rows")
    start = requests[0]["ts"]
    end = requests[-1]["ts"]
    windows = []
    cursor = start
    while cursor <= end:
        window_end = cursor + window_s
        members = [item for item in requests if cursor <= item["ts"] < window_end]
        if members:
            truth, shares = _majority_label(members, margin)
            prediction = _prediction_at(occ_rows, cursor)
            phases = {item["phase"] for item in members if item["phase"]}
            boundary = len(phases) > 1
            windows.append({
                "start_ts": round(cursor, 3),
                "end_ts": round(window_end, 3),
                "phase": next(iter(phases)) if len(phases) == 1 else "boundary",
                "truth": "boundary" if boundary else (truth or "mixed"),
                "prediction": prediction or "unknown",
                "correct": bool(not boundary and truth and prediction == truth),
                "shares": shares,
            })
        cursor += step_s

    eligible = [
        window for window in windows
        if window["truth"] not in ("mixed", "boundary")
    ]
    known = [window for window in eligible if window["prediction"] != "unknown"]
    accuracy = (
        sum(window["correct"] for window in known) / len(known)
        if known else None
    )
    by_phase: Dict[str, Dict[str, Any]] = {}
    for window in windows:
        phase = window["phase"] or "unlabeled"
        stats = by_phase.setdefault(phase, {"eligible": 0, "known": 0, "correct": 0})
        if window["truth"] in ("mixed", "boundary"):
            continue
        stats["eligible"] += 1
        if window["prediction"] != "unknown":
            stats["known"] += 1
            stats["correct"] += int(window["correct"])
    for stats in by_phase.values():
        stats["accuracy"] = (
            stats["correct"] / stats["known"] if stats["known"] else None
        )
    return {
        "window_s": window_s,
        "step_s": step_s,
        "dominance_margin": margin,
        "num_windows": len(windows),
        "eligible_windows": len(eligible),
        "known_windows": len(known),
        "coverage": len(known) / len(eligible) if eligible else 0.0,
        "accuracy": accuracy,
        "by_phase": by_phase,
        "windows": windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-csv", required=True)
    parser.add_argument("--occ-log", required=True)
    parser.add_argument("--task-labels-json")
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--step-s", type=float, default=30.0)
    parser.add_argument("--dominance-margin", type=float, default=0.10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.window_s <= 0 or args.step_s <= 0:
        raise SystemExit("window and step must be positive")
    occ = _occ_observations(args.occ_log)
    if not occ:
        raise SystemExit(
            f"no [QueueOcc] samples in {args.occ_log}: the queue-occupancy sampler "
            "was removed from radix_cache; only [HeterCache profile] is emitted now."
        )
    result = evaluate(
        _read_rows(args.requests_csv),
        occ,
        _task_labels(args.task_labels_json),
        args.window_s,
        args.step_s,
        args.dominance_margin,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        key: result[key]
        for key in ("accuracy", "coverage", "eligible_windows", "known_windows")
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
