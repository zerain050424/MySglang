"""堆发射器完整回归单测（基于标准库 unittest，无需 GPU / torch / aiohttp / pytest）。

被测对象：`Heter-Bench/scripts/bench_heap.py`
覆盖：
  - ArrivalHeap 调度语义与同刻顺序
  - _resolve_prompt_ids 前缀一致性检测
  - stable_seed_from_str / load_sessions / assert_turn_policy / build_hybrid_arrival_schedule
  - percentile 与 Stats 预热窗切分、JSON 汇总键
  - HeapEmitter 异步闭环（多轮推进、并发上限、异常容错、lag_p50 单位修复、write_outputs）
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "Heter-Bench", "scripts"
)
sys.path.insert(0, _SCRIPTS)

import bench_heap  # noqa: E402


EXPECTED_CSV_COLUMNS = [
    "warmup",
    "task",
    "phase",
    "session_id",
    "turn",
    "prior_tool_call",
    "prompt_tokens",
    "cached_tokens",
    "recompute_tokens",
    "ttft_ms",
    "e2e_ms",
    "gap_since_prev_end_s",
    "send_lag_ms",
    "queue_time_ms",
    "prefill_ms",
    "ts",
]

EXPECTED_SUMMARY_KEYS = {
    "warmup_s",
    "warmup_requests",
    "warmup_sessions",
    "warmup_prompt_tokens",
    "done_requests",
    "num_sessions",
    "num_requests",
    "num_errors",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "cache_hit_rate",
    "cache_hit_rate_pct",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "e2e_mean_ms",
    "e2e_p99_ms",
    "per_task",
}


def _turn(session_id: str, turn_idx: int = 1) -> bench_heap.PendingTurn:
    return bench_heap.PendingTurn(
        task_name="t",
        task_cfg={"name": "t"},
        session={"id": session_id},
        turn_idx=turn_idx,
    )


class TestArrivalHeapAndPrompt(unittest.TestCase):
    def test_pop_due_only_returns_due_events_in_time_order(self):
        heap = bench_heap.ArrivalHeap()
        for due, sid in ((30.0, "c"), (10.0, "a"), (20.0, "b")):
            heap.push(due_ts=due, turn=_turn(sid))

        self.assertEqual(heap.next_due_ts, 10.0)
        self.assertEqual([s.session["id"] for _, s in heap.pop_due(now=20.0, limit=1)], ["a"])
        self.assertEqual([s.session["id"] for _, s in heap.pop_due(now=20.0)], ["b"])
        self.assertEqual(len(heap), 1)
        self.assertEqual([s.session["id"] for _, s in heap.pop_due(now=30.0)], ["c"])
        self.assertFalse(heap)

    def test_same_timestamp_pops_in_push_order(self):
        heap = bench_heap.ArrivalHeap()
        for sid in ("first", "second", "third"):
            heap.push(due_ts=5.0, turn=_turn(sid))

        self.assertEqual(
            [s.session["id"] for _, s in heap.pop_due(now=5.0)],
            ["first", "second", "third"],
        )

    def test_earlier_event_pushed_later_becomes_next_due(self):
        heap = bench_heap.ArrivalHeap()
        heap.push(due_ts=100.0, turn=_turn("far"))
        heap.push(due_ts=1.0, turn=_turn("near"))

        self.assertEqual(heap.next_due_ts, 1.0)
        self.assertEqual([s.session["id"] for _, s in heap.pop_due(now=1.0)], ["near"])

    def test_resolve_prompt_literal_suffix_and_fallback(self):
        qwen_session = {
            "num_turns": 2,
            "turn_1": {"input_ids": [1, 10], "response_ids": [11, 12]},
            "turn_2": {"input_ids": [1, 10, 11, 12, 20]},
        }
        ids, concat_ok = bench_heap._resolve_prompt_ids(session=qwen_session, turn_idx=2)
        self.assertTrue(concat_ok)
        self.assertEqual(ids, [1, 10, 11, 12, 20])

        llama_session = {
            "num_turns": 2,
            "turn_1": {"input_ids": [1, 10], "response_ids": [11, 12]},
            "turn_2": {"input_ids": [1, 10, 91, 92, 11, 12, 20]},
        }
        ids, concat_ok = bench_heap._resolve_prompt_ids(session=llama_session, turn_idx=2)
        self.assertFalse(concat_ok)
        self.assertEqual(ids, llama_session["turn_2"]["input_ids"])


class TestScheduleAndPolicy(unittest.TestCase):
    def test_stable_seed_golden_values(self):
        self.assertEqual(bench_heap.stable_seed_from_str("ShareGPT-0", base_seed=2042), 2914695463)
        self.assertEqual(bench_heap.stable_seed_from_str("apps_0", base_seed=42), 1054942099)
        self.assertEqual(bench_heap.stable_seed_from_str("alfred_1", base_seed=3042), 2122452089)

    def test_load_sessions_and_assert_turn_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sessions.jsonl")
            rows = [
                {"s1": {"id": "s1", "num_turns": 3}},
                {"s2": {"id": "s2", "num_turns": 2}},
                {"s3": {"id": "s3", "num_turns": 1}},
            ]
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            loaded = bench_heap.load_sessions(path, num_sessions=2, shuffle=False)
            self.assertEqual([s["id"] for s in loaded], ["s1", "s2"])
            bench_heap.assert_turn_policy({"name": "mt", "min_turns": 2}, loaded, path)

            all_sessions = bench_heap.load_sessions(path, num_sessions=-1, shuffle=False)
            with self.assertRaises(ValueError):
                bench_heap.assert_turn_policy({"name": "mt", "min_turns": 2}, all_sessions, path)
            with self.assertRaises(ValueError):
                bench_heap.assert_turn_policy(
                    {"name": "mt", "min_turns": 3, "max_turns": 2}, loaded, path
                )

    def test_build_hybrid_arrival_schedule_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "fixed.jsonl")
            p2 = os.path.join(tmp, "poisson.jsonl")
            with open(p1, "w", encoding="utf-8") as f:
                for i in range(2):
                    f.write(json.dumps({f"f{i}": {"id": f"f{i}", "num_turns": 1}}) + "\n")
            with open(p2, "w", encoding="utf-8") as f:
                for i in range(3):
                    f.write(json.dumps({f"p{i}": {"id": f"p{i}", "num_turns": 2}}) + "\n")

            cfg = {
                "tasks": [
                    {
                        "name": "fixed_task",
                        "data_path": p1,
                        "task_id": 0,
                        "seed": 42,
                        "shuffle": False,
                        "arrival": {"mode": "fixed", "offset_s": 0.1, "step_s": 0.5},
                    },
                    {
                        "name": "poisson_task",
                        "data_path": p2,
                        "task_id": 1,
                        "seed": 42,
                        "lambda_rate": 2.0,
                        "min_turns": 2,
                    },
                ]
            }
            s1 = bench_heap.build_hybrid_arrival_schedule(cfg)
            s2 = bench_heap.build_hybrid_arrival_schedule(cfg)
            self.assertEqual(
                [(e.task_name, e.session["id"], e.arrival_offset) for e in s1],
                [(e.task_name, e.session["id"], e.arrival_offset) for e in s2],
            )
            offsets = [e.arrival_offset for e in s1]
            self.assertEqual(offsets, sorted(offsets))


class TestMetricsAndStats(unittest.TestCase):
    def test_percentile(self):
        self.assertIsNone(bench_heap.percentile([], 50))
        self.assertEqual(bench_heap.percentile([7.0], 99), 7.0)
        self.assertAlmostEqual(bench_heap.percentile([10.0, 20.0, 30.0], 50), 20.0)
        self.assertAlmostEqual(bench_heap.percentile([10.0, 20.0], 25), 12.5)

    def test_stats_warmup_cutoff_and_summary_keys(self):
        stats = bench_heap.Stats()
        stats.configure_warmup(10.0, now_wall=100.0)

        warm_res = bench_heap.RequestMetrics(
            ttft=0.1, e2e=0.2, prompt_tokens=100, cached_tokens=50,
            completion_tokens=10, queue_time_ms=1.0, prefill_ms=2.0,
        )
        cold_res = bench_heap.RequestMetrics(
            ttft=0.05, e2e=0.15, prompt_tokens=200, cached_tokens=150,
            completion_tokens=20, queue_time_ms=0.5, prefill_ms=1.5,
        )

        row_base = {
            "task": "chat",
            "phase": "p1",
            "session_id": "s1",
            "turn": 1,
            "prior_tool_call": False,
            "prompt_tokens": 100,
            "cached_tokens": 50,
            "recompute_tokens": 50,
            "ttft_ms": 100.0,
            "e2e_ms": 200.0,
            "gap_since_prev_end_s": None,
            "send_lag_ms": 0.0,
            "queue_time_ms": 1.0,
            "prefill_ms": 2.0,
        }
        stats.record_turn(task_name="chat", result=warm_res, row=row_base, now_wall=105.0)
        stats.record_session_finished("chat", now_wall=105.0)
        stats.record_turn(task_name="chat", result=cold_res, row=row_base, now_wall=115.0)
        stats.record_session_finished("chat", now_wall=115.0)

        summary = stats.summary()
        self.assertEqual(set(summary.keys()), EXPECTED_SUMMARY_KEYS)
        self.assertEqual(summary["warmup_requests"], 1)
        self.assertEqual(summary["warmup_sessions"], 1)
        self.assertEqual(summary["warmup_prompt_tokens"], 100)
        self.assertEqual(summary["num_requests"], 1)
        self.assertEqual(summary["num_sessions"], 1)
        self.assertEqual(summary["done_requests"], 2)
        self.assertEqual(summary["cache_hit_rate_pct"], 75.0)
        self.assertEqual(list(stats.rows[0].keys()), EXPECTED_CSV_COLUMNS)
        self.assertTrue(stats.rows[0]["warmup"])
        self.assertFalse(stats.rows[1]["warmup"])


class TestHeapEmitter(unittest.TestCase):
    def test_multiturn_max_inflight_error_resilience_and_progress_lag_unit(self):
        sessions = [
            {
                "id": "s1",
                "num_turns": 2,
                "turn_1": {"input_ids": [1, 2], "response_ids": [3], "tool_call": {"k": 1}},
                "turn_2": {"input_ids": [1, 2, 3, 4], "response_ids": [5], "tool_call": None},
            },
            {
                "id": "s2",
                "num_turns": 2,
                "turn_1": {"input_ids": [6], "response_ids": [7], "tool_call": None},
                "turn_2": {"input_ids": [6, 7], "response_ids": [8], "tool_call": None},
            },
        ]

        async def run():
            heap = bench_heap.ArrivalHeap()
            for sess in sessions:
                heap.push(
                    due_ts=9.95,  # 比 mono_clock=10.0 早 0.05s → send_lag = 50.0ms
                    turn=bench_heap.PendingTurn(
                        task_name="agentic",
                        task_cfg={"name": "agentic", "seed": 42, "task_id": 0},
                        session=sess,
                        turn_idx=1,
                    ),
                )

            inflight_now = 0
            max_seen_inflight = 0
            seen_calls = []

            async def fake_req(http, url, payload, timeout_s):
                nonlocal inflight_now, max_seen_inflight
                inflight_now += 1
                max_seen_inflight = max(max_seen_inflight, inflight_now)
                try:
                    await asyncio.sleep(0.005)
                    seen_calls.append((payload["rid"], payload["tool_call"]))
                    if payload["rid"] == "s2_turn_1":
                        raise RuntimeError("boom")
                    return bench_heap.RequestMetrics(
                        ttft=0.01,
                        e2e=0.02,
                        prompt_tokens=len(payload["input_ids"]),
                        cached_tokens=1,
                        completion_tokens=1,
                        queue_time_ms=0.5,
                        prefill_ms=1.0,
                    )
                finally:
                    inflight_now -= 1

            now_wall = 1000.0
            stats = bench_heap.Stats()
            cfg = bench_heap.EmitterConfig(
                wait_scale=0.0,
                max_inflight=1,
                progress_every=1,
                timeout_s=5.0,
            )
            emitter = bench_heap.HeapEmitter(
                config=cfg,
                stats=stats,
                heap=heap,
                http=None,
                url="http://unused/generate",
                request_fn=fake_req,
                mono_clock=lambda: 10.0,
                wall_clock=lambda: now_wall,
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                await emitter.run()
            return stats, emitter.summary_extra(), max_seen_inflight, seen_calls, buf.getvalue()

        stats, extra, max_seen, seen_calls, stdout = asyncio.run(run())
        self.assertEqual(max_seen, 1)
        self.assertEqual(stats.errors, 1)
        self.assertEqual(stats.num_requests, 3)  # s2_turn_1 失败，s2_turn_2 仍继续完成
        self.assertEqual(stats.num_sessions, 2)
        self.assertIn(("s1_turn_1", None), seen_calls)
        self.assertIn(("s1_turn_2", {"k": 1}), seen_calls)
        self.assertIn(("s2_turn_2", None), seen_calls)
        # 验证 lag_p50 单位已修复为 ms（前两轮 0.05s -> 50.0ms，全部 4 轮中位 25.0ms）
        self.assertIn("lag_p50=50.0ms", stdout)
        self.assertEqual(extra["emitter"], "heap")
        self.assertEqual(extra["send_lag"]["p50_ms"], 25.0)
        self.assertEqual(extra["send_lag"]["max_ms"], 50.0)

        with tempfile.TemporaryDirectory() as tmp:
            out_json = os.path.join(tmp, "res.json")
            bench_heap.write_outputs(out_json, stats.summary(), stats.rows)
            self.assertTrue(os.path.isfile(out_json))
            out_csv = os.path.join(tmp, "res.csv")
            with open(out_csv, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, EXPECTED_CSV_COLUMNS)
                self.assertEqual(len(list(reader)), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
