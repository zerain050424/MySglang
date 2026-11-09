"""Behavior tests for the incremental session-type classifier.

Run with:
    /root/sl_old/bin/python -m pytest test/manual/test_session_type_classifier.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

from sglang.srt.mem_cache.heter_utils import SessionRegistry, SessionType


def observe(registry, sid, turn, at, prompt_len, tool_call=None):
    return registry.observe_request(
        sid, rid=f"{sid}-turn-{turn}", prompt_len=prompt_len, now=at, tool_call=tool_call
    )


def test_short_unshared_session_stays_trans_through_four_turns():
    registry = SessionRegistry()
    for turn in range(1, 5):
        state = observe(registry, "short", turn, turn * 2.0, 500 + turn * 100)
    assert state.turn_count == 4
    assert state.session_type is SessionType.TRANS


def test_short_session_with_cross_session_prefix_is_struct():
    registry = SessionRegistry()
    registry.record_cross_session_hit("short", owner_session_id="older-session")
    state = observe(registry, "short", 1, 1.0, 700)
    assert state.cross_session_shared is True
    assert state.session_type is SessionType.STRUCT


def test_same_session_owner_does_not_count_as_cross_session_sharing():
    registry = SessionRegistry()
    state = observe(registry, "short", 1, 1.0, 700)
    registry.record_cross_session_hit("short", owner_session_id="short")
    assert state.cross_session_shared is False
    assert state.session_type is SessionType.TRANS


def test_tool_call_after_fifth_turn_classifies_agentic():
    registry = SessionRegistry()
    timestamps = (0.0, 25.0, 50.0, 75.0)
    for turn, at in enumerate(timestamps, 1):
        state = observe(registry, "agent", turn, at, 100 + turn * 100)
    state = observe(registry, "agent", 5, 100.0, 900, tool_call={"action": "move_ahead"})
    assert state.turn_count == 5
    assert state.tool_call_count == 1
    assert state.session_type is SessionType.AGENTIC


def test_long_session_uses_gap_when_no_tool_calls():
    registry = SessionRegistry()
    for turn, at in enumerate((0.0, 5.0, 10.0, 15.0, 20.0), 1):
        state = observe(registry, "fast", turn, at, 1000 + (turn - 1) * 500)
    assert state.session_type is SessionType.AGENTIC

    registry = SessionRegistry()
    for turn, at in enumerate((0.0, 25.0, 50.0, 75.0, 100.0), 1):
        state = observe(registry, "chat", turn, at, 1000 + (turn - 1) * 100)
    assert state.session_type is SessionType.CHAT


def test_ambiguous_gap_falls_back_to_context_growth():
    timestamps = (0.0, 15.0, 30.0, 45.0, 60.0)
    registry = SessionRegistry()
    for turn, at in enumerate(timestamps, 1):
        state = observe(registry, "growing", turn, at, 100 + (turn - 1) * 500)
    assert state.session_type is SessionType.AGENTIC

    registry = SessionRegistry()
    for turn, at in enumerate(timestamps, 1):
        state = observe(registry, "small-growth", turn, at, 100 + (turn - 1) * 100)
    assert state.session_type is SessionType.CHAT


def test_duplicate_rid_does_not_count_extra_turn_or_tool_call():
    registry = SessionRegistry()
    call = {"action": "turn_left"}
    state = registry.observe_request("session", "same-rid", 100, now=1.0, tool_call=call)
    state = registry.observe_request("session", "same-rid", 100, now=1.2, tool_call=call)
    assert state.turn_count == 1
    assert state.tool_call_count == 1
    assert state.classifier_gaps == []


def test_generate_request_accepts_optional_tool_call_metadata():
    from sglang.srt.managers.io_struct import GenerateReqInput

    call = {"action": "move_ahead"}
    req = GenerateReqInput(input_ids=[1, 2], tool_call=call)
    assert req.tool_call == call


def test_tokenized_request_schema_carries_tool_call_metadata():
    import msgspec

    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

    field_names = {field.name for field in msgspec.structs.fields(TokenizedGenerateReqInput)}
    assert "tool_call" in field_names


def test_scheduler_request_accepts_tool_call_metadata():
    import inspect

    from sglang.srt.managers.schedule_batch import Req

    assert "tool_call" in inspect.signature(Req).parameters


def test_radix_cross_session_hit_promotes_short_session_to_struct():
    from array import array
    from types import SimpleNamespace

    import torch

    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
    from sglang.srt.mem_cache.heter_utils import HeterCacheQueueTag

    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            eviction_policy="heter_cache",
        )
    )
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", [1, 2, 3])),
            value=torch.tensor([1, 2, 3]),
            session_id="owner",
        )
    )
    req = SimpleNamespace(
        rid="short-turn-1",
        session_id="short",
        origin_input_ids=array("q", [1, 2, 3]),
        tool_call=None,
    )
    result = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3])), req=req)
    )
    state = cache.session_registry.get("short")
    assert state.session_type is SessionType.STRUCT
    assert result.last_device_node.queue_tag is HeterCacheQueueTag.QSTRUCT


def test_classified_agentic_session_tags_new_nodes(tmp_path):
    from array import array

    import torch

    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
    from sglang.srt.mem_cache.heter_utils import HeterCacheQueueTag

    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            eviction_policy="heter_cache",
        )
    )
    for turn in range(1, 6):
        cache.session_registry.observe_request(
            "agent", f"agent-turn-{turn}", 1000 + turn * 1000,
            tool_call={"action": "move_ahead"} if turn > 1 else None,
            now=float(turn * 2),
        )
    result = cache.insert(
        InsertParams(
            key=RadixKey(array("q", [10, 11, 12])),
            value=torch.tensor([10, 11, 12]),
            session_id="agent",
        )
    )
    assert result.last_device_node.queue_tag is HeterCacheQueueTag.QAGENT

    import json

    dump_path = tmp_path / "labels.jsonl"
    assert cache._dump_node_labels(str(dump_path), 1) == 1
    row = json.loads(dump_path.read_text())
    assert row["nodes"][0]["stype"] == "agentic"


def test_classified_chat_session_tags_new_nodes():
    from array import array

    import torch

    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
    from sglang.srt.mem_cache.heter_utils import HeterCacheQueueTag

    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            eviction_policy="heter_cache",
        )
    )
    for turn, at in enumerate((0.0, 25.0, 50.0, 75.0, 100.0), 1):
        cache.session_registry.observe_request(
            "chat", f"chat-turn-{turn}", 1000 + (turn - 1) * 100,
            now=at,
        )
    result = cache.insert(
        InsertParams(
            key=RadixKey(array("q", [20, 21, 22])),
            value=torch.tensor([20, 21, 22]),
            session_id="chat",
        )
    )
    assert result.last_device_node.queue_tag is HeterCacheQueueTag.QCHAT


def test_non_heter_cache_preserves_legacy_delta_tags():
    from array import array
    from types import SimpleNamespace
    import time

    import torch

    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
    from sglang.srt.mem_cache.heter_utils import HeterCacheQueueTag

    cache = RadixCache.create_simulated(page_size=1)
    inserted = cache.insert(
        InsertParams(
            key=RadixKey(array("q", [30, 31])),
            value=torch.tensor([30, 31]),
            session_id="legacy-session",
        )
    )
    inserted.last_device_node.last_access_time = time.monotonic() - 1.0
    req = SimpleNamespace(
        rid="legacy-session-turn-2",
        session_id="legacy-session",
        origin_input_ids=array("q", [30, 31]),
        tool_call=None,
    )
    cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", [30, 31])), req=req)
    )
    assert inserted.last_device_node.queue_tag is HeterCacheQueueTag.QAGENT


def test_tokenized_tool_call_round_trips_through_msgpack():
    from array import array

    from sglang.srt.managers.io_struct import (
        TokenizedGenerateReqInput,
        msgpack_decode,
        msgpack_encode,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams

    call = {"action": "move_ahead"}
    request = TokenizedGenerateReqInput(
        input_text="",
        input_ids=array("q", [1, 2]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=[0, 0],
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        tool_call=call,
    )
    decoded = msgpack_decode(msgpack_encode(request))
    assert decoded.tool_call == call


def test_bench_passes_only_completed_tool_calls(monkeypatch):
    import asyncio
    from argparse import Namespace

    bench_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Heter-Bench"))
    sys.path.insert(0, bench_path)
    import bench_hetercache

    sent_tool_calls = []

    async def fake_one_request(http, url, payload, timeout_s):
        sent_tool_calls.append(payload["tool_call"])
        return {
            "ttft": 0.01,
            "e2e": 0.02,
            "prompt_tokens": len(payload["input_ids"]),
            "cached_tokens": 0,
            "completion_tokens": 1,
            "queue_time_ms": 0.0,
            "prefill_ms": 0.0,
        }

    monkeypatch.setattr(bench_hetercache, "one_request", fake_one_request)
    first_call = {"action": "move_ahead"}
    session = {
        "id": "alfred_test_session",
        "num_turns": 2,
        "turn_1": {"input_ids": [1], "response_ids": [2], "tool_call": first_call},
        "turn_2": {"input_ids": [1, 2], "response_ids": [3], "tool_call": None},
    }
    stats = bench_hetercache.Stats()
    asyncio.run(
        bench_hetercache.run_session(
            http=None,
            sem=asyncio.Semaphore(1),
            sess=session,
            args=Namespace(wait_scale=0.0, timeout_s=1.0, seed=42),
            stats=stats,
            url="http://unused/generate",
            task_cfg={"name": "AgentBank_alfred", "seed": 42, "task_id": 3},
        )
    )
    assert sent_tool_calls == [None, first_call]
    assert [row["prior_tool_call"] for row in stats.rows] == [False, True]
