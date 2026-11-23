"""R4 验收：无 response 时哨兵唯一 + 超大请求的显式处理。

两个都会静默改变树形，所以必须钉死：

1. **response token 每请求唯一**：所有请求共用同一串 token 时，同一个 prompt 的
   重放会命中上一次的 response 段（树节点数、占用、duplicate 统计全都偏）；
2. **超大请求**：生产用 chunked prefill / retraction 伺候，仿真装不下就必须
   `error`（默认，逼调用方直面）或 `skip`（显式跳过并计数），不能静默丢。

    PYTHONPATH=$PWD/python python -m pytest -q test/manual/test_sim_oversize.py
"""

import os
import sys
from array import array

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from sim.core.allocator import OutOfTokens, tree_tokens  # noqa: E402
from sim.core.config import SimConfig  # noqa: E402
from sim.core.request import OUTPUT_SENTINEL_BASE, SimReq  # noqa: E402
from sim.simulator import Simulator  # noqa: E402
from sim.workload.trace import Session, Turn  # noqa: E402


def _turn(prompt, out=0):
    return Turn(t=0.0, prompt=array("q", prompt), out=out)


def test_sim_req_uses_a_distinct_sentinel_per_request():
    a = SimReq("r1", "s", [1, 2, 3], output_len=4, output_token=OUTPUT_SENTINEL_BASE + 1)
    b = SimReq("r2", "s", [1, 2, 3], output_len=4, output_token=OUTPUT_SENTINEL_BASE + 2)
    assert set(a.output_ids) == {OUTPUT_SENTINEL_BASE + 1}
    assert set(b.output_ids) == {OUTPUT_SENTINEL_BASE + 2}
    assert a.output_ids != b.output_ids
    # 远高于任何真实词表（Qwen2.5 = 151k），同时留出 int32/uint32 的余量：
    # OPT 会拿它算 TreeNode.hash_value（C++ checked_u32），opt_map 用 array("i")。
    assert OUTPUT_SENTINEL_BASE > 1 << 20
    assert OUTPUT_SENTINEL_BASE + (1 << 20) < 1 << 31


def test_identical_prompts_do_not_share_the_response_segment():
    """同一 prompt + 同 out 的两次请求：decode 段必须各写一份。"""

    prompt = list(range(64))
    sim = Simulator(SimConfig(capacity=4096, seed=1))
    sim.lifecycle.prepare(width=80)
    for i in range(2):
        sim.lifecycle.serve(_turn(prompt, out=16), f"apps_{i}", task_name="chat")

    assert sim.lifecycle.duplicate_insert_tokens == 0, "response 段被合并了（fallback 哨兵没生效）"
    # prompt 共享 64；两段 response 各写 16（共用一串 0 时树里只有 80）
    assert tree_tokens(sim.cache) == 64 + 16 + 16
    assert sim.alloc.used == tree_tokens(sim.cache)


def test_oversize_raises_by_default():
    session = Session("apps_0", (_turn(list(range(500))),), task_name="chat")
    sim = Simulator(SimConfig(capacity=100, seed=1))
    with pytest.raises(OutOfTokens, match="oversize"):
        sim.run_replay([session])


def test_oversize_skip_counts_and_keeps_the_cache_consistent():
    prompt = list(range(500))
    sessions = [
        Session("apps_0", (_turn(prompt),), task_name="chat"),
        Session("apps_1", (_turn(list(range(100))),), task_name="chat"),
    ]
    sim = Simulator(SimConfig(capacity=100, seed=1, oversize="skip"))
    result = sim.run_replay(sessions)

    assert [r.skipped for r in result.records] == [True, False]
    assert result.summary["skipped_requests"] == 1
    # The skipped record keeps its raw length for debugging, but the summary
    # ratios only average served requests (PrefixSim's n_req_skipped rule).
    assert result.records[0].prompt_len == 500
    assert result.summary["requests"] == 1
    assert result.summary["total_requests"] == 2
    assert result.summary["skipped_prompt_tokens"] == 500
    assert result.summary["prompt_tokens"] == 100
    # 跳过的请求不落树：树里只有第二条的 100 token
    assert tree_tokens(sim.cache) == 100
    assert sim.alloc.used == tree_tokens(sim.cache) + sim.inflight_tokens
    assert sim.inflight_tokens == 0


def test_skip_leaves_no_dangling_lock():
    """跳过路径也要放锁，否则后续淘汰全被挡住。"""

    sessions = [
        Session("apps_0", (_turn(list(range(80))),), task_name="chat"),
        Session("apps_1", (_turn(list(range(200))),), task_name="chat"),  # 超容量
        Session("apps_2", (_turn(list(range(200, 290))),), task_name="chat"),
    ]
    sim = Simulator(SimConfig(capacity=100, seed=1, oversize="skip"))
    result = sim.run_replay(sessions)
    assert [r.skipped for r in result.records] == [False, True, False]
    # 第三条要写 90 token，池里已有 80：必须能腾出来（说明锁没漏）
    assert result.records[2].evicted >= 70
    assert sim.alloc.used <= 100
