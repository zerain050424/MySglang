"""chunked prefill 的回扫不得计命中：`MatchPrefixParams.chunked` 的契约测试。

背景（`radix_cache.py` 的三处写路径）：

* `cache_unfinished_req` 插完一块后会再走一次 `match_prefix`，只为拿索引；
  此时匹配到的节点里有本请求刚创建的那些。若照常 `_record_node_hit`，
  `hit_count` 就变成「被扫过多少遍」而不是「被复用多少次」——chunk 越多越
  大，而 `HeterCacheStrategy.is_one_shot` 正是靠它判「有没有被复用」。
* `cache_unfinished_req` 的 `chunked=False` 走 `_insert_helper` 时会把整条
  prompt 路径 +1（`_inc_hit_count`），那一分是"自己写的自己走一遍"。
* 真正的复用是「别的请求 prefill 时的 match」（+1）与「它自己结束时的
  insert 回扫」（+1）。

所以本文件锁的是四条：`chunked=True` 不加分也不重标、`chunked=False` 照常加分、
回扫幂等、`chunked` 缺省为 False（现有调用点行为不变）。

顺带记一个本机踩过的坑：改完 `evict_policy.py` 里一行同长度的比较（`!= 0` →
`!= 1`）后立刻重跑，Python 会因「源文件 mtime 落在同一秒且 size 不变」而复用
旧 `.pyc`，测试看起来通过/失败都可能与源码不符。切变体时请设
`PYTHONDONTWRITEBYTECODE=1` 或先清 `__pycache__`。

不需要 GPU / 不需要 sglang 的完整导入链：`python/` 下的模块按路径装载，
`mem_cache.utils` / `registry` / `metrics_collector` 等重依赖用 stub 顶掉
（同 `test_queue_selector.py` 的手法，见 AGENTS.md 5.2）。

跑法：
    PYTHONPATH=$PWD/python python test/manual/test_chunked_hit_count.py
"""

import importlib.util
import os
import sys
import types

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_PY = os.path.join(_ROOT, "python")
_CACHE_DIR = os.path.join(_PY, "sglang", "srt", "mem_cache")
_HC_DIR = os.path.join(_CACHE_DIR, "heter_cache")

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


for _pkg in (
    "sglang",
    "sglang.srt",
    "sglang.srt.mem_cache",
    "sglang.srt.mem_cache.heter_cache",
    "sglang.srt.observability",
    "sglang.srt.mem_cache.unified_cache",
):
    _stub_package(_pkg)

# --- 重依赖用 stub 顶掉：这些模块的真实实现在 import 期会拉起 torch 模型栈 ---
_stub_module("sglang.srt.mem_cache.allocator", BaseTokenToKVPoolAllocator=object)
_stub_module("sglang.srt.mem_cache.memory_pool", ReqToTokenPool=object)
_stub_module("sglang.srt.mem_cache.unified_cache.component_type", ComponentType=object)
_stub_module(
    "sglang.srt.observability.metrics_collector",
    SchedulerMetricsCollector=object,
    STAT_LOGGER_ROLE_RADIX_CACHE="radix_cache",
    RadixCacheMetricsCollector=object,
    radix_cache_metric_labels=lambda *a, **k: {},
    resolve_collector_class=lambda *a, **k: object,
)
_stub_module(
    "sglang.srt.runtime_context",
    get_observability=lambda: None,
    get_parallel=lambda: None,
)


def _fake_register_backend(*args, **kwargs):
    return None


_stub_module(
    "sglang.srt.mem_cache.registry", register_radix_cache_backend=_fake_register_backend
)
# events.py 会拉起 disaggregation 与 mem_cache.utils 的哈希工具；本测试只用类型名。
_stub_module("sglang.srt.mem_cache.events", KVCacheEventRecorder=object)

hu = _load(
    "sglang.srt.mem_cache.heter_cache.utils", os.path.join(_HC_DIR, "utils.py")
)
_load(
    "sglang.srt.mem_cache.heter_cache.occupancy", os.path.join(_HC_DIR, "occupancy.py")
)
_load("sglang.srt.mem_cache.heter_cache.value", os.path.join(_HC_DIR, "value.py"))
_load("sglang.srt.mem_cache.heter_cache.selector", os.path.join(_HC_DIR, "selector.py"))
_load(
    "sglang.srt.mem_cache.heter_cache.node_share", os.path.join(_HC_DIR, "node_share.py")
)
_load(
    "sglang.srt.mem_cache.heter_cache.override", os.path.join(_HC_DIR, "override.py")
)
_load(
    "sglang.srt.mem_cache.heter_cache.bench_bridge",
    os.path.join(_HC_DIR, "bench_bridge.py"),
)
ep = _load(
    "sglang.srt.mem_cache.evict_policy", os.path.join(_CACHE_DIR, "evict_policy.py")
)

# `mem_cache.utils` 会拉起 sglang 的 server 栈；这里只需要两个符号。
mem_utils = _stub_module(
    "sglang.srt.mem_cache.utils",
    get_eviction_strategy=lambda policy, config=None: ep.HeterCacheStrategy(
        **(config or {})
    ),
    split_node_hash_value=lambda h, n, p: (None, None),
)
sys.modules["sglang.srt.mem_cache.utils"] = mem_utils

# cache_init_params 只被 radix_cache 的构造路径用；本测试走 __new__，用空壳顶掉。
_stub_module("sglang.srt.mem_cache.cache_init_params", CacheInitParams=object)
bpc = _load(
    "sglang.srt.mem_cache.base_prefix_cache",
    os.path.join(_CACHE_DIR, "base_prefix_cache.py"),
)
rc = _load("sglang.srt.mem_cache.radix_cache", os.path.join(_CACHE_DIR, "radix_cache.py"))

import torch  # stub 装配完之后再引，避免提前拉起无关模块


def _make_cache(now=1000.0):
    """最小可用的 RadixCache：只装 `match_prefix` / `_record_node_hit` 要用的字段。"""
    cache = rc.RadixCache.__new__(rc.RadixCache)
    cache._now = lambda: now
    cache.page_size = 1
    cache.is_eagle = False
    cache.disable = False
    cache._heter = None            # 关掉标签重写与跨会话记账，只测计数
    cache._hit_recorder = None
    cache.force_agentic = False
    cache.share_on_hit = None
    cache.share_on_split = None
    cache.session_registry = hu.SessionRegistry()
    cache._empty_match_result = bpc.MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=None,
        last_host_node=None,
        best_match_node=None,
    )
    cache.root_node = rc.TreeNode()
    cache.root_node.key = rc.RadixKey([], None)
    return cache


def _attach(cache, tokens):
    """把一段 tokens 挂成一个根的直接子节点（等价于首次 insert 建出的节点）。"""
    node = rc.TreeNode(abs_start=0)
    node.key = rc.RadixKey(list(tokens), None)
    node.value = torch.arange(len(tokens), dtype=torch.int64)
    node.parent = cache.root_node
    node.session_id = "s1"
    cache.root_node.children[node.key.child_key(cache.page_size)] = node
    return node


def _match(cache, tokens, chunked=False):
    return cache.match_prefix(
        bpc.MatchPrefixParams(
            key=rc.RadixKey(list(tokens), None), req=None, chunked=chunked
        )
    )


def test_contract_default_is_not_chunked():
    """默认 False：现有全部调用点（scheduler 的真实匹配）行为不变。"""
    print("[1] MatchPrefixParams.chunked defaults to False")
    params = bpc.MatchPrefixParams(key=rc.RadixKey([1], None))
    check(params.chunked is False, f"default must be False, got {params.chunked!r}")


def test_chunked_rescan_does_not_bump_hit_count():
    """chunked=True 的回扫：hit_count 不动，也不产生 VD 归因。"""
    print("[2] chunked rescan leaves hit_count untouched")
    cache = _make_cache()
    node = _attach(cache, [1, 2, 3])
    feed = {}
    cache._hit_recorder = lambda req, last, tokens, by_queue: feed.update(by_queue)

    result = _match(cache, [1, 2, 3], chunked=True)
    check(len(result.device_indices) == 3, "indices must still be returned")
    check(result.last_device_node is node, "must still resolve the terminal node")
    check(node.hit_count == 0, f"hit_count must stay 0, got {node.hit_count}")
    check(not feed, f"chunked rescan must not feed VD attribution, got {feed}")


def test_plain_match_still_bumps():
    """chunked=False（真实复用）：照常 +1 并给出 by_queue。"""
    print("[3] plain match still counts as a reuse")
    cache = _make_cache()
    node = _attach(cache, [1, 2, 3])
    feed = {}
    cache._hit_recorder = lambda req, last, tokens, by_queue: feed.update(by_queue)

    class _Req:
        rid = "r1"
        session_id = "s1"
        tool_call = None

    cache.match_prefix(
        bpc.MatchPrefixParams(key=rc.RadixKey([1, 2, 3], None), req=_Req())
    )
    check(node.hit_count == 1, f"one reuse must be +1, got {node.hit_count}")
    check(sum(feed.values()) == 3, f"VD attribution must see 3 tokens, got {feed}")


def test_rescan_is_idempotent():
    """多次回扫不得累加：chunked prefill 每块都会回扫一次。"""
    print("[4] repeated rescans are idempotent")
    cache = _make_cache()
    node = _attach(cache, [1, 2, 3])
    for _ in range(5):
        _match(cache, [1, 2, 3], chunked=True)
    check(node.hit_count == 0, f"5 rescans must not accumulate, got {node.hit_count}")

    _match(cache, [1, 2, 3])          # 一次真实匹配
    for _ in range(5):
        _match(cache, [1, 2, 3], chunked=True)
    check(node.hit_count == 1,
          f"real hit + 5 rescans must be 1, got {node.hit_count}")


def test_chunked_rescan_does_not_retag_or_touch():
    """回扫连标签与时间戳都不该改：它们由 `_insert_helper` 在写路径上盖。

    `cache_unfinished_req` 的回扫不带 `req`，所以 `session_id is None` 会让
    `_record_node_hit` 落到 legacy 分支，按 `delta < 8s` 把节点重写成 QAGENT。
    刚写进去的节点 delta≈0 → 每个 chunk 都把 prompt 节点刷成 agentic。这是
    回扫计命中的第二个（更贵的）副作用：不只 hit_count 失真，队列归属也被
    覆盖，而 VD 的按队列归因与 `queue_order` 都吃这个标签。
    """
    print("[5] chunked rescan does not retag or restamp")
    cache = _make_cache()  # now = 1000.0
    # 挂一个真策略，让 legacy 重标分支真的可达。
    cache._heter = ep.HeterCacheStrategy()

    fresh = _attach(cache, [1, 2, 3])
    fresh.queue_tag = hu.HeterCacheNodeTag.QCHAT
    fresh.last_access_time = 1000.0  # delta≈0：legacy 分支会判 < 8s → QAGENT

    old = _attach(cache, [4, 5, 6])
    old.queue_tag = hu.HeterCacheNodeTag.QCHAT
    old.last_access_time = 42.0      # 时间戳必须停在这个写路径值上

    _match(cache, [1, 2, 3], chunked=True)
    _match(cache, [4, 5, 6], chunked=True)

    check(fresh.queue_tag is hu.HeterCacheNodeTag.QCHAT,
          f"a just-written node must keep its tag, got {fresh.queue_tag}")
    check(old.queue_tag is hu.HeterCacheNodeTag.QCHAT,
          f"queue_tag must be untouched, got {old.queue_tag}")
    check(fresh.last_access_time == 1000.0 and old.last_access_time == 42.0,
          "node timestamps must stay at the write-path values, got "
          f"{fresh.last_access_time} / {old.last_access_time}")


def main():
    tests = [
        test_contract_default_is_not_chunked,
        test_chunked_rescan_does_not_bump_hit_count,
        test_plain_match_still_bumps,
        test_rescan_is_idempotent,
        test_chunked_rescan_does_not_retag_or_touch,
    ]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            FAILS.append(f"{test.__name__} raised {type(exc).__name__}: {exc}")
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)} problems)")
        for fail in FAILS:
            print("  -", fail)
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
