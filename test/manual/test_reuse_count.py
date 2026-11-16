"""`reuse_count` 的契约测试：只有 admission match 才算一次「重读」。

两个计数器各管一件事，不能混：

* `hit_count` —— 上游口径，数「写路径与匹配路径走过多少遍」。一次请求自己就会
  加 1（chunked prefill 逐块回扫还会再加）。LFU / SLRU / unified 写穿判断读它，
  HeterCache **不再**用它判 one-shot。
* `reuse_count` —— 只数「请求确定自己 KV 缺口时的那次前缀匹配」，一次请求最多
  +1。`is_one_shot` 只读它。

所以要区分三种触达：
  1. admission match（`schedule_batch` / `schedule_policy` 传 req 的那次）→ 两个都 +1；
  2. `cache_unfinished_req` 插完块后的回扫（`count_reuse=False`）→ 只有 `hit_count` +1；
  3. 写路径 `_insert_helper`（不经过 match）→ 只有 `hit_count` +1（`_inc_hit_count`）。

不需要 GPU / 不需要 sglang 的完整导入链：`python/` 下的模块按路径装载，
`mem_cache.utils` / `registry` / `metrics_collector` 等重依赖用 stub 顶掉
（同 `test_queue_selector.py` 的手法，见 AGENTS.md 5.2）。

跑法：
    PYTHONPATH=$PWD/python python test/manual/test_reuse_count.py

顺带记一个本机踩过的坑：改动若是「同长度的一行」（例如把 `!= 0` 改成 `!= 1`），
源文件 mtime 可能落在同一秒且 size 不变，Python 会复用旧 `.pyc`，测试结果与源码
不符。切变体时请设 `PYTHONDONTWRITEBYTECODE=1` 或先清 `__pycache__`。
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


def _match(cache, tokens, count_reuse=True):
    return cache.match_prefix(
        bpc.MatchPrefixParams(
            key=rc.RadixKey(list(tokens), None), req=None, count_reuse=count_reuse
        )
    )


def test_contract_default_counts_reuse():
    """默认 True：现有全部 admission 调用点行为不变（不传就记重读）。"""
    print("[1] MatchPrefixParams.count_reuse defaults to True")
    params = bpc.MatchPrefixParams(key=rc.RadixKey([1], None))
    check(params.count_reuse is True, f"default must be True, got {params.count_reuse!r}")


def test_admission_match_counts_both():
    """admission match：reuse_count 与 hit_count 都 +1，并喂 VD 归因。"""
    print("[2] an admission match counts as one reuse")
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
    check(node.reuse_count == 1, f"reuse_count must be 1, got {node.reuse_count}")
    check(node.hit_count == 1, f"hit_count must be 1, got {node.hit_count}")
    check(sum(feed.values()) == 3, f"VD attribution must see 3 tokens, got {feed}")


def test_rewalk_counts_only_hit_count():
    """回扫（count_reuse=False）：hit_count 照上游口径 +1，reuse_count 不动。"""
    print("[3] the post-insert rewalk counts hit_count only")
    cache = _make_cache()
    node = _attach(cache, [1, 2, 3])

    _match(cache, [1, 2, 3], count_reuse=False)
    check(node.reuse_count == 0,
          f"a rewalk is not a reuse, got reuse_count={node.reuse_count}")
    check(node.hit_count == 1,
          f"hit_count must still follow the upstream rule, got {node.hit_count}")


def test_rewalk_is_idempotent_for_reuse_count():
    """连扫 5 次回扫：reuse_count 恒 0；再来一次 admission 才是 1。"""
    print("[4] repeated rewaks do not accumulate reuse_count")
    cache = _make_cache()
    node = _attach(cache, [1, 2, 3])
    for _ in range(5):
        _match(cache, [1, 2, 3], count_reuse=False)
    check(node.reuse_count == 0, f"5 rewaks must not count, got {node.reuse_count}")

    _match(cache, [1, 2, 3])          # 一次 admission match
    check(node.reuse_count == 1, f"one admission must be 1, got {node.reuse_count}")
    for _ in range(5):
        _match(cache, [1, 2, 3], count_reuse=False)
    check(node.reuse_count == 1,
          f"rewaks after an admission must not add, got {node.reuse_count}")


def test_split_inherits_reuse_count():
    """分裂出的父节点必须继承 reuse_count，否则命中过的段会被当成没命中。"""
    print("[5] a split node inherits reuse_count")

    # (a) 回扫触发的分裂：只继承，不额外计数
    cache = _make_cache()
    node = _attach(cache, [1, 2, 3])
    _match(cache, [1, 2, 3])                       # reuse_count = 1
    check(node.reuse_count == 1, "sanity: one reuse")
    split = _match(cache, [1, 2], count_reuse=False)
    check(len(split.device_indices) == 2, "the short match must return 2 tokens")
    parent = split.last_device_node
    check(parent.reuse_count == 1,
          f"inherited only (rewalk, no bump), got {parent.reuse_count}")

    # (b) admission 触发的分裂：继承 + 本次重读
    cache = _make_cache()
    node = _attach(cache, [4, 5, 6])
    _match(cache, [4, 5, 6])
    split = _match(cache, [4, 5])
    parent = split.last_device_node
    check(parent.reuse_count == 2,
          f"inherited 1 + this read 1 = 2, got {parent.reuse_count}")
    child = next(iter(parent.children.values()))
    check(child.reuse_count == 1,
          f"the untouched child keeps its old count, got {child.reuse_count}")


def test_rewalk_does_not_retag():
    """回扫不得改写 queue_tag：它没有会话上下文，legacy 分支会按 delta<8s 猜 QAGENT。

    `cache_unfinished_req` 的回扫不带 req → `session_id is None`，`_record_node_hit`
    会落进 `elif state is None or session_id is None:` 分支，把刚在 insert 时按会话
    类型盖好章的节点改成 QAGENT（delta≈0）。这个假标签会同时带偏 VD 的按队列归因
    与 `queue_order`，所以 `count_reuse=False` 时整段标签逻辑都要跳过。
    """
    print("[6] a rewalk never retags")
    cache = _make_cache()  # now = 1000.0
    # 挂一个真策略，让 legacy 重标分支真的可达。
    cache._heter = ep.HeterCacheStrategy()

    just_written = _attach(cache, [1, 2, 3])
    just_written.queue_tag = hu.HeterCacheNodeTag.QCHAT
    just_written.last_access_time = 1000.0  # delta≈0 → legacy 分支会判成 QAGENT

    _match(cache, [1, 2, 3], count_reuse=False)
    check(just_written.queue_tag is hu.HeterCacheNodeTag.QCHAT,
          f"a rewalk must not retag, got {just_written.queue_tag}")

    # 真读一次之后照常按会话类型改写（此处无 req，走 legacy → QAGENT，属预期）
    _match(cache, [1, 2, 3])
    check(just_written.queue_tag is hu.HeterCacheNodeTag.QAGENT,
          "an admission match may still retag via the legacy branch, got "
          f"{just_written.queue_tag}")


def main():
    tests = [
        test_contract_default_counts_reuse,
        test_admission_match_counts_both,
        test_rewalk_counts_only_hit_count,
        test_rewalk_is_idempotent_for_reuse_count,
        test_split_inherits_reuse_count,
        test_rewalk_does_not_retag,
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
