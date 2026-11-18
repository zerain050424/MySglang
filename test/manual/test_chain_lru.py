"""ChainLRUStrategy：LRU + 链形修正的纯逻辑单测（不需要 GPU / torch）。

被测对象：`python/sglang/srt/mem_cache/evict_policy.py` 的 ``chain_lru``。

背景（tail-stress 诊断）：match 刷新整条匹配路径的 recency，一条私有单子链
要么全新要么全旧；链成为全局最旧后，``evict`` 里「叶子出堆 → 父变叶子再出堆」
会一路剥穿，整链归零后进入「整链重插 → 再被剥穿」的死循环。该策略做两件事：

* 浅段保护（``floor_tokens``）：淘汰叶子后链上剩余前缀低于下限时，把
  ``weight × 缺口`` 秒加进 LRU 键；``gate="rerode"`` 只护有重读证据的链。
* 砍链尾（``tail_first``）：平票时先剥 ``remain`` 大的叶子。

重读证据只认 ``reuse_count``（admission match 计数），``hit_count`` 含写路径
自增，会把刚写的段误判成「被重读过」——所以替身节点两个都给，断言里分开验。
"""

import importlib.util
import os
import sys
import types

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_CACHE_DIR = os.path.join(_ROOT, "python", "sglang", "srt", "mem_cache")
_HC_DIR = os.path.join(_CACHE_DIR, "heter_cache")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _real_evict_policy():
    """环境完整（torch 链能起来）时直接用真包，不污染 sys.modules。

    替身装载会往 sys.modules 里塞空壳包，同一进程里随后导入真 sglang 的
    测试（如 test_radix_eviction_policy_cli）会被毒害。先试真导入，失败才装替身。
    """
    try:
        import sglang.srt.mem_cache.evict_policy as m
        return m
    except Exception:
        return None


def _stub_package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


ep = _real_evict_policy()
if ep is None:
    for _pkg in (
        "sglang",
        "sglang.srt",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.heter_cache",
    ):
        _stub_package(_pkg)

if ep is None:
    hu = _load(
        "sglang.srt.mem_cache.heter_cache.utils",
        os.path.join(_HC_DIR, "utils.py"),
    )
else:
    import sglang.srt.mem_cache.heter_cache.utils as hu

if ep is None:
    _load("sglang.srt.mem_cache.heter_cache.utils", os.path.join(_HC_DIR, "utils.py"))
    _load(
        "sglang.srt.mem_cache.heter_cache.occupancy",
        os.path.join(_HC_DIR, "occupancy.py"),
    )
    _load("sglang.srt.mem_cache.heter_cache.value", os.path.join(_HC_DIR, "value.py"))
    _load(
        "sglang.srt.mem_cache.heter_cache.selector",
        os.path.join(_HC_DIR, "selector.py"),
    )
    _load(
        "sglang.srt.mem_cache.heter_cache.node_share",
        os.path.join(_HC_DIR, "node_share.py"),
    )
    _load(
        "sglang.srt.mem_cache.heter_cache.override",
        os.path.join(_HC_DIR, "override.py"),
    )
    _load(
        "sglang.srt.mem_cache.heter_cache.bench_bridge",
        os.path.join(_HC_DIR, "bench_bridge.py"),
    )
    _load(
        "sglang.srt.mem_cache.heter_cache.regret_gate",
        os.path.join(_HC_DIR, "regret_gate.py"),
    )
    ep = _load(
        "sglang.srt.mem_cache.evict_policy",
        os.path.join(_CACHE_DIR, "evict_policy.py"),
    )

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


class _Node:
    """链节点替身：只带 get_priority 用到的字段。root 的 parent 为 None。"""

    _next_id = 0

    def __init__(self, tokens, lat, reuse_count=0, hit_count=0, parent=None,
                 session_id=None, queue_tag=None):
        _Node._next_id += 1
        self.id = _Node._next_id
        self.key = [0] * tokens
        self.last_access_time = lat
        self.reuse_count = reuse_count
        self.hit_count = hit_count
        self.parent = parent
        self.session_id = session_id
        self.queue_tag = queue_tag


def _chain(specs, base_lat):
    """specs: [(tokens, reuse_count, hit_count), ...] 从浅到深挂成单子链，返回叶子。

    顶端先挂一个空 root（tokens=0，parent=None），与真树同构：链段从 root 之下
    开始计数，root 自身不进 remain / 证据扫描。
    """
    parent = _Node(0, base_lat)
    for tokens, rc, hc in specs:
        base_lat += 1.0
        parent = _Node(tokens, base_lat, reuse_count=rc, hit_count=hc, parent=parent)
    return parent


def test_floor_penalty_and_slope():
    """remain 低于下限才有罚，罚 = weight × 缺口，随深度递增（越剥越贵）。"""
    # 12k 链按轮分段：remain(叶子) = 10500 > floor → 无罚
    leaf = _chain([(1500, 1, 2)] * 8, 0.0)
    s = ep.ChainLRUStrategy(floor_tokens=4096, weight=0.02, gate="rerode")
    deep_key = s.get_priority(leaf)
    check(deep_key[0] == leaf.last_access_time, f"remain≥floor 无罚：{deep_key}")
    check(deep_key[1] == -10500.0, f"砍链尾平票键 = -remain：{deep_key}")
    check(isinstance(deep_key[2], int), f"末位是节点 id：{deep_key}")

    # 把链剥到只剩一段（remain=1500 < 4096）→ 罚 = 0.02 × 2596
    leaf2 = _chain([(1500, 1, 2), (4500, 1, 2)], 0.0)
    shallow_key = s.get_priority(leaf2)
    expect = leaf2.last_access_time + 0.02 * (4096 - 1500)
    check(
        abs(shallow_key[0] - expect) < 1e-9,
        f"罚 = weight × 缺口：{shallow_key[0]} != {expect}",
    )


def test_rerode_gate():
    """gate="rerode"：路径上无重读证据（reuse_count 全 0）不护浅段。"""
    # 全 0 reuse_count 的短链：hit_count 高也不算（hit_count 含写路径自增）
    leaf = _chain([(1500, 0, 5), (1500, 0, 1)], 0.0)
    s = ep.ChainLRUStrategy(floor_tokens=4096, weight=0.02, gate="rerode")
    key = s.get_priority(leaf)
    check(
        key[0] == leaf.last_access_time,
        f"无重读证据不护浅段（hit_count 不作数）：{key}",
    )

    # 任一祖先 reuse_count>0 → 护
    leaf2 = _chain([(1500, 1, 2), (1500, 0, 1)], 0.0)
    key2 = s.get_priority(leaf2)
    check(
        key2[0] > leaf2.last_access_time,
        f"路径上有重读证据 → 护浅段：{key2}",
    )

    # gate="none"：不看证据，一律护
    s_none = ep.ChainLRUStrategy(floor_tokens=4096, weight=0.02, gate="none")
    key3 = s_none.get_priority(leaf)
    check(key3[0] > leaf.last_access_time, f"gate=none 不看证据：{key3}")


def test_tail_first_tiebreak():
    """平票（同 lat、同罚）时先剥 remain 大的叶子——砍链尾。"""
    short = _chain([(1500, 1, 2)], 0.0)      # remain = 0
    long_ = _chain([(1500, 1, 2)] * 4, 0.0)  # remain = 4500 > floor → 无罚
    short.last_access_time = long_.last_access_time
    s = ep.ChainLRUStrategy(floor_tokens=4096, weight=0.02, gate="rerode")
    check(
        s.get_priority(long_)[0] < s.get_priority(short)[0],
        "同 lat：长链的尾先出（短链近根被罚）",
    )
    # tail_first=False：平票键不再含 -remain
    s2 = ep.ChainLRUStrategy(floor_tokens=4096, weight=0.02, gate="rerode",
                             tail_first=False)
    check(s2.get_priority(short)[1] == 0.0, "tail_first=False 平票键为 0")
    # floor=0 退化为纯 LRU + 砍链尾平票键，无罚
    s3 = ep.ChainLRUStrategy(floor_tokens=0, weight=0.02, gate="rerode")
    key = s3.get_priority(short)
    check(key[0] == short.last_access_time, f"floor=0 无罚：{key}")


def test_multiturn_gate():
    """gate="multiturn"：rerode 之外，归属会话已观测到轮间隔也护。

    场景：tail-stress 的 warm 第 1 轮——链还没被重读过（reuse_count=0），
    但会话已经回来过一次（gaps ≥ 1）。rerode 门下它裸奔，multiturn 门下被护。
    """
    hu.SessionRegistry.reset_global()
    reg = hu.SessionRegistry.global_instance()  # 策略读的是进程单例
    reg._sessions["warm-1"] = hu.SessionState(session_id="warm-1", gaps=[45.0])
    reg._sessions["cold-1"] = hu.SessionState(session_id="cold-1", gaps=[])

    leaf = _chain([(1500, 0, 1), (1500, 0, 1)], 0.0)
    leaf.session_id = "warm-1"
    leaf.parent.session_id = "warm-1"
    s = ep.ChainLRUStrategy(floor_tokens=4096, weight=0.02, gate="multiturn")
    key = s.get_priority(leaf)
    check(key[0] > leaf.last_access_time, f"会话回来过 → 护浅段：{key}")

    # 单轮会话（gaps 为空）：不护
    leaf2 = _chain([(1500, 0, 1), (1500, 0, 1)], 0.0)
    leaf2.session_id = "cold-1"
    leaf2.parent.session_id = "cold-1"
    key2 = s.get_priority(leaf2)
    check(
        key2[0] == leaf2.last_access_time,
        f"单轮会话不被 multiturn 门护：{key2}",
    )

    # 无 session_id 的节点：退化为 rerode 证据
    leaf3 = _chain([(1500, 0, 1), (1500, 0, 1)], 0.0)
    key3 = s.get_priority(leaf3)
    check(
        key3[0] == leaf3.last_access_time,
        f"无 session_id 且无重读 → 不护：{key3}",
    )

    # rerode 证据仍在 multiturn 门下生效
    leaf4 = _chain([(1500, 1, 2), (1500, 0, 1)], 0.0)
    key4 = s.get_priority(leaf4)
    check(key4[0] > leaf4.last_access_time, f"重读证据仍生效：{key4}")
    hu.SessionRegistry.reset_global()


def test_sla_multiplier():
    """per-queue SLA 乘数：同形状两叶，SLA 大的队列锚更贵；未列出队列恒 1。"""
    hu.SessionRegistry.reset_global()
    reg = hu.SessionRegistry.global_instance()
    reg._sessions["c1"] = hu.SessionState(session_id="c1", gaps=[45.0])
    reg._sessions["a1"] = hu.SessionState(session_id="a1", gaps=[6.0])

    TAG_CHAT = hu.HeterCacheNodeTag.QCHAT
    TAG_AGENTIC = hu.HeterCacheNodeTag.QAGENT

    def mk(sid, tag, lat):
        return _Node(1500, lat, reuse_count=0, parent=None,
                     session_id=sid, queue_tag=tag)

    leaf_chat = mk("c1", TAG_CHAT, 10.0)
    leaf_agentic = mk("a1", TAG_AGENTIC, 10.0)
    s = ep.ChainLRUStrategy(floor_tokens=4096, weight=0.02, gate="none",
                            sla={"chat": 2.0})
    # 同形状（remain=0），chat 罚 = 0.02×2×4096，agentic 罚 = 0.02×1×4096
    k_chat = s.get_priority(leaf_chat)
    k_agentic = s.get_priority(leaf_agentic)
    check(abs(k_chat[0] - (10.0 + 0.02 * 2.0 * 4096)) < 1e-9,
          f"sla_chat=2 → 罚加倍：{k_chat[0]}")
    check(abs(k_agentic[0] - (10.0 + 0.02 * 4096)) < 1e-9,
          f"未列出队列恒 1：{k_agentic[0]}")
    check(k_chat[0] > k_agentic[0], "SLA 大的队列更难被削")

    # 证据门仍然在前：rerode 门下无证据的链不享受 SLA 保护
    s2 = ep.ChainLRUStrategy(floor_tokens=4096, weight=0.02, gate="rerode",
                             sla={"chat": 2.0})
    leaf_noev = mk("c9", TAG_CHAT, 10.0)   # 会话未回来过（gaps 空）
    reg._sessions["c9"] = hu.SessionState(session_id="c9", gaps=[])
    k_noev = s2.get_priority(leaf_noev)
    check(k_noev[0] == 10.0, f"无证据不享受 SLA 保护：{k_noev[0]}")
    hu.SessionRegistry.reset_global()


def test_param_validation_with_sla():
    for sla in ({"chat": 0}, {"chat": -1.0}, {"chat": "x"}, ["chat"]):
        try:
            ep.ChainLRUStrategy(sla=sla)
        except ValueError:
            continue
        check(False, f"非法 sla 未拒绝：{sla}")


def test_param_validation():
    for kwargs in (
        {"floor_tokens": -1},
        {"weight": -0.1},
        {"gate": "whatever"},
        {"min_gaps": 0},
    ):
        try:
            ep.ChainLRUStrategy(**kwargs)
        except ValueError:
            continue
        check(False, f"非法参数未拒绝：{kwargs}")


if __name__ == "__main__":
    test_floor_penalty_and_slope()
    test_rerode_gate()
    test_multiturn_gate()
    test_tail_first_tiebreak()
    test_sla_multiplier()
    test_param_validation_with_sla()
    test_param_validation()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        sys.exit(1)
    print("ALL PASSED")
