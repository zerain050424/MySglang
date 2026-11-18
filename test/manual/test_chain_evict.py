"""ChainEvict：chat/agentic 队内链形保护的纯逻辑单测（不需要 GPU / torch）。

被测对象：`python/sglang/srt/mem_cache/heter_cache/override.py` 的 ``ChainEvict``
（经 ``HeterCacheStrategy.queue_victim`` 全链路驱动）。

三条实测依据（mixexp/chain_shape_report.md）：

* 自适应锚：护到「链上最深重读节点」的深度——admission match 读连续前缀，
  重读集合就是链前缀，最深重读节点深度 = 该会话上次被命中的前缀长度；
* 证据门 = 重读过 ∪ 会话回来过（gaps ≥ min_gaps），无证据不护（不然一次性
  流量靠锚吃容量，四 cell 实测 gate=none 全线跌）；
* 锚出口 = 路径上所有会话终结才释放（``session_expired`` 口径：共享段不终结、
  证据不足不终结），防死会话占着保护。
"""

import importlib.util
import os
import sys
import time
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


def _stub_package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


def _real_heter():
    """环境完整时直接用真包，不污染 sys.modules（与 test_chain_lru 同约定）。"""
    try:
        import sglang.srt.mem_cache.evict_policy as ep
        import sglang.srt.mem_cache.heter_cache.utils as hu
        return ep, hu
    except Exception:
        return None, None


ep, hu = _real_heter()
if ep is None:
    for _pkg in (
        "sglang",
        "sglang.srt",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.heter_cache",
    ):
        _stub_package(_pkg)

    hu = _load("sglang.srt.mem_cache.heter_cache.utils", os.path.join(_HC_DIR, "utils.py"))
    _load("sglang.srt.mem_cache.heter_cache.occupancy", os.path.join(_HC_DIR, "occupancy.py"))
    _load("sglang.srt.mem_cache.heter_cache.value", os.path.join(_HC_DIR, "value.py"))
    _load("sglang.srt.mem_cache.heter_cache.selector", os.path.join(_HC_DIR, "selector.py"))
    _load("sglang.srt.mem_cache.heter_cache.node_share", os.path.join(_HC_DIR, "node_share.py"))
    _load("sglang.srt.mem_cache.heter_cache.override", os.path.join(_HC_DIR, "override.py"))
    _load("sglang.srt.mem_cache.heter_cache.bench_bridge", os.path.join(_HC_DIR, "bench_bridge.py"))
    ep = _load("sglang.srt.mem_cache.evict_policy", os.path.join(_CACHE_DIR, "evict_policy.py"))

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


class _Node:
    """链节点替身：parent 链 + get_priority / session_expired 用到的字段。

    必须可弱引用（NodeShareLedger 是 WeakKeyDictionary）。
    """

    _next_id = 0

    def __init__(self, tokens, lat, reuse_count=0, parent=None,
                 session_id=None, tag=None):
        _Node._next_id += 1
        self.id = _Node._next_id
        self.key = [0] * tokens
        self.last_access_time = lat
        self.reuse_count = reuse_count
        self.parent = parent
        self.session_id = session_id
        self.queue_tag = tag


def _chain(segments, base_lat):
    """segments: [(tokens, reuse_count, session_id), ...] 从浅到深挂单子链。

    顶端先挂空 root（与真树同构：root 不进 remain / 证据扫描）。
    返回 (叶子, 全部节点含 root)。
    """
    root = _Node(0, base_lat)
    parent = root
    nodes = [root]
    for tokens, rc, sid in segments:
        base_lat += 1.0
        parent = _Node(tokens, base_lat, reuse_count=rc, parent=parent, session_id=sid)
        nodes.append(parent)
    return parent, nodes


QCHAT = hu.HeterCacheNodeTag.QCHAT
QAGENT = hu.HeterCacheNodeTag.QAGENT


def _registry_now(now):
    """建一个可控时钟的会话账本单例，返回 (registry, set_now)。"""
    hu.SessionRegistry.reset_global()
    reg = hu.SessionRegistry.global_instance()
    reg.use_clock(lambda: now[0])
    return reg


def _strategy(chain_cfg, now):
    st = ep.HeterCacheStrategy(chain=chain_cfg, now=lambda: now[0])
    return st


def _tag_chat(node):
    node.queue_tag = QCHAT
    return node


def test_adaptive_anchor():
    """锚 = max(floor, 最深重读深度)；缺口 = 锚 − remain，剥到重读边界以下才罚。"""
    print("[1] adaptive anchor at deepest re-read node")
    now = [1000.0]
    reg = _registry_now(now)
    st = _strategy({"enabled": True, "floor_tokens": 0, "weight": 0.02}, now)
    # 三段链：浅两段被重读（rc=1），深一段是新写入（rc=0）。会话 warm-1 已回来过。
    leaf, nodes = _chain([(2000, 1, "s1"), (2000, 1, "s1"), (2000, 0, "s1")], 0.0)
    for n in nodes:
        _tag_chat(n)
    reg = _registry_now(now)
    reg._sessions["s1"] = hu.SessionState(session_id="s1", gaps=[30.0])
    st.chain_evicts["chat"].bind_registry(reg)

    key = st.chain_evicts["chat"].victim_key(leaf)
    # remain = 4000，最深重读深度 = 4000 → 锚 = 4000 → 缺口 0 → 无罚
    check(key[0] == leaf.last_access_time, f"remain ≥ 重读边界无罚：{key}")
    check(key[1] == -4000.0, f"砍链尾平票键 = -remain：{key}")

    # 剥到重读段本身成为叶子：remain=0 < 锚 2000 → 罚 0.02×2000
    # （这条链被继续剥时，罚在重读段上出现——正是「别剥穿锚」的刹车点）
    leaf2, nodes2 = _chain([(2000, 1, "s1")], 0.0)
    for n in nodes2:
        _tag_chat(n)
    key2 = st.chain_evicts["chat"].victim_key(leaf2)
    expect = leaf2.last_access_time + 0.02 * 2000
    check(abs(key2[0] - expect) < 1e-9, f"罚 = weight × (锚 − remain)：{key2[0]} != {expect}")

    # floor 先验抬锚：floor=6000 > 重读边界 2000 → 锚取 6000
    st2 = _strategy({"enabled": True, "floor_tokens": 6000, "weight": 0.02}, now)
    st2.chain_evicts["chat"].bind_registry(reg)
    key3 = st2.chain_evicts["chat"].victim_key(leaf2)
    expect3 = leaf2.last_access_time + 0.02 * 6000
    check(abs(key3[0] - expect3) < 1e-9, f"floor 先验抬锚：{key3[0]} != {expect3}")


def test_evidence_gate():
    """证据门 = 重读过 ∪ 会话回来过；无证据不护。"""
    print("[2] evidence gate: rerode ∪ multiturn")
    now = [1000.0]
    reg = _registry_now(now)
    st = _strategy({"enabled": True, "floor_tokens": 4096, "weight": 0.02}, now)
    # s-new：会话回来过一次（gaps=1），但链没被重读过（rc 全 0）
    reg._sessions["s-new"] = hu.SessionState(session_id="s-new", gaps=[45.0])
    # s-dead：单轮会话（gaps=0）
    reg._sessions["s-dead"] = hu.SessionState(session_id="s-dead", gaps=[])
    st.chain_evicts["chat"].bind_registry(reg)

    leaf_new, nodes_new = _chain([(2000, 0, "s-new"), (2000, 0, "s-new")], 0.0)
    for n in nodes_new:
        _tag_chat(n)
    key_new = st.chain_evicts["chat"].victim_key(leaf_new)
    check(key_new[0] > leaf_new.last_access_time, f"会话回来过 → 护：{key_new}")

    leaf_dead, nodes_dead = _chain([(2000, 0, "s-dead"), (2000, 0, "s-dead")], 0.0)
    for n in nodes_dead:
        _tag_chat(n)
    key_dead = st.chain_evicts["chat"].victim_key(leaf_dead)
    check(
        key_dead[0] == leaf_dead.last_access_time,
        f"单轮会话且无重读 → 不护：{key_dead}",
    )

    # 重读证据不依赖会话账本（gaps=0 但 rc>0）
    leaf_rc, nodes_rc = _chain([(2000, 1, "s-dead"), (2000, 0, "s-dead")], 0.0)
    for n in nodes_rc:
        _tag_chat(n)
    key_rc = st.chain_evicts["chat"].victim_key(leaf_rc)
    check(key_rc[0] > leaf_rc.last_access_time, f"重读过 → 护（与 gaps 无关）：{key_rc}")


def test_expiry_release():
    """锚出口：路径上所有会话终结 → 释放；共享段活着 → 保守保留。"""
    print("[3] expiry release: all-dead releases, shared survives")
    now = [1000.0]
    reg = _registry_now(now)
    st = _strategy(
        {"enabled": True, "floor_tokens": 4096, "weight": 0.02, "expire_anchors": True},
        now,
    )
    # s-old：回来了两次（gaps=2），寿命 = k×p90 = 3×45=135s；last_active=100 → 1000 时刻已终结
    reg._sessions["s-old"] = hu.SessionState(
        session_id="s-old", gaps=[45.0, 45.0], last_active=100.0
    )
    st.chain_evicts["chat"].bind_registry(reg)

    leaf, nodes = _chain([(2000, 1, "s-old"), (2000, 1, "s-old")], 0.0)
    for n in nodes:
        _tag_chat(n)
    key = st.chain_evicts["chat"].victim_key(leaf)
    check(
        key[0] == leaf.last_access_time,
        f"会话已终结 → 锚释放（不护）：{key}",
    )

    # 会话还活着（last_active 刚刚）→ 继续护
    reg._sessions["s-alive"] = hu.SessionState(
        session_id="s-alive", gaps=[45.0, 45.0], last_active=999.0
    )
    leaf2, nodes2 = _chain([(2000, 1, "s-alive"), (2000, 1, "s-alive")], 0.0)
    for n in nodes2:
        _tag_chat(n)
    key2 = st.chain_evicts["chat"].victim_key(leaf2)
    check(key2[0] > leaf2.last_access_time, f"会话存活 → 继续护：{key2}")

    # expire_anchors=False：终结的会话也不释放（关闭出口的对照）
    st3 = _strategy(
        {"enabled": True, "floor_tokens": 4096, "weight": 0.02, "expire_anchors": False},
        now,
    )
    st3.chain_evicts["chat"].bind_registry(reg)
    key3 = st3.chain_evicts["chat"].victim_key(leaf)
    check(key3[0] > leaf.last_access_time, f"出口关闭 → 终结会话仍被护：{key3}")


def test_shared_segment_not_released():
    """session_expired 口径：跨会话共享段永不「终结」，锚保守保留。"""
    print("[4] shared segment survives expiry")
    now = [1000.0]
    reg = _registry_now(now)
    st = _strategy({"enabled": True, "floor_tokens": 4096, "weight": 0.02}, now)
    reg._sessions["s-old"] = hu.SessionState(
        session_id="s-old", gaps=[45.0, 45.0], last_active=100.0
    )
    st.chain_evicts["chat"].bind_registry(reg)
    # 把共享段上游节点标成「已共享 ≥ share_min」：账本里记两个不同会话
    leaf, nodes = _chain([(2000, 1, "s-old"), (2000, 1, "s-old")], 0.0)
    for n in nodes:
        _tag_chat(n)
    share = st.node_share
    for n in nodes:
        share.on_hit(n, "other-a")
        share.on_hit(n, "other-b")
    key = st.chain_evicts["chat"].victim_key(leaf)
    check(key[0] > leaf.last_access_time, f"共享段不终结 → 锚保留：{key}")


def test_queue_dispatch_and_abstain():
    """queue_victim 全链路：只接管 chat/agentic；struct 不受影响；空池弃权。"""
    print("[5] dispatch via queue_victim, struct untouched, abstain on empty pool")
    now = [1000.0]
    reg = _registry_now(now)
    st = _strategy({"enabled": True, "floor_tokens": 4096, "weight": 0.02}, now)
    reg._sessions["s1"] = hu.SessionState(session_id="s1", gaps=[30.0])
    st.chain_evicts["chat"].bind_registry(reg)

    leaf, nodes = _chain([(2000, 1, "s1"), (2000, 1, "s1")], 0.0)
    for n in nodes:
        _tag_chat(n)
    victim = st.queue_victim("chat", [leaf])
    check(victim is leaf, "chat 队内由 ChainEvict 给出候选")

    # struct 队列不归 ChainEvict 管（chain_evicts 只有 chat/agentic）
    leaf_s, nodes_s = _chain([(2000, 1, "s1")], 0.0)
    for n in nodes_s:
        n.queue_tag = hu.HeterCacheNodeTag.QSTRUCT
    # struct 无共享证据、未老化 → struct_tier 仍可用（原路径不受影响）
    victim_s = st.queue_victim("struct", [leaf_s])
    check(victim_s is leaf_s, "struct 仍走 struct_tier")

    # 空池：弃权返回 None（VD 仲裁会落到下一个队列）
    check(st.queue_victim("agentic", []) is None, "空池弃权")
    check(st.queue_victim("chat", []) is None, "空池弃权（chat）")

    # 默认不装配：chain 未开时 chat 仍走 ReuseScoreEvict 路径（对象类型）
    st_off = _strategy(None, now)
    check(not st_off.chain_evicts, "默认不装配 chain_evicts")
    check("chat" in st_off.reuse_evicts, "默认走 ReuseScoreEvict")


def test_param_validation():
    print("[6] param validation")
    for kwargs in (
        {"floor_tokens": -1},
        {"weight": -0.1},
        {"min_gaps": 0},
    ):
        try:
            hu.ChainParams(**kwargs)
        except ValueError:
            continue
        check(False, f"非法参数未拒绝：{kwargs}")
    # 默认关闭
    check(hu.ChainParams().enabled is False, "ChainParams 默认 enabled=False")


if __name__ == "__main__":
    test_adaptive_anchor()
    test_evidence_gate()
    test_expiry_release()
    test_shared_segment_not_released()
    test_queue_dispatch_and_abstain()
    test_param_validation()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        sys.exit(1)
    print("ALL PASSED")
