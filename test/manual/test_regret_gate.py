"""RegretGate：跨队列仲裁后悔弃权门（档 A）的纯逻辑单测（不需要 torch）。

被测对象：
* ``heter_cache/regret_gate.py`` 的 ``RegretGate``（R 计算、弃权判定、min-R 兜底）；
* ``heter_cache/selector.py`` 的 ``pick_victim``（弃权门接入后的全链路裁决）。

口径（mixexp/chain_shape_report.md §六 + RegretGateParams 文档）：
* R_q = 窗口内 join 回队列 q 的淘汰内容 token ÷ 同窗从 q 削掉的 token；
* 一个 stamp 只归因一次（join 即摘除）；
* 弃权 = R_q ≥ max(r_min, k × 已采样队列 R 中位数)，样本不足不弃权；
* 全弃权 → 已采样队列里 min R；无人采样 → None（调用方回退 VD 首队）。
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


def _stub_package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


def _real():
    try:
        import sglang.srt.mem_cache.evict_policy as ep
        import sglang.srt.mem_cache.heter_cache.utils as hu
        import sglang.srt.mem_cache.heter_cache.regret_gate as rg
        import sglang.srt.mem_cache.heter_cache.selector as qs
        import sglang.srt.mem_cache.heter_cache.override as ov
        return ep, hu, rg, qs, ov
    except Exception:
        return None


loaded = _real()
if loaded is None:
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
    _load("sglang.srt.mem_cache.heter_cache.node_share", os.path.join(_HC_DIR, "node_share.py"))
    _load("sglang.srt.mem_cache.heter_cache.override", os.path.join(_HC_DIR, "override.py"))
    _load("sglang.srt.mem_cache.heter_cache.bench_bridge", os.path.join(_HC_DIR, "bench_bridge.py"))
    rg = _load("sglang.srt.mem_cache.heter_cache.regret_gate", os.path.join(_HC_DIR, "regret_gate.py"))
    qs = _load("sglang.srt.mem_cache.heter_cache.selector", os.path.join(_HC_DIR, "selector.py"))
    ep = _load("sglang.srt.mem_cache.evict_policy", os.path.join(_CACHE_DIR, "evict_policy.py"))
    ov = _load("sglang.srt.mem_cache.heter_cache.override", os.path.join(_HC_DIR, "override.py"))
else:
    ep, hu, rg, qs, ov = loaded

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return cond


def _gate(now, **kw):
    params = hu.RegretGateParams(enabled=True, **kw)
    gate = rg.RegretGate(params)
    gate.use_clock(lambda: now[0])
    return gate


def test_r_and_window():
    """R = join 回的 span ÷ 削掉的 span；滑窗过期后老样本退出。"""
    print("[1] R = joined_span / evicted_span, sliding window")
    now = [1000.0]
    g = _gate(now, window_s=100.0, retain_s=1000.0, min_evicted=100)
    # chat: 削 3 次 × 1000 = 3000；join 回 2 × 1000 = 2000 → R = 2/3
    for i in range(3):
        g.on_evict(key=100 + i, queue="chat", span=1000)
    g.on_request(key=100)
    g.on_request(key=101)
    check(g.r("chat") is not None and abs(g.r("chat") - 2000 / 3000) < 1e-9,
          f"R = 2/3, got {g.r('chat')}")
    # agentic: 削 1000，无人回来 → R = 0
    g.on_evict(key=200, queue="agentic", span=1000)
    check(g.r("agentic") == 0.0, f"无人 join → R=0, got {g.r('agentic')}")
    # 一个 stamp 只归因一次：join 已摘除
    g.on_request(key=100)
    check(g._rg_sum["chat"] == 2000, f"重复 join 不再归因：{g._rg_sum}")
    # 滑窗：时间推过 100s，chat 窗口里的样本全部过期
    now[0] += 150.0
    check(g.sampled("chat") is False, "过期后不再采样")
    check(g.r("chat") is None, "过期后 R = None")


def test_abstain_rule():
    """弃权 = R_q ≥ max(r_min, k × 已采样中位数)；样本不足 / 单队采样不弃权。"""
    print("[2] abstain = max(r_min, k*median), sample floors")
    now = [1000.0]
    g = _gate(now, window_s=1_000_000.0, retain_s=2_000_000.0, min_evicted=100, r_min=0.5, k=1.0)
    # chat: R=0.8；agentic: R=0.1 → 中位数 0.45，阈值 max(0.5, 0.45)=0.5 → chat 弃权
    for i in range(8):
        g.on_evict(key=300 + i, queue="chat", span=1000)
    g.on_request(key=300); g.on_request(key=301); g.on_request(key=302)
    g.on_request(key=303); g.on_request(key=304); g.on_request(key=305)
    for i in range(10):
        g.on_evict(key=400 + i, queue="agentic", span=1000)
    g.on_request(key=400)
    order = ["agentic", "chat"]
    check(g.abstain("chat", order) is True, f"R=0.6+ 中位 0.05 → 弃权, got {g.r('chat')}")
    check(g.abstain("agentic", order) is False, "R=0.1 < r_min → 不弃权")
    check(g.abstains.get("chat") == 1, "弃权计数")
    # 未采样队列：R=None → 永不弃权
    check(g.abstain("struct", order) is False, "无样本不弃权")
    # 单队采样：没有对照 → 不弃权
    g2 = _gate(now, window_s=1_000_000.0, retain_s=2_000_000.0, min_evicted=100)
    g2.on_evict(key=500, queue="chat", span=1000)
    g2.on_request(key=500)
    check(g2.abstain("chat", ["chat"]) is False, "只有自己在采样 → 不弃权")


def test_cheapest_and_fallback():
    """全弃权 → 已采样队列里 min R；无人采样 → None（回退现状）。"""
    print("[3] cheapest = min R among sampled; None when nobody sampled")
    now = [1000.0]
    g = _gate(now, window_s=1_000_000.0, retain_s=2_000_000.0, min_evicted=100)
    # struct: R=0（无人 join）；agentic: R=0.5；chat: R=0.8
    for i in range(4):
        g.on_evict(key=600 + i, queue="struct", span=1000)
    for i in range(4):
        g.on_evict(key=700 + i, queue="agentic", span=1000)
    g.on_request(key=700); g.on_request(key=701)
    for i in range(4):
        g.on_evict(key=800 + i, queue="chat", span=1000)
    g.on_request(key=800)
    g.on_request(key=801); g.on_request(key=803)
    order = ["chat", "agentic", "struct"]
    check(g.cheapest(order) == "struct", f"min R = struct(0), got {g.cheapest(order)}")
    # 无人采样 → None
    g2 = _gate(now, window_s=1_000_000.0, retain_s=2_000_000.0, min_evicted=100)
    check(g2.cheapest(order) is None, "无人采样 → None（调用方回退 VD 首队）")


def test_overwrite_and_retention():
    """同键覆盖（重算重插后再被逐出）不产生双重归因；retain 过期后 join 不上。"""
    print("[4] stamp overwrite + retention")
    now = [1000.0]
    g = _gate(now, window_s=100.0, retain_s=1000.0, min_evicted=100)
    g.on_evict(key=900, queue="chat", span=1000)
    now[0] += 50.0
    g.on_evict(key=900, queue="agentic", span=1000)  # 覆盖：归因应到 agentic
    g.on_request(key=900)
    check(g._rg_sum.get("chat", 0) == 0, "覆盖后旧队列不归因")
    check(g._rg_sum.get("agentic", 0) == 1000, "归因到新队列")
    # retain 过期：stamp 消失，join 不上
    now[0] += 150.0
    g.on_evict(key=901, queue="chat", span=1000)
    now[0] += 1100.0
    g.on_request(key=901)
    check(g._rg_sum.get("chat", 0) == 0, "retain 过期后 join 不上")


class _FakeStrategy:
    """duck-typed 策略：只提供 pick_victim 用到的接口。"""

    def __init__(self, order, victims, gate=None):
        self._order = order
        self._victims = victims      # queue -> victim or None
        self.regret_gate = gate
        self.fallback_called = False

    def is_one_shot(self, node):
        return False

    def one_shot_key(self, node):
        return 0

    def queue_order(self):
        return list(self._order)

    def queue_victim(self, queue, leaves):
        return self._victims.get(queue)

    def fallback(self, leaves):
        self.fallback_called = True
        return "FB"


def test_pick_victim_gate_routing():
    """pick_victim 全链路：弃权跳队 → 全弃权走 min-R 跨队 → 无人采样走 VD。"""
    print("[5] pick_victim with regret gate")
    now = [1000.0]

    # 场景 1：chat 弃权（R 高），VD 序里 agentic 接住
    g = _gate(now, window_s=1_000_000.0, retain_s=2_000_000.0, min_evicted=100, r_min=0.5)
    for i in range(8):
        g.on_evict(key=1000 + i, queue="chat", span=1000)
    for i in range(6):
        g.on_request(key=1000 + i)
    for i in range(4):
        g.on_evict(key=1100 + i, queue="agentic", span=1000)
    st = _FakeStrategy(["chat", "agentic"], {"agentic": "V-AG"}, gate=g)
    victim, source = qs.pick_victim(strategy=st, leaves=["L1"])
    check((victim, source) == ("V-AG", "value:agentic"),
          f"chat 弃权 → agentic 接住: {victim}/{source}")

    # 场景 2：全弃权 → min-R 跨队裁决（struct R=0）
    g2 = _gate(now, window_s=1_000_000.0, retain_s=2_000_000.0, min_evicted=100, r_min=0.5)
    for i in range(8):
        g2.on_evict(key=1200 + i, queue="chat", span=1000)
    for i in range(6):
        g2.on_request(key=1200 + i)
    for i in range(8):
        g2.on_evict(key=1300 + i, queue="agentic", span=1000)
    for i in range(4):
        g2.on_request(key=1300 + i)
    for i in range(4):
        g2.on_evict(key=1400 + i, queue="struct", span=1000)   # 无人 join → R=0
    st2 = _FakeStrategy(["chat", "agentic", "struct"],
                        {"struct": "V-ST"}, gate=g2)
    victim2, source2 = qs.pick_victim(strategy=st2, leaves=["L1"])
    check((victim2, source2) == ("V-ST", "value:struct"),
          f"全弃权 → min R 队列: {victim2}/{source2}")

    # 场景 3：全弃权且无人采样 → 回退 VD 首队有候选者；池全空走 fallback
    g3 = _gate(now, window_s=1_000_000.0, retain_s=2_000_000.0, min_evicted=100000)
    st3 = _FakeStrategy(["chat", "agentic"], {}, gate=g3)
    victim3, source3 = qs.pick_victim(strategy=st3, leaves=["L1"])
    check((victim3, source3) == ("FB", "value:fallback"),
          f"无人采样且池全空 → fallback: {victim3}/{source3}")
    check(st3.fallback_called, "fallback 被调用")

    # 场景 4：无门（gate=None）→ 行为与现状一致（VD 首队优先）
    st4 = _FakeStrategy(["chat", "agentic"], {"chat": "V-CHAT"})
    victim4, source4 = qs.pick_victim(strategy=st4, leaves=["L1"])
    check((victim4, source4) == ("V-CHAT", "value:chat"),
          f"无门走现状: {victim4}/{source4}")


def test_param_validation():
    print("[6] param validation")
    for kwargs in (
        {"window_s": 0},
        {"k": 0},
        {"r_min": 1.5},
        {"min_evicted": -1},
        {"retain_s": 60.0, "window_s": 120.0},   # retain < window
        {"max_stamps": 0},
        {"context": 0},
        {"floor_tokens": -1},
        {"weight": -0.1},
        {"beta": -0.1},
    ):
        try:
            hu.RegretGateParams(**kwargs)
        except ValueError:
            continue
        check(False, f"非法参数未拒绝：{kwargs}")
    check(hu.RegretGateParams().enabled is False, "默认 enabled=False")
    check(hu.RegretGateParams().priced_fallback is False, "默认 priced_fallback=False")


class _LeafNode:
    """可弱引用的链节点替身（NodeShareLedger 是 WeakKeyDictionary）。"""

    def __init__(self, nid, tokens, lat, rc, parent, sid, tag=None):
        self.id = nid
        self.key = [0] * tokens
        self.last_access_time = lat
        self.reuse_count = rc
        self.parent = parent
        self.session_id = sid
        self.queue_tag = tag


def _leaf_chain(segments, base_lat, tag=None):
    """segments: [(tokens, reuse_count, session_id), ...] 从浅到深，返回叶子。"""
    parent = None
    for tokens, rc, sid in segments:
        base_lat += 1.0
        parent = _LeafNode(_FakeId.next(), tokens, base_lat, rc, parent, sid, tag)
    return parent


class _FakeId:
    next_id = 0

    @classmethod
    def next(cls):
        cls.next_id += 1
        return cls.next_id


def test_priced_fallback_beta_weighting():
    """档 B：R 归因加权——同形状的两个叶子，历史亏得多的队列更贵。"""
    print("[7] priced fallback: beta * R attribution weighting")
    now = [1000.0]
    g = _gate(now, window_s=1_000_000.0, retain_s=2_000_000.0, min_evicted=100)
    # chat R≈0.8，agentic R=0
    for i in range(10):
        g.on_evict(key=2000 + i, queue="chat", span=1000)
    for i in range(8):
        g.on_request(key=2000 + i)
    for i in range(10):
        g.on_evict(key=3000 + i, queue="agentic", span=1000)

    params = hu.RegretGateParams(enabled=True, priced_fallback=True,
                                 floor_tokens=4096, weight=0.02, beta=1.0)
    fb = ov.PricedFallback(params, g)

    def mk_leaf(sid, tag=None):
        # 两条同形状的单段链（remain=0，缺罚 = 0.02×4096），只差队列归属
        return _LeafNode(_FakeId.next(), 1000, 10.0, 1, None, sid, tag=tag)

    chat_leaf = mk_leaf("c1")
    agentic_leaf = mk_leaf("a1")
    chat_leaf.queue_tag = hu.HeterCacheNodeTag.QCHAT
    agentic_leaf.queue_tag = hu.HeterCacheNodeTag.QAGENT
    pick = fb.pick([chat_leaf, agentic_leaf])
    check(pick is agentic_leaf,
          f"R 高的队列叶子更贵 → 选便宜的 agentic 叶子: {pick is agentic_leaf}")

    # β=0：纯形状定价，同形状叶子平票 → 由 id 定序，两叶价格相等
    key_chat = ov.chain_leaf_key(
        chat_leaf, floor_tokens=4096, weight=0.02, registry=None,
        expired_of=None, weight_scale=1.0)
    key_agentic = ov.chain_leaf_key(
        agentic_leaf, floor_tokens=4096, weight=0.02, registry=None,
        expired_of=None, weight_scale=1.0)
    check(key_chat[0] == key_agentic[0] == 10.0 + 0.02 * 4096,
          f"β=0 同形状同价：{key_chat[0]} / {key_agentic[0]}")


def test_priced_fallback_routing():
    """pick_victim：全弃权 + priced_fallback 装配 → 走 priced 兑底并标 priced 来源。"""
    print("[8] pick_victim routes all-abstain to priced fallback")
    now = [1000.0]
    # k=0.5：阈值 = max(0.5, 0.5×中位数)，两队列 R≈0.9 都越过 → 全弃权
    g = _gate(now, window_s=1_000_000.0, retain_s=2_000_000.0,
              min_evicted=100, r_min=0.5, k=0.5,
              priced_fallback=True, floor_tokens=4096, weight=0.02, beta=1.0)
    # chat: R≈0.9 → 弃权；agentic: R≈0.9 → 弃权（全体弃权触发 priced 兑底）
    for i in range(10):
        g.on_evict(key=4000 + i, queue="chat", span=1000)
    for i in range(9):
        g.on_request(key=4000 + i)
    for i in range(10):
        g.on_evict(key=5000 + i, queue="agentic", span=1000)
    for i in range(8):
        g.on_request(key=5000 + i)   # agentic R=0.8，仍是 min-R

    params = hu.RegretGateParams(enabled=True, priced_fallback=True,
                                 floor_tokens=4096, weight=0.02, beta=1.0)
    fb = ov.PricedFallback(params, g)
    st = _FakeStrategy(["chat", "agentic"], {"chat": "V-CHAT", "agentic": "V-AG"},
                       gate=g)
    st.priced_fallback = fb
    st.node_tag = lambda n: hu.HeterCacheNodeTag.coerce(getattr(n, "queue_tag", None))
    st.session_registry = None

    cheap = _LeafNode(_FakeId.next(), 1000, 5.0, 0, None, None,
                      tag=hu.HeterCacheNodeTag.QAGENT)
    victim, source = qs.pick_victim(strategy=st, leaves=[cheap])
    check(source.startswith("priced:"), f"全弃权走 priced 兑底: {source}")
    check(victim is cheap, "priced 兑底选中的叶子")
    # min-R 档 A 路径保留为退路：priced 不装配时仍走 min-R
    st2 = _FakeStrategy(["chat", "agentic"], {"agentic": "V-AG"}, gate=g)
    st2.priced_fallback = None
    st2.node_tag = st.node_tag
    st2.session_registry = None
    victim2, source2 = qs.pick_victim(strategy=st2, leaves=[cheap])
    check(source2 == "value:agentic", f"无 priced 时回退 min-R 档 A: {source2}")


def _strategy(chain_cfg, now):
    return ep.HeterCacheStrategy(chain=chain_cfg, now=lambda: now[0])


def test_chain_leaf_key_shared_with_chain_evict():
    """口径单一来源：chain_leaf_key 与 ChainEvict.victim_key 同键。"""
    print("[9] chain_leaf_key == ChainEvict.victim_key")
    now = [1000.0]
    st = _strategy({"enabled": True, "floor_tokens": 4096, "weight": 0.02}, now)
    ce = st.chain_evicts.get("chat")
    if ce is None:
        print("  SKIP  chain 未装配，跳过同键断言")
        return
    leaf = _leaf_chain([(2000, 1, "s1"), (2000, 0, "s1")], 0.0,
                       tag=hu.HeterCacheNodeTag.QCHAT)
    hu.SessionRegistry.reset_global()
    reg = hu.SessionRegistry.global_instance()
    reg._sessions["s1"] = hu.SessionState(session_id="s1", gaps=[30.0])
    ce.bind_registry(reg)
    leaf.session_id = "s1"
    leaf.parent.session_id = "s1"
    k1 = ce.victim_key(leaf)
    k2 = ov.chain_leaf_key(
        leaf, floor_tokens=ce.params.floor_tokens, weight=ce.params.weight,
        tail_first=ce.params.tail_first, registry=reg,
        min_gaps=ce.params.min_gaps,
        expired_of=st.session_expired if ce.params.expire_anchors else None,
    )
    check(k1 == k2, f"同键：{k1} != {k2}")
    hu.SessionRegistry.reset_global()


if __name__ == "__main__":
    test_r_and_window()
    test_abstain_rule()
    test_cheapest_and_fallback()
    test_overwrite_and_retention()
    test_pick_victim_gate_routing()
    test_param_validation()
    test_priced_fallback_beta_weighting()
    test_priced_fallback_routing()
    test_chain_leaf_key_shared_with_chain_evict()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        sys.exit(1)
    print("ALL PASSED")
