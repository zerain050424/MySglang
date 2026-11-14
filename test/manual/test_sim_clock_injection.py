"""虚拟时钟注入验收：sglang 的 radix 缓存改读模拟器的逻辑时钟。

上线路径：模拟器在 scheduler 进程里安装 `C_RadixCacheClockHook`，它在
`RadixCache.__init__` 里把 `CacheInitParams.now` 指到 `StateManager.get_global_clock`。
本测试不用起 serving 栈（本机 transformers 冲突），而是直接验证：

1. hook 的**匹配**逻辑（真的按模块名 + 类名命中，不是盲调）；
2. 命中后用**真实** `HeterCacheRadixCache` 构造，`now` 被注入且传播到
   占用账本 / 遥测 / 会话账本；
3. 节点时间戳与老化 TTL 真的跟模拟时钟走（步进 179.9 / 180.0 翻转）；
4. 显式传入的 `now`（离线仿真 / 单测）不被覆盖。

    PYTHONPATH=$PWD/python python -m pytest -q test/manual/test_sim_clock_injection.py
"""

import os
import sys
from array import array
from types import SimpleNamespace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIM_SRC = os.path.join(ROOT, "tools", "sglang-simulator", "src")
for path in (os.path.join(ROOT, "python"), SIM_SRC, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

def _install_optional_stubs() -> None:
    """AIConfigurator 是可选依赖，但 simulation/utils.py:5 在模块级 import 了它。

    本机没装（`pip install -e "tools/sglang-simulator[aic]"` 才有），所以按
    sim/bootstrap.py 同一套做法装「真模块优先、缺失才替身」的惰性桩；真装了就走真模块。
    """

    import importlib.util
    import types

    if importlib.util.find_spec("aiconfigurator") is not None:
        return

    def _stub_attr(name):
        # dunder 必须照常抛 AttributeError：inspect.getfile / dataclasses 会读
        # __file__、__all__ 之类，返回替身会让 os.path 直接炸。
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Any

    class _AnyMeta(type):
        """任何属性都是它自己；因为类体里有 `X | None` 注解，必须是**类**。"""

        def __getattr__(cls, name):
            return _stub_attr(name)

        def __call__(cls, *args, **kwargs):
            return super().__call__()

    class _Any(metaclass=_AnyMeta):
        pass

    for name in (
        "aiconfigurator",
        "aiconfigurator.sdk",
        "aiconfigurator.sdk.models",
        "aiconfigurator.sdk.backends",
        "aiconfigurator.sdk.backends.factory",
        "aiconfigurator.sdk.common",
        "aiconfigurator.sdk.config",
        "aiconfigurator.sdk.inference_session",
        "aiconfigurator.sdk.perf_database",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        module.__getattr__ = _stub_attr
        sys.modules[name] = module


_install_optional_stubs()

from sglang_simulator.hook import install_class_hooks, remove_class_hooks  # noqa: E402
from sglang_simulator.hook.class_hook_entry import (  # noqa: E402
    is_class_hook_matched,
)
from sglang_simulator.simulation.manager.state import StateManager  # noqa: E402
from sglang_simulator.hook.class_hook_entry import CLASS_HOOKS  # noqa: E402
from sglang_simulator.simulation.sglang import hook_bootstrap  # noqa: E402
from sglang_simulator.simulation.sglang.radix_cache import (  # noqa: E402
    C_RadixCacheClockHook,
)

WEIGHTS = {"struct": 0.2, "agentic": 0.5, "chat": 0.3}


@pytest.fixture(autouse=True)
def _clean_hooks():
    yield
    remove_class_hooks()
    # install_simulator_hooks() 有一次性开关；不复位会让后续用例静默跳过安装
    hook_bootstrap._HOOKS_INSTALLED = False


def _build(sym, *, now=None):
    from sim.allocator import TokenAllocator

    return sym["HeterCacheRadixCache"].create_simulated(
        mock_allocator=TokenAllocator(4096),
        page_size=1,
        eviction_policy="heter_cache",
        eviction_policy_config={"weight": {"weights": dict(WEIGHTS)}},
        now=now,
    )


def _insert(sym, cache, tokens, session_id="ShareGPT-a"):
    torch = __import__("torch")
    return cache.insert(
        sym["InsertParams"](
            key=sym["RadixKey"](array("q", tokens)),
            value=torch.arange(len(tokens), dtype=torch.int64),
            decode_start=len(tokens),
            session_id=session_id,
        )
    ).last_device_node


def test_hook_bootstrap_registers_the_clock_hook():
    """注入必须真的挂在启动路径上，而不只是定义了一个类。"""

    hook_bootstrap.install_simulator_hooks()
    assert any(h is C_RadixCacheClockHook for h in CLASS_HOOKS)


def test_hook_matches_module_and_class_name():
    """真的走匹配器：只有 sglang.srt.mem_cache.radix_cache.RadixCache 会被挂上。"""

    install_class_hooks([C_RadixCacheClockHook])
    assert not is_class_hook_matched(C_RadixCacheClockHook)

    namespace = {"__name__": "sglang.srt.mem_cache.radix_cache"}
    exec(
        "class RadixCache:\n"
        "    def __init__(self, params):\n"
        "        self.params = params\n",
        namespace,
    )
    assert is_class_hook_matched(C_RadixCacheClockHook)

    StateManager.reset()
    StateManager.step_global_clock(12.5)
    cache = namespace["RadixCache"](SimpleNamespace(now=None))
    # 注入的是可调用的时钟，取到当前模拟时刻
    assert cache.params.now() == pytest.approx(12.5)


def test_wrong_module_is_not_hooked():
    install_class_hooks([C_RadixCacheClockHook])
    namespace = {"__name__": "somewhere.else"}
    exec(
        "class RadixCache:\n"
        "    def __init__(self, params):\n"
        "        self.params = params\n",
        namespace,
    )
    assert not is_class_hook_matched(C_RadixCacheClockHook)
    cache = namespace["RadixCache"](SimpleNamespace(now=None))
    assert cache.params.now is None


def test_real_heter_cache_follows_the_simulator_clock():
    from sim import bootstrap

    sym = bootstrap.load()
    # 类已经 import 完了，匹配器不会回溯；直接对真实类执行同一段 hook
    C_RadixCacheClockHook.hook(sym["RadixCache"])
    StateManager.reset()

    cache = _build(sym)  # 不传 now -> 由 hook 注入

    # 1) 缓存层的时钟 == 模拟器逻辑时钟
    assert cache._now() == pytest.approx(StateManager.get_global_clock())
    # 2) 经 use_clock 传播到三个协作者（会话账本是 is_one_shot 的差值来源）
    assert cache.occupancy._now() == pytest.approx(StateManager.get_global_clock())
    assert cache.telemetry._now() == pytest.approx(StateManager.get_global_clock())
    assert cache.session_registry._now() == pytest.approx(
        StateManager.get_global_clock()
    )

    # 3) 节点时间戳跟模拟时钟（不是墙钟）
    leaf = _insert(sym, cache, list(range(0, 64)))
    assert leaf.last_access_time == pytest.approx(StateManager.get_global_clock())
    assert leaf.last_access_time < 1e6, "看起来是墙钟（monotonic 量级）"

    # 4) 老化 TTL 真的按模拟秒翻转
    leaf.queue_tag = sym["HeterCacheNodeTag"].QCHAT
    ttl = cache.eviction_strategy.slru.ageing_ttl_s["chat"]
    StateManager.step_global_clock(ttl - 0.1)
    assert cache.eviction_strategy.is_aged(leaf) is False
    StateManager.step_global_clock(0.1)
    assert cache.eviction_strategy.is_aged(leaf) is True


def test_explicit_clock_is_not_overridden():
    from sim import bootstrap
    from sim.clock import VirtualClock

    sym = bootstrap.load()
    C_RadixCacheClockHook.hook(sym["RadixCache"])
    StateManager.reset()
    StateManager.step_global_clock(1000.0)

    mine = VirtualClock(0.0)
    cache = _build(sym, now=mine)
    assert cache._now is mine
    leaf = _insert(sym, cache, list(range(0, 32)))
    assert leaf.last_access_time == 0.0


def test_hook_is_idempotent():
    from sim import bootstrap

    sym = bootstrap.load()
    C_RadixCacheClockHook.hook(sym["RadixCache"])
    first = sym["RadixCache"].__init__
    C_RadixCacheClockHook.hook(sym["RadixCache"])
    assert sym["RadixCache"].__init__ is first
