"""把 sglang 的 radix 缓存接到模拟器的逻辑时钟上。

背景：模拟器 OFFline 模式**不推进真墙钟**（`scheduler.py:464-470` 只把预测延迟
记进 stats），而 sglang 缓存的秒级判据全读 `time.monotonic()`。不接这根线，
HeterCache 的老化 TTL（120/180/300s）、`ONE_SHOT_IDLE_S=90`、复用窗口 90s、承压
校准窗口 3s 全部永不触发——策略还在跑，但时间维度是死的。

接线口：`CacheInitParams.now`（sglang 侧已支持：节点时间戳由 `RadixCache._stamp`
盖，再经 `HeterCacheStrategy.use_clock` 推给占用账本 / 遥测 / 会话账本 / 队内改写器）。

为什么注入点是 `RadixCache.__init__` 而不是 `build_kv_cache`：
`build_kv_cache` 内部就地构造 `params` 并**立刻** `create_tree_cache(...)`，
函数返回时缓存已经建好，事后改 `params.now` 已经晚了。而 `RadixCache.__init__`
是 `params` 与 `self._now` 之间唯一的缝隙；`HeterCacheRadixCache` /
`PureSWARadixCache` 都不定义自己的 `__init__`，继承这一个，所以一次覆盖整个家族。

覆盖范围（写清楚）：
  * 覆盖：`RadixCache` 及其子类（classic / heter_cache / pure_swa）；
  * 不覆盖：`UnifiedRadixCache`（独立实现，内部读的不是 `_now`；需要另行接线）。
"""

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager.state import StateManager
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


class C_RadixCacheClockHook(BaseHook):
    """在缓存构造前把 `CacheInitParams.now` 指向模拟器的全局逻辑时钟。"""

    HOOK_CLASS_NAME = "RadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.radix_cache"
    # UnifiedRadixCache 路径不会创建这个类，所以不能设为必需（否则启动即失败）。
    REQUIRED = False

    @classmethod
    def hook(cls, target) -> None:
        original_init = target.__init__
        if getattr(original_init, "_sim_clock_injected", False):
            return

        def wrapped_init(self, params, *args, **kwargs):
            # 显式传入的 now（单测 / 离线仿真）优先，不覆盖。
            if getattr(params, "now", None) is None:
                params.now = StateManager.get_global_clock
                logger.info(
                    "[cache-clock] RadixCache.now -> StateManager.get_global_clock()"
                )
            return original_init(self, params, *args, **kwargs)

        # 幂等标记：spawn 子进程可能重复安装 hook
        wrapped_init._sim_clock_injected = True
        target.__init__ = wrapped_init
