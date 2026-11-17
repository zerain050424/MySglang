"""CPU 仿真用的 ``sgl_kernel`` 替身（不是真实现）。

为什么需要它
------------
真正的 ``sgl_kernel``（PyPI 包 ``sglang-kernel``）在 CPU-only 环境里整包 import
不成功：``__init__.py`` 会顺序 import 一批 CUDA 扩展模块，缺 ``libnvrtc.so.13``
就抛 ``ImportError``。而 ``sglang.srt.mem_cache.memory_pool`` 在模块顶层通过
``sglang.kernels.ops.kvcache.cache_move`` 间接 import 了 ``sgl_kernel``，
于是整条 ``build_kv_cache`` 装配链在 CPU 上直接断掉，模拟器起不来。

上游模拟器的官方路径是跑在配套的 SGLang 镜像里（有 CUDA），所以它只 stub
``sgl_kernel.load_utils`` 防止预加载 CUDA 动态库，没有 stub 整个包。

边界（重要）
------------
本替身**只保证 import 能过**。任何一个符号被真正调用都会立刻抛
``RuntimeError``，不会静默返回 None 或者假值——否则「跑通了」和「根本没算」
在外观上没有区别，排查代价极高。

实际影响范围（已核对 ``python/sglang/srt`` 全树）：

* 模块级 ``from sgl_kernel import ...`` 只出现在 flashmla / xpu / npu /
  minicpm 这些 attention backend 与硬件后端里。模拟器固定用
  ``torch_native``，不会走这些模块；
* ``copy_all_layer_kv_cache_cpu``（``kvcache/cache_move.py`` 的 CPU 分支）
  是唯一的 CPU 侧真实调用点。它用裸地址搬运 KV 行（``data_ptrs`` 是 uint64
  地址张量），没法用 torch 等价实现，所以只能占位。CPU 仿真不触发
  HiCache 多池搬运，正常路径不会调到它；真调到了请改回 GPU 镜像跑。

用法：把本目录放到 ``PYTHONPATH`` 最前面，遮蔽 site-packages 里的真包。
"""

from __future__ import annotations

_UNAVAILABLE_HINT = (
    "sgl_kernel.{name} 在 CPU 仿真环境下不可用（替身模块只保证 import 通过）。"
    "这个符号属于 sglang 的 AOT kernel，需要配套的 CUDA 运行库；"
    "本地 CPU 仿真不走到这里。如果确实被调用了，说明所选路径超出了 "
    "tools/sglang-simulator/cpu_shim 的覆盖范围，"
    "请改用官方 SGLang 镜像（见 tools/sglang-simulator/README.md）。"
)


class _Unavailable:
    """任何调用都失败的占位符。属性访问返回自身，便于 ``mod.x.y(...)`` 形式。"""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *_args, **_kwargs):
        raise RuntimeError(_UNAVAILABLE_HINT.format(name=self._name))

    def __getattr__(self, item: str):
        return _Unavailable(f"{self._name}.{item}")

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<unavailable sgl_kernel.{self._name}>"


def __getattr__(name: str):
    """``from sgl_kernel import whatever`` 一律拿到占位符，而不是 ImportError。

    这里故意不抛 ImportError：真正需要的符号缺失会在 import 期暴露成
    ``ImportError``，而上游代码大量写法是 ``try: from sgl_kernel import X /
    except ImportError: <降级>``。抛 ImportError 会让这些降级分支悄悄生效，
    把「替身覆盖不全」伪装成「本来就该降级」。
    """
    return _Unavailable(name)
