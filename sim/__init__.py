"""HeterCache 离线仿真器（CPU）。

只做装配与驱动：前缀缓存、队列标签、权重仲裁、队内改写器全部 import 生产代码，
不在本目录留第二份实现。入口见 ``sim/__main__.py``，里程碑见 ``sim/README.md``。
"""

from sim.bootstrap import load

__all__ = ["load"]
