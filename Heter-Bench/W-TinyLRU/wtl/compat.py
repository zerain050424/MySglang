"""离线回放的确定性补丁（**不改框架源码**，只在回放进程内替换排序键）。

为什么需要
----------
``RadixCache.evict`` 用 ``min(leaves, key=...)`` 在 ``set`` 上选候选，而 ``leaves`` 是
``evictable_leaves`` 这个集合。仿真时钟下同一请求命中的节点时间戳完全相同，平票极多，
``min`` 的结果就取决于 set 的遍历顺序 —— 而 set 顺序依赖对象地址，于是**同一配置两次
进程内回放的淘汰序列会不同**（门控 G1 实测出现）。

做法
----
给排序键末尾追加 ``(路径深度, 路径 token 滚动哈希)``，构成全局唯一的稳定后缀。
这只改变「平票时选谁」，不改变策略语义；**在线服务不安装**（真实时钟下平票很少，
且不应该改变生产行为）。

可用 ``WTL_COMPAT=0`` 或 ``--unstable-ties`` 关闭，便于对照原始行为。
"""

from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger("wtl.compat")

APPLIED: List[str] = []

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_TOKEN_SALT = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1


def _enabled() -> bool:
    return os.environ.get("WTL_COMPAT", "1") not in ("0", "false", "False")


def apply_all(enabled: bool = True) -> List[str]:
    """安装进程级补丁，返回本次新安装的补丁名列表。

    目前没有需要进程级安装的补丁（确定性平票由回放器按需调用
    ``install_stable_ties``）；保留入口是为了以后有兼容修复时有统一位置。
    """
    APPLIED.clear()
    if not enabled or not _enabled():
        return []
    return list(APPLIED)


def install_stable_ties(strategy) -> bool:
    """把淘汰排序键的**平票**变成确定性的（仅离线回放使用）。"""
    patched = []

    def wrap(name):
        if not hasattr(strategy, name):
            return
        orig = getattr(strategy, name)

        def key_fn(node, _orig=orig):
            base = _orig(node)
            depth, ph = _node_identity(node)
            if isinstance(base, tuple):
                return base + (depth, ph)
            return (base, depth, ph)

        setattr(strategy, name, key_fn)
        patched.append(name)

    # 只剩 one_shot_key：victim_key 与 HeterCache 的 get_priority 已随
    # 全局复用分一起删除，没有稳定排序键可打补丁。
    for name in ("one_shot_key",):
        wrap(name)
    if patched:
        logger.debug("[compat] stable ties installed on %s", patched)
    return bool(patched)


def _node_identity(node):
    """(绝对深度, 路径 token 滚动哈希)，记忆化在节点属性上。"""
    chain = []
    cur = node
    while cur is not None and getattr(cur, "_wtl_pd", None) is None:
        chain.append(cur)
        cur = getattr(cur, "parent", None)
    if cur is None:
        depth, ph = 0, _FNV_OFFSET
    else:
        depth, ph = cur._wtl_pd, cur._wtl_ph
    for nd in reversed(chain):
        key = getattr(nd, "key", None)
        toks = key.token_ids if key is not None else ()
        for t in toks:
            ph = ((ph ^ (int(t) + _TOKEN_SALT)) * _FNV_PRIME) & _MASK
        depth += len(toks)
        try:
            nd._wtl_pd, nd._wtl_ph = depth, ph
        except Exception:
            pass
    return depth, ph
