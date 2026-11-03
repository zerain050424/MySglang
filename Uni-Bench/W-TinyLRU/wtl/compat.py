"""框架缺陷的**离线/运行时兼容补丁**（不改框架源码，只在本进程内替换方法）。

每个补丁都满足三条：
1. 只在「确实命中已知缺陷」时安装（用源码文本特征判定，避免覆盖上游修复）；
2. 安装时打 WARNING，并把补丁名写进结果 JSON 的 ``compat`` 字段，便于归因；
3. 可用 ``WTL_COMPAT=0`` 或 ``--no-compat`` 关闭，便于对照「原始框架行为」。

当前只有一个补丁
----------------
``struct_promote_no_window_recursion``
    ``StructQueueState._promote()`` 在「候选仍属 window 区」时会递归调用
    ``_drain_window()``；当 probation/protected 均为空（没有可比较的 victim）
    时形成无界递归，实测在 150 个「一次性分叉前缀」的 trace 上直接
    ``RecursionError``。设计文档 §5.2 的语义是「promote(cand) → probation 尾」，
    即晋升必须离开观察窗。补丁按该语义实现，并去掉递归（外层 while 负责继续 drain）。
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import List

logger = logging.getLogger("wtl.compat")

APPLIED: List[str] = []


def _enabled() -> bool:
    return os.environ.get("WTL_COMPAT", "1") not in ("0", "false", "False")


def apply_all(enabled: bool = True) -> List[str]:
    """安装全部补丁，返回本次新安装的补丁名列表。"""
    APPLIED.clear()
    if not enabled or not _enabled():
        return []
    _fix_struct_promote()
    return list(APPLIED)


def install_stable_ties(strategy) -> bool:
    """把淘汰排序键的**平票**变成确定性的（仅离线回放使用）。

    为什么需要：``RadixCache.evict`` 取 ``min(leaves, key=...)``，而 ``leaves`` 是
    ``set``。仿真时钟下同一请求命中的节点时间戳完全相同，平票极多，
    ``min`` 的结果就取决于 set 的遍历顺序 —— 而 set 顺序依赖对象地址，
    同一配置两次进程内回放的淘汰序列会不同（G1 实测出现）。

    做法：给排序键末尾追加 ``(路径深度, 路径 token 哈希)``，构成全局唯一的稳定后缀。
    这只改变「平票时选谁」，不改变策略语义；在线服务不安装（真实时钟下平票很少，
    且我们不应该改变生产行为）。
    """
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

    for name in ("victim_key", "one_shot_key", "get_priority"):
        wrap(name)
    return bool(patched)


def _node_identity(node):
    """(绝对深度, 路径 token 滚动哈希)，记忆化在节点属性上。"""
    chain = []
    cur = node
    while cur is not None and getattr(cur, "_wtl_pd", None) is None:
        chain.append(cur)
        cur = getattr(cur, "parent", None)
    if cur is None:
        depth, ph = 0, 0xCBF29CE484222325
    else:
        depth, ph = cur._wtl_pd, cur._wtl_ph
    for nd in reversed(chain):
        key = getattr(nd, "key", None)
        toks = key.token_ids if key is not None else ()
        for t in toks:
            ph = ((ph ^ (int(t) + 0x9E3779B97F4A7C15)) * 0x100000001B3) & ((1 << 64) - 1)
        depth += len(toks)
        try:
            nd._wtl_pd, nd._wtl_ph = depth, ph
        except Exception:
            pass
    return depth, ph


def _fix_struct_promote() -> None:
    try:
        from sglang.srt.mem_cache.struct_queue_policy import (
            REGION_PROBATION,
            REGION_WINDOW,
            StructQueueState,
        )
    except Exception:  # 框架不可导入（如 fake 回放）：无需补丁
        return

    try:
        src = inspect.getsource(StructQueueState._promote)
    except (OSError, TypeError):
        return
    if "_drain_window" not in src:  # 已被上游修复
        return

    def _promote(self, node) -> None:
        # 晋升 = 离开观察窗（文档 §5.2：promote(cand) → probation 尾）。
        # 原实现在 band 仍为 window 时递归 drain，victim 为空时无限递归。
        band = self._band_of(node)
        if band == REGION_WINDOW:
            band = REGION_PROBATION
        self._index_remove(node)
        self._push(band, node, at_head=False)
        self.stats["admit"] += 1
        self._note_event()

    StructQueueState._promote = _promote
    APPLIED.append("struct_promote_no_window_recursion")
    logger.warning(
        "[compat] 已安装补丁 struct_promote_no_window_recursion："
        "框架 struct_queue_policy.StructQueueState._promote 会无限递归，"
        "本补丁按设计文档 §5.2 语义把「晋升」实现为离开 window。"
        "（建议框架侧修复；设置 WTL_COMPAT=0 可关闭）"
    )
