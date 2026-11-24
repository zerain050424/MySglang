"""HeterCache 的共享数据模型：任务/会话分类、参数组、会话账本。

三个消费者都依赖这里，所以口径必须单点定义，避免"策略算一套、指标算一套"：

* ``evict_policy``：跨队列仲裁与 one-shot 判定；
* ``override``：改写器的作用域、共享数阈值、会话类型，以及 chat/agentic
  队内共用的复用分（lognormal 条件生存概率，``ReuseScoreEvict``）；
* ``selector`` / profile：队列标签、占用口径与已兑现价值密度（VD）。

命名约定：``HeterCacheNodeTag`` 是**节点级**标签（KV 片段出生时盖章），
``SessionType`` 是**会话级**分类（由分类器推得）。两者经常不一致，判定归属时
必须明确用哪一个（见 ``is_one_shot`` 与 ``ReuseScoreEvict.pick``）。

进程级状态只有一处：``SessionRegistry.global_instance()``。它被所有 radix cache
实例共享，因此 ``clear()`` 是全局语义（这正是 ``/flush_cache`` 需要的）。
"""

import math
import zlib
import dataclasses
import enum
import logging
import os
import math
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

# SPDX-License-Identifier: Apache-2.0
from enum import Enum

class SessionType(str, enum.Enum):
    """会话级分类结果，由 ``SessionRegistry._classify`` 依据轮数/工具调用/间隔推出。

    与节点标签分属两层：一个 TRANS 会话插入的片段后来可能被多会话复用，
    那时标签仍是 QTRANS，但已不该算 one-shot。
    """

    TRANS = "trans"
    STRUCT = "struct"
    CHAT = "chat"
    AGENTIC = "agentic"


class HeterCacheNodeTag(str, enum.Enum):
    """节点级标签：KV 片段出生时由会话分类盖章，之后只被少数规则改写。

    与 ``SessionType`` 是两个维度：本枚举回答“这段 KV 是谁创建的”，
    ``SessionType`` 回答“这个会话现在是什么”。两者会不同步，升级规则见
    ``RadixCache._record_node_hit``。评分模型、队列分池、profile 列名都只读本枚举。

    值（``qtrans`` 等）随节点标签快照落盘（JSONL 的 ``tag`` 字段），改值等于改
    数据格式，故保持原样。
    """

    QTRANS = "qtrans"
    QSTRUCT = "qstruct"
    QAGENT = "qagent"
    QCHAT = "qchat"

    @classmethod
    def coerce(cls, tag) -> "HeterCacheNodeTag":
        """把节点上的原始值收敛成枚举；缺失/未知按多轮 chat 处理。

        回落到 chat 而不是 QTRANS：QTRANS 会被排进 one-shot 带直接清掉，
        多轮归类最多是多留一段时间。
        """
        if isinstance(tag, cls):
            return tag
        try:
            return cls(tag)
        except (ValueError, TypeError):
            return cls.QCHAT

    @property
    def queue_name(self) -> str:
        """归属队列名（``VALUE_QUEUES`` 的键、profile 的列名）。

        把枚举投影成字符串只是为了对接 CLI 配置（`caps` / `reservations` 的键
        必须是字符串）与日志列名；判定一律用枚举本身。
        """
        return _QUEUE_NAME[self]

    @classmethod
    def from_queue(cls, name) -> "HeterCacheNodeTag | None":
        """队列名反查枚举；没有对应关系时返回 None。

        反向表包含全部四项（含 ``QTRANS`` ↔ ``random``），因此与旧的"队列名
        等于字符串"判定逐字等价。仲裁层只会传入 ``VALUE_QUEUES`` 里的三个
        名字，其余名字走 None 分支。
        """
        return _QUEUE_TO_TAG.get(name)

    @property
    def reuse_config(self) -> Optional[Tuple[float, float]]:
        """复用间隔的 lognormal 参数 ``(mu, sigma)``；无时间维度时返回 None。"""
        return _REUSE_CONFIG.get(self)


def session_bloom_bits(session_id) -> int:
    """把session_id哈希成 64 bit 布隆位图,用于共享数去重."""
    # 1. 序列化
    text = f"sid:{session_id}"
    
    # 2. 基础哈希 (CRC32 结果为无符号32位整数)
    h = zlib.crc32(text.encode("utf-8", "replace"))
    
    # 3. 增强雪崩扩散 (Knuth乘法散列 + 双向异或位移)
    # 解决CRC32线性特征以及低位混合不充分的问题
    h = (h ^ (h >> 16)) * 0x85EBCA6B & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 0xC2B2AE35 & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF

    # 4. 提取4个[0, 63]的索引
    # 分散利用32位的不同区间，避免只用低 24 位
    bits = 0
    bits |= 1 << (h & 63)              # bits 0~5
    bits |= 1 << ((h >> 8) & 63)       # bits 8~13
    bits |= 1 << ((h >> 16) & 63)      # bits 16~21
    bits |= 1 << ((h >> 24) | ((h & 0x3) << 4))  # 组合高位与剩余位，打乱相关性
    
    return bits


# 共享会话数的饱和上界：8 bit，够用且能原样塞进诊断快照。
SESSION_SHARE_MAX = 255


def session_type_to_node_tag(session_type: SessionType) -> HeterCacheNodeTag:
    """把会话分类结果映射成“新写入 KV 该归哪个队列”的标签。

    只在节点出生时使用：出生标签回答“谁创建了这段”，之后的复用事实由共享数
    账本记录，不回写标签。
    """
    return {
        SessionType.TRANS: HeterCacheNodeTag.QTRANS,
        SessionType.STRUCT: HeterCacheNodeTag.QSTRUCT,
        SessionType.AGENTIC: HeterCacheNodeTag.QAGENT,
        SessionType.CHAT: HeterCacheNodeTag.QCHAT,
    }[session_type]


# ---------------------------------------------------------------------------
# HeterCache 的可调参数组
#
# 用 dataclass 而非裸 kwargs：让 eviction_policy_config 有层级、可校验、可整体
# 传递。配置只接受分组形式
# {"reuse":{...},"share":{...},"value":{...}}，
# 拼错的键会在构造处报错——实验配置静默失效比启动失败更难查。
#
#   ReuseParams        : CHAT / AGENTIC 共用的队内复用分排序（两队列同一口径）
#   ShareParams        : 共享会话数阈值（决定 one-shot 分界与保护范围）
#   QueueValueParams   : 已兑现价值密度（VD）的时间尺度与统计下限
# ---------------------------------------------------------------------------


def _coerce_params(cls, value, *, name: str):
    """把 None / dict / 参数组实例统一成实例。

    Args:
        cls: 参数组类。
        value: None（取默认值）、dict（按关键字展开）或已有实例。
        name: 错误消息里的配置分组名（如 ``"value"``）。

    Returns:
        参数组实例。

    Raises:
        ValueError: dict 里出现未知键；消息会列出该组已知键名。
        TypeError: 传入既不是 None/dict 也不是该组的实例。
    """
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        try:
            return cls(**value)
        except TypeError as exc:
            known = ", ".join(f.name for f in dataclasses.fields(cls))
            raise ValueError(
                f"Invalid {name} parameters {sorted(value)}: {exc}. Known keys: {known}."
            ) from None
    raise TypeError(
        f"{name} must be {cls.__name__}, dict or None, got {type(value).__name__}."
    )


# 跨会话模板的共享会话数阈值。
DEFAULT_SHARE_MIN = 2

# 一次性带的时间豁免窗口（秒）：新写入的 TRANS/STRUCT 节点在归属会话活跃期内不进 one-shot 带。
ONE_SHOT_IDLE_S = 20.0

# 队内复用分的预测窗口 H（秒）：P(reuse in (a, a+H] | not reused by a)。
DEFAULT_REUSE_WINDOW_S = 90.0

# 按生效标签的老化阈值（秒）：节点超过该时长未被访问即降级到 one-shot 带（0 表示不老化）。
DEFAULT_AGEING_TTL_S = {
    "struct": 120.0,
    "chat": 300.0,
    "agentic": 300.0,
    "random": 0.0,
}


@dataclasses.dataclass
class ShareParams:
    """共享会话数阈值与按类型的节点老化 TTL。

    Attributes:
        share_min: 受保护的共享会话数阈值，必须 >= 2。
        ageing_ttl_s: 按队列名的老化 TTL（秒），0 表示该类型不老化。只写想
            覆盖的键即可，其余类型沿用 DEFAULT_AGEING_TTL_S。
    """

    share_min: int = DEFAULT_SHARE_MIN
    ageing_ttl_s: Dict[str, float] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_AGEING_TTL_S)
    )
    # 老化判据的两种口径：
    #   "queue_ttl"    —— 按队列 TTL（历史口径，节点"M 秒没被访问"就老化）
    #   "session_life" —— 按会话寿命（方案 A）：只看该会话自己的轮间隔分布，
    #                     超过 k×p90 才算终结；终结后的**私有**内容才可回收。
    # 前者分不清"会话还活着只是这轮间隔长"，后者把"会话终结"当成唯一判据。
    # struct 始终走 TTL：它是模板族，没有会话级证据（单轮、gaps 为空）。
    ageing_mode: str = "queue_ttl"
    session_ttl_k: float = 3.0
    session_ttl_min_s: float = 60.0
    session_ttl_max_s: float = 1800.0
    session_ttl_min_gaps: int = 2

    def __post_init__(self):
        if self.share_min < 2:
            raise ValueError(
                "share.share_min must be >= 2 (>1 个外部会话才算共享), "
                f"got {self.share_min}"
            )
        unknown = set(self.ageing_ttl_s) - set(QUEUE_NAMES)
        if unknown:
            raise ValueError(
                f"share.ageing_ttl_s has unknown queues {sorted(unknown)}; "
                f"expected a subset of {list(QUEUE_NAMES)}."
            )
        if self.ageing_mode not in ("queue_ttl", "session_life"):
            raise ValueError(
                "share.ageing_mode must be 'queue_ttl' or 'session_life', "
                f"got {self.ageing_mode!r}"
            )
        if self.session_ttl_k <= 0:
            raise ValueError(f"share.session_ttl_k must be > 0, got {self.session_ttl_k}")
        if self.session_ttl_min_s > self.session_ttl_max_s:
            raise ValueError(
                "share.session_ttl_min_s must be <= session_ttl_max_s, "
                f"got {self.session_ttl_min_s} > {self.session_ttl_max_s}"
            )
        if self.session_ttl_min_gaps < 1:
            raise ValueError(
                f"share.session_ttl_min_gaps must be >= 1, got {self.session_ttl_min_gaps}"
            )
        merged = {**DEFAULT_AGEING_TTL_S, **self.ageing_ttl_s}
        for queue, ttl in merged.items():
            if ttl < 0:
                raise ValueError(
                    f"share.ageing_ttl_s[{queue!r}] must be >= 0, got {ttl}"
                )
        self.ageing_ttl_s = merged

    @classmethod
    def from_any(cls, value=None) -> "ShareParams":
        return _coerce_params(cls, value, name="share")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ReuseParams:
    """CHAT / AGENTIC 共用的队内复用分排序参数。

    两个队列用同一套公式与同一组旋钮，只有 mu/sigma 不同（按节点生效标签取
    ``HeterCacheNodeTag.reuse_config``）：

        score = P(reuse in (elapsed, elapsed + reuse_window_s] | not reused by elapsed)

    分数越小越先淘汰。会话级轮询（RR）已整体删除：它的排序键是「最久没被削过」，
    与「最不可能被复用」正交，会让刚起、30s 后必然回来的会话排在最前。

    Attributes:
        enabled: 是否启用复用分排序；关闭时 chat/agentic 队内弃权（走兜底）。
        reuse_window_s: 预测窗口 H（秒）。
    """

    enabled: bool = True
    reuse_window_s: float = DEFAULT_REUSE_WINDOW_S

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError(f"reuse.enabled must be bool, got {self.enabled!r}")
        if self.reuse_window_s <= 0:
            raise ValueError(
                f"reuse.reuse_window_s must be > 0, got {self.reuse_window_s}"
            )

    @classmethod
    def from_any(cls, value=None) -> "ReuseParams":
        return _coerce_params(cls, value, name="reuse")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ChainParams:
    """chat / agentic 队内的链形保护参数（chain_lru 的 heter 队内版）。

    与独立策略 ``ChainLRUStrategy`` 同一条实测结论：match 刷新整条匹配路径的
    recency，私有单子链要么全新要么全旧；链成为全局最旧后会被一路剥穿，
    归零后进入「整链重插 → 再被剥穿」的死循环（tail-stress 实测 warm 整链丢失
    52%，P90 是 OPT 上界的 5 倍）。队内版排序键 =
    ``(last_access_time + penalty, -remain, node.id)``，``min`` 即淘汰：

    * 自适应锚（区别于独立策略的全局 floor）：锚 = max(锚下限先验,
      **链上最深重读节点的深度**)——重读边界 = 该会话上次被命中的前缀长度，
      正是它下一轮要读的位置。罚 ``weight × (锚 − remain)`` 秒，剥得越深罚越重。
    * 证据门固定为「重读过 ∪ 会话回来过」（四 cell 实测两档单门互堵，
      multiturn 并集两头通吃，见 mixexp/chain_shape_report.md §五）；
      会话终结（session_expired 口径）时释放锚，防死会话占着保护不放。
    * 砍链尾：平票时先剥 remain 大的叶子（长链砍尾只损失尾巴）。

    Attributes:
        enabled: 默认 False——不装配时 chat/agentic 队内仍走 ReuseScoreEvict，
            默认行为逐位不变。
        floor_tokens: 锚下限先验（token）。0 = 纯自适应（只护重读边界）。
        weight: 缺口罚斜率（秒/token）。
        min_gaps: 「会话回来过」的最小轮间隔观测数（SessionRegistry.gaps）。
        tail_first: 平票时是否先剥长链的尾。
        expire_anchors: 会话终结是否释放锚（生产建议 True）。
    """

    enabled: bool = False
    floor_tokens: int = 8192
    weight: float = 0.02
    min_gaps: int = 1
    tail_first: bool = True
    expire_anchors: bool = True

    def __post_init__(self):
        if self.floor_tokens < 0:
            raise ValueError(f"chain.floor_tokens must be >= 0, got {self.floor_tokens}")
        if self.weight < 0:
            raise ValueError(f"chain.weight must be >= 0, got {self.weight}")
        if self.min_gaps < 1:
            raise ValueError(f"chain.min_gaps must be >= 1, got {self.min_gaps}")

    @classmethod
    def from_any(cls, value=None) -> "ChainParams":
        return _coerce_params(cls, value, name="chain")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RegretGateParams:
    """跨队列仲裁的后悔弃权门参数（档 A：VD 排序不动，R 做弃权）。

    背景（mixexp/chain_shape_report.md §六）：tail-stress 实测账本证明，
    Stage 0 已把垃圾流量吃干净（one_shot ≈ 冷尾巴全部写入量），但残余压力经
    VD 路由到密度最低的队列（锚所在队）后被迫队内挑人，正反馈放大成整链重算
    （warm 写入量被放大 2.4 倍）。本门给仲裁加实测反馈：

        R_q = 窗口 W 内 join 回队列 q 的淘汰内容 token ÷ 同窗从 q 削掉的 token

    削了没人回来读（冷尾巴）→ R≈0；削了锚（会话回来 miss）→ R 飙升 →
    弃权跳队。正反馈在第一圈被打断，不再需要 floor/weight 调参。

    Attributes:
        enabled: 默认 False——不装配时 queue_order/queue_victim 行为逐位不变。
        window_s: 滑窗 W（秒）。对齐 VD 的政策口径：视野是政策、窗口只是统计，
            所有队列共用，不按队拆。
        k: 弃权阈值 = k × 全体已采样队列 R 的中位数。
        r_min: 绝对下限：R_q < r_min 永不弃权（正常 regime 不触发的保险）。
        min_evicted: 采样下限：窗口内削掉量不足的队列不参与弃权判定
            （避免 0/0 的空队列永远赢得 min-R 兜底）。
        retain_s: stamp 保留时长（秒），须覆盖最大的会话轮间隔，
            否则 join 不上、R 被低估。与 regret_ledger.DEFAULT_RETAIN_S 同值。
        max_stamps: stamp 容量上限（FIFO），内存护栏。
        context: content_key 的上下文 token 数，与 regret_ledger.CONTEXT_TOKENS
            同值——观测账本在场时以账本的为准（键同源才有 join），仅当账本缺席
            而本门单独启用时使用本值。

    档 B（全弃权兑底的全局定价）：

        priced_fallback: 全弃权后用「全树叶子按链形定价取最便宜」替代随机
            fallback 与 min-R 跨队。定价 = lat + weight × (1 + β·R_队列) ×
            max(0, 锚 − remain)，锚 = max(floor 先验, 最深重读深度)。
        floor_tokens / weight / beta: 定价参数；β=0 退化为纯形状定价（与
            chain_lru 同构），β>0 把实测后悔密度乘进锚罚（历史亏得多的队列
            锚罚更重）。
    """

    enabled: bool = True
    window_s: float = 120.0
    k: float = 1.2
    r_min: float = 0.3
    min_evicted: int = 4096
    retain_s: float = 900.0
    max_stamps: int = 200_000
    context: int = 16
    priced_fallback: bool = True
    early_priced: bool = True
    floor_tokens: int = 8192
    floor_by_queue: Dict[str, int] = dataclasses.field(
        default_factory=lambda: {"struct": 0, "agentic": 4096, "chat": 8192}
    )
    weight: float = 0.02
    alpha: float = 0.5
    beta: float = 1.0
    scale_min: float = 0.25
    scale_max: float = 4.0
    min_gaps: int = 1

    def __post_init__(self):
        if self.window_s <= 0:
            raise ValueError(f"regret_gate.window_s must be > 0, got {self.window_s}")
        if self.k <= 0:
            raise ValueError(f"regret_gate.k must be > 0, got {self.k}")
        if not (0.0 <= self.r_min <= 1.0):
            raise ValueError(f"regret_gate.r_min must be in [0, 1], got {self.r_min}")
        if self.min_evicted < 0:
            raise ValueError(
                f"regret_gate.min_evicted must be >= 0, got {self.min_evicted}"
            )
        if self.retain_s < self.window_s:
            raise ValueError(
                f"regret_gate.retain_s ({self.retain_s}) must be >= window_s "
                f"({self.window_s}): stamp 活得比窗口短会让 join 迟到丢失"
            )
        if self.max_stamps < 1:
            raise ValueError(f"regret_gate.max_stamps must be >= 1, got {self.max_stamps}")
        if self.context < 1:
            raise ValueError(f"regret_gate.context must be >= 1, got {self.context}")
        if self.floor_tokens < 0:
            raise ValueError(
                f"regret_gate.floor_tokens must be >= 0, got {self.floor_tokens}"
            )
        if self.weight < 0:
            raise ValueError(f"regret_gate.weight must be >= 0, got {self.weight}")
        if self.alpha < 0:
            raise ValueError(f"regret_gate.alpha must be >= 0, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"regret_gate.beta must be >= 0, got {self.beta}")
        if self.scale_min <= 0 or self.scale_min > self.scale_max:
            raise ValueError(
                f"regret_gate.scale_min ({self.scale_min}) must satisfy 0 < scale_min <= scale_max ({self.scale_max})"
            )
        if self.min_gaps < 1:
            raise ValueError(
                f"regret_gate.min_gaps must be >= 1, got {self.min_gaps}"
            )
        merged_floors = {"struct": 0, "agentic": 4096, "chat": 8192}
        if self.floor_by_queue:
            for q, fl in self.floor_by_queue.items():
                if fl < 0:
                    raise ValueError(f"regret_gate.floor_by_queue[{q!r}] must be >= 0, got {fl}")
                merged_floors[q] = int(fl)
        self.floor_by_queue = merged_floors

    @classmethod
    def from_any(cls, value=None) -> "RegretGateParams":
        return _coerce_params(cls, value, name="regret_gate")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# 参与跨队列仲裁的三级队列。`random` 不在此列：它整行都是 one-shot，
# 在 Stage 0 就被清掉，永远不会计入 VD 账本。
VALUE_QUEUES = ("struct", "agentic", "chat")

# 队内按复用概率排序的两个队列：两者用同一套公式与同一组旋钮（``ReuseParams``），
# 只有 mu/sigma 随标签不同（chat T50≈118s、agentic T50≈6.1s）。
REUSE_QUEUES = ("chat", "agentic")

# VD 的决策视野（秒）：所有队列共享，VD 的量纲由它定义——未来一个视野窗口内
# 每个驻留 token 预期换回的命中量。跨队列比较只在同一视野下成立，所以它是政策
# 旋钮，不随队列变化。
DEFAULT_VALUE_HORIZON_S = 120.0

# VD 估计窗的平滑倍数 k：W_q = max(视野, k * T50_q)。W_q 只影响命中速率估计的
# 方差，不改变比较尺度（投影回视野后速率估计与 k 无关）。
DEFAULT_VALUE_SMOOTHING_K = 4.0

# VD 分母的统计下限。0 = 不做下限，密度就是 H/max(occ, 0)。
#
# 为什么不再用 2048：它把「小队列」系统性地算成低价值。exp3 scan cell 实测
# agentic 驻留中位 530 token、真实密度 3.2（每驻留 token 每窗换回 3.2 个命中
# token），除掉 floor 后只剩 0.684，与 struct 的 0.68 打平/略低，于是每次都
# 先削它；削完 occ 更小、分母恒被 floor 顶住 → 密度恒低 → 正反馈饿死
# （0.57M token 淘汰换掉 0.45M 命中，命中率 96.9%→77.8%）。
# 反事实：floor=1 时同一批 982 次决策有 66% 不再选 agentic。
# 它不是保留量，取下限也没有「1 token 队列被当成高价值」的实际风险：
# 空队列（occ=0）不进入 queue_order 的候选，densities() 对 0 分母返回 0。
DEFAULT_VALUE_FLOOR_TOKENS = 0.0


@dataclasses.dataclass
class QueueValueParams:
    """已兑现价值密度（VD）的参数。

    跨队列仲裁只读 VD（``selector.queue_order`` 按 VD 升序，占用不参与排序），
    没有权重、没有 cap / reservation / deadzone / chunk。

    Attributes:
        horizon_s: 跨队列共享的决策视野 horizon（秒）。VD 的量纲由它定义：未来 horizon 秒
            内每个驻留 token 预期换回的命中量。
        smoothing_k: 估计窗的平滑倍数 k，W_q = max(horizon, k * T50_q)。只影响速率
            估计的方差，不改变跨队列比较的尺度。
        floor_tokens: VD 分母的统计下限（token），不是保留量。
        seed: 兜底随机抽样的种子（同 seed 可复现）。
    """

    horizon_s: float = DEFAULT_VALUE_HORIZON_S
    smoothing_k: float = DEFAULT_VALUE_SMOOTHING_K
    floor_tokens: float = DEFAULT_VALUE_FLOOR_TOKENS
    seed: int = 0

    def __post_init__(self):
        if self.horizon_s <= 0:
            raise ValueError(f"value.horizon_s must be > 0, got {self.horizon_s}")
        if self.smoothing_k < 1.0:
            raise ValueError(
                "value.smoothing_k must be >= 1 (W_q < horizon would push the estimator "
                f"into the under-sampled regime), got {self.smoothing_k}"
            )
        if self.floor_tokens < 0:
            raise ValueError(
                f"value.floor_tokens must be >= 0 (0 = 不做下限), "
                f"got {self.floor_tokens}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"value.seed must be int, got {self.seed!r}")

    @classmethod
    def from_any(cls, value=None) -> "QueueValueParams":
        return _coerce_params(cls, value, name="value")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@lru_cache(maxsize=8)
def _cached_lognorm_params(mu: float, sigma: float):
    """缓存 lognormal 的派生常数。"""
    return mu, sigma, sigma * math.sqrt(2.0)


def _cached_lognorm_cdf(x: float, mu: float, sigma: float) -> float:
    """lognormal CDF（erf 近似）；x <= 0 时返回 0 避免 log(0)。"""
    if x <= 0:
        return 0.0
    mu, sigma, denom = _cached_lognorm_params(mu, sigma)
    z = (math.log(x) - mu) / denom
    return 0.5 * (1.0 + math.erf(z))


def median(values: List[float]) -> float:
    """计算非空数值序列的中位数。"""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def reuse_window_probability(
    elapsed_s: float, horizon_s: float, mu: float, sigma: float
) -> float:
    """P(reuse in (elapsed, elapsed+horizon] | not reused by elapsed)."""
    elapsed_s = max(float(elapsed_s), 0.0)
    survived = 1.0 - _cached_lognorm_cdf(elapsed_s, mu, sigma)
    if survived <= 0.0:
        return 0.0
    survived_after = 1.0 - _cached_lognorm_cdf(elapsed_s + horizon_s, mu, sigma)
    return max(0.0, 1.0 - survived_after / survived)


# 多轮任务的复用间隔分布参数 (mu, sigma)，lognormal。
_REUSE_CONFIG = {
    HeterCacheNodeTag.QCHAT: (4.77, 1.279),
    HeterCacheNodeTag.QAGENT: (1.81, 1.092),
}

# 节点标签到队列名映射
_QUEUE_NAME = {
    HeterCacheNodeTag.QCHAT: "chat",
    HeterCacheNodeTag.QAGENT: "agentic",
    HeterCacheNodeTag.QSTRUCT: "struct",
    HeterCacheNodeTag.QTRANS: "random",
}
_QUEUE_TO_TAG = {name: tag for tag, name in _QUEUE_NAME.items()}


def reuse_interval_median(tag: HeterCacheNodeTag) -> Optional[float]:
    """复用间隔的 lognormal 中位数 T50 = e^mu；无时间维度时返回 None。"""
    cfg = tag.reuse_config
    return None if cfg is None else math.exp(cfg[0])


def estimation_window_s(queue: str, horizon_s: float, smoothing_k: float) -> float:
    """VD 的估计窗 W_q = max(horizon, k * T50_q)，单位秒。

    W_q 只用来平滑命中速率估计 ``r_q = H_q(W_q) / W_q``：窗口小于复用间隔
    时，窗口里常常一次命中都没有，读数会在 0 与尖峰之间摆动。没有复用间隔
    分布数据的队列（struct）直接回退 horizon，不给它拍一个无依据的常数。
    """
    tag = _QUEUE_TO_TAG.get(queue)
    t50 = reuse_interval_median(tag) if tag is not None else None
    if t50 is None:
        return float(horizon_s)
    return max(float(horizon_s), float(smoothing_k) * t50)

# profile 的四列，顺序即打印顺序。
# random 不参与跨队列仲裁：它整行都是 one-shot，在阶段 0 就被清掉。
QUEUE_NAMES = ("chat", "agentic", "struct", "random")



def node_num_tokens(node) -> int:
    """节点的片段长度（radix 片段的 token 数），非驻留节点返回 0。"""
    return len(node.value) if node.value is not None else 0




@dataclasses.dataclass
class SessionState:
    """单条会话的 KV 记账 + 分类证据。
    * KV记录：L（最近一次 prompt 长度）与d（当前缺口）。d只在
      两处变化——会话自己的请求写入时归零、其节点被删时累加
      两者只用于观测（summary 的d_sum/d_p50/d_max 与告警），不参与选点。
    * 分类依据：classifier_gaps/turn_count/tool_call_count/
      cross_session_shared，只用于推session_type。

    分开的原因是：分类器改一次阈值就会把历史replay的分类结果改掉，如果两者
    共用gaps，连带会把 KV 缺口的口径也改掉。

    Attributes:
        L: 最近一次请求的 prompt 长度（token）。
        d: 当前 KV 缺口。
        node_count: 仍驻留的节点数。
        last_rid: 用于 chunked prefill 去重——同一 rid 重入不算新的一轮。
    """

    session_id: str
    L: int = 0                  # 最近一次请求的 prompt 长度
    d: int = 0                  # 当前缺口（token）
    node_count: int = 0         # 仍驻留的节点数
    last_active: float = 0.0    # monotonic
    last_rid: str = ""          # 用于 chunked prefill 去重
    gaps: List[float] = dataclasses.field(default_factory=list)
    attached: int = 0           # 累计挂到该会话名下的 token 数（调试用）
    charged: int = 0            # 累计被删的 token 数（调试用）
    turn_count: int = 0
    tool_call_count: int = 0
    cross_session_shared: bool = False
    session_type: SessionType = SessionType.TRANS
    first_context_len: int = 0
    latest_context_len: int = 0
    classifier_gaps: List[float] = dataclasses.field(default_factory=list)
    classifier_last_rid: str = ""
    classifier_last_request_start: Optional[float] = None


logger = logging.getLogger(__name__)


class SessionProbe:
    """会话账本的诊断探针：生产态是基类，什么都不做。

    诊断（单会话追踪、缺口越界告警、sweep 耗时采样）全部收在子类里，账本只在
    关键点回调查询。这样 ``charge`` / ``sweep`` 保持纯算法，排查代码不侵入热路径。

    默认不挂探针（账本里是 ``None``）而不是挂一个空对象：少一次方法调用。
    基类只作为注入点与类型边界存在。
    """

    __slots__ = ()

    # 是否需要 sweep 计时。基类不测，账本据此跳过两次 perf_counter。
    wants_sweep_timing = False

    def on_charge(self, node, state: "SessionState", num_tokens: int) -> None:
        """节点被逐出、缺口累加之后、截断之前回调。"""

    def on_sweep(self, elapsed_s: float, scanned: int, dead: int) -> None:
        """一次 sweep 结束后回调。"""

    def sweep_profile(self) -> Dict[str, float]:
        """sweep 耗时统计；不采样的探针返回全零。"""
        return {
            "calls": 0,
            "avg_us": 0.0,
            "max_us": 0.0,
            "last_scan": 0,
            "last_dead": 0,
        }


class DebugSessionProbe(SessionProbe):
    """排查用探针：单会话追踪 + 缺口越界告警 + sweep 耗时采样。

    三项各自独立开关，全关时不建实例（见 ``_probe_from_env``）。

    Attributes:
        trace_sid: 只盯一条会话，逐次打印逐出细节。
        enable_break_log: 缺口远超链长时告警（通常意味着节点 session_id 归属串标）。
        enable_sweep_timing: 记录 sweep 耗时与扫描规模。
    """

    __slots__ = (
        "trace_sid",
        "enable_break_log",
        "wants_sweep_timing",
        "_calls",
        "_us_total",
        "_us_max",
        "_scan",
        "_dead",
    )

    SWEEP_REPORT_EVERY = 64    # 每 N 次 sweep 打一行汇总，太小刷屏、太大漏尖峰
    SWEEP_WARN_US = 2000.0     # 单次 sweep 超过 2ms 就告警
    BREAK_SLACK_TOKENS = 32    # 缺口超出链长多少才值得告警

    def __init__(
        self,
        trace_sid: str = "",
        enable_break_log: bool = False,
        enable_sweep_timing: bool = False,
    ):
        self.trace_sid = trace_sid
        self.enable_break_log = enable_break_log
        self.wants_sweep_timing = enable_sweep_timing
        self._calls = 0
        self._us_total = 0.0
        self._us_max = 0.0
        self._scan = 0
        self._dead = 0

    def on_charge(self, node, state: "SessionState", num_tokens: int) -> None:
        """打印单会话逐出细节与缺口越界告警。"""
        session_id = state.session_id
        if self.trace_sid and session_id == self.trace_sid:
            logger.warning(
                "[session trace charge] nid=%s abs_start=%s key=%d val=%d d=%d L=%d",
                getattr(node, "id", -1),
                getattr(node, "abs_start", None),
                len(getattr(node, "key", []) or []),
                num_tokens,
                state.d,
                state.L,
            )
        if (
            self.enable_break_log
            and state.L
            and state.d > state.L + self.BREAK_SLACK_TOKENS
        ):
            logger.warning(
                "[session d break] sid=%s d=%d L=%d +%d abs_start=%s "
                "attached=%d charged=%d live=%d node_count=%d",
                session_id,
                state.d,
                state.L,
                num_tokens,
                getattr(node, "abs_start", None),
                state.attached,
                state.charged,
                state.attached - state.charged,
                state.node_count,
            )

    def on_sweep(self, elapsed_s: float, scanned: int, dead: int) -> None:
        """累加一次 sweep 样本，按周期或超阈打印。

        同时记录耗时与扫描规模：单看耗时分不清“表小但慢”还是“表大但正常”，
        二者的优化手段完全不同。两者相除就是单会话成本，能直接外推到更大规模。
        """
        us = elapsed_s * 1e6
        self._calls += 1
        self._us_total += us
        self._scan = scanned
        self._dead = dead
        if us > self._us_max:
            self._us_max = us
            # 尖峰立即报，不等周期，否则会被均值抹平。
            if us > self.SWEEP_WARN_US:
                logger.warning(
                    "[Session sweep] SLOW %.1fus (scan=%d, dead=%d) 阈%.0fus",
                    us,
                    scanned,
                    dead,
                    self.SWEEP_WARN_US,
                )
        if self._calls % self.SWEEP_REPORT_EVERY == 0:
            per_session = us / scanned if scanned else 0.0
            logger.info(
                "[Session sweep] n=%d last=%.1fus avg=%.1fus max=%.1fus "
                "scan=%d dead=%d per_session=%.3fus",
                self._calls,
                us,
                self._us_total / self._calls,
                self._us_max,
                scanned,
                dead,
                per_session,
            )

    def sweep_profile(self) -> Dict[str, float]:
        """导出 sweep 耗时统计。"""
        calls = self._calls
        return {
            "calls": calls,
            "avg_us": self._us_total / calls if calls else 0.0,
            "max_us": self._us_max,
            "last_scan": self._scan,
            "last_dead": self._dead,
        }


def _probe_from_env() -> Optional[SessionProbe]:
    """按环境变量装配探针；三项都没开时返回 None（零开销）。

    * ``SGLANG_SESSION_TRACE_SID``：只追踪这一条会话的逐出细节；
    * ``SGLANG_SESSION_D_BREAK_LOG=1``：缺口远超链长时告警；
    * ``SGLANG_SESSION_SWEEP_PROFILE=1``：记录 sweep 耗时。
    """
    trace_sid = os.environ.get("SGLANG_SESSION_TRACE_SID", "")
    break_log = os.environ.get("SGLANG_SESSION_D_BREAK_LOG", "0") == "1"
    sweep_prof = os.environ.get("SGLANG_SESSION_SWEEP_PROFILE", "0") == "1"
    if not (trace_sid or break_log or sweep_prof):
        return None
    return DebugSessionProbe(
        trace_sid=trace_sid,
        enable_break_log=break_log,
        enable_sweep_timing=sweep_prof,
    )

class SessionRegistry:
    """会话KV缺口统计：维护 KV Cache 缺口（d/L 记录）并在线推断会话负载类型。

    职责边界：
    - observe_request / record_cross_session_hit：提取特征更新分类（TRANS/STRUCT/AGENTIC/CHAT）。
    - touch / charge / attach：维护 KV 节点增删及缺失（d/L）。


    设计约束：
    1. 进程级单例：多 cache 实例通过 global_instance() 共享同一份状态，防止分类器与调度层策略出现分歧。
    2. 无锁安全：所有入口严格限定在调度主线程（match / insert / evict）执行，禁止跨线程并发。
    3. 鸭子类型：仅读取 node.session_id 与 node.value，与底层 Radix 树实现解耦。
    """

    _global: Optional["SessionRegistry"] = None

    @classmethod
    def global_instance(cls) -> "SessionRegistry":
        """获取进程单例，初次调用按默认参数惰性创建。"""
        inst = SessionRegistry._global
        if inst is None:
            inst = SessionRegistry()
            SessionRegistry._global = inst
        return inst

    @classmethod
    def reset_global(cls) -> Optional["SessionRegistry"]:
        """销毁单例并返回旧对象。用于单测隔离、重置参数或回放对比。"""
        old = SessionRegistry._global
        SessionRegistry._global = None
        return old

    GAP_WINDOW = 8          # 轮间间隔滑动窗口大小（采样最近 8 次）
    SWEEP_EVERY = 256       # 惰性回收周期：每 N 次 touch 触发一次 sweep
    # 判定「多轮」的最小 turn 阈值；低于该值归为 TRANS/STRUCT。
    MULTI_TURN_MIN = 2
    GAP_SPLIT_S = 15.0      # Agent 与 Chat 的间隔判定临界值（秒）
    AGENTIC_CONTEXT_GROWTH_TOKENS = 1500  # 边界兜底：判定上下文持续暴涨的 token 阈值

    def __init__(
        self,
        idle_ttl: float = 3600.0,
        empty_grace: float = 60.0,
        probe: Optional[SessionProbe] = None,
        now: Optional[Callable[[], float]] = None,
    ):
        """构造账本。

        Args:
            idle_ttl: 会话无请求的最大存活时间（秒）。
            empty_grace: 节点清空后保留分类证据的宽限期（秒）。
            probe: 诊断探针；``None`` 表示按环境变量装配，全关时不挂探针。
            now: 时间源；``None`` 表示 ``time.monotonic``。这个时钟必须与
                缓存层的 ``CacheInitParams.now`` 同源：``last_active`` 要拿去跟
                节点时间戳做差（one-shot 时间豁免），混源等于拿假 now 减真时间。
        """
        self._sessions: Dict[str, SessionState] = {}
        self.idle_ttl = idle_ttl          # 会话无请求的最大存活时间
        self.empty_grace = empty_grace    # 节点清空后保留分类证据的宽限期
        self._now = now if now is not None else time.monotonic
        self._touches = 0
        self._probe = _probe_from_env() if probe is None else probe
        # 绑定回调而不是每次都判断并调一个空对象：默认态只剩一次 None 判断。
        self._on_charge = self._probe.on_charge if self._probe is not None else None
        # 只有需要计时的探针才挂 on_sweep，这样“不测速”连两次 perf_counter 都省了。
        self._on_sweep = (
            self._probe.on_sweep
            if self._probe is not None and self._probe.wants_sweep_timing
            else None
        )

    # ---- 查询 ----
    def use_clock(self, now: Callable[[], float]) -> None:
        """接管时间源（默认墙钟下等价于无操作）。

        由 ``HeterCacheStrategy.use_clock`` 从缓存层推过来：会话账本里的
        ``last_active`` 与节点 ``last_access_time`` 必须由同一个时钟产生。
        """
        self._now = now

    def get(self, session_id: Optional[str]) -> Optional[SessionState]:
        return self._sessions.get(session_id) if session_id else None

    def __len__(self) -> int:
        return len(self._sessions)

    # ---- 特征提取与分类 ----
    def _get_or_create(self, session_id: str) -> SessionState:
        st = self._sessions.get(session_id)
        if st is None:
            st = self._sessions[session_id] = SessionState(session_id=session_id)
        return st

    def observe_request(
        self,
        session_id: Optional[str],
        rid: str = "",
        prompt_len: int = 0,
        *,
        tool_call=None,
        now: Optional[float] = None,
    ) -> Optional[SessionState]:
        """记录请求特征并刷新分类，基于 rid 去重以规避 chunked prefill 重入。"""
        if not session_id:
            return None
        now = self._now() if now is None else now
        st = self._get_or_create(session_id)
        if rid and rid == st.classifier_last_rid:
            return st

        if st.classifier_last_request_start is None:
            st.first_context_len = prompt_len
        else:
            st.classifier_gaps.append(max(0.0, now - st.classifier_last_request_start))
            if len(st.classifier_gaps) > self.GAP_WINDOW:
                del st.classifier_gaps[0]
        st.turn_count += 1
        st.tool_call_count += int(bool(tool_call))
        st.latest_context_len = prompt_len
        st.classifier_last_request_start = now
        st.classifier_last_rid = rid
        st.session_type = self._classify(st)
        return st

    def record_cross_session_hit(
        self, session_id: Optional[str], owner_session_id: Optional[str]
    ) -> Optional[SessionState]:
        """标记跨会话命中，用于将单轮共享前缀会话归为 STRUCT。"""
        if not session_id or not owner_session_id or session_id == owner_session_id:
            return self.get(session_id)
        st = self._get_or_create(session_id)
        st.cross_session_shared = True
        st.session_type = self._classify(st)
        return st

    @classmethod
    def _classify(cls, state: SessionState) -> SessionType:
        """会话类型决策树：轮数 -> 工具调用 -> 交互间隔中位数 -> 上下文膨胀量。"""
        if state.turn_count < cls.MULTI_TURN_MIN:
            return SessionType.STRUCT if state.cross_session_shared else SessionType.TRANS
        if state.tool_call_count:
            return SessionType.AGENTIC
        if state.classifier_gaps:
            median_gap = median(state.classifier_gaps)
            if median_gap < cls.GAP_SPLIT_S:
                return SessionType.AGENTIC
            if median_gap > cls.GAP_SPLIT_S:
                return SessionType.CHAT
        growth = state.latest_context_len - state.first_context_len
        return (
            SessionType.AGENTIC
            if growth >= cls.AGENTIC_CONTEXT_GROWTH_TOKENS
            else SessionType.CHAT
        )

    # ---- 缓存水位与缺口记账 ----
    def touch(
        self,
        session_id: Optional[str],
        rid: str = "",
        prompt_len: int = 0,
        now: Optional[float] = None,
    ) -> Optional[SessionState]:
        """新请求落盘时调用：重设有效链长 L，缺口 d 清零，重置活跃时钟。"""
        if not session_id:
            return None
        now = self._now() if now is None else now
        st = self._get_or_create(session_id)
        if rid != st.last_rid and st.last_active > 0.0:
            # 过滤 chunked prefill 重入，确保间隔反映真实轮次交互延时
            st.gaps.append(now - st.last_active)
            if len(st.gaps) > self.GAP_WINDOW:
                del st.gaps[0]
        st.last_rid = rid
        st.L = prompt_len
        st.d = 0
        st.last_active = now

        # 挂在调度热路径上做摊销清理，避免常驻后台回收线程
        self._touches += 1
        if self._touches % self.SWEEP_EVERY == 0:
            self.sweep(now)
        return st

    def attach(self, session_id: Optional[str], num_tokens: int = 0) -> None:
        """KV 树挂载新节点时累加存活计数，供淘汰及 sweep 存活判定使用。"""
        if not session_id:
            return
        st = self._get_or_create(session_id)
        st.node_count += 1
        st.attached += num_tokens

    def charge(self, node) -> None:
        """KV 节点逐出时调用：累加会话缺口 d，递减节点计数。"""
        session_id = getattr(node, "session_id", None)
        if not session_id:
            return
        value = getattr(node, "value", None)
        num_tokens = len(value) if value is not None else 0
        st = self._get_or_create(session_id)
        st.d += num_tokens
        st.charged += num_tokens
        if st.node_count > 0:
            st.node_count -= 1
        # 探针必须在截断之前回调：截断后 d <= L 恒成立，d > L + slack 分支将不可达
        on_charge = self._on_charge
        if on_charge is not None:
            on_charge(node, st, num_tokens)
        if st.L and st.d > st.L:
            st.d = st.L

    # ---- 状态回收 ----
    def clear(self) -> None:
        """全量清空账本。在 /flush_cache 或测试用例重置时调用，防止实验间脏数据污染。"""
        self._sessions.clear()

    def sweep(self, now: Optional[float] = None) -> int:
        """基于 TTL 与节点存活状态惰性回收僵尸会话。

        回收判定：
        1. 超过 idle_ttl 未活跃；
        2. 节点全被逐出且超过 empty_grace 宽限期。

        这是本模块唯一会全量遍历会话表的地方（O(会话数)），由 ``touch`` 按
        ``SWEEP_EVERY`` 计数触发。开关 ``SGLANG_SESSION_SWEEP_PROFILE=1`` 时
        记录本次耗时与扫描规模，供高负载下定夺要不要做增量回收。

        Args:
            now: 注入时钟，便于回放与测试。

        Returns:
            本次回收的会话数。
        """
        now = self._now() if now is None else now
        # 只有挂了计时探针才取时钟与规模，默认路径不付这个成本。
        on_sweep = self._on_sweep
        if on_sweep is not None:
            started = time.perf_counter()
            scanned = len(self._sessions)
        dead = [
            sid
            for sid, st in self._sessions.items()
            if now - st.last_active > self.idle_ttl
            or (st.node_count <= 0 and now - st.last_active > self.empty_grace)
        ]
        for sid in dead:
            del self._sessions[sid]
        if on_sweep is not None:
            on_sweep(time.perf_counter() - started, scanned, len(dead))
        return len(dead)

    def sweep_profile(self) -> Dict[str, float]:
        """导出 sweep 耗时统计，供压测脚本直接消费。

        样本持有在探针里；没挂探针（默认）时返回全零，不区分“没测”还是“零开销”。

        Returns:
            ``calls`` / ``avg_us`` / ``max_us`` / ``last_scan`` / ``last_dead``。
        """
        probe = self._probe
        if probe is None:
            return SessionProbe().sweep_profile()
        return probe.sweep_profile()

    # ---- 监控指标 ----
    def summary(self) -> Dict[str, float]:
        """导出会话缺口分布。重点关注 d_nonzero（实际发生截断的受损会话量）。"""
        ds = sorted(st.d for st in self._sessions.values())
        if not ds:
            return {"sessions": 0, "d_sum": 0, "d_p50": 0, "d_max": 0, "d_nonzero": 0}
        n = len(ds)
        return {
            "sessions": n,
            "d_sum": sum(ds),
            "d_p50": ds[n // 2],
            "d_max": ds[-1],
            "d_nonzero": sum(1 for x in ds if x > 0),
        }