"""实验臂矩阵（设计文档 §6）。

单一事实来源：离线回放、真机启动、分析脚本都从这里读配置，
避免「跑了 A4 却记成 A3」这类实验事故。

约定
----
* ``policy`` 直接对应 ``--radix-eviction-policy``；
* ``config`` 直接对应 ``--radix-eviction-policy-config``（即策略构造 kwargs）；
* ``A0/A1`` 是上游能力边界（lru / slru）；
* ``A2`` 是「只有 one-shot 带」的基线（simple_hybrid + share_min 调到不可达），
  ``A3`` 是主对象（两带 SLRU：one-shot 带 + 跨会话保护），``A4`` 是阈值消融。

共享数阈值只有一个旋钮 ``slru.share_min``（框架侧 ``SlruParams``）：
它是「退出 one-shot 带 / 进入 SLRU 保护段」的分界，``share_min`` 越大保护越难触发。
"""

from __future__ import annotations

import dataclasses
import json
from typing import Dict, List, Optional

# 框架侧默认值（与 sglang.srt.mem_cache.uni_utils.SlruParams 保持一致）
DEFAULT_SHARE_MIN = 2
# 不可达阈值：任何节点都进不了保护段 → 等价于「只有 one-shot 带」。
BAND_ONLY_SHARE_MIN = 10 ** 6
@dataclasses.dataclass(frozen=True)
class Arm:
    id: str
    label: str
    policy: str
    config: Optional[dict]
    note: str = ""
    # 判定「策略未被行使」时用来跳过结论（见设计文档 §8.5）
    expect_struct: bool = False

    @property
    def slru(self) -> dict:
        return (self.config or {}).get("slru", {})

    @property
    def share_min(self):
        return self.slru.get("share_min")

    @property
    def is_baseline(self) -> bool:
        return self.id in ("A0", "A2")

    def config_json(self) -> Optional[str]:
        return None if self.config is None else json.dumps(self.config, sort_keys=True)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "policy": self.policy,
            "config": self.config,
            "note": self.note,
            "expect_struct": self.expect_struct,
        }


ARMS: Dict[str, Arm] = {
    "A0": Arm("A0", "lru", "lru", None, "SGLang 默认基线（LRU）", False),
    "A1": Arm(
        "A1", "slru", "slru", {"protected_threshold": 2},
        "上游分段 LRU：能力上限参照（无准入/无 aging）", False,
    ),
    "A2": Arm(
        "A2", "hybrid_band_only", "simple_hybrid",
        {"slru": {"share_min": BAND_ONLY_SHARE_MIN}},
        "只有 one-shot 带：保护段不可达（隔离「保护机制」的贡献）", False,
    ),
    "A3": Arm(
        "A3", "slru_two_band", "simple_hybrid", {"slru": {"share_min": 2}},
        "主对象：两带 SLRU（one-shot 带 + 跨会话共享保护段 · 段内 LRU）", True,
    ),
    "A4": Arm(
        "A4", "slru_min3", "simple_hybrid", {"slru": {"share_min": 3}},
        "消融：要 3 个会话命中过才进保护段（阈值敏感性）", True,
    ),
}

DEFAULT_ARMS = ["A0", "A2", "A3"]
ALL_ARMS = list(ARMS)


def get(arm_id: str) -> Arm:
    try:
        return ARMS[arm_id.strip().upper()]
    except KeyError:
        raise KeyError(
            f"未知实验臂 {arm_id!r}；可用：{', '.join(ARMS)}"
        ) from None


def select(expr: Optional[str]) -> List[Arm]:
    """解析 ``A0,A2-A4`` / ``all`` / ``default`` 表达式，保持输入顺序。"""
    if not expr or expr in ("default", "DEFAULT"):
        return [ARMS[a] for a in DEFAULT_ARMS]
    if expr in ("all", "ALL", "*"):
        return [ARMS[a] for a in ALL_ARMS]
    out: List[Arm] = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and not part.startswith("A-"):
            lo, hi = part.split("-", 1)
            lo_i, hi_i = int(lo.strip().lstrip("Aa")), int(hi.strip().lstrip("Aa"))
            for i in range(lo_i, hi_i + 1):
                out.append(get(f"A{i}"))
        else:
            out.append(get(part))
    # 去重保序
    seen, uniq = set(), []
    for a in out:
        if a.id not in seen:
            seen.add(a.id)
            uniq.append(a)
    return uniq


# # 压力点：pool_ratio = pool_tokens / 校准工作集（见 replay 的 --pool-ratio）
DEFAULT_POOL_RATIOS = [0.3, 0.6, 1.0]
SANITY_POOL_RATIO = 2.0  # §10 无压力对照点：各臂必须趋同


if __name__ == "__main__":  # 便于人工核对矩阵
    for a in ARMS.values():
        print(f"{a.id:4s} {a.label:22s} {a.policy:14s} {a.config_json()}")
