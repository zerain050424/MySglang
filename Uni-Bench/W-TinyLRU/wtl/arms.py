"""实验臂矩阵（设计文档 §6）。

单一事实来源：离线回放、真机启动、分析脚本都从这里读配置，
避免「跑了 A4 却记成 A3」这类实验事故。

约定
----
* ``policy`` 直接对应 ``--radix-eviction-policy``；
* ``config`` 直接对应 ``--radix-eviction-policy-config``（即策略构造 kwargs）；
* ``A2`` 是隔离框架增益的基线（simple_hybrid + tiny_lru=off），
  ``A0/A1`` 是上游能力边界（lru / slru），``A3`` 是主对象。
"""

from __future__ import annotations

import dataclasses
import json
from typing import Dict, List, Optional

# 默认参数（与 TAG_LRU.md §6 一致）
# 标签版 TinyLRU 的默认值（与 TAG_LRU.md 一致）
DEFAULT_STRUCT_SHARE_MIN = 2
DEFAULT_LOCAL_HOT = "chat"

# 注意：tiny_lru 缺省 policy="off"。这里显式写全，便于逐臂对照。
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
    def tiny_lru(self) -> dict:
        return (self.config or {}).get("tiny_lru", {})

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
        "A2", "hybrid_off", "simple_hybrid", {"tiny_lru": {"policy": "off"}},
        "隔离框架增益：多队列 + one-shot 带 + 分类器标签，Q_struct 策略关闭", False,
    ),
    "A3": Arm(
        "A3", "tag_lru", "simple_hybrid", {"tiny_lru": {"policy": "tag_lru"}},
        "默认：标签版 TinyLRU（分类即晋升，跨会话共享→QSTRUCT·LRU）", True,
    ),
    "A4": Arm(
        "A4", "tag_lru_one_shot", "simple_hybrid",
        {"tiny_lru": {"policy": "tag_lru", "local_hot": "one_shot"}},
        "消融：会话内热点也进 one-shot 硬优先带（规则原样）", True,
    ),
    "A5": Arm(
        "A5", "tag_lru_min3", "simple_hybrid",
        {"tiny_lru": {"policy": "tag_lru", "struct_share_min": 3}},
        "消融：要 3 个会话共享才晋升 QSTRUCT", True,
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
    """解析 ``A0,A3-A5`` / ``all`` / ``default`` 表达式，保持输入顺序。"""
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
