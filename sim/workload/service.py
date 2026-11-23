"""请求服务时长模型（虚拟时钟推进依据）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ServiceModel:
    """线性近似请求在服务端的 Prefill 与 Decode 耗时。

    虚拟时钟环境没有真实流式读取过程，必须显式估计完成时刻，否则闭环多轮间隔将退化为零延迟，导致缓存老化判据失真。
    """

    prefill_base_s: float = 0.05
    prefill_tokens_per_s: float = 2000.0
    decode_tokens_per_s: float = 40.0

    def __post_init__(self) -> None:
        if self.prefill_tokens_per_s <= 0 or self.decode_tokens_per_s <= 0:
            raise ValueError("token 速率必须为正")

    def __call__(self, prompt_tokens: int, out_tokens: int = 0) -> float:
        return (
            self.prefill_base_s
            + max(0, prompt_tokens) / self.prefill_tokens_per_s
            + max(0, out_tokens) / self.decode_tokens_per_s
        )

    def prefill(self, tokens: int) -> float:
        """计算单批 extend tokens 的 Prefill 执行耗时。"""
        return self.prefill_base_s + max(0, tokens) / self.prefill_tokens_per_s

    def decode_step(self) -> float:
        """计算单步 Decode 耗时（按恒定 TPOT 建模）。"""
        return 1.0 / self.decode_tokens_per_s

    @classmethod
    def from_calibration(
        cls, path: str, *, decode_tokens_per_s: Optional[float] = None
    ) -> "ServiceModel":
        """从 calibrate.py 产出的 JSON 标定文件加载 Prefill 速率与基础延迟。"""
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        rate = data.get("prefill_rate_tok_per_s")
        if not rate:
            raise ValueError(f"{path} 里没有 prefill_rate_tok_per_s，标定未完成？")
        return cls(
            prefill_base_s=float(data.get("ttft_intercept_ms", 0.0)) / 1000.0,
            prefill_tokens_per_s=float(rate),
            decode_tokens_per_s=(
                decode_tokens_per_s
                if decode_tokens_per_s is not None
                else cls.decode_tokens_per_s
            ),
        )
