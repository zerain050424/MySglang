"""基于前缀和的块级 Prefill 重算代价计价器。"""

from __future__ import annotations

from typing import List, Tuple

from sim.analysis.cost import CostModel


class PromptPricer:
    """按页粒度预计算块代价前缀和，并以 O(1) 计算每请求的 total_cost 与 saved_cost。"""

    __slots__ = ("page_size", "cost_model", "_prefix")

    def __init__(self, cost_model: CostModel, page_size: int, max_width: int = 0) -> None:
        self.cost_model = cost_model
        self.page_size = int(page_size)
        self._prefix: List[float] = [0.0]
        if max_width > 0:
            self.prepare(max_width)

    def prepare(self, max_width: int) -> None:
        """按最大序列宽度构造代价前缀和数组。"""
        max_blocks = (int(max_width) + self.page_size - 1) // self.page_size
        prefix = [0.0]
        for index in range(max_blocks):
            prefix.append(prefix[-1] + self.cost_model.cost(index))
        self._prefix = prefix

    def cost_of(self, tokens: int) -> float:
        """计算从位置 0 起算 tokens 个 token 的累计代价。"""
        page = self.page_size
        full_blocks, tail_tokens = divmod(int(tokens), page)
        full_blocks = min(full_blocks, len(self._prefix) - 1)
        total = page * self._prefix[full_blocks]
        if tail_tokens:
            total += tail_tokens * self.cost_model.cost(full_blocks)
        return total

    def price(self, prompt_len: int, hit_len: int) -> Tuple[float, float]:
        """返回单请求的 (total_cost, saved_cost)。"""
        return self.cost_of(prompt_len), self.cost_of(hit_len)
