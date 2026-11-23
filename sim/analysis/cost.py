"""块级 Prefill 重算代价模型（独立于驱动层与核心层）。"""

from __future__ import annotations

from typing import List, Optional


class CostModel:
    """按缓存块位置定义的 Prefill 计算代价模型。"""

    __slots__ = ("_a", "_b", "_spec", "_table")

    def __init__(
        self,
        spec: str,
        *,
        a: int = 1,
        b: int = 0,
        table: Optional[List[int]] = None,
    ) -> None:
        self._spec = spec
        self._a = int(a)
        self._b = int(b)
        self._table = table
        if self._a < 1 or self._b < 0:
            raise ValueError(
                f"cost model {spec!r} needs A >= 1 and B >= 0, got A={a} B={b}"
            )
        if table is not None and not table:
            raise ValueError(f"cost model {spec!r} has an empty cost table")

    @classmethod
    def parse(cls, spec: str) -> "CostModel":
        """从配置串解析代价模型，支持 uniform / position / linear:A,B / table:FILE。"""
        text = spec.strip()
        lowered = text.lower()
        if lowered == "uniform":
            return cls(text, a=1, b=0)
        if lowered == "position":
            return cls(text, a=1, b=1)
        if lowered == "qwen3coder30b_blksz_16":
            return cls(text, a=865, b=2)
        if lowered.startswith("linear:"):
            body = text.split(":", 1)[1]
            parts = body.split(",")
            if len(parts) != 2:
                raise ValueError(f"linear cost model needs 'linear:A,B', got {spec!r}")
            try:
                a, b = (int(part) for part in parts)
            except ValueError as exc:
                raise ValueError(
                    f"linear cost model needs integer A and B, got {spec!r}"
                ) from exc
            return cls(text, a=a, b=b)
        if lowered.startswith("table:"):
            path = text.split(":", 1)[1]
            return cls(text, table=_load_table(path))
        raise ValueError(
            f"unknown cost model {spec!r}; use uniform, position, "
            "linear:A,B, table:FILE or qwen3coder30b_blksz_16"
        )

    def cost(self, block_index: int) -> int:
        """返回第 block_index 个缓存块的相对重算代价。"""
        index = int(block_index)
        if index < 0:
            raise ValueError(f"block index must be non-negative, got {block_index}")
        if self._table is not None:
            return self._table[min(index, len(self._table) - 1)]
        return self._a + self._b * index

    @property
    def position_dependent(self) -> bool:
        return self._table is not None or self._b != 0

    def __str__(self) -> str:
        return self._spec


def _load_table(path: str) -> List[int]:
    if not path:
        raise ValueError("table cost model needs a file path")
    values: List[int] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            try:
                value = int(text)
            except ValueError as exc:
                raise ValueError(
                    f"cost table {path!r} has a non-integer line: {line.rstrip()!r}"
                ) from exc
            if value < 0:
                raise ValueError(f"cost table {path!r} has a negative value: {value}")
            values.append(value)
    if not values:
        raise ValueError(f"cost table {path!r} is empty")
    return values


__all__ = ["CostModel"]
