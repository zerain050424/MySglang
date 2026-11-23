"""兼容导出层，实际实现位于 sim.core.allocator。"""

from sim.core.allocator import OutOfTokens, TokenAllocator, tree_tokens

__all__ = ["OutOfTokens", "TokenAllocator", "tree_tokens"]
