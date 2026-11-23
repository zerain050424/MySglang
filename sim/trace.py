"""兼容导出层，实际实现位于 sim.workload.trace。"""

from sim.workload.trace import Session, Turn, by_class, max_total_len, synthetic_sessions

__all__ = ["Session", "Turn", "by_class", "max_total_len", "synthetic_sessions"]
