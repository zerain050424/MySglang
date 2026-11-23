"""兼容导出层，实际实现位于 sim.workload.session_data。"""

from sim.workload.session_data import load_task, load_tasks

__all__ = ["load_task", "load_tasks"]
