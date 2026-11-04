"""W-TinyLRU 效果实验工具包（离线回放 / 真机埋点 / 分析 / 门控）。

实现文档：``python/sglang/srt/mem_cache/TAG_LRU.md``
本包**不修改**框架代码：所有观测都通过子类、monkeypatch、或运行时属性实现。

模块分工
--------
arms.py         实验臂（A0-A5）矩阵，单一事实来源
simclock.py     事件驱动仿真时钟（替换 sglang 模块内的 time）
trace.py        session_data.jsonl 读取与到达调度生成
synth.py        合成「共享前缀强度阶梯」trace（场景 A）
prefix_stats.py 共享前缀 LCP 统计（M1 指标的基准）
backends.py     回放后端：RealBackend（真 SimpleHybridRadixCache）/ FakeBackend（smoke）
metrics.py      保留度 / 后悔率 / 保护浪费率
replay.py       离线回放驱动（L1）
run_matrix.py   离线矩阵并发执行器
struct_profile.py 真机 struct 状态 JSONL 埋点（monkeypatch）
serve.py        python -m wtl.serve 启动服务并安装埋点
analyze.py      汇总表 + 图 + results.csv
gate.py         Phase-0 前置门控（§10）
"""

__version__ = "1.0.0"
SCHEMA_REPLAY = "wtl.replay/1"
