"""HeterCache 的全部组件：按队列分带、按已兑现价值密度（VD）的跨队列淘汰。

- **utils**：会话类型 / 节点标签 / 参数组 / `SessionRegistry` / 探针，是其余模块的公共底座。
- **selector**（纯函数）：`queue_order` / `pick_victim` / `struct_tier`，只做判据与排序。
- **occupancy** / **value** / **node_share**：三本账（可调度占用、已兑现价值、跨会话共享数）。
- **override**：队内候选改写器（RR / MT / MultiAgentic）。
- **radix_cache**：`HeterCacheRadixCache`，执行层（内存回收、树拓扑、级联修剪）。
- **bench_bridge**：通往 `Heter-Bench/` 观测实现的唯一接缝。
"""
