"""HeterCache 组件包：分带过滤 + 已兑现价值密度（VD）跨队列淘汰。

- utils：会话类型、节点标签、参数组与 SessionRegistry。
- selector：纯策略裁决（queue_order / pick_victim / struct_tier）。
- occupancy / value / node_share：队列占用、价值密度与跨会话共享数账本。
- regret_gate：跨队列后悔弃权门。
- override：队内与全树候选改写器（ReuseScoreEvict / ChainEvict / PricedFallback）。
- radix_cache：HeterCacheRadixCache，负责单节点释放与树观测。
- bench_bridge：对接 Heter-Bench 观测实现的桥接层。
"""
