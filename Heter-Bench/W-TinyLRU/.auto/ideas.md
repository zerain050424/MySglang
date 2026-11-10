# 回放器待验证的想法

- **按 session 统计淘汰归因**：现在的 `victim_src` 只按队列聚合。给 `charge`
  加一个 per-session 计数器，就能回答「一条 40 轮会话的 KV 到底被谁删掉的」。
- **分叉拓扑差异**：真机 struct 命中 8.49M vs 回放 0.73M。怀疑回放里同一前缀
  没有形成 `len(children)>=2` 的分叉（`_insert_helper` 只在插入新节点时检查），
  而真机因为并发交错更频繁地分叉。可对比两边的节点数 / 平均分支数。
- **账本存活数**：真机 586 vs 回放 37。先确认 `sweep` 的 `empty_grace`/`idle_ttl`
  在仿真时钟下的行为是否符合真机（真机 30s 一次 profile、idle_ttl=3600s）。
- **`--inflight-reserve` 的物理化**：现在是 `并发数 × 94` 的拟合值。应改成
  「在途请求数 × 平均未插入 token 数」，由 e2e 与吞吐算出，去掉拟合。
- **mid 档补一条真机基线**：现在只有 smoke / full 两档有真机结果，mid 无法校验
  模型是否在中间规模也成立。

- **（最高优先）真机节点级拓扑 dump**：`SGLANG_NODE_LABEL_DUMP=<path>` 会让
  `node_label_logger` 逐节点输出标签/深度，本轮所有关于 struct 命中 8.49M、
  分叉、QSTRUCT 黏性的推断都只能靠聚合数字反推，无法证伪。下一次真机跑
  hc_full 时带上这个环境变量，就能直接对拍回放的树形状。
- **响应段的真实去向**：两处证据互相矛盾——(a) 逐请求看 `cached ≈ 上一轮 prompt`，
  说明 response 段不驻留；(b) 真机 struct 命中 8.49M ≈ 2 万条 alfred 请求 × 420 token，
  说明存在大量 QSTRUCT 标签，而 alfred 会话之间没有公共前缀（实测最长 3 token），
  只能来自会话内分叉。分叉要求某个节点有 ≥2 个子节点，这与 (a) 矛盾。
  解开它需要节点级 dump。
- **hold-ms（信号量额外持有 e2e）已试**：full/lru 从 60.10 掉到 52.84，discard。
  说明配置里的 mu/sigma 已经包含服务时长。
- **mid 档真机基线**：尝试启动时与另一个并行实验抢了 30000 端口，已终止；
  等对方跑完再补 mid 档，作为第二个标定点（防止只拟合 full 一档）。
