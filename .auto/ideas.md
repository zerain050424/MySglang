# 未执行的优化想法

- **弃用内容的会话活跃度优先**（针对 25k 极端压力下 A3 落后 A0 0.7pp）：
  带内排序现在是纯 LRU，可以再加一维「所属会话是否已闲置」——先清已结束会话的内容。
  难点：需要一个「闲置」判据，任何固定秒数都会变成对 trace 的调参，
  可考虑用 `SessionState.gaps` 的中位数自校准（`now - last_active > median(gaps)`）。
  只有在主指标仍有 headroom 时才值得做；当前主指标已饱和。
- **保护段降级（demotion）**：`share_sessions` 只增不减，protected 只进不出。
  真实负载下若某模板长期不再被复用，应降回带内。需要「保护年龄」字段。
- **`pick()` 实际不可达**：只要请求流持续产生一次性内容，带就永不为空，
  `elif leaves:` 分支与 `victim_key` 在实践中不执行（本 trace `slru_picks` 恒 0）。
  若后续要考察「protected 溢出」场景，需要专门的负载（受保护集 > 池容量）。
- **`_reuse_score` 的 0 分陷阱**：struct 节点 `end_offset >= max_offset(2048)` 时位置分为 0，
  在 `victim_key` 里会变成首选。当前因 `pick()` 不可达而无害，但一旦 `elif` 分支可达就会咬人。
  候选修法：非一次性节点不要用位置打分，直接用 `(1, last_access_time)`。

## 已搁置的下一靶心（未执行）

主指标 `template_intact_min` 已到天花板（93.33 = 1−64/960）。若要继续优化，
建议把主指标换成 **`hit_min_over_pools` = min(A3 总命中率) over {25k, 250k}**
（当前 50.77，A0 同口径 51.49，有 0.7pp headroom），并以
`template_intact_min >= 93` 作为硬约束。对应假设：带内改 LRU 后预热已被覆盖，
`share_min` 从 2 提到 3 可缩小保护段、给会话内私有内容腾地方。
`.auto/remote_eval.py` 曾改到一半（POOLS=[25000,250000] + 新增 5 个指标）后回滚，
未执行测量。
