# 未执行的优化想法（随实验补充，含已定位的机制证据）

## F1（最高优先）：once-read 节点离开 one-shot 带

**机制证据**（quick_wt1 / fast_mid 双重复现的「均匀税」）：
- 会话前 4 轮（`turn_count < MULTI_TURN_MIN=5`）创建的节点被标 QTRANS/one-shot，
  在扫描流冲刷下活不过创建者自己的下一次重读（alfred 6.1s、chat 63s 间隔都被清）。
- 表现：hc 下所有多轮 task 每个 turn 桶均匀掉 ~20pp（LRU t1 53.4% → hc 22.3%）；
  ShareGPT t1（跨会话共享问候前缀）39.7% → 7.5%；codeparrot 68.2% → 36.0%。
- 上游 SLRU 用 `hit_count>=2`（创建者自己的 turn2/turn3 重读就够了）正是这个道理。
- victim_src 佐证：quick_wt1 one_shot=157160 / fast_mid one_shot=992943 ——
  大量被反复重建的会话内容在 one-shot 带里死亡。

**改动**（`evict_policy.py::is_one_shot`，一行语义 + 记账转移）：
```python
# 现判定：TRANS/STRUCT 且 share<share_min → one-shot
# 加一条件：node.hit_count == 0 才算 one-shot；
# 即被任何请求匹配过一次（读命中或 insert 路径前缀匹配，_inc_hit_count 两个来源
# 都是「内容被再次用到」的证据）的节点离开优先淘汰带。
```
- `hit_count` 的自膨胀已有防护（chunked insert `chunked=True` 不 bump；
  decode 后 prompt 重插入也带 chunked=True，见 radix_cache L605-630 注释）。
- QSTRUCT 分裂继承 hit_count（L905），语义连续。
- **记账转移**：one-shot→regular 的节点从「occupancy 跳过」变为计入
  `random` 队列（QTRANS.queue_name=="random"）。配额仲裁只对
  struct/agentic/chat（quota_ranking 过滤 random），random 行仅 profile 展示——
  首次晋升时在 record_hit 补 `occupancy.insert("random", tokens)`
  （guard：hit_count==1 时插一次），evict 对称扣减；random 行允许暂时漂移。
- 预期：alfred/sharegpt/codeparrot 早期轮 + 跨会话共享前缀存活 →
  codeparrot 33→~70、sharegpt t1 7.5→~40；scan 内容（真一次性）不受影响。

## F2：cap 仅在跨队列饥饿时强制（Min-Max 水填充的替代仲裁）

**机制证据**：quick_wt1 victim_src `quota:agentic=445536`，agentic occ 38409 > cap
30000，同时 chat occ 仅 3602（cap 18000 闲置）、targets 合计 < pool；
fast_mid `quota:agentic=4008790`，targets 96k < pool 120k。
cap 是硬上限：即使其他队列不要空间、池有空闲，也把占大头的队列强拆到 cap。
（demandcap 消融在 mid 上无效是因为该 cell 有分类伪影 + M1 主导，不代表 F2 无效。）

**改动方向**（`quota_selector.py`）：队列 q 超过 cap 时，仅当
(a) 池已满 且 (b) 存在其他队列占用 < 其 reservation（真饥饿）时才触发配额淘汰；
否则超 cap 部分按队列内 victim 顺序正常参与全局淘汰。
或更彻底：把 caps 从「硬上限」改为「争用时才生效的目标份额」。

## F3（次级）：保护段降级 / 会话活跃度判据
（同旧会话记录；等 F1/F2 落地后重新评估优先级。）

## 已搁置
- exp3 压缩时间 cell（arrival/wait 0.1）：会把 chat 轮间压到 15s 以下触发
  GAP_SPLIT_S 分类伪影 → 已从注册表移除，exp3 只做全保真最终验证（3.3h/臂）。
