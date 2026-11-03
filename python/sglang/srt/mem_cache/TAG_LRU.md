# 标签版 TinyLRU（Q_struct 队内策略）

一句话：**分类即晋升**。负载分类器已经知道「这个会话是不是模板/共享前缀型」，
于是节点归属不需要队列状态机去推断，直接用两条证据贴标签：

```
会话类型 == STRUCT（模板/共享前缀型）:
    share_sessions(node) >= struct_share_min (默认 2)  -> QSTRUCT   # 跨会话共享 → 类内 LRU
    share_sessions(node) <  struct_share_min:
        rids(node) >= struct_share_min                 -> QCHAT     # 会话内复用（默认）
        （local_hot="one_shot" 时 -> QTRANS）                       # 规则原样：进 one-shot 带
    其余                                               -> QTRANS    # 真一次性，进 one-shot 带
其他会话类型（chat / agentic / trans）:
    沿用会话类型映射（QCHAT / QAGENT / QTRANS），不参与晋升
```

实现：`tag_lru.TagLruPolicy`（约 200 行），由 `SimpleHybridStrategy.tagState` 持有，
`radix_cache` 在**命中与插入时**调用；`tiny_lru.policy ∈ {"tag_lru"(默认), "off"}`。

## 1. 为什么它替代了队列状态机

上一版是「三区（window/probation/protected）+ 频数准入 + aging + protected 预算」的状态机，
实测下来三件事让它收益有限（该实现已从本分支移除，仅保留结论）：

1. **准入与 aging 基本不触发**：window 条目一旦成为叶子往往立刻被淘汰，堆不到
   `window_entries=64`；`epoch_events=8192` 对几百~几千请求的 trace 也到不了
   （实测 `admit=reject=ages=0`）。
2. **成员必须先变成可淘汰叶子**：而「成为叶子」的时刻恰好是它即将被淘汰的时刻，
   于是 protected 很难积累，预算机制几乎没有触发机会。
3. **瓶颈根本不在队内顺序**：`victim_src.one_shot` 显示一次性硬优先带吃掉了 90%+ 的
   淘汰压力，队内策略根本没有决策机会。

标签版把这三件事一起绕开：晋升发生在**分类/命中**时，不依赖「成为叶子」；
标签直接进入全局排序键，不再需要插件与三区索引。

## 2. 淘汰顺序（`SimpleHybridStrategy.victim_key`）

```python
tokens = _node_num_tokens(node)
if tagState is not None:
    task = _task_type_of(node)
    if task == STRUCT_SINGLE_TURN:  return (2, node.last_access_time, -tokens)  # 共享前缀：类内 LRU
    if task == RANDOM_SINGLE_TURN:  return (0, node.last_access_time, -tokens)  # 一次性
    return (1, self._reuse_score(node), -tokens)                               # chat/agentic 原曲线
```

**为什么 QSTRUCT 必须显式给 LRU**：`_reuse_score` 对 STRUCT 用的是
`alpha * (1 - end_offset/max_offset)`（HeterCache 的位置打分）。长度 ≥ `max_offset`（默认 2048）
的共享前缀会被算成 0 分，反而**最先**被淘汰——与「保护共享前缀」正好相反。

## 3. 死链上溯闸

one-shot 收割会沿父链级联删除，必须停在共享前缀上，否则会把模板前缀一起删掉。
三区版曾用 `structState.owns(parent)`；现在统一为：

```python
def owns_struct(self, node) -> bool:            # SimpleHybridStrategy
    return self.tagState is not None and self.tagState.owns(node)   # 即 tag == QSTRUCT
```

被 `simple_hybrid_radix_cache._evict_one` 的级联循环调用；门控 G5 专门量它（实测 0 token）。

## 4. `local_hot`：会话内热点怎么处置（关键参数）

「会话共享 == 1」既可能是**真一次性内容**，也可能是**同一会话多轮复用的历史**，
两者最优处置相反：前者该进 one-shot 带先清，后者清了就要重算。

| 值 | 会话共享==1 的节点 | ShareGPT 命中率 | 合成多轮 struct 命中率 |
|---|---|---|---|
| `"one_shot"`（规则原样） | `QTRANS`（rank 0，one-shot 带） | 54.61% | 75.67% |
| **`"chat"`（默认）** | `QCHAT`（rank 1，复用概率曲线） | **78.10%** | **82.46%** |

（离线 L1，300 会话，pool/工作集=0.3；同日对照 LRU：66.90% / 81.09%）

差别 +23.5pp / +6.8pp，所以默认 `chat`；`one_shot` 保留为消融（实验臂 A4）。

## 5. 实测（离线 L1，真实 `SimpleHybridRadixCache`，300 会话，pool/工作集=0.3）

| 负载 | 臂 | 配置 | 命中率 | 会话内历史保留 | 后悔率 |
|---|---|---|---|---|---|
| ShareGPT | A0 | LRU | 66.90% | 0.736 | 0.0638 |
| ShareGPT | A2 | `tiny_lru=off` | 49.26% | 0.551 | 0.1810 |
| ShareGPT | A4 | 标签版 `local_hot=one_shot` | 54.61% | 0.738 | 0.1393 |
| ShareGPT | **A3** | **标签版默认** | **78.10%** | **0.892** | **0.0202** |
| synth p2048/b10 | A0 | LRU | 81.09% | – | 0.0190 |
| synth p2048/b10 | A2 | off | 80.37% | – | 0.0263 |
| synth p2048/b10 | A4 | 标签版 one_shot | 75.67% | – | 0.0732 |
| synth p2048/b10 | **A3** | **标签版默认** | **82.46%** | – | **0.0053** |

一次性带占比从 90%+ 降到 34%（ShareGPT：qtrans 143k / 总 297k）——把会话内复用从
一次性带里救回来，正是收益来源。

## 6. 参数（`TinyLRUParams`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `policy` | `"tag_lru"` | `"tag_lru"` / `"off"` |
| `share_metric` | `"session"` | 共享判定口径：不同会话数 / `"rid"`（消融） |
| `struct_share_min` | 2 | 会话共享数达到该值才晋升 QSTRUCT（=「>1 个会话」） |
| `local_hot` | `"chat"` | 会话共享==1 的处置：`"chat"` / `"one_shot"` |

`tag_lru` 与 `tlru.enabled` 互斥（两者都改写 QAGENT/QSTRUCT 的候选顺序，后者会静默遮蔽），
启动期直接 `ValueError`。

## 7. 已知限制

1. `share_sessions` 依赖 64-bit bloom（4 函数），约 8 个会话后误判率上升；误判方向是**少计**（保守）。
2. 标签只增不减：`share_sessions` 单调，QSTRUCT 事实上不会退回 QTRANS（没有 aging）。
   目前靠「节点被淘汰即消失」自然收敛；若长跑出现陈旧模板，需要补时间闸。
3. `local_hot="chat"` 借用 `QCHAT` 标签表达「会话内复用」，会让 profile 的 workload 归属偏 chat；
   语义上更准确的做法是新增 `QLOCAL` 标签（需同步 `task_type_of_tag` / `is_one_shot` /
   `workload_label_of` / alpha keys），本次未做。
