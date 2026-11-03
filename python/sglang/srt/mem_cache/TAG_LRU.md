# 标签版 TinyLRU（TinyLRU 分支默认实现）

一句话：**分类即晋升**。负载分类器已经知道「这个会话是不是模板/共享前缀型」，
于是节点归属不需要再靠三区状态机推断，直接用两条证据贴标签：

```
会话类型 == STRUCT（模板/共享前缀型）:
    share_sessions(node) >= struct_share_min (默认 2)  -> QSTRUCT  # 跨会话共享，LRU 淘汰
    share_sessions(node) <  struct_share_min:
        rids(node) >= struct_share_min                 -> QCHAT    # 会话内复用（默认）
        （local_hot="one_shot" 时 -> QTRANS）                      # 规则原样：进 one-shot 带
    其余                                               -> QTRANS   # 真一次性，进 one-shot 带
其他会话类型（chat / agentic / trans）:
    沿用会话类型映射（QCHAT / QAGENT / QTRANS），不参与晋升
```

实现：`tag_lru.TagLruPolicy`（约 200 行），由 `SimpleHybridStrategy.tagState` 持有，
`radix_cache` 在**命中与插入时**调用。替代关系：`tiny_lru.policy = "tag_lru"`（默认）
vs `"wtinylru"`（三区版，保留用于对照）。

## 1. 与三区版的关键差别

| | 三区版 `wtinylru` | 标签版 `tag_lru` |
|---|---|---|
| 晋升条件 | 拓扑分叉 / 频数 + **必须先是可淘汰叶子** | 只需「分类器判为 STRUCT + 第二个会话命中」 |
| 队列状态 | window/probation/protected/parked 四索引 | 无（标签本身就是状态） |
| 准入 / aging / 预算 | 有（实测基本不触发） | 无 |
| 淘汰顺序 | 全局 `victim_key` → 插件改写为队内区头 | `victim_key` 里直接编码类间 rank + 类内 LRU |
| 观测 | `victim_src.qstruct`、`struct.snapshot()` | `evicted_by_tag`、`tag.snapshot()` |
| 代码量 | 状态机 695 行 + 适配器 + 8 钩子 | ~200 行 + 3 个调用点 |

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

`_evict_one` 的 one-shot 级联删除必须停在共享前缀上。三区版用的是
`structState.owns(parent)`；标签版没有状态机，于是统一到策略方法：

```python
def owns_struct(self, node):
    if self.structState is not None: return self.structState.owns(node)   # 三区版
    if self.tagState   is not None: return self.tagState.owns(node)       # 标签版 = tag==QSTRUCT
    return False
```

## 4. `local_hot`：会话内热点怎么处置（实测的关键）

「会话共享 == 1」既可能是**真一次性内容**，也可能是**同一会话多轮复用的历史**。
两者的最优处置相反：

- 真一次性 → `QTRANS` → one-shot 硬优先带（先清）；
- 会话内复用 → 不能进 one-shot 带，否则每轮都被清空重算。

所以给了开关 `local_hot`：

| 值 | 会话共享==1 的节点 | 实测（ShareGPT / 合成多轮 struct，pool=0.3） |
|---|---|---|
| `"chat"`（默认） | `QCHAT`（rank 1，按复用概率曲线） | **78.10%** / **82.46%** |
| `"one_shot"`（规则原样） | `QTRANS`（rank 0，one-shot 带） | 54.61% / 75.67% |

差别达 **+23.5pp / +6.8pp**，因此默认必须是 `chat`；`one_shot` 保留为消融（实验臂 A16）。

## 5. 实测（离线 L1，真实 `SimpleHybridRadixCache`，300 会话，pool/工作集=0.3）

| 负载 | 臂 | 配置 | 命中率 | 会话内历史保留 | 后悔率 |
|---|---|---|---|---|---|
| ShareGPT | A0 | LRU | 66.90% | 0.736 | 0.0638 |
| ShareGPT | A2 | `tiny_lru=off` | 49.26% | 0.551 | 0.1810 |
| ShareGPT | A3 | 三区版（hybrid+预算） | 49.74% | 0.593 | 0.1921 |
| ShareGPT | A16 | 标签版 `local_hot=one_shot` | 54.61% | 0.738 | 0.1393 |
| ShareGPT | **A18** | **标签版默认** | **78.10%** | **0.892** | **0.0202** |
| synth p2048/b10 | A0 | LRU | 81.09% | – | 0.0190 |
| synth p2048/b10 | A2 | off | 80.37% | – | 0.0263 |
| synth p2048/b10 | A3 | 三区版 | 80.37% | – | 0.0263 |
| synth p2048/b10 | A16 | 标签版 one_shot | 75.67% | – | 0.0732 |
| synth p2048/b10 | **A18** | **标签版默认** | **82.46%** | – | **0.0053** |

结论：标签版（默认 `chat`）同时优于 LRU 与三区版；ShareGPT 上比 LRU **+11.2pp**、
比三区版 **+28.4pp**；合成共享前缀负载比 LRU **+1.4pp**、比三区版 **+2.1pp**。
`one-shot` 占比也从"吃掉 90%+ 压力"降到 34%（ShareGPT 143k/297k），说明
「把会话内复用从一次性带里救出来」正是收益来源。

## 6. 参数（`TinyLRUParams`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `policy` | `"tag_lru"` | `"tag_lru"` / `"wtinylru"`（三区版）/ `"off"` |
| `struct_share_min` | 2 | 会话共享数达到该值才晋升 QSTRUCT（=「>1 个会话」） |
| `local_hot` | `"chat"` | 会话共享==1 的处置：`"chat"` / `"one_shot"` |
| `share_metric` | `"hybrid"` | 标签版只用其中的 `"session"` / `"rid"` 两档（hybrid→session） |

`tag_lru` 与 `tlru.enabled` 互斥（两者都改写 QAGENT/QSTRUCT 的候选顺序，后者会静默遮蔽），
启动期直接 `ValueError`。

## 7. 已知限制

1. `share_sessions` 依赖 64-bit bloom（4 函数），约 8 个会话后误判率上升；误判方向是**少计**（保守）。
2. 标签只增不减：`share_sessions` 单调，QSTRUCT 事实上不会退回 QTRANS（没有 aging）。
   目前靠「节点被淘汰即消失」自然收敛；若长跑出现陈旧模板，需要补一个时间闸。
3. `local_hot="chat"` 借用 `QCHAT` 标签表达「会话内复用」，会让 profile 的 workload 归属偏 chat；
   语义上更准确的做法是新增一个 `QLOCAL` 标签（需要同步 `task_type_of_tag` / `is_one_shot` /
   `workload_label_of` / alpha keys），本次未做。
