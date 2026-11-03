# 结构复用队列（Q_struct）的 W-TinyLRU 队内淘汰设计

范围：**仅 Q_struct 队列内部**的淘汰策略
非目标：跨队列仲裁、队列容量预算、alpha 配额重分配、淘汰后频数记忆（跨队列设计下一期）

---

## 0. 一句话概述

在 `SimpleHybridStrategy` 内部挂一个 **只认 Q_struct 成员** 的子策略 `StructQueuePolicy`，
用成员判定 M3 + 三区（window/probation/protected）+ 频数准入 + 事件驱动 aging
替换当前位置项 + LRU的队内排序，并在全局 victim 落在 struct 成员上时**委托**它给出更优 victim。
所有 KV 释放仍然只由 `SimpleHybridRadixCache._evict_one()` 执行。

| 需求 | 机制 | 生效点 |
|---|---|---|
| 按频数保持驻留 | 节点级 distinct-rid 计数 `share_count` + 分档 | 三区排序、protected 门限 |
| 单次对文档的询问被删去 | 准入比较失败 → 成员**降级出队**为 `QTRANS` | window 出队 |
| 高频共享前缀可以驻留 | 拓扑成员 M3 + 降级继承 + 上溯级联闸 | `_update_leaf_status` 钩子 |

---

## 1. 动机：现有实现的三个坑

### 成员集合单调增长，队列没有边界

```python
# radix_cache.py:838
def _record_node_hit(self, node, access_time):
    node.hit_count += 1
    if node.queue_tag != SimpleHybridQueueTag.QSTRUCT:      # 只升不降
        node.queue_tag = QAGENT if delta < 8.0 else QCHAT
```

`_update_leaf_status()`（`radix_cache.py:995`）只维护 `evictable_leaves`，**完全不碰 tag**。
于是分支节点的子树被其他队列删光后，它自己成了叶子却仍然带着 `QSTRUCT`。
不重新定义成员，就没有队列也没有window 容量」。

### struct 队列的可淘汰叶子是别人制造出来的


struct 成员只有在子链被删光之后才降级成叶子。**队列自己不产生淘汰机会**，
因此必须在 `_update_leaf_status` 挂事件钩子，而不能等到缺页时才去扫。

### 坑 3 — `hit_count` 不是共享次数

一次 chunked 分块、一次重试、一次同会话多轮，都会给路径上的每个节点 +1
而单次对文档的询问恰恰会因为分块或重试拿到 2~3 次命中，被误判成高频。
**频数口径必须是不同请求数**，这是整个设计的地基。

---

## 2. 术语与不变量

### 2.1 术语

| 术语 | 定义 |
|---|---|
| 成员（member） | 满足成员谓词 M3 的树节点（见 §3） |
| 可淘汰成员 | 成员 ∧ `value is not None` ∧ `lock_ref == 0` ∧ 无未驱逐子节点 |
| 挂起（parked） | 是成员但当前不可淘汰（有子节点或被 lock） |
| 活跃区（region） | window / probation / protected，**只装可淘汰成员** |
| `share_count` | 穿过该节点的**不同请求（rid）数**，u8 饱和 |
| epoch | aging 的计数纪元，按 struct 淘汰/入队事件计数推进，与 wall-clock 无关 |

### 2.2 不变量（可在调试构建里断言）

1. **INV-1**：任一成员同时只存在于 {parked, window, probation, protected} 之一。
2. **INV-2**：三个 region 的并集 ≡ 可淘汰成员集合；无重复、无 stale、无已驱逐节点。
3. **INV-3**：只有 `_evict_one()` 可以释放 KV；struct 策略从不调用 `free_segment` / `_delete_leaf`。
4. **INV-4**：淘汰必须单调推进。准入可以拒绝，淘汰不能拒绝；淘汰只有顺序。
5. **INV-5**：非 struct 节点永远不被 struct 策略触碰（读也只读 `parent` 链用于判定）。
6. **INV-6**：`hit_count` 语义不变（其他策略/指标依赖），`share_count` 是独立字段。

---

## 3. 成员判定：M3

### 3.1 谓词

```
is_member(node) :=
    node is not root
    and not node.evicted
    and (
        len(node.children) >= 2                      # 分叉点
        or node.struct_ever_branch                   # 曾经分叉过（黏性结构证据）
        or (share_count(node) >= 2 and is_member(node.parent))   # 被共享的主干
    )
```

- 第一项就是现有 `_insert_helper` 的晋升条件（`radix_cache.py:955`）。
- 第二项补上「分叉点上游的主干」：这些节点是 `_split_node` 产生的前缀半段
  （`radix_cache.py:850`，它继承 `queue_tag`），正是「高频共享前缀」的物理载体。
  判据用 `share_count >= 2` 而不是「有分叉后代」，避免把「一次请求 + 一次重试」也算共享。
- 迭代判定的终止条件：沿 parent 一直走到根也算成立（共享到序列开头的整条前缀）。
- **`struct_ever_branch`（实现期新增的黏性标记）**：节点只要曾经出现过 `len(children) >= 2`，
  就保留成员身份，即使后来子链被删光、`share_count` 仍然很低。没有这个标记，
  窗口区在 M3 下**永远不可达**：成员若 `share<2` 必然还有子节点（→ parked），
  一旦子节点删光就同时失去成员身份，直接退队——准入比较与「拒绝降级」整条路径都不会执行。
  标记只在两种情况下清除：准入拒绝（§5.2）与 aging 退队（§6.3）。

### 3.2 入队（`on_member_promote`）

| 触发 | 动作 |
|---|---|
| 插入新段使 `parent.children` 从 1 → 2 | `parent` 打上 `struct_ever_branch` 并晋升成员；有子节点 → parked |
| 可淘汰叶子命中使 `share_count` 从 1 → 2，且父链是成员 | 该节点晋升成员；若可淘汰 → 入 probation |
| 父成员降级（§3.4） | 见继承表 |
| `_split_node` 产生的后缀半段 | 若原节点是成员 → 后缀半段成为成员；可淘汰则入**原区**（保持区归属，不重新观察） |

### 3.3 退队（坑 1 的解）

```
退队 = 准入拒绝（§5.2，立即发生）
     | aging 判定结构证据与共享证据都没了（§6.3）
  → 移出 region、清 share_count、bloom 归零、清 struct_ever_branch、queue_tag 降为 QTRANS
```

**区别于「离开索引」**：`_leave_struct()` 只把节点移出 region，**保留** `share_count`
（频数要在节点成为成员之前就开始累积：一个前缀被 3 个请求用过之后才分叉，它一入队就该
直接进 protected）。真正清计数的是 `_retire()`，只在准入拒绝与 aging 时调用。

**退队判定只在两个时刻求值**：`_update_leaf_status(node)` 之后、以及 epoch 衰减之后（§6）。
不做全树扫描。

### 3.4 降级继承（需求 4）

父成员因唯一子链被删而变成可淘汰叶子时，按继承表落区，**不重置 `share_count`**：

| 父节点原状态 | 落区 |
|---|---|
| protected（`share_count >= K`） | protected 尾（不再观察） |
| probation | probation 头 |
| window | window 尾（继续观察） |
| parked（首次入区） | `share_count >= K` → protected 尾；`== 2` → probation 头；`<= 1` → window 尾 |

理由：同一前缀的共享证据已经存在，重走一遍观察期只会让整条模板链在每轮淘汰后反复「重新证明自己」。

---

## 5. 三区与准入（需求 2、3 的核心）

### 5.1 区含义

| 区 | 语义 | 入区条件 | 淘汰端 | 区内顺序 |
|---|---|---|---|---|
| window | 观察期：成员刚变为可淘汰，或从未被第二个请求共享 | `share_count <= 1`（或继承表指定） | head | FIFO |
| probation | 证据成立：恰好被 2 个不同请求共享 | `share_count == 2`，或准入通过 | head | LRU（命中移尾） |
| protected | 高频共享前缀 | `share_count >= K`（默认 K=3） | head（兜底） | LRU |

> **deviation 说明**：教科书 W-TinyLRU 的 window 按 token 容量比例给（1%~40%）。
> 本方案在没有队列预算的前提下**用条目数配额**（`window_entries`，默认 64），
> 只表达刚入队 vs 已验证的区分，不做 token 预算，也不做 hill-climbing（无跨队列预算时调 window 无意义）。

### 5.2 准入比较（只在 window 溢出时发生）

```
on_window_overflow():
    cand = window.head                      # 最老的观察者
    victim = probation.head if probation else protected.head
    if victim is None:                      # 队内没有可比较对象
        promote(cand); return
    if share(cand) >= 2:                    # 已具备共享证据，不再拒绝（否则破坏 INV-2）
        promote(cand); return
    if share(cand) > share(victim):
        promote(cand)                       # cand → probation 尾
    else:
        demote(cand)                        # 拒绝：交回上层
```

- `promote(cand)` 后在 `share >= K` 时直接跳 protected，不必在 probation 停留。
- `share(cand) >= 2` 的候选**无条件晋升**：它已经是「被两个请求共享」的证据，
  拒绝会让它成为「成员但不在任何区」的孤儿（破坏 INV-2）。拒绝只对 `share <= 1` 的观察者生效。
- `demote(cand)` 的动作**只有三件**：移出 region、`queue_tag = QTRANS`、记账 `struct_demoted`。
  **不释放 KV、不删节点**（INV-3）。
- `share(cand) == share(victim)` 时的判定：默认拒绝（保守，宁留老的），可配 `admission_tie = "reject" | "admit"`。

### 5.3 降级之后发生什么（闭环）

```
demote(cand)  →  cand.queue_tag = QTRANS
             →  SimpleHybridStrategy.is_one_shot(cand) == True    # evict_policy.py:305
             →  下一轮全局选择器按 one_shot_key 立即收割它         # simple_hybrid_radix_cache.py:49
             →  _evict_one(cand) 释放 KV
             →  父节点（可能是 protected 成员）失去最后子节点 → on_became_leaf → 按 §3.4 落区
```

即：**struct 队列拒绝一个成员 = 把它踢出保护伞，让现有的 one-shot 带执行处决**。
这条链完全复用现有机制，不需要新枚举值、不需要新的释放路径。

### 5.4 为什么不需要大小归一（重要）

TinyLFU 假设 entry 成本相同，radix 节点长度从 1 token 到 20k token，直觉上需要 GDSF 式的
`freq / bytes` 归一。这里不需要，原因：

- 存活一个长度 L 的片段，成本 = L tokens；
- 它的期望收益 = `s × L` 个命中 token（被 s 个未来请求整段命中）；
- 密度 = `sL / L = s`，与 L 无关。

`share_count` 本身就是密度，**只要准入单位是「会被整段命中的段」**。
反例只出现在「段首不会命中、只命中中段」的情形，而 radix 前缀命中天然是从段首开始匹配的，
所以该假设成立。

> 留一个开关 `admission_metric = "share" | "share_per_token"`（默认 `"share"`）。
> 如果实验发现长尾文档段仍然占优，再用 `share_per_token` 做对照——但按上面的推导它不该赢。

### 5.5 淘汰顺序（无准入判断）

```
victim():
    for region in (window, probation, protected):
        if region:
            return region.head
    return None                      # 队列暂时没有可淘汰成员
```

- 只有顺序，没有拒绝（INV-4）。protected 的 head 是**兜底**，即「队内已无更差选择」。
- `victim()` 返回 `None` 时，上层必须回退到原始全局 victim 选择，**不能 break、不能自旋**。

---

## 6. 频数口径与 aging

### 6.1 `share_count` 的维护

节点新增三个字段（`TreeNode.__init__`，`radix_cache.py:262`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `share_count` | u8（饱和 255） | 不同 rid 计数 |
| `rid_bloom` | u64 | 已计入的 rid 哈希，4 函数 bloom |
| `struct_region` | u8 / None | `0=parked 1=window 2=probation 3=protected` |

```
on_hit(node, rid):
    if node is not member: return
    h = hash(rid)                                     # 已有 rid 可直接复用 64-bit 哈希
    if bloom_maybe_contains(node.rid_bloom, h): return  # 同一请求重复命中：不计
    node.rid_bloom |= bloom_bits(h)
    node.share_count = min(255, node.share_count + 1)
    touch_region(node)                                 # 区内则移到尾部
```

- bloom 大小 64bit / 4 函数，饱和在 ~8 个 rid 后误判率上升；对 `K <= 3` 的门限判定，误判方向是
  **少计**（把新 rid 误判为已计），即偏保守，不会把单次文档误升为高频——这是可接受的偏差方向。
- 需要精确的实验环境可把 bloom 换成 `u64[2]`，内存代价 +8B/成员。

### 6.2 分裂时的 bloom

`on_split(prefix_half, suffix_half)`：前缀半 `rid_bloom = child.rid_bloom`、`share_count` 继承；
后缀半 `rid_bloom = child.rid_bloom`（同一批 rid 也经过它）、`share_count` 继承。
两边都继承是安全的：分裂不改变「哪些请求经过了这段前缀」。

### 6.3 aging（需求 5）

**事件驱动，不用 wall-clock**，以便回放可复现。时钟由三类事件构成：
**每个不同 rid 的首个命中**（一次请求只记 1 次，不随路径长度膨胀）+ 入区（promote/place）
+ 淘汰（on_delete）。把命中算进来是必要的——长期所有成员都处于 parked（有子链）时
不会产生入区事件，只靠入区/淘汰计数会让 aging 永远不触发。

```
every E struct 入区/淘汰事件 (默认 E = 8192):
    epoch += 1
    for node in active_regions:
        node.share_count >>= 1
        node.rid_bloom = 0                 # 允许旧 rid 重新计数
        if node.share_count < 2 and node is not 分叉点:
            demote_out_of_struct(node)     # 退队（§3.3）
    rebuild_regions_by_band()              # protected/probation/window 重排
```

- `share >>= 1` 而不是 `-= 1`：衰减与计数同量级，`K=3` 的成员大约 2 个 epoch 后失去保护，量级可控。
- 清 bloom 是必要的：否则「三个月前访问过一个模板的 rid」会被永久判为已计数，模板再也涨不上去。
- `rebuild_regions_by_band()` 是 O(成员数)，每 8192 次事件一次，摊薄成本可忽略。
- **aging 不改变 `hit_count`**（INV-6）。

---

## 7. 集成契约

### 7.1 装载方式

不新增 eviction policy 名——`simple_hybrid_radix_cache.py:27` 用
`isinstance(self.eviction_strategy, SimpleHybridStrategy)` 做分派，
新增 policy 名会绕过它。改为组合，且三组参数分别用 `uni_utils` 里的 dataclass 包裹：

| 参数组 | 类 | 管什么 |
|---|---|---|
| `alpha` | `AlphaParams` | alpha 在线权重的反馈控制（update_interval / min_cap_share / max_alpha_step / smooth / temperature / balance） |
| `rr` | `RrParams` | Agentic 会话轮询配额（fair_rr / kappa / min_retain_tokens / min_retain_frac） |
| `tiny_lru` | `TinyLRUParams` | Q_struct 队内 W-TinyLRU（policy / k / window_entries / epoch_events / admission_tie / admission_metric / inherit_share） |

`max_offset` 不属于任何一组：它是 Q_struct 复用分数里的位置项上界（HeterCache 的 offset 语义），
与频数准入无关，仍是顶层参数。

```jsonc
"eviction_policy_config": {
  "max_offset": 2048,
  "alpha":  {"smooth": 0.7, "temperature": 0.5, "update_interval": 30.0},
  "rr":     {"fair_rr": false, "kappa": 0.5},
  "tiny_lru": {"policy": "wtinylru", "k": 3, "window_entries": 64, "epoch_events": 8192}
}
```

旧的扁平键仍可用（`alpha_smooth` / `fair_rr` / `struct_k` ... 见
`uni_utils.LEGACY_PARAM_ALIASES`），由 `split_legacy_params()` 路由；
**分组显式给出的字段优先**，未给出的字段由扁平键补齐；未知键直接 `ValueError`
（拼错不再静默失效）。三个 dataclass 都在 `__post_init__` 里做范围校验。

`SimpleHybridStrategy.__init__` 里构造 `StructQueuePolicy(cfg)`，字段名 `self.struct`；
`struct_policy == "off"` 时全部钩子为 no-op 且 `victim()` 恒返回 `None`（零开销路径）。

### 7.2 淘汰委托点

在 `simple_hybrid_radix_cache.py:49-57` 选出 `victim` 之后、`_evict_one(victim)`（同一循环末尾）之前插入：

```
if struct_policy.enabled and struct_policy.owns(victim):
    alt = struct_policy.victim()
    if alt is not None:
        victim = alt
    # alt is None → 保留原 victim，绝不阻塞（INV-4）
```

`owns(node)` 只做一次集合/字段查，不遍历树。

**为什么是「委托」而不是「预算」**：本期不做跨队列仲裁，所以 struct 队列不能自己决定
我该留多少，只能在对局部的淘汰顺序上提出更优候选。这样 `evict()` 的推进语义、one-shot 优先带、
alpha 更新链路全部不变。

### 7.3 单调推进的硬约束

`evict()` 是 `while num_evicted < num_tokens` 的硬循环。为保证不死循环：

1. `victim()` 返回的必须是 `evictable_leaves` 中的节点（INV-2 的推论）；
2. `_evict_one()` 必须返回 `len(value) > 0`（现有一致）；
3. 加断言：每次循环 `num_evicted` 必须严格递增，否则 `logger.error` + 强制回退到全局 victim。

### 7.4 与 one-shot 带的关系

`_evict_one()`（`simple_hybrid_radix_cache.py:65`）对 one-shot 节点会**向上追溯删除整条死链**：

```python
# simple_hybrid_radix_cache.py:88-92
while ... parent.lock_ref == 0 and not parent.children:   # 父已无子
    cur = parent; continue                                # 连带删除
```

由于 §5.3 会把被拒成员降级为 `QTRANS`（→ `is_one_shot == True`），一次降级收割可能顺带
上溯删掉刚刚进入 `protected` 的模板前缀。**这是本方案最大的正确性风险**，必须在
上溯循环里加闸：

```
# _evict_one 上溯条件追加：
and not struct_policy.is_member(parent)          # 模板前缀不参与死链级联
```

加闸后，闭环变成：文档段被删 → 父（成员）成为叶子 → `on_became_leaf` → protected。

---

## 8. 复杂度与内存

| 操作 | 复杂度 | 说明 |
|---|---|---|
| 命中 | O(1) | bloom 4 次位运算；区内移动 O(1) |
| 插入 / 分叉晋升 | O(1) | 只看 parent 的 children 数 |
| 变叶子 / 变非叶子 | O(1) | 现有 `_update_leaf_status` 已是 O(children) |
| 分裂 | O(1) | 重绑两个引用 |
| victim 选择 | **O(1)** | 三个 deque 取 head；对比现状 O(N) 扫 `evictable_leaves` + 堆重建 |
| aging | O(成员数) / 每 E 次事件 | 摊薄可忽略 |
| 内存 | +12~20 B / 成员节点 | 相比 `TreeNode` 现有开销（dict + 多个字段）可忽略 |

---

## 9. 观测与指标

复用 `_profile_token_counts()`（`radix_cache.py:750`）与 `profile_summary`，新增一段
`[struct wtinylru]`（env 门控 `SGLANG_STRUCT_WTINYLRU_LOG=1`，默认关）：

| 指标 | 含义 | 期望 |
|---|---|---|
| `members` / `parked` / `active` | 成员数、挂起数、活跃区数 | 成员数随拓扑收敛，不单调爆炸 |
| `w/p/r` 三区条目与 token | 区占用 | window 接近 `window_entries` 上界 |
| `admit` / `reject` | 准入通过/拒绝次数 | reject 比例应集中在单次文档 |
| `demoted` | 降级次数 | 与 reject 一致 |
| `force_protected` | 从 protected 兜底淘汰的次数 | 应趋零；显著 >0 说明 K 太高或 window 太小 |
| `share_hist` | 被淘汰成员的 `share_count` 直方图 | `share >= K` 的淘汰 token 占比应趋零 |
| `template_survival` | 分叉点平均存活时长 / 被删次数 | 相比基线显著提升 |
| `one_shot_residency` | 单次文档成员平均驻留时长 | 应压缩到 1 个淘汰周期 |
| `victim_us` | `victim()` 平均耗时 | 相比基线 O(N) 应有量级下降 |

回放实验：`Uni-Bench/bench_unicache.py` + `cachewise-coding-traces`，与
`--eviction-policy=simple_hybrid`（现状）对比，同时监控 chat/agent 命中率**不回退**
（验证「只在本队列内动手」确实没有漏到外面）。

---

## 10. 分阶段落地

| 阶段 | 内容 | 交付 |
|---|---|---|
| P0 | 只观测：`share_count` 与 `hit_count` 的偏差倍数、成员 `share` 分布、被删成员的历史 `share` 直方图 | 一页数据，用来证明坑 3 的严重程度 |
| P1 | M3 成员判定 + `share_count` + bloom + 钩子 4.1/4.2/4.3/4.4/4.5，`struct_policy="slru"`（只按 band 排序，无准入、无 aging） | 可直接 A/B 的最小可用版 |
| P2 | 三区 + 准入 + 降级继承 + 上溯闸（§7.4） | S2 完整版 |
| P3 | aging + 指标 + 自动重排 | 长期稳定性 |
| — | 结构签名（S3）、队列预算与跨队列仲裁 | 下一期 |

每阶段默认关闭（`struct_policy="off"`），保证 merge 后主路径行为不变。

---

## 11. 测试用例（`test/manual/test_struct_wtinylru.py`，与现有 `test_simple_hybrid_abs_offset.py` 同风格）

1. **成员判定**：一子 → 二子晋升；子链删光后仍是成员且可淘汰；再插入子段 → parked。
2. **rid 去重**：同 rid 三次 match → `share_count == 1`；三个 rid → 3。
3. **bloom 饱和**：8 个 rid 后新 rid 被误判 → `share_count` 不涨（保守方向，明确断言）。
4. **分裂继承**：前缀半/后缀半的 `share_count`、`rid_bloom`、区归属。
5. **准入拒绝**：单次文档成员在 window 溢出时被降级 → `queue_tag == QTRANS`、**`evictable_size_` 不变**（KV 未释放）。
6. **准入通过**：`share >= K` 直接进 protected。
7. **降级继承**：父 `share=4` → 父成为叶子时直接进 protected，不经过 window。
8. **淘汰顺序**：window → probation → protected；`force_protected` 计数正确。
9. **兜底推进**：只有 protected 成员时仍能淘汰且返回 token > 0；断言 `num_evicted` 严格递增。
10. **上溯闸**：降级 QTRANS 文档段被 one-shot 收割时，**不**连带删除 protected 的父成员。
11. **aging**：跨 epoch `share` 减半、protected → probation、bloom 清空后同 rid 可重新计数、`hit_count` 不变。
12. **一致性（随机化）**：任意 insert/match/evict 序列后，INV-2 成立（三区并集 == 可淘汰成员集）。
13. **隔离**：非成员节点的 `queue_tag`、`last_access_time`、KV 不被策略改写；`struct_policy="off"` 时与基线逐字节等价。
14. **reset/disable**：无残留引用、无 bloom 泄漏。





## 13. 风险与回滚

| 风险 | 影响 | 缓解 |
|---|---|---|
| 上溯级联删掉 protected 前缀 | 模板丢失，最坏情况下本方案无效 | §7.4 上溯闸 + 用例 10 |
| 成员集合不收敛 | window 无界、`share_count` 噪声 | §3.3 退队 + aging 重排 + 用例 12 |
| 降级→one-shot 收割链与 `_evict_one` 交互 | 死循环 / token 数不递增 | §7.3 断言 + 回退全局 victim |
| bloom 误判「少计」 | 新模板晋升更慢 | `K`/bloom 位数可配；实验环境换精确集合 |
| `_record_node_hit` 缺 rid | 无法去重 | 从 `match_prefix` 透传 `params.req`（只透传） |
| 采样线程访问 region | 并发结构性修改 | 观测只读 `_per_workload_token_count()`（遍历树），不碰 region 索引 |
| 性能回退 | 钩子加在每次 match 上 | 全部 O(1) 位运算；`off` 时零开销；`victim()` 相对基线是净收益 |

回滚：单一开关 `struct_policy="off"`，不涉及数据结构迁移（新增字段默认 `0`/`None`）。
`share_count` / `rid_bloom` 即使关闭也留空，不参与任何逻辑。


---

## 16. 实现落点（第一版已落地，未提交）

### 16.1 文件

| 文件 | 角色 |
|---|---|
| `python/sglang/srt/mem_cache/uni_utils.py` | 新增 `AlphaParams` / `RrParams` / `TinyLRUParams` 三组参数 dataclass + `split_legacy_params()` / `merge_param_overrides()` |
| `python/sglang/srt/mem_cache/struct_queue_policy.py` | 新增：`StructQueuePolicy`（成员判定 / 三区 / 准入 / aging / victim）。纯 Python，只依赖 `uni_utils`，不碰 KV |
| `python/sglang/srt/mem_cache/radix_cache.py` | `TreeNode` 新增 `share_count` / `rid_bloom` / `struct_region` / `struct_ever_branch`；6 个钩子 + `rid` 透传 + `reset()` |
| `python/sglang/srt/mem_cache/evict_policy.py` | `SimpleHybridStrategy.__init__` 改为三组参数（`alpha` / `rr` / `tiny_lru`）+ 旧扁平键兼容路由，构造 `self.struct`（默认 `policy="off"` → `None`） |
| `python/sglang/srt/mem_cache/simple_hybrid_radix_cache.py` | 淘汰委托 + `released <= 0` 防自旋 + `_evict_one` 上溯闸 |

### 16.2 钩子与代码位置

| 钩子 | 调用点 | 说明 |
|---|---|---|
| `on_hit(node, rid)` | `_record_node_hit()`（`radix_cache.py`） | `rid` 来自 `match_prefix(params.req)`；`rid is None`（chunked 复查）只刷新顺序不计数 |
| `on_children_added(node)` | `_insert_helper()` 挂上新子节点之后 | 打 `struct_ever_branch` 并重新判定成员 |
| `on_became_leaf(node)` | `_update_leaf_status()` 的 `add(node)` 之后 | 按继承表落区（热点模板在这里被保住） |
| `on_no_longer_leaf(node)` | `_update_leaf_status()` 的两个 `remove` 分支 | 挂起，保留频数 |
| `on_delete(node)` | `_delete_leaf()` 中 `_update_leaf_status(parent)` 之前 | 清索引 + aging 时钟 |
| `on_split(new_node, child)` | `_split_node()` 末尾 | 前缀半段重新判定；后缀半段对象未变，索引自然有效 |
| `reset()` | `RadixCache.reset()` 开头 | 清空索引与计数 |

`struct_region` 编码：`0=非成员 / 1=window / 2=probation / 3=protected / 4=parked`。
`owns(node)` 只读这一个字段，所以淘汰热路径是 O(1)。

### 16.3 与设计稿的三处偏差（均为实现期发现）

1. **窗口区在纯 M3 下不可达** → 引入 `struct_ever_branch` 黏性标记（§3.1）。
   否则「成员 ⇔ 有子节点或 share≥2」，`share≤1` 的成员只能是 parked，准入路径永不执行。
2. **`share >= 2` 的候选不能拒绝**（§5.2）：拒绝会留下「成员但不在任何区」的孤儿，
   破坏 INV-2。拒绝只对 `share <= 1` 的观察者生效。
3. **aging 时钟必须包含命中，且按请求去重**（§6.3）：长期所有成员都有子链（parked）时，
   只有入区/淘汰事件计数会让 aging 永不触发；而按「每个节点一次命中」计数又会把 aging 频率
   与路径长度耦合，所以按不同 rid 的首个命中计一次。

另有一处防御性补充：`evict()` 循环里若 `_evict_one()` 返回 0，记 `logger.error` 并 break，
避免「victim 不可删」导致的死循环（§7.3 第 3 条）。

### 16.4 开关与观测

```bash
--radix-eviction-policy=simple_hybrid \
--radix-eviction-policy-config '{"tiny_lru": {"policy": "wtinylru", "k": 3}}'
```

`StructQueuePolicy.snapshot()` 返回 `window/probation/protected/parked` 条目与 token、
`admit/reject/retire/park/ages/dedup_hits/forced` 计数，供 §9 的指标接入。
`forced` 是从 protected 兜底淘汰的次数：显著大于 0 说明 `struct_k` 太高或 window 太小。
