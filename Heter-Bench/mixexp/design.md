# heter_cache 保护 TTFT 的淘汰设计

> 载体：`python/sglang/srt/mem_cache/heter_cache/` + `evict_policy.HeterCacheStrategy`
> 状态：档 A（后悔弃权门 `RegretGate`）、档 B（全弃权全局定价 `PricedFallback`）
> 均已实现，**默认关闭**；不带参数时其他策略行为逐位不变。
> 实测口径：tail-stress（cap=198k、page=8、engine FCFS、1700 请求）与 exp3 四 cell。
> 数字都在 `chain_shape_report.md` / `best_configs.md` 里，本文只引结论。

---

## 一、这个版本要解决什么

一句话：**淘汰器的目标从「命中率最大」改成「别让人等」**。落到指标上，是不让
TTFT 的 p90 被「整条私有前缀被剥穿」顶起来，而不是去追总命中率。

### 1.1 为什么看 TTFT 而不看 hit

tail-stress 上 LRU 和 OPT 的 hit 只差 7.2pp（77.63% vs 84.84%），p90 差 5 倍
（920ms vs 184ms）。原因很直白：

```
TTFT = (派发 − 到达) + (prompt_len − hit) / prefill 吞吐
```

同样是丢 12k 个 token：分 20 次每次丢 600，每次只多 72ms；一次丢 12k，单次多
1.44s，直接进 p90。**总量相同、形状不同，TTFT 差一倍**。所以本设计的主指标是
「warm 会话整链丢失率」，总 miss 只作为「不许变差」的约束。

### 1.2 整链丢失不是逐渐剥光，是断崖加死循环

对 LRU 每个会话逐轮打印 `(轮次, prompt, hit)`，看到的是：

```
warm-000  (0,1500,0) (1,3000,1696) ... (6,10500,9200) (7,12000,0)
warm-004  (0,1500,0) (1,3000,1696) ... (5,9000,0) (6,10500,0) (7,12000,0)
```

三个事实叠在一起：

1. `match_prefix` 刷新的是**整条匹配路径**的 recency，一条私有单子链不存在
   「中间有点旧」，要么全新鲜要么全旧；
2. 链一旦成为全局最旧，`evict` 的「叶子出堆 → 父变新叶子再出堆」会在同一次
   淘汰里一路剥穿，归零是断崖；
3. 归零后会话下一轮 hit=0、整链重插、几十秒后又全旧被剥——进入死循环。

实测 52.2% 的整链丢失里，**只有 12.5% 是第 1 轮的天然 hit=0**，其余全是这个循环。

### 1.3 为什么 heter 在 tail-stress 上比 LRU 还差

这是本设计存在的直接理由。两条独立的机制叠在一起：

**（1）one-shot 判据几乎正好命中 warm 的画像。** `is_one_shot` 要求「未重读
（`reuse_count == 0`）+ 归属会话已搁置 > `ONE_SHOT_IDLE_S`(20s) + 跨会话共享数
< 2 + 会话类型属 TRANS/STRUCT」。warm 会话是私有的、轮间 45s、第一次命中前
`reuse_count` 一直是 0——严格符合。于是它比 LRU 削得更狠：warm 整链丢失 76.9%
vs LRU 52.2%，p90 1646ms vs 920ms。

**（2）Stage 0 清完之后，残余压力被 VD 引向锚队。** VD 是「已兑现价值密度」
（每驻留 token 换回多少命中）。锚所在的队列（chat）命中稀疏——它一次命中就省
12k，但一天命中不了几次——按密度算就是最低的，于是每次仲裁都点名它。队内只能
在本队池子里挑人，挑满就整队剥穿。实测 chat 被点名 609 次、agentic 731 次，
终态 agentic 占用被削到 0；warm 的写入量被放大 2.4 倍（287 万 token vs
chain_lru 全局版 215 万）——**同一个内容反复写、反复删**。

一句话总结缺陷：**VD 是一个自洽的排序键，但它是个正反馈环**——越削越贵、越贵
越削。而队内隔离拿走了「去别的池子找便宜受害者」这个逃逸口。

---

## 二、结构：三层分离 + 五步流水线

### 2.1 层位

| 层 | 文件 | 职责 | 不做什么 |
|---|---|---|---|
| 状态层 | `heter_cache/occupancy.py`、`value.py`、`node_share.py`、`utils.SessionRegistry` | 三本账：队列占用、已兑现价值、共享数与会话史 | 不做任何裁决 |
| 判据层 | `heter_cache/selector.py`、`regret_gate.py`、`override.py` | 纯策略：给定叶子集合与账本读数，返回「谁」和「为什么」 | 不碰树、不碰 allocator |
| 执行层 | `heter_cache/radix_cache.py` | 批量淘汰循环、物理释放、树拓扑、观测接缝 | 不重算判据 |
| 观测层 | `Heter-Bench/profile/*`（经 `bench_bridge`） | 后悔率账本、profile 打点 | 不参与控制 |

判据层全部**无状态或纯计数器**：`queue_order` 只读 VD 与占用，`chain_leaf_key`
是纯函数，`RegretGate` 是滑窗计数器。这样同一个种子能复现，也能在单测里用假树
直接验证。

### 2.2 一次淘汰的完整路径

`HeterCacheRadixCache.evict()` 的主循环（`radix_cache.py`）：

```
evict(num_tokens):
    观测对齐  →  _observe_occupancy(available, capacity)
    while 已释放 < num_tokens:
        (victim, source) = pick_victim(可淘汰叶子)
        source 暂存（释放完节点就没了，归因要用）
        弃权门 stamp（必须在 _evict_one 之前，需要完整父链）
        released = _evict_one(victim)          # 单节点，不级联
        if released <= 0: break                # 防自旋
        record_victim_source(source, released)
```

`selector.pick_victim()` 的五步裁决：

| 步 | 名称 | 判据 | 产出 `source` |
|---|---|---|---|
| 0 | one-shot 带 | `strategy.is_one_shot(n)` 为真 | `one_shot` |
| 1 | VD 排序 | `VD_q = 已兑现命中 / max(占用, floor)` 升序 | —（只是候选序） |
| 2 | 后悔弃权门 | `R_q ≥ max(r_min, k × 已采样中位数)` 则跳过该队 | —（只是过滤） |
| 3 | 队内选点 | struct 按共享数分档、chat/agentic 按复用分 | `value:<queue>` |
| 4 | 全弃权兜底 | 全树叶子按链形定价取最便宜 | `priced:<queue>` / `value:fallback` |

关键设计：**第 1、2 步只决定「削哪个队列」，第 3、4 步才决定「削哪个节点」**。
上一层永远不改下一层的排序键，这样每一层的效果都能单独测量、单独证否。

---

## 三、逐步展开

### 3.1 步 0：占用校准（先对齐水位，再裁决）

VD 的分母是队列占用，占用错了排序就错。所以每轮淘汰前先看两个时钟是否到期：

- **校准时钟**（状态正确性）：常规 30s，承压期 3s。承压判据是
  `available / capacity < 5%`——水位变化最快的时候必须用新鲜读数，否则按过期
  水位点名。
- **打印时钟**（观测）：固定 30s。

任一到期就做**一次**全树遍历（`_per_queue_token_count`），同一份快照先喂给校准、
再喂给 profile。这是硬约束：拆成两次调用会让树遍历翻倍。校准之后立刻
`realized.materialize()`，保证 VD 的分子（命中质量）与分母（驻留量）同刻。

增量维护（`record_insert` / `record_evict`）只负责两拍之间的漂移，`reconcile`
是唯一校准源。漂移的已知方向也写在代码里：one-shot 晋升会让队列短暂「欠账」、
时间豁免会让占用短暂虚高，两者都由校准吸收。

### 3.2 步 1：one-shot 带（保持不变）

排序键 `(0 若已老化 else 1, last_access_time, id)`，即老化节点最先清。

这一层**故意不动**，尽管它正是 tail-stress 崩掉的原因之一。理由来自 exp3 的
实测：早先的实现会沿 one-shot 私有祖先链向上级联删除，一次决策实际释放量是被
选节点的 5.5 倍（14.57M 里只有 2.67M 是选中节点本身），还连带删掉了 2.88M 个
struct 标签的 token——**决策与结果没法归因**，配额、密度、后悔率三本账全部失真。
现在退回「一次淘汰一个节点」，粒度回到节点级，每一步都可复盘。

真正的修法不在这一层，而在第 2、4 步：让脆弱的锚（会回来但暂时空的链）不被
点名，而不是放宽 one-shot 的判据。放宽判据在真实混合负载上养死锚，四 cell 全跌。

### 3.3 步 2：VD 仲裁（削哪个队列）

```
r_q  = H_q(W_q) / W_q                    命中速率，W_q = max(horizon, k·T50_q)
VD_q = r_q · horizon / max(occ_q, floor) 每驻留 token 在一个视野内换回的命中量
```

- `horizon = 120s` 是**政策**：三个队列共享一个决策视野，跨队列比较只在同一
  视野下成立，所以它不按队列拆。
- `W_q` 只是**统计**：用来平滑命中速率的估计方差，投影回视野后量纲不变。把时间
  常数按队列拆开，稳态下等价于给队列加隐性权重。
- `floor` 默认 0。曾经用 2048，实测把「小队列」系统性算成低价值（agentic 真实
  密度 3.2，除掉 floor 后只剩 0.684），于是每次都先削它，削完占用更小、分母恒
  被 floor 顶住 → 饿死。floor 不是保留量。
- 排序键只有 VD；**占用不参与排序**，只在 VD 完全相同时按占用降序兜底。把占用
  乘进排序键等价于一个隐式配额。

`queue_order` 只排 `occ > 0` 的队列（空队列没有驻留物可评密度，也没东西可削）。

### 3.4 步 3：后悔弃权门（档 A，打断正反馈）

VD 的唯一问题是它的信号是**预测**（「这段还值钱」），预测错了会自我强化。弃权门
给它加一路**实测反馈**。

**信号定义**：窗口 W（默认 120s）内

```
R_q = join 回队列 q 的淘汰内容 token ÷ 同窗从 q 削掉的 token
```

- 削了没人回来读（冷尾巴）→ R ≈ 0；
- 削了锚（会话回来 miss）→ R 飙升。

**怎么测出来**（`regret_gate.py`）：淘汰决策时把「断点页的内容键 → (时刻, 队列,
腾出 token 数)」登记成一个 stamp；admission 匹配时用同一种键去查，命中就说明
「刚删的内容马上被要回去了」，把该 stamp 的 span 记进队列的后悔窗口，并**摘除
stamp**（一个 stamp 只归因一次）。内容键 = `blake2b(页号 + 首页前 16 个 token +
本页 token + 命名空间)`——不用缓存自己的累进 SHA256，那个要从根重量整条前缀。

为什么不按请求的「重算量」归因：第 k 页 miss 之后的内容反正都要重算，更深的页
被删对这个请求没有额外损失。所以键取链上**最浅**那一页，损失按 stamp 的 span
计。这一条口径与观测账本（`Heter-Bench/profile/regret_ledger.py`）完全一致，
两边同源才能对账。

**判据**（三个条件同时满足才弃权）：

1. `R_q ≥ r_min`（绝对下限，正常状态不触发）；
2. `R_q > k × 已采样队列 R 的中位数`（用中位数做对照，不写死阈值——队列尺度
   不同、负载会漂）；
3. 已采样队列 ≥ 2 且窗口内削掉量 ≥ `min_evicted`（4096 token）。样本不足的队列
   不参与判定，否则 0/0 的空队列永远赢得「最便宜」的兜底。

**语义边界**：门**不改 VD 的排序键**，只决定「这个队列这一轮削不削」。按 VD 序
逐队问，超阈值就跳过。全弃权时按 min-R 跨队裁决（全局最便宜处下刀）；无人采样
则回退 VD 首队，即回到现状。

**实测**：默认参数（k=1.0/r_min=0.5）就把 heter 从 p90 1646 → 902ms、warm 整链
76.9% → 57.5%，hit 73.4% → 76.5%。机械上确认生效：chat 弃权 1431 次，VD 不再
确定性地把削减引向锚队。扫参平台是 k ∈ [0.4, 0.8] × r_min = 0.3，取
k=0.6/r_min=0.3 → p90 622ms、hit 78.5%。

**过敏的下界**：k=0.5/r_min=0.1 时 p90 反而 1218ms、p95 4677ms——全弃权太频繁，
随机兜底开始乱削。

### 3.5 步 4：队内选点（削哪个节点）

队内策略按队列分开，因为三种负载的形状不一样：

| 队列 | 队内键 | 依据 |
|---|---|---|
| `struct` | `(0 if share ≤ share_min else 1, last_access_time, id)` | 模板族按跨会话共享数分档，低共享先走 |
| `chat` / `agentic` | `(P(未来 H 秒内被复用), −节点 token 数)` | lognormal 条件生存概率 `1 − S(a+H)/S(a)`，只有 mu/sigma 分队列（chat T50≈118s、agentic T50≈6.1s） |
| 可选 `ChainEvict` | `chain_leaf_key` | 链形保护，**默认关**，见 3.7 |

chat/agentic 共用一套公式，只在参数上分开。这不是为了省代码，而是实测结论：
用「最久没被削过」这种与复用概率正交的键，会让刚起、30s 后必然回来的会话排在
最前面（会话级轮询 RR 已整体删除）。

### 3.6 步 5：全弃权兜底（档 B，把随机换成全局定价）

档 A 修好了「削哪个队列」，但全弃权之后仍是**随机**挑一个叶子。tail-stress 里
warm 锚占叶子池的大头，随机照样削到锚——所以 warm 整链停在 ~49%，追不上
chain_lru 的 23.8%。

档 B 把这一步换成「全树最便宜的叶子」，价格用与 `ChainLRUStrategy` 完全相同的
函数 `chain_leaf_key`（口径单一来源）：

```
price(leaf) = last_access_time
            + weight × (1 + β · R_队列) × max(0, anchor − remain)
anchor      = max(floor_tokens , 链上最深重读节点的 root→node token 数)
remain      = 删掉这个叶子后，链上仍然连续缓存的前缀 token 数
```

逐项解释：

- **`remain`**：admission match 读的是连续前缀，删掉叶子后 `root → parent` 那
  一段还在，就是「部分丢失」；删到根就是「整链丢失」。
- **`anchor − remain`**：缺口越大罚越重。别的受害者必须比它还旧
  `weight × 缺口` 秒才轮得到它——剥到锚附近自然刹住。
- **`anchor` 的物理含义**：从根到「最深重读节点」累计的 token 数 = 该会话上次
  被命中到的位置 = 它下一轮要读的位置。`floor_tokens` 是全局先验，两者取大。
- **证据门**：无重读**且**会话没回来过（`SessionRegistry.gaps` 为空）时罚项归零。
  「会回来过一次」是「还会再回来」最强的可观测信号，正好补上「还没被重读过、但
  会回来」的第一轮；而一次性冲刷链不享受保护，死锚问题消失。证据只认
  `reuse_count`（admission match 计数），**不认 `hit_count`**——后者把写路径和
  匹配路径混着数，一次请求自己就会加 1。
- **锚出口**：路径上所有会话都终结（`session_expired` 口径，共享段不算终结）
  时罚项归零。死会话不许一直占着保护。
- **`β · R_q`**：把实测后悔密度乘进罚斜率。历史削了亏得多的队列，它的锚更贵。
  β=1 是甜点，β=2 把 chat 锚罚过重、削减被挤到别的池子（cold hit 从 0.868 掉到
  0.851）。

**实测**：k=0.6/r_min=0.3/β=1 时 p90 581ms（档 A 622、原版 heter 1646、LRU 920），
hit 78.4%，四 cell 加权 64.42%（heter 64.24，无劣化，chat +0.64pp）。

**它为什么还是追不上 chain_lru 全局版**（404ms）：门放行的队列仍然可以自由削
自己的叶子，全局定价只在「全弃权」这个状态下打开；chain_lru 是**每一次削减**都
全局定价。要把差距补平，等价于把每次削减都换成全局定价——那就是 chain_lru 本身。
所以 heter 侧到此收手。

### 3.7 队内链形保护（ChainEvict，默认关，已证否）

`ChainEvict` 把同一套 `chain_leaf_key` 放进队内（chat/agentic）。实测是**无效
组件**：8 组参数（floor {0,4096,8192} × weight {0.01,0.02}）在四 cell 上全部
±0.1pp 贴着原版；tail-stress p90 1646ms（原版 1620）。

诊断：独立策略能赢，靠的是**全局候选池里的逃逸阀**——冷尾巴的罚项比锚小一个
量级（remain 7k 贴 floor，缺口≈0），容量压力先吃它，碰不到锚。队内化把这个阀
拿走了，削减又必须交给 VD 路由，于是火力集中到锚队，两个池子都被强制剥穿。

更糟的是「三件套同开」（弃权门 + 队内 ChainEvict + 定价兜底）：p90 1239ms，比
档 B 单项的 581ms 差一倍——队内锚保护 + 门放行 = 把削减挤到剩下的池子里，互相
打架。**这三个机制不要同开。**

---

## 四、物理释放与三条不变量

### 4.1 一次释放的顺序

`_release_node` 的调用序是硬约束：

```
record_evict(node)                            # ① 先扣队列占用
token_to_kv_pool_allocator.free_segment(...)  # ② 再还显存
kv_events.record_remove(node)                 # ③ 对外事件
_delete_leaf(node)                            # ④ 摘树
```

**①必须早于②，也必须早于同一轮淘汰循环里的下一次 `_pick_victim`。** 占用账本
在同一次循环里被反复读取，扣减晚了就会按虚高水位点名下一个受害者。

`_evict_one` 只释放一个节点，不向上级联（理由见 3.2）。

### 4.2 不变量清单

1. `record_evict` 早于 `free_segment`，且早于同轮下一次 `_pick_victim`；
2. 观测快照一次遍历、两家消费（校准 + 渲染），不许拆成两次；
3. VD 物化必须在占用校准**之后**（分子分母同刻）；
4. stamp 与 join 指在 `_evict_one` **之前**（需要完整父链），并且只在
   `params.count_reuse` 为真时 join（回扫不是重读）；
5. 所有入口只在调度主线程（`match` / `evict`）执行，与观测账本同一约束，不加锁。

另有一条与本设计相邻但同样不许动的：`SessionProbe.on_charge` 必须在 `d` 截断
之前回调——截断后 `d <= L` 恒成立，`d > L + slack` 的告警分支永远进不去。

---

## 五、四种队列的判据差异

| 队列 | 是否会进 one-shot 带 | 是否参与 VD 仲裁 | 老化口径 | 队内键 |
|---|---|---|---|---|
| `random`（TRANS 一次性） | 整行进带 | 否 | 无增量 | 无 |
| `struct`（模板） | 低共享 + 未重读 + 搁置超窗 | 是 | 固定 TTL 120s | 共享数分档 |
| `chat` | 未重读 + 搁置超窗 + 私有 | 是 | TTL 300s 或会话寿命 | 复用概率 |
| `agentic` | 同上 | 是 | 同上 | 复用概率 |

`force_agentic` 会把标签统一成 agentic（对早期无法分类的会话兜底），观测层必须
读 `strategy.node_tag(node)` 而不是自己 `coerce(queue_tag)`，否则回报的队列是假的。

---

## 六、实测证据

### 6.1 tail-stress（p90 / warm 整链丢失）

| 版本 | hit | p90 | warm 整链 | 说明 |
|---|---|---|---|---|
| OPT 上界 | 84.84% | 184ms | 14.1% | 有未来知识，仅作参照 |
| chain_lru 全局版 | 80.05% | 404ms | 23.8% | 独立策略，部署推荐 |
| **heter 档 B（k0.6/r0.3/β=1）** | **78.4%** | **581ms** | 49.1% | 本设计的终态 |
| heter 档 A（k0.6/r0.3） | 78.5% | 622ms | 49.1% | 档 B 的退路 |
| LRU | 77.63% | 920ms | 52.2% | 上游基线 |
| heter 档 A（默认 k1.0/r0.5） | 76.5% | 902ms | 57.5% | 参数不敏感时的水平 |
| 三件套同开 | 76.6% | 1239ms | 53.1% | ❌ 互相打架 |
| heter（原版） | 73.4% | 1646ms | 76.9% | 正反馈失控 |

### 6.2 四 cell（prompt token 加权 hit）

| | LRU | heter | 档 A | 档 B | chain_lru | OPT |
|---|---|---|---|---|---|---|
| 加权 | 62.54% | 64.24% | 64.36% | 64.42% | **66.25%** | 71.54% |
| chat | 57.28 | 58.07 | 58.60 | 58.71 | **62.61** | 79.45 |
| scan | 29.65 | 33.59 | 33.62 | 33.76 | **36.09** | 38.42 |

档 A/B 在真实混合负载上**不劣化**，chat 有小幅增益——这是它能作为「保守改进
开关」的前提。四 cell 数字来自实验机 `/root/sim_runs/`（`engine_rgate_pf/` 等），
本地无 exp3 数据。

### 6.3 已经证否、不要复活

| 方向 | 结论 |
|---|---|
| 深度代价加权（cost-oracle gap） | gap ≤ 0，不值做 |
| 队内 ChainEvict | 四 cell ±0.1pp，无效组件 |
| 三件套同开 | p90 1239ms vs 单项 581ms |
| 全弃权过敏（k=0.5 / r_min=0.1） | p90 1218、p95 4677，随机兜底乱削 |
| 配额（caps / reservations / 充水 / 死区） | 已整体删除，等价于隐式按规模配额 |
| 全局复用分（`_reuse_score` / `victim_key`） | 已整体删除 |
| 用单节点分数做跨队列排序（`get_priority`） | 保持抛 `NotImplementedError`——没有跨队列可比的节点分数，退回任何默认序都是假语义 |

---

## 七、配置与开关

默认全关（`RegretGateParams.enabled = False`、`priced_fallback = False`），
不装配时 `pick_victim` 的 `gate` 分支整段跳过，行为与引入前逐位一致。

### 7.1 参数

| 参数 | 默认 | 推荐（档 B） | 含义 |
|---|---|---|---|
| `enabled` | False | True | 是否装配弃权门 |
| `window_s` | 120 | 120 | 后悔滑窗，与 VD 视野同值 |
| `k` | 1.0 | 0.6 | 弃权阈值 = k × 已采样中位数 |
| `r_min` | 0.5 | 0.3 | R 的绝对下限 |
| `min_evicted` | 4096 | 4096 | 窗口内削掉量下限（样本量） |
| `retain_s` | 900 | 900 | stamp 保留时长，必须 ≥ 窗口 |
| `max_stamps` | 200000 | 200000 | stamp 容量上限（FIFO），内存边界 |
| `context` | 16 | 16 | 内容键带的上下文 token 数 |
| `priced_fallback` | False | True | 全弃权时是否用全局定价 |
| `floor_tokens` | 8192 | 8192 | 锚下限先验（token） |
| `weight` | 0.02 | 0.02 | 缺口罚斜率（秒/token） |
| `beta` | 1.0 | 1.0 | 后悔归因加权系数 |
| `min_gaps` | 1 | 1 | 「会话回来过」所需的轮间隔观测数 |

配置入口（JSON）：

```json
{
  "regret_gate": {
    "enabled": true,
    "k": 0.6, "r_min": 0.3, "window_s": 120,
    "priced_fallback": true,
    "floor_tokens": 8192, "weight": 0.02, "beta": 1.0
  }
}
```

### 7.2 有效参数域

档 A/B 实际扫过的是阈值与归因系数：

- `k ∈ [0.4, 0.8]` × `r_min = 0.3` 是平台（p90 611~652ms），不是单点运气；
- `β = 1` 是甜点；`β = 2` 把 chat 锚罚压过别的池子，cold hit 从 0.868 掉到
  0.851；
- `k = 0.5` / `r_min = 0.1` 是过敏区，不要用。

`floor_tokens` / `weight` 这一档**没有单独扫**，直接沿用了 chain_lru 全局版的
平台值（floor 6144~9216 × weight 0.02~0.03，p90 420~530ms）。沿用是合理的，
因为两边用的是同一个 `chain_leaf_key`；但换硬件或换代时必须重新标定 `weight`
的量纲（它在仿真里是「秒/token」，0.02 比真实 prefill 速率高两个量级）。

### 7.3 上线前必须做的事

- 真机 bench 端到端（仿真里 `weight` 的量纲是「秒/token」，0.02 比真实 prefill
  速率高两个量级，换硬件要重标定）；
- chunked prefill 场景（当前实测都是关 chunked prefill 的）；
- `get_priority` 风格的父链遍历是 O(链深)：本负载链深 ≈ 8，agentic 长会话 60+
  轮会到几十，线上要实测，必要时把 `remain` 增量缓存到节点上。

---

## 八、怎么验证

### 8.1 单测（不需要 torch）

```bash
PYTHONPATH=$PWD/python python test/manual/test_regret_gate.py   # ALL PASSED
PYTHONPATH=$PWD/python python test/manual/test_chain_lru.py
PYTHONPATH=$PWD/python python test/manual/test_evict_loop.py
```

`test_regret_gate.py` 覆盖的关键口径：

- R 的窗口与切片（过期即消失、stamp 只归因一次）；
- 弃权判据的三个条件与「已采样队列 < 2 不弃权」；
- 全弃权 → min-R；无人采样 → None（回退现状）；
- 档 B 的 β 归因：同形状两叶，`R` 高的队列叶子更贵；
- `chain_leaf_key == ChainEvict.victim_key`（口径单一来源）。

切换参数变体时先 `export PYTHONDONTWRITEBYTECODE=1` 或删对应 `.pyc`：同长度的一
行改动可能落在同一秒且 size 不变，Python 会复用旧字节码，测试结果与源码不符。

### 8.2 仿真

```bash
# tail-stress
PYTHONPATH=$PWD:$PWD/python ~/venvs/sim/bin/python \
  Heter-Bench/mixexp/tools/build_tail_stress.py \
  --arms lru,heter_rgate --rg-k 0.6 --rg-rmin 0.3 --pf --pf-beta 1.0

# 四 cell 在实验机 /root/sim_runs/sim_engine_cell_v2.py，口径见 best_configs.md
```

仿真直接复用 `python/sglang/srt/mem_cache/` 里的同一份策略代码，不是另写一份，
所以单测与仿真的结论能互相印证。

### 8.3 负载自检三条（换参数必须复验）

| 判据 | 本配置实测 | 含义 |
|---|---|---|
| cold 的 p90 ≤ 300ms | 230ms | 没有排队积压，p90 不是被拥塞顶上去的 |
| warm 整链丢失 ≥ 50% | 52.2% | 缺陷确实被触发 |
| LRU p90 / OPT p90 ≥ 3 | 5.0× | 有足够可回收空间，值得当靶子 |

第三条是这条负载最容易失效的地方：cap 调到 120k 或 warm 轮间调到 70s 都能让整链
丢失升到 80% 以上，但同时把 FCFS 队列压出积压——那时测的是拥塞，不是淘汰粒度。

### 8.4 一致性约束

改完先确认旧符号在目标范围内归零，再确认默认行为逐位不变：

```bash
rg -n "旧符号" python/ test/ Heter-Bench/
git diff --stat          # 预期删 N 行却显示 N×7 行，就是切多了
```

---

## 九、已知边界

1. **heter 侧的上限就是 581ms。** 再往下走等价于「每次削减都全局定价」，那就是
   `chain_lru` 独立策略。两条路并存：部署看 TTFT 用 chain_lru 全局版，要求留在
   heter 架构内、只想保守改进时开档 A+B。
2. **`gate=none`（不看证据一律护）只在人造负载上最优。** tail-stress p90 421ms，
   但四 cell 全线跌——一次性冲刷链的死锚吃容量。所以证据门不是可选项。
3. **淘汰粒度是节点（轮段）。** 一条 warm 链 ≈ 8 个节点 × ~1500 token，部分丢失
   的分辨率受此限制。页粒度剥链是另一个话题。
4. **后悔信号依赖内容键 join。** `join_rate` 为 0 说明 page_size / extra_key /
   页对齐没对上，此时读到的 `R = 0` 是采集失败，不是策略完美。账本把这个量单列
   成字段并在低于 50% 时告警。
5. **裁剪窗口是删失样本。** chat 的复用间隔 T50≈118s，统计期只有 600s，不报
   删失率就会系统性低估 chat 的后悔。账本有独立字段，本设计不据它做决策。

---

## 十、一页速记

- 目标：保住「会回来的长前缀」，别让它被整条剥掉；指标是 p90 TTFT 与整链丢失率。
- 手段 1：VD 排序决定削哪个队列（已兑现密度，不许把占用乘进排序键）。
- 手段 2：后悔弃权门打断 VD 的正反馈——削了马上被要回去的队列弃权。
- 手段 3：全弃权时不再随机，全树叶子按「删了还剩多少前缀」定价，最便宜的挨刀。
- 边界：默认关闭；证据门必开；三件套不同开；`record_evict` 永远排第一。
