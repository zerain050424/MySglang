# chain_shape：按「下游依赖体量」选受害者——浅段优先保护、砍链尾（tail-stress + 四 cell 实测）

> 【长尾优化专题 · 机制推演报告】按下游依赖体量选受害者：浅段优先保护、砍链尾机制推演与全线实测

## ⓪ 全线总结（2026-11-18，所有实测数字的汇总表）

**tail-stress（cap=198k、page=8、engine FCFS、1700 请求；p90 / warm 整链丢失）**

| 臂 | hit | p90 | warm 整链 | 说明 |
|---|---|---|---|---|
| **OPT 上界** | 84.84% | **184ms** | 14.1% | 未来知识，仅供参照 |
| **chain_lru 全局**（multiturn f8192 w0.02） | **80.05%** | **404ms** | **23.8%** | ⭐ TTFT 优先部署推荐 |
| chain_lru（gate=none f6144~8192） | 79.6~79.9% | 421~454 | 12.5~13.1% | 真实负载上会劣化，勿用 |
| heter+档B（k0.6/r0.3/β=1） | 78.4% | 581ms | 49.1% | heter 内最优（默认关） |
| heter+档A（k0.6/r0.3） | 78.5% | 622ms | 49.1% | 档B 的退路 |
| LRU | 77.63% | 920ms | 52.2% | 上游基线 |
| heter+档A（默认 k1.0/r0.5） | 76.5% | 902ms | 57.5% | 阈值不敏感时的水平 |
| 三件套（gate+队内chain+pf） | 76.6% | 1239ms | 53.1% | ❌ 互相打架，勿同开 |
| heter(TTL300) | 73.4% | 1646ms | 76.9% | 正反馈失控 |

**四 cell（原始 hit_rate；加权 = prompt token 加权）**

| cell | LRU | heter | chain_lru全局 | 队内chain | 档A | 档B |
|---|---|---|---|---|---|---|
| scan | 29.65 | 33.59 | **36.09** | 34.59 | 33.62 | 33.76 |
| struct | 81.64 | 82.41 | **82.68** | 82.18 | 82.41 | 82.41 |
| chat | 57.28 | 58.07 | **62.61** | 59.22 | 58.60 | 58.71 |
| agentic | 81.07 | 82.36 | **83.12** | 82.46 | 82.28 | 82.28 |
| **加权** | 62.54 | 64.24 | **66.25** | 64.42 | 64.36 | 64.42 |
| OPT 加权 | — | — | 差距 5.29pp | — | — | — |

**机制结论（每条都有实测支撑，详见各节）**

1. 整链丢失 = 断崖+死循环（§一）：match 刷新整条路径 → 私有链全旧 → 剥穿
   → 归零重插 → 再剥穿。52.2% 整链丢失里仅 12.5% 是天然首轮。
2. 独立 chain_lru 两靶同赢（§三/五）：四 cell 全胜 heter +2.01pp、
   tail-stress p90 −56%；代价感知的深度加权被 cost-oracle gap 证否（≤0）。
3. 队内化证否（§六）：锚保护的收益来自「全树挑最便宜的受害者」；
   队内隔离拿走逃逸阀，还把火力集中到锚队（gate/chain 仌仌）。
4. 证据门三档（§五）：rerode 漏第一轮、none 养死锚；**multiturn 并集两头通吃**。
5. 正反馈的账本证据（§六）：heter 的 value:chat 淘汰量 ≈ warm 写入量 1.65M，
   写入被放大 2.4×（287万 vs chain_lru 215万）。
6. 档 A/B（§七）：实测后悔密度做弃权门 + 全弃权后全局叶子定价，
   把 heter 从 p90 1646 拉到 581（−64%）、hit +3.0pp、四 cell 无劣化；
   β=1 是甜点；三件套劣化。与 chain_lru 的剩余差距是结构性的：
   VD+门只对全弃权状态启用全局定价，chain_lru 每次削减都全局定价。

**部署建议**：TTFT 优先 → chain_lru 全局版（待 chunked prefill 验证 +
weight 量纲校准）；heter 保守改进 → 档 A+档 B（默认关，真机验证后再开）。

> 时间：2026-11-18 · 分支 `W1`（`00e8985` → `5d1f90e`）· 载体 `sim/` engine 仿真器
> 目标：把「整条前缀丢失」变成「部分丢失」。这是 cost_oracle_gap 证否深度代价加权后，
> 唯一还没证否、且 tail-stress 上有 920ms → 184ms 空间的方向。
> 新策略：``--radix-eviction-policy chain_lru``（`ChainLRUStrategy`，全部 opt-in，
> 不带参数时其他策略行为逐位不变）。
>
> **终态配置：``floor_tokens=8192, weight=0.02, gate=multiturn``** —— 两个靶子同时赢：
> 四 cell 加权 66.25%（LRU 62.54 / heter 64.24 / OPT 上界 71.54），四 cell 全胜 heter；
> tail-stress P90 920 → **404ms**（OPT 184，2.2×），warm 整链 52.2% → 23.8%。

---

## 一、先摸清机制：整链丢失不是「逐渐剥光」，是断崖 + 死循环

对 LRU 臂逐会话打印 warm 轨迹（40 会话 × 8 轮）：

```
warm-000  (0,1500,0) (1,3000,1696) (2,4500,3200) ... (6,10500,9200) (7,12000,0)
warm-004  (0,1500,0) (1,3000,1696) ... (4,7500,6200) (5,9000,0) (6,10500,0) (7,12000,0)
```

每轮都有 17~21/40 个会话 hit=0，且**一旦归零，之后每轮都是 0**。机制三条：

1. match 刷新的是**整条匹配路径**的 recency ⇒ 一条私有单子链要么全新要么全旧
   （节点按轮成段，warm 链 ≈ 8 个节点，不存在中间新鲜度）。
2. 链一旦成为全局最旧，``evict`` 的「叶子出堆 → 父变新叶子再出堆」会在同一次
   （以及随后连续几次）淘汰里**剥穿**——归零是断崖，不是渐变。
3. 归零后会话下轮 hit=0、整链重插（全新）、45s 后又整链变旧被剥——**死循环**。
   52.2% 的整链丢失全部来自这个循环（第 1 轮天然 hit=0 只占 12.5%）。

## 二、策略：``chain_lru`` = LRU + 两个树形修正项

victim 仍是叶子，排序键 = ``(lat + penalty, -remain, node.id)``，``heapq`` 取最小。
全部只读树形与上游已有计数，不引新账本、不改 evict 循环：

| 项 | 语义 | 对应「下游依赖体量」的哪一面 |
|---|---|---|
| 浅段保护 | 淘汰叶子后链上剩余前缀（root→parent token 数）``remain`` 低于 ``floor_tokens`` 时，罚 ``weight × 缺口`` 秒加进 LRU 键；剥得越深罚越重，链在锚附近自然刹住 | **浅块优先保护**：浅段是链的命中锚，砍掉它整条链的已缓存前缀作废（前缀命中要连续） |
| 砍链尾 | 平票时先剥 ``remain`` 大的叶子 | 长链砍尾只损失尾巴（部分丢失），短链砍根损失整链 |
| ``gate`` | ``none``＝一律护浅段；``rerode``＝只护路径上有重读证据的链 | 证据只认 ``reuse_count``（admission match 计数）；``hit_count`` 含写路径自增，会把刚写的段误判成被重读过（AGENTS 口径） |

``floor_tokens=0`` 或 ``weight=0`` 退化回纯 LRU（保留砍链尾平票键）。

## 三、实测（tail-stress：cap=198k、page=8、engine FCFS、1700 请求）

| 臂 | 总体 hit | TTFT p50 | **p90** | p95 | p99 |
|---|---|---|---|---|---|
| LRU | 77.63% | 135ms | **920ms** | 1719ms | 4590ms |
| **chain_lru**（floor=8192, weight=0.02, gate=none） | **79.85%** | 134ms | **421ms** | 916ms | 1542ms |
| OPT 上界 | 84.84% | 133ms | 184ms | 215ms | 524ms |

分类明细：

| 类 | LRU hit / 整链 / p90 | chain_lru hit / 整链 / p90 | OPT hit / 整链 / p90 |
|---|---|---|---|
| hot | 0.845 / 12.5% / 232ms | 0.836 / 12.5% / 213ms | 0.845 / 12.5% / 118ms |
| **warm** | **0.426 / 52.2% / 2783ms** | **0.542 / 12.5% / 1098ms** | 0.785 / 14.1% / 244ms |
| cold | 0.868 / 0.8% / 230ms | 0.868 / 0.8% / 207ms | 0.868 / 0.8% / 157ms |

要点：

* **warm 整链丢失 52.2% → 12.5%**，正好回到第 1 轮天然 hit=0 的基线——死循环被消除。
* warm mean miss 3875 → 3089 token（−20%）：**总 miss 是降的**，不是拿总量换形状。
  省下来的是死循环里反复整链重插的那部分。tail-stress 设计文档的验收判据
  「P90 下降、总 miss 基本不变」以更强的方式满足。
* cold p90 反而变好（230 → 207ms）：浅段保护顺带护住了共享模板的浅段，
  cold 模板命中更稳。
* 与 OPT 的剩余差距（421 vs 184ms）在 warm 的**部分命中深度**：chain_lru 的锚
  ≈ 5~6k（warm hit 0.542 vs OPT 0.785）。它证明了「不剥穿」能拿走大部分 P90，
  但「锚多深才对」是 Belady 用未来知识解的，本策略只能靠 floor 全局近似。

## 四、扫参与稳健性（tail-stress 本机）

* **floor 6144~9216 × weight 0.02~0.03 是一个平台**（p90 420~530、warm 整链
  12.5~13%），不是单点运气。floor ≤ 2048 或 weight ≥ 0.1 开始劣化：
  weight 太大 ⇒ 被护段吃掉容量 ⇒ cold 模板被挤掉（cold p90 冲到 855ms），
  周转反而加速，warm 丢得更狠（整链 81%）——保护是有容量代价的。
* **gate=none 全面优于 gate=rerode**（最优 rerode 点只有 p90 724 / warm 整链 39%）。
  原因：rerode 门下 warm 第 1 轮的 1500 token 链还没被读过（reuse_count=0），
  出生即裸奔，被剥掉就直接进死循环；纯树形保护从出生就护。
* 换 seed（1/2/3）数字逐位不变——负载生成用固定种子，``SimConfig.seed`` 只喂
  heter 兜底随机源；本对比是确定性的。
* 参数变体（不挑好情况）：
  * ``warm_gap=70s``（设计文档已注明 LRU 基线混入拥塞）：p90 1268 → 774ms，
    warm 整链 80.3% → 47.8%；
  * ``cap=120k``（拥塞档，隔离性丢失）：p90 5233 → 3111ms，warm 整链 96.9% → 74.4%。
  两个压力更大的档位方向一致，无反向劣化，但绝对值都被排队污染，不作数。

## 五、四 cell 验证（实验机 bitahub，`e4c7407`+`5d1f90e`）

同一驱动（`/root/sim_runs/sim_engine_cell_v2.py`，page=8、cap=198k、FCFS、seed=0，
与 `exp3_cells_ttl300.md` 完全同口径；LRU/heter/OPT 行取自同盘 `engine_full/` 现有结果）：

| cell | LRU | heter(TTL300) | **chain_lru** | OPT 上界 |
|---|---|---|---|---|
| scan（均衡） | 29.65% | 33.59% | **36.09%**（+2.50 vs heter） | 38.42% |
| struct 主导 | 81.64% | 82.41% | **82.68%**（+0.27） | 83.49% |
| chat 主导 | 57.28% | 58.07% | **62.61%**（**+4.54**） | 79.45% |
| agentic 主导 | 81.07% | 82.36% | **83.12%**（+0.76） | 84.43% |
| **加权** | **62.54%** | **64.24%** | **66.25%** | 71.54% |

四 cell 全胜 heter，加权 **+2.01pp**（heter 对 LRU 只有 +1.69pp）；距 OPT 上界 5.29pp。

### 两个证据门的冲突（重要发现）

gate 是这条策略唯一的「负载敏感旋钮」，三档在两个靶子上方向相反：

| gate | tail-stress p90 | 四 cell 加权 | 解释 |
|---|---|---|---|
| none | **421ms**（最优） | 全线跌（scan −1.54pp…） | 无证据一律护：一次性冲刷链的死锚吃容量 |
| rerode | 724ms（只修一半） | 64.74%（小胜 heter） | 漏掉「还没被重读过、但会回来」的链（warm 第 1 轮出生裸奔） |
| **multiturn** | **404ms** | **66.25%** | 重读过 ∪ 会话已回来过（观测到 ≥1 个轮间隔）——两者互堵对方的洞 |

multiturn 的物理含义：**「会回来过一次」是「还会再回来」最强的可观测信号**，
恰好补上 rerode 漏掉的第一轮；而冲刷/一次性链（gaps 空）不护，死锚问题消失。
这等于把 heter 的会话分类证据用在了树形保护的门上，但只做「是否续命」一个判断，
不做 one-shot 判定——heter 在 tail-stress 上翻车（P90 1620ms）的正是那个判定。

floor 在 multiturn 门下可以放大：8192 → 16384 时 scan 继续升（36.96%），但 chat
过 8192 后回落（62.61 → 61.41），取 8192~10240 为工作点（加权 66.25/66.26%）。

## 六、队内化尝试（heter_chain）——负结果，含架构结论

把 chain 保护做进 heter 队内（`ChainEvict`，`enabled=False` 默认关，
`07a7197`）后在两个靶子上实测：

**四 cell（8 组参数：floor {0,4096,8192} × weight {0.01,0.02}，全部 ±0.1pp 贴着
原版 heter）**：scan 33.56~33.61（heter 33.59）、struct 82.40~82.41（82.41）、
chat 57.79~58.12（58.07）、agentic 82.35~82.38（82.36）。floor/weight 几乎
不起作用——队内版是无效组件。

**tail-stress：p90 1646ms（原版 heter 1620，比 LRU 920 还差），warm 整链 76.9%。**

诊断（victim_source + 改写器计数 + 终态占用）：

* 独立臂能赢，靠的是**全局候选池里的逃逸阀**：冷尾巴（一次性、不回来）的
  罚项比锚小一个量级（remain=7k 贴着 floor，缺口 ≈0 vs 锚缺口 ≥8k），
  每周期 ~42k 的容量压力先吃它，碰不到锚。
* heter 结构里这个阀没了：冷尾巴走 Stage 0 one-shot 清掉后，剩下的削减
  全部由 VD 路由——VD 按已兑现密度排序，chat 队列（warm 锚，命中稀 → 密度
  最低）每次都是被削对象。ChainEvict 罚项只能在池内重排谁先死，
  **pick 必须还人**：实测 chat picks=609 / agentic picks=731，两个池子都被
  强制剥穿（终态 agentic occ=0、chat 只剩 122k）。
* 换句话说：**锚保护的收益来自「淘汰器能在全树范围挑最便宜的受害者」；
  队内隔离恰好拿走了这个能力，还把火力集中到锚所在的队列。**
  这也解释了为什么 heter 本身在 tail-stress 比 LRU 差：
  Stage 0 + VD 没有「廉价受害者」池，非 one-shot 削减集中在 min-VD 队列。

### 对下一步的含义

* 队内化（本次方向 1）**证否**：ChainEvict 保留在代码里（默认关）作对照，
  但 TTFT 优先的部署应使用**全局版 chain_lru 独立臂**（tail-stress p90 404 /
  四 cell 加权 66.25%，两个靶子都赢）。
* 若坚持 heter 架构内做锚保护，逃逸阀必须跨队列——比如「本队全部候选罚项
  超阈值时弃权，VD 跳到下一个队列，全弃权时按各队最小罚项跨队裁决」。
  这等于给仲裁加第二根轴，动 VD 口径，需要单独讨论（AGENTS 口径：仲裁只按
  已兑现密度），不在本批做。

## 七、后悔弃权门（档 A）实测：正反馈被打断，但追不上全局定价

档 A（``RegretGate``，`cc58225`，默认关）：VD 出队序不动，按实测后悔密度
R_q = 窗口内 join 回队列 q 的淘汰 token ÷ 同窗从 q 削掉的 token，超阈值弃权
跳队；全弃权 → min R 跨队；无人采样 → 回退现状。

**tail-stress**（W=120s，p90 / warm 整链）：

| 臂 | hit | p90 | warm 整链 |
|---|---|---|---|
| LRU | 77.63% | 920ms | 52.2% |
| heter | 73.4% | 1646ms | 76.9% |
| **heter+档A**（k=1.0, r_min=0.5 默认） | 76.5% | 902ms | 57.5% |
| **heter+档A**（k=0.6, r_min=0.3） | **78.5%** | **622ms** | 49.1% |
| chain_lru 全局版 | 80.0% | 404ms | 23.8% |

门机械上生效且可观测（gate.stats）：默认参数下 chat 弃权 1431 次——VD 不再
把削减确定性地引向锚队，正反馈断掉：p90 1646 → 902（回到 LRU 水平），warm
整链 76.9% → 57.5%。扫参方向明确：阈值越敏感越好，k ∈ {0.4~0.8} ×
r_min=0.3 是平台（p90 611~652、hit ≈78%）；k=0.5/r_min=0.1 过敏（p90 1218、
p95 4677——全弃权后的随机兑底开始乱削）。

**四 cell**（k=0.6, r_min=0.3）：scan 33.62 / struct 82.41 / **chat 58.60** /
agentic 82.28，对照 heter 33.59 / 82.41 / 58.07 / 82.36——无劣化，chat +0.53pp，
cheapest_pick 在 chat cell 触发 1014 次。

**为什么追不上 chain_lru（404 / 23.8%）**：门只回答「这个队这一轮削不削」，
全弃权后的兑底仍是**随机** fallback——而 tail-stress 里 warm 锚占叶子池的大
头，随机兑底照样削到锚（warm 整链 ~48~50%）。chain_lru 赢在全树统一的
**形状定价**（每个叶子都有 key，兑底也是全池最便宜者），不只是「躲开锚队」。
两个机制互补：弃权门是实测反馈（不需调参的形状知识），全局定价是结构先验。

**档 B（全弃权兑底换全局叶子定价）已实施**（`PricedFallback`，`18576c2`）：
全弃权时每个叶子按 `price = lat + weight × (1 + β·R_队列) × (锚 − remain)`
定价（链形定价与 ChainEvict 共用 `chain_leaf_key`，口径单一来源），全树取
最便宜。tail-stress（k=0.6/r_min=0.3）：

| 配置 | hit | p90 | warm 整链 |
|---|---|---|---|
| 档 A 最优（pf 关） | 78.5% | 622ms | 49.1% |
| **档 B β=1** | 78.4% | **581ms** | 49.1% |
| 档 B β=2 | 76.7% | 782ms | 51.9%（cold hit 掉到 0.851） |
| gate+ChainEvict+pf 三件套 | 76.6% | 1239ms | 53.1% |

* β=1 是甜点：β=2 把 chat 锚罚过重、削减挤到其它池（cold 受损）；
* **三件套反而劣化**（1239 vs 581）：队内 ChainEvict 的锚保护 + 门放行的
  队列 = 把削减挤到剩下的池里，互相打架——三个机制不要同开；
* 四 cell（档 B）：scan 33.76 / struct 82.41 / **chat 58.71** /
  agentic 82.28——无劣化，chat +0.64pp。

## 八、边界与下一步

* ~~本机没有 exp3 四 cell 数据，「其他负载不劣化」尚未验证~~ 已在实验机复跑
  （§五、§七），四 cell 全胜或平 heter。剩余未验：drift3、真机 bench 端到端。
* **heter 侧收尾结论（档 B 后）**：档 A+档 B 把 heter 从 p90 1646 拉到 581
  （−64%）、总 hit +3.0pp，四 cell 无劣化；但 warm 整链停在 ~49%，距
  chain_lru 全局版（404 / 23.8%）还有结构性差距——VD+门只对「全弃权」状态
  启用全局定价，门放行的队列仍可自由削自己的叶子，而 chain_lru 是每一次
  削减都全局定价。heter 内继续逼近等价于把每次削减都换全局定价（那实际上
  就是 chain_lru），不再值得做。**TTFT 优先部署直接用 chain_lru 全局版**，
  档 A+档 B 作为 heter 内的保守改进开关（默认关）。
* 淘汰粒度是「轮段」节点（一条 warm 链 ≈ 8 节点 × ~1500 token），部分丢失的
  分辨率受此限制；页粒度剥链是另一个话题，不在这批里做。
* ``get_priority`` 每次沿父链走到根（求 remain + 证据），代价 O(链深)。本负载
  链深 ≈ 8；agentic 长链（60+ 轮）会到几十，仿真无感，线上要实测再定，
  必要时可增量缓存 ``remain``。
* gate=none 在有真实「永不回来」流量的负载上会积累死锚。tail-stress 里 cold
  尾巴天然在深层（remain=7k）不受影响，但一般负载需要一个「锚老化」出口——
  与 heter 的 TTL/会话寿命判据合并是自然方向，这批不做。

## 九、复现

```bash
cd <repo-root>
PYTHONPATH=$PWD:$PWD/python ~/venvs/sim/bin/python \
  Heter-Bench/mixexp/tools/build_tail_stress.py \
  --arms lru,chain_lru,opt --floor 8192 --weight 0.02 --gate multiturn
# 单测（不需要 torch）：python test/manual/test_chain_lru.py  → ALL PASSED
# 回归：test_sim_determinism / test_sim_metrics / test_sim_opt / test_sim_oversize /
#       test_radix_eviction_policy_cli  → 37 passed
#
# 四 cell（实验机 bitahub /root/sg-unicache）：
#   for c in scan struct chat agentic; do
#     CELL=$c PAGE=8 CAP=198000 ARMS=chain_lru CHAIN_FLOOR=8192 \
#     CHAIN_WEIGHT=0.02 CHAIN_GATE=multiturn OUT_DIR=/root/sim_runs/engine_chain/$c \
#     /root/sl_old/bin/python -u /root/sim_runs/sim_engine_cell_v2.py
#   done
```
