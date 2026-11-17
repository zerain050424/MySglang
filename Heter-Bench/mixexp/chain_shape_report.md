# chain_shape：按「下游依赖体量」选受害者——浅段优先保护、砍链尾（tail-stress 实测）

> 时间：2026-11-18 · 分支 `W1`（`00e8985` + 本批）· 载体 `sim/` engine 仿真器
> 目标：把「整条前缀丢失」变成「部分丢失」。这是 cost_oracle_gap 证否深度代价加权后，
> 唯一还没证否、且 tail-stress 上有 920ms → 184ms 空间的方向。
> 新策略：``--radix-eviction-policy chain_lru``（`ChainLRUStrategy`，全部 opt-in，
> 不带参数时其他策略行为逐位不变）。

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

## 四、扫参与稳健性

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

## 五、边界与下一步

* **本机没有 exp3 四 cell 数据**，「其他负载不劣化」尚未验证——回归只能证明
  默认行为不变（37 个 sim/CLI 回归全过）+ tail-stress 三档压力方向一致。
  上实验机后应把 chain_lru 加进四 cell 矩阵复跑，重点看 scan（纯树形保护在
  冲刷负载上等于给每条一次性链留 floor×驻留链数的死重，需要 gate 或容量证据）。
* 淘汰粒度是「轮段」节点（一条 warm 链 ≈ 8 节点 × ~1500 token），部分丢失的
  分辨率受此限制；页粒度剥链是另一个话题，不在这批里做。
* ``get_priority`` 每次沿父链走到根（求 remain + 证据），代价 O(链深)。本负载
  链深 ≈ 8；agentic 长链（60+ 轮）会到几十，仿真无感，线上要实测再定，
  必要时可增量缓存 ``remain``。
* gate=none 在有真实「永不回来」流量的负载上会积累死锚。tail-stress 里 cold
  尾巴天然在深层（remain=7k）不受影响，但一般负载需要一个「锚老化」出口——
  与 heter 的 TTL/会话寿命判据合并是自然方向，这批不做。

## 六、复现

```bash
cd <repo-root>
PYTHONPATH=$PWD:$PWD/python ~/venvs/sim/bin/python \
  Heter-Bench/mixexp/tools/build_tail_stress.py \
  --arms lru,chain_lru,opt --floor 8192 --weight 0.02 --gate none
# 单测（不需要 torch）：python test/manual/test_chain_lru.py  → ALL PASSED
# 回归：test_sim_determinism / test_sim_metrics / test_sim_opt / test_sim_oversize /
#       test_radix_eviction_policy_cli  → 37 passed
```
