# TinyLRU 正式实验方案（one-shot × 结构复用）

> 范围声明：**只考察 one-shot 硬优先带与 Q_struct 队内策略（标签版 TinyLRU）**。
> 不考察 Agentic 队列、RR 会话公平、T-LRU、跨队列 alpha 仲裁——这些一律关闭或冻结。
> 对照组只用 **SGLang 原生 LRU / SLRU**。

---

## 1. 待验证假设与预注册判据

| # | 假设 | 判据（跑之前定死） |
|---|---|---|
| **H1** | 有 KV 压力时，TinyLRU 让**结构复用前缀**更久驻留 | `ret@P full_rate` 相对 A2（one-shot 带、无 TinyLRU）≥ **+5pp**，请求级配对 bootstrap 95% CI 排除 0 |
| **H2** | 该驻留转化为命中率，且**不靠牺牲干扰项** | 结构复用子集 hit ≥ **+3pp**；干扰（burst）子集 hit 回退 ≤ **1pp** |
| **H3** | 收益来自「会话内复用不被打进 one-shot 带」 | A3 − A4 ≥ **+3pp**（A4 = `local_hot=one_shot`，规则原样），且 A4 的 `qtrans` 淘汰占比显著更高 |
| **H4** | 无压力时策略应退化为等价 | `pool/工作集 = 1.0` 时 A0/A2/A3 的 hit 极差 < **0.5pp** |
| **H5** | 命中率传导到 TTFT | 在线 `ttft_p50` ≤ **−3%**，`ttft_p99` 回退 ≤ **5%** |
| **H6** | 干扰强度越大，TinyLRU 的相对收益越大 | 收益随「每块 burst token / 池容量」单调上升（比值 0.5 → 1 → 2） |

**否决条件**：干扰子集回退 > 1pp，或 H4 不成立（说明测到的是噪声/框架差异）。

---

## 2. 对照链路（三点分解）

实验只关心两条机制：**one-shot 带** 与 **TinyLRU 标签**。因此对照是三点链，不是大矩阵：

| 臂 | 配置 | 隔离出的贡献 |
|---|---|---|
| **A0** | `--radix-eviction-policy lru` | SGLang 原生基线（无 one-shot 带、无队列标签） |
| **A1** | `--radix-eviction-policy slru` | 上游分段 LRU（外部参照：说明「自研是否值得」） |
| **A2** | `simple_hybrid` + `tiny_lru.policy="off"` | **one-shot 带 + 分类器标签** 的贡献 = A2 − A0 |
| **A3** | `simple_hybrid` + `tiny_lru.policy="tag_lru"` | **TinyLRU** 的贡献 = A3 − A2（主对象） |
| **A4** | A3 + `local_hot="one_shot"` | H3 消融：把会话内复用也打进 one-shot 带（只在主压力点跑） |

固定配置（所有臂一致，避免混入跨队列因素）：

```jsonc
{
  "alpha": {"update_interval": 1e9},   // 冻结 alpha 在线重平衡（不做跨队列仲裁）
  "rr":    {"fair_rr": false},         // 关闭会话公平轮询
  "tlru":  {"enabled": false}          // 关闭 T-LRU
}
```

> `alpha` 仍会作为静态权重乘进 `reuse_score`，但默认各队列 1.0 且不再更新，
> 因此本实验里不存在「按命中效率重新分配容量」的跨队列行为。

---

## 3. 负载构造：结构复用 + 单轮 burst 干扰

用 `wtl/synth.py` 生成，**两个成分必须可分辨**（分析时按 `session_id` 前缀分组，不需要改代码）：

| 成分 | session id 前缀 | 形态 | 作用 |
|---|---|---|---|
| **结构复用** | `synth-v<k>-s<j>` | `variants` 个模板，每模板 `breadth` 个会话，每会话 `turns` 轮 | 被保护的共享前缀（M1/H1 的对象） |
| **单轮 burst 干扰** | `synth-noise-<i>` | 单轮、一次性、无复用 | 制造 one-shot 压力并冲刷 KV 池 |

### 3.1 关键参数约束

1. **`turns = 3`（必须 < 5）**：分类器在 `turn_count < 5` 时才可能判 `STRUCT`；
   ≥5 轮会被判 `AGENTIC/CHAT`，就混进了本实验明确排除的 agentic 路径。
2. **`prefix_depth P = 1024`**，分层取 `[256, 512, 1024]`；M1 报 `ret@1024`（整段模板）与
   `cover@1024`（部分保留）。
3. **`phased=True`**：模板按变体分块，**每块之间插入一段 burst**。
   这样 burst 会把池子刷一遍，下一个模板块必须重新从缓存里找前缀——
   这正是「单轮 burst 对话干扰结构复用」的可控形态。
4. **burst 长度 = 800 token**（对话量级，不是长文档），块内 burst 会话数由 `noise_frac` 决定。
5. **干扰强度**（H6 的横轴）：
   `ρ = 每块 burst token / 池容量`。设计取 `ρ ≈ 0.5 / 1 / 2` 三档，
   通过「burst 每块条数」或「池容量」调节，并在结果里记录实测 ρ。

### 3.2 数据量升级（三档，×2 递增）

| 档 | 模板数 M | 每模板会话 S | 轮数 T | 模板前缀 P | burst 会话数 | 模板请求数 | 总 session |
|---|---|---|---|---|---|---|---|
| **D1** | 8 | 8 | 3 | 1024 | 128 | 192 | 192 |
| **D2** | 16 | 8 | 3 | 1024 | 256 | 384 | 384 |
| **D3** | 32 | 8 | 3 | 1024 | 512 | 768 | 768 |

生成命令（D2 为例）：

```bash
python -m wtl.synth --out-dir data/exp/D2 \
  --prefix-depth 1024 --variants 16 --breadth 8 --turns 3 \
  --body-mu 5.5 --body-sigma 0.6 \
  --noise-frac 2.0 --doc-len 800 --phased
```

> `noise_frac=2.0` ⇒ burst 会话数 = 2 × 模板会话数；`--phased` 把 burst 均分到每个模板块之后。
> 需要 `wtl/replay.py --no-shuffle`（约 5 行改动）才能保住块顺序；在线端用
> `bench_unicache.py --no-shuffle`（已支持）。

---

## 4. 显存压力阶梯（与数据量正交）

用 `pool_ratio = 池容量 / 工作集`（工作集 = 不淘汰时的树占用，由回放器自动校准），
保证不同数据量之间可比：

| 档 | pool_ratio | 含义 |
|---|---|---|
| P1 | **1.0** | 无压力对照（H4：四臂必须趋同） |
| P2 | **0.6** | 轻度压力 |
| P3 | **0.35** | 主压力点（H1/H2/H3 的主战场） |
| P4 | **0.2** | 重度压力（看是否出现保护反噬） |

在线（真机）用绝对 token 数换算：`--max-total-tokens ∈ {200k, 100k, 50k}`，
对应同一份 D2 负载的 P1/P2/P4（实验前用 `nvidia-smi` 与 `--max-total-tokens` 试跑确认可达）。

---

## 5. 指标（只报用户要的三个 + 两个护栏 + 两个机制）

### 5.1 报告指标

| 指标 | 定义 | 来源 |
|---|---|---|
| **命中率** | `Σcached_tokens / Σprompt_tokens`，**按成分拆分**（struct / burst） | 离线 rows、在线逐请求 CSV |
| **结构复用保留度** | `ret@P full_rate`（模板前缀整段命中比例）、`cover@P`（部分保留） | `RetentionTracker` |
| **TTFT** | `p50 / p90 / p99`，并给 `prefill_ms` 与 `queue_time_ms` 分解 | 在线 `bench_unicache` CSV |

### 5.2 护栏

| 指标 | 判据 |
|---|---|
| burst 子集命中率 | 回退 ≤ 1pp（H2） |
| 无压力趋同 | P1 上 A0/A2/A3 极差 < 0.5pp（H4） |

### 5.3 机制证据（用来解释「为什么」）

| 指标 | 来源 | 期望 |
|---|---|---|
| `qtrans` 淘汰 token 占比（one-shot 带压力） | `evicted_by_tag` | A2 高（>80%）、A3 显著下降（<50%） |
| `tag.promote` / `counted` / `dedup` | `tag.snapshot()` | promote > 0 且随数据量上升 |

---

## 6. 实验矩阵（组数刻意压小）

### L1 离线回放（快、可复现；只出命中率与保留度）

| 阶段 | 目的 | 臂 | 数据量 | 压力 | seed | 组数 |
|---|---|---|---|---|---|---|
| **S1 主链路** | H1/H2/H4 | A0, A1, A2, A3 | D2 | P1, P2, P3, P4 | 42 | **16** |
| **S2 数据量升级** | 数据量 ×2 时收益是否保持 | A0, A2, A3 | D1, D2, D3 | P3 | 42 | **9** |
| **S3 干扰强度** | H6 | A0, A2, A3 | D2 | P3（ρ≈0.5/1/2） | 42 | **9** |
| **S4 消融** | H3 | A3, A4 | D2 | P3 | 42, 43 | **4** |
| **S5 复现性** | 单 seed 结论是否稳 | A0, A2, A3 | D2 | P3 | 43 | **3** |

合计 **41 组**；D1/D2/D3 单组约 20/40/80 s，`--jobs 12` ⇒ **约 4–6 分钟机时**。
种子只用 42（主扫）+ 43（仅主压力点复核），符合「减少种子数目」。

### L2 在线真机（只验证 TTFT 传导）

| 阶段 | 臂 | 负载 | 池 | seed | 次数 |
|---|---|---|---|---|---|
| **S6** | A0, A2, A3 | D2 | 200k / 50k tokens | 42 | **6 次** |

每次 `bench_unicache` 跑 D2（768 session、~1150 请求、并发 32）约 8–12 分钟 ⇒
**约 1–1.5 小时机时**。只跑 6 次是有意的：TTFT 用**同池同负载的配对残差**判断，
不需要多 seed 覆盖（每请求都是一次配对样本）。

---

## 7. 统计与判定

1. **配对**：同一份 trace、同一 seed 下，按 `(task, sid, turn)` 配对；配对样本数 = 请求数。
2. **区间**：对「汇总命中率差」做请求级重采样 bootstrap（10k 次）——已实现
   （`analyze.paired_ratio_delta`，统计量是 `Σcached/Σprompt`，避免逐请求比率方差爆炸）。
3. **多 seed**：主压力点用 42/43 两次，报「两 seed 的方向是否一致」；不一致即判为不稳定。
4. **多重比较**：只做 A2−A0、A3−A2 两组（外加 A4−A3），用 Holm 校正 α=0.05。
5. **TTFT**：报 p50/p90/p99 与配对 Δ，并给 `prefill_ms ≈ α + β·recompute_tokens` 的斜率
   一致性检查（β 应各臂一致，否则说明测的不是同一种东西）。

---

## 8. 执行步骤

```bash
cd Uni-Bench/W-TinyLRU
export PYTHONPATH=/path/to/sglang/python

# 0) 前置门控（机制没被行使就别跑正式矩阵）
python -m wtl.gate --backend real --arm A3

# 1) 生成三档负载（D1/D2/D3）
for d in D1:8 D2:16 D3:32; do
  n=${d%%:*}; m=${d##*:}
  python -m wtl.synth --out-dir data/exp/$n --prefix-depth 1024 \
    --variants $m --breadth 8 --turns 3 --body-mu 5.5 --body-sigma 0.6 \
    --noise-frac 2.0 --doc-len 800 --phased
done

# 2) S1 主链路（16 组）
python -m wtl.run_matrix --no-shuffle --data-dir data/exp/D2 --arms A0,A1,A2,A3 \
  --ratios 1.0,0.6,0.35,0.2 --seeds 42 --jobs 12 --backend real

# 3) S2/S3/S4/S5 见 §6 表格（同一入口，换 --data-dir / --ratios / --arms / --seeds）

# 4) 汇总（配对 CI + 判定表 + 两张图）
python -m wtl.analyze --offline result/offline --prefix-stats result/prefix_stats.json

# 5) 在线 TTFT（GPU 机器，6 次）
ARMS="A0 A2 A3" MODES=exp_D2 TOKENS_LIST="200000 50000" SEEDS=42 \
  bash scripts/run_online.sh
```

> `--no-shuffle` 与 `--data-dir` 是本次需要补的两个小参数（各 ~5 行）；
> 其余命令、指标、判定都已实现。

---

## 9. 交付物

1. **主表**：`数据量 × 压力 × 臂` → 命中率（struct / burst 分列）、`ret@P`、`cover@P`。
2. **图 1**：x = `pool/工作集`（P1→P4），y = 结构复用子集命中率，线 = 臂。
3. **图 2**：x = 模板前缀深度（256/512/1024），y = 保留度，线 = 臂。
4. **图 3**：在线 TTFT CDF（A0 vs A2 vs A3）+ 同 `recompute_tokens` 桶的配对残差。
5. **机制表**：`qtrans` 淘汰占比、`tag.promote/counted`、ρ（每块 burst token / 池容量）。
6. **判定表**：H1–H6 逐条 PASS/FAIL（`analyze.judge` 直接输出）。

---

## 10. 风险与控制

| 风险 | 控制 |
|---|---|
| 分类器把 3 轮模板会话判成 AGENTIC | `turns=3 < 5`，且 `tlru/rr` 关闭；结果里核对 `stype` 分布 |
| 干扰项与结构复用比例失衡（要么没压力、要么全被冲光） | 以 ρ = 每块 burst token / 池容量为控制变量，取 0.5/1/2 三档 |
| 单 seed 侥幸 | 主压力点跑 seed 42/43，方向必须一致 |
| 离线与在线分类口径不同（离线无 `tool_call`） | 负载不含 agentic；在线记录 `stype` 分布并与离线对照 |
| TTFT 抖动掩盖效应 | 只比同池同负载的配对残差；先看 `precompute_tokens` 的效应量再谈 TTFT |
| 保留度饱和（P1 上人人 100%） | P1 只作 H4 对照，不进 H1 统计 |

---

## 11. 与早期方案的关系

早先的三区版方案（window/probation/protected + 频数准入 + aging）已随实现一起从本分支移除。
本方案只保留它的两个有用结论作为设计前提：

1. **one-shot 硬优先带会吃掉 80%+ 的淘汰压力**（所以 A2−A0 是必须单独测的一项）；
2. **准入与 aging 在现实 trace 上基本不触发**（所以不再为它们设计实验）。
