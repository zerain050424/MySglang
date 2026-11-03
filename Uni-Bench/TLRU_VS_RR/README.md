# T-LRU vs RR-EVICT（Agentic 负载）

对比两种 **QAGENT 候选改写器** 在同一棵 tree、同一套全局评分下的表现：

| 臂 | 插件 | 判定 |
|---|---|---|
| `a0_control` | 无 | 纯 `victim_key`（alpha × lognormal 复用分数） |
| `a1_rr` | `RrEvictPolicy` | 会话级轮询 + 单会话配额 `loss_cap(L)` |
| `a2_tlru` | `TlruEvictPolicy` | tel-safe 尾巴优先，用尽后退化为 LRU |

三臂都跑 `--radix-eviction-policy simple_hybrid --enable-simple-hybrid`，
差别只有 `--radix-eviction-policy-config` 里的一个分组。因此差值可归因到插件层。

---

## 1. 为什么这样比（设计取舍）

| 潜在歧义 | 本方案的处理 |
|---|---|
| TLRU 原生实现锁死在 `UnifiedRadixCache`（`registry.py` 硬校验），换成 classic/simple_hybrid 会因 `_tlru_*` 字段缺失而崩 | 不跨后端对比。TLRU 以插件形式复用同一棵树，`_tlru_history_len` 由插入期回调维护 |
| `one_shot` 硬优先带会在插件之前抢走决策 | **未消除，但被量化**：`verify_effective.py` 从 profile 读出 `victim_src: one_shot=<tokens>` 的占比。占比高则该轮实验只能表述为「可调度带残余部分」 |
| 三种评分公式（QTRANS/QCHAT/QSTRUCT/QAGENT）并存 | 两臂完全相同，做差值时抵消；报告中注明 `occ` 行不代表评分分组 |
| alpha 在线更新干扰 | 单负载下只有一个 alpha key，`eff≡1`，alpha 恒为 1（惰性）。默认 `update_interval=30s` 保留 profile 打印 |
| TLRU 的 L 口径 | 已用**插入期高水位**（`_tlru_history_len`）复刻 unified 语义，不是 `SessionRegistry.st_l` |

**不做**：混合负载、线上延迟剖析、修改 `unified` 侧代码、跨后端（`a3_tlru_native`）。

---

## 2. 负载与固定量

- 数据：`TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl`（多轮 agentic，`mu=1.81, sigma=1.092`）
- 所有臂必须一致的量：模型、`--max-total-tokens`、`--page-size`、`--tp`、数据、`--num-sessions`、`--concurrency`、`--arrival-scale`、`--wait-scale`、`--seed`
- 压力梯度：`--max-total-tokens ∈ {100000, 200000, 400000}`（策略差异只在有淘汰压力时出现）
- 重复：`--seed ∈ {42, 43, 44}`，同一 seed 下各臂看到**完全相同**的到达时序 → 可配对

矩阵规模：3 臂 × 3 预算 × 3 seed = 27 次。

---

## 3. 参数校准（不要拿调好的 RR 打没调的 TLRU）

```bash
/root/sl_old/bin/python calibrate_tlru.py \
  --data-path ../TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.jsonl
```

输出 `Q_hat`（下一轮 prompt 增量中位数）与 3 档阈值候选（`threshold = Q_hat + x·median(L)`）。
`TlruParams` 要求 `threshold > next_prompt_estimate`，否则启动即报错。

### 已校准结果（AgentBank alfred，前 800 个 session）

| 统计量 | 值 |
|---|---|
| session 数 / 轮数 | 800 / 31199 |
| 每轮 prompt 长度 p50 / p90 / max | 2001 / 4986 / 9547 |
| 每个 session 末轮长度 p50 / p90 | 3487 / 6187 |
| **下一轮增量 `Q_hat`**（p50） | **71**（p90=193，max=494） |
| 阈值候选（0.10/0.25/0.50 × L_p50） | **420 / 943 / 1815** |

`configs/a2_tlru.json` 默认取中间档 `threshold=943, next_prompt_estimate=71`。

### 两个"退化端"（扫参时必须避开）

因为 `budget = max(L + Q_hat - threshold, 0)`，两端都会退化成纯 LRU：

| 条件 | 后果 |
|---|---|
| `threshold <= Q_hat` | `budget > L`，没有任何节点 tel-safe → 纯 LRU（`TlruParams` 直接报错拦住） |
| `threshold >= max(L) + Q_hat` | `budget = 0`，**所有**节点 tel-safe → 全部同一优先级，仍按 `last_access_time` 排序 = 纯 LRU |

所以只有 `Q_hat < threshold < max(L)+Q_hat` 区间才有真实效果。在 alfred 上大约就是
**71 … 9600**；若用 `threshold=9216` 这种值，实验会得到“TLRU 与 LRU 完全一致”的假象。
`verify_effective.py` 用 `tel_safe` 计数占比来识别这种情形（全 tel-safe 时报吿）。

RR 侧默认 `kappa=0.5 / min_retain_tokens=512 / min_retain_frac=0.3`；
若要对 TLRU 扫阈值，就同步对 RR 扫 `kappa ∈ {0.3, 0.5, 0.7}`，两边都报「默认」与「各自最优」。

---

## 4. 运行

```bash
# 单点
BUDGETS=200000 SEEDS=42 ARMS=a2_tlru bash run_matrix.sh

# 全矩阵（约 27 次；每次 3–10 分钟 + server 启动）
bash run_matrix.sh

# 汇总
/root/sl_old/bin/python summarize.py            # 表 + 配对差值
/root/sl_old/bin/python verify_effective.py     # 插件是否真的生效
```

`run_matrix.sh` 每次运行会：停旧 server → 起新 server → 等 `/health` → 跑 bench → 存 JSON → 抓 profile 关键行。
已存在的 `result/<arm>/<budget>/<seed>.json` 会跳过（可续跑）。

---

## 5. 指标与判据

主指标 `cache_hit_rate_pct`（`Σcached/Σprompt`，bench 直接输出）。
辅指标 `ttft_p50/p90/p99_ms`、`num_errors`。

判据（跑之前先写死，避免事后解释）：

1. 主结论看**配对差**：同一 `(seed, budget)` 下 `a1 − a2`；
2. 3 个 seed 的 3 个差值**同号**才算结论，否则报「无显著差异」；
3. `|Δ| < 1` 个百分点视为持平（单次运行本身有波动）。

---

## 6. 有效性检查（必做）

`verify_effective.py` 会读 `profile_<seed>.txt` 并给出：

| 检查 | 期望 | 不满足的含义 |
|---|---|---|
| `a2` 的 `tlru: tel_safe=` | > 0 | 阈值配错（`Q_hat ≥ threshold`），TLRU 实际退化成 LRU |
| `a2` 的 `tlru: fallback=` | 远小于淘汰次数 | 该轮没有 QAGENT 叶子，插件没被询问 |
| `victim_src` 的 `one_shot` 占比 | 越低越好 | 高占比说明 one-shot 带抢走了决策权，实验只能部分解释 |
| `victim_src` 的 `qagent` 占比 | 高 | 说明标签状态机把流量判成了 agentic |

---

## 7. 产物

```
result/<arm>/<budget>/
  server_<seed>.log      # server 原始日志（含 [SimpleHybrid profile]）
  profile_<seed>.txt     # 抽出的 victim_src / tlru / one_shot_evicted 行
  <seed>.json            # bench 结果（含 per_task 细分）
  <seed>.csv
  meta_<seed>.json       # 本轮的 arm config + bench 参数（复现用）
```

---

## 8. 已知限制

1. **TLRU 只在 QAGENT 上生效**：它是「候选修正」而不是全局接管，非 QAGENT 叶子仍由 `victim_key` 决定。这是为了与 RR 同域可比，代价是它不是 unified 版 TLRU 的等价物。
2. **one-shot 带仍在插件之前**：`QTRANS`（新插入、尚未被再次命中的节点）默认就是 one-shot，所以真实负载下这条带不是空的。
3. 绝对命中率不可与历史（不强制标签的）结果直接对比，只能组内对比。
