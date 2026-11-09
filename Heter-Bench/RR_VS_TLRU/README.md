# RR vs T-LRU：burst / pause 两阶段 + 同时回归 vs 分批回归

目标：暴露小流量涌入把长挂起会话的 KV 挤掉这个场景，并在同一场景下比较
`RrEvictPolicy`（会话轮询 + 配额）与 `TlruEvictPolicy`（尾部预算）。

## 1. 场景与时序

```
t0        victim 同相位起跑（arrival.mode=fixed, step_s=0.05）
          victim = agentic_idle（alfred 多轮，截断到 12 轮），只跑 3 轮
t_pause   全部 victim 越过 pause 点 → BurstGate 放行搅拌流
t_pause   ── burst 窗口开始：agentic_active（apps 单轮，λ=6/s）持续涌入 ──
          这段窗口内 victim 的 KV 被挤出；挤掉多少取决于淘汰策略
t_open    burst 窗口关闭（burst_duration_s）
t_open    ── 回归方式（本实验的自变量之一）──
            simultaneous：全部 victim 同时恢复 → 一次性重 prefill（thundering herd）
            batched     ：victim i 在 t_open + (i // wave_size) * interval 恢复
t_tail    victim 跑完剩余 9 轮
```

victim 与搅拌流**都**被强制标成 QAGENT（`force_agentic`），所以：
- 不再有 QTRANS 被 one-shot 带先清掉；
- 也不再有「间隔 ≥ 8s → QCHAT」把长挂起会话踢出插件作用域；
- RR 与 T-LRU 拿到 100% 的流量，差别只剩候选规则本身。

## 2. 三个臂（`force_agentic` 是环境，不是处理）

| 臂 | 配置 | 说明 |
|---|---|---|
| `a0_control` | `{"force_agentic": true}` | 纯全局 `victim_key`（`_reuse_score` = lognormal 复用概率） |
| `a1_rr` | `+ rr.fair_rr=true, kappa=0.5` | 会话级轮询，单会话最多削 `loss_cap(L)` |

插件形态的 T-LRU 臂（原 `a2_tlru`，`{"tlru": {...}}` 分组）已从主干移除，
归档在分支 `exp/tlru-plugin-arm`（对应 tag `pre-alpha-removal`）。
原生 T-LRU 臂见下节 `a3_native_tlru` / `a4_tlru_t90` / `a5_tlru_t200`。

三臂共用同一棵 tree、同一套评分、同一个 one-shot 带（空）。差值可归因到插件。

## 3. 矩阵与耗时

| 轴 | 取值 |
|---|---|
| 臂 | a0 / a1 / a2 |
| 回归方式 | simultaneous / batched |
| KV 预算 | **22000 / 30000**（实测标定，见下） |
| seed | **42 / 43**（默认 2 个，够做同号校验） |

### 压力标定（两轮实测的教训）

**第一轮：压力过大（6×）。** 搅拌流 `max_record=300, lambda_rate=6`，60 s 窗口涌入
约 360k tokens = 60k 预算的 6 倍。三个臂的 resume 命中率完全相同（6.17 / 6.14），
因为 6 倍周转下任何策略都保不住东西。归档：`/root/rr_vs_tlru_runs/result_saturated_6x/`。

**第二轮：压力过小（0×）。** 把搅拌流降到 70×387 token 后，整轮**零淘汰**：
`victim_src` 一行都没打印，常驻峰值 35.7k < 预算 60k。归档：
`/root/rr_vs_tlru_runs/result_no_eviction_60k_100k/`（12 个点，可作「无淘汰」对照）。

**实测数据（budget=60k 时单 run）**：

| 任务 | 请求数 | prompt tokens | 每条均值 | 命中率 |
|---|---|---|---|---|
| agentic_idle | 96 | 57,535 | 599 | 84.4 % |
| agentic_active | 70 | 27,111 | **387** | 6.4 % |
| 合计 | 166 | 84,646 | — | 59.4 % |
| **常驻峰值** | | | | **35.7k** |

关键结论：**预算是压力杠杆，搅拌流不是**。常驻 35.7k 是上界，预算必须低于它：

| 预算 | 相对常驻 | 预计淘汰 | 说明 |
|---|---|---|---|
| 22000 | 1.62× | ~13.7k | 压力明显，策略有充足介入空间 |
| 30000 | 1.19× | ~5.7k | 温和压力，对策略差异最敏感 |
| ~~60000 / 100000~~ | 0.60× / 0.36× | **0** | 无淘汰，三臂必然相同 |

**耗时构成**（用 `result/burst/burst_n4_base.json` 实测校准：144 请求 / 157 s）：

| 项 | 单次 |
|---|---|
| server 启动 | ~90 s ← **最大固定开销** |
| bench（396 请求 + 60 s burst 窗口） | ~110 s |
| `/flush_cache` | ~2 s |

因此 `run_matrix.sh` 默认按 `(arm, budget)` **复用 server**：server 只依赖这两个轴，
`resume_mode` / `seed` 全在 bench 侧。seed 之间用 `/flush_cache` 隔离（见 §5）。

| 场景 | 运行数 | server 启动 | 预估 |
|---|---|---|---|
| 试点（`ARMS="a1_rr a3_native_tlru" RESUME_MODES=simultaneous BUDGETS=30000 SEEDS="42"`） | 2 | 2 | **~7 min** |
| 快跑（默认：3 臂 × 2 模式 × 2 预算 × 2 seed） | 24 | **6** | **~1 h** |
| 全量（3 seed） | 36 | 6 | ~1.4 h |
| 不复用（`GROUP_BY_SERVER=0`，24 次启动） | 24 | 24 | ~1.5 h |

```bash
# 试点（7 分钟，先确认脚本/相位都对）
ARMS="a1_rr a3_native_tlru" RESUME_MODES=simultaneous BUDGETS=30000 SEEDS="42" bash run_matrix.sh

# 快跑（默认）
bash run_matrix.sh

# 全量
SEEDS="42 43 44" bash run_matrix.sh
```

## 3.5 第四个臂：原生 TLRU（unified 后端）

前面的 a2 是「TLRU 公式做成插件」；`a3_native_tlru` 是**仓库原生的 TLRU**：

| 项 | a2（插件） | a3（原生） |
|---|---|---|
| 后端 | `HeterCacheRadixCache` | `UnifiedRadixCache`（registry 硬校验） |
| 启动参数 | `--enable-heter-cache --radix-eviction-policy heter_cache` | `--radix-eviction-policy tlru`（**不加** `--enable-heter-cache`） |
| 配置键 | `{"tlru": {...}}` | `{"threshold": ..., "next_prompt_estimate": ...}` |
| 作用域 | 仅 QAGENT（`force_agentic` 下即全部） | 全局所有叶子 |
| 记账 | `_tlru_*` 由插件在插入期维护 | `unified_tree_core` 原生维护 |
| profile | `[HeterCache profile]` 有 `victim_src` / `tlru:` | **无**（没有这套日志），只能看 resume 指标 |

```bash
# 原生 TLRU，与三个插件臂完全相同的负载/预算/seed
ARMS=a3_native_tlru bash run_matrix.sh

# 阈值敏感性（不动代码，用 POLICY_CONFIG 覆盖）
POLICY_CONFIG=configs/arm_a3_native_tlru_200.json \
ARMS=a3_native_tlru RESUME_MODES=simultaneous BUDGETS=22000 SEEDS="42" bash run_matrix.sh
```

阈值语义：`budget = L + Q_hat − threshold`，`threshold` 越大越激进。
挂起点链长约 400–500 token（不是会话最终长度 3487），所以默认取 **320**
（= 只让最后 ~250 token 可淘汰）。可选档位已备：`_200`（更保守）/ `_500`（更激进）。

## 4. 判据：看 resume 轮，不看全局均值

关键行是 `resume_phase == "resume"` 的那些请求（挂起后第一个回归请求）：

| 指标 | 含义 | 期望（策略有效时） |
|---|---|---|
| `resume_hit_rate` | `Σcached/Σprompt`（仅 resume 轮） | 越高越好 |
| `zero_cache_victims` | resume 轮 `cached_tokens == 0` 的 victim 数 | **越少越好**（说明整条链没被掏空） |
| `resume_ttft_p99` | resume 轮 TTFT 尾部 | 越低越好 |
| `resume_recompute` | resume 轮重算 token 总量 | 越低越好 |

`speedup`（同时 vs 分批）单独看：`simultaneous` 的 p99 应当更差（thundering herd），
而 `batched` 抹平峰值——但**它也可能掩盖策略差异**（压力被均摊）。
这正是本实验要区分的：策略差异是否只在「同时回归」时才暴露。

## 5. 有效性检查（先看这个，否则实验白跑）

`summarize.py` 会打印：

| 检查 | 期望 | 不满足的含义 |
|---|---|---|
| `force_agentic` 生效 | profile 的 `victim_src` 里 `qagent` 占 ≈100%、`one_shot=0` | 标签强制没生效，插件仍在抢残余流量 |
| burst 真的产生压力 | 记录 `phases.burst_start_s / burst_end_s`，且 resume 轮 `cached_tokens` 有明显下降 | 预算太松，没有淘汰发生 |
| victim 真的挂起 | `phases.victim_resumed_s` 与 `burst_end_s` 的差值符合模式 | 回归逻辑没接上 |
| seed 之间真的隔离 | 每次开始前都成功 `/flush_cache`（脚本会打印 WARN） | 复用 server 时上一轮的状态会串味 |

## 6. 产物

```
result/<arm>/<resume_mode>/<budget>/
  server_<seed>.log         # 含 [HeterCache profile]
  profile_<seed>.txt
  <seed>.json               # summary，含 phases{burst_start_s, burst_end_s, victim_resumed_s}
  <seed>.csv                # 逐请求明细，含 resume_phase 列
  bench_<seed>.log
```

## 7. 与 TLRU_VS_RR 的区别

| | TLRU_VS_RR | RR_VS_TLRU（本目录） |
|---|---|---|
| 负载 | 单负载 agentic，无相位 | burst/pause 两阶段，victim + 搅拌双流 |
| agentic 覆盖率 | ~60%（受 8s 阈值限制） | **100%**（`force_agentic`） |
| 自变量 | 只有插件 | 插件 × 回归方式（同时/分批） |
| 指标 | 全局命中率 | resume 轮命中率、零命中 victim 数、resume TTFT |
