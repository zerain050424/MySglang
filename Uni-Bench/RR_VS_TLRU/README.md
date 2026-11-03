# RR vs T-LRU：burst / pause 两阶段 + 同时回归 vs 分批回归

目标：暴露「小流量涌入把长挂起会话的 KV 挤掉」这个场景，并在同一场景下比较
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
| `a0_control` | `{"force_agentic": true}` | 纯全局 `victim_key`（alpha × lognormal） |
| `a1_rr` | `+ rr.fair_rr=true, kappa=0.5` | 会话级轮询，单会话最多削 `loss_cap(L)` |
| `a2_tlru` | `+ tlru.enabled=true, threshold=943, Q=71` | tel-safe 尾巴先走 |

三臂共用同一棵 tree、同一套评分、同一个 one-shot 带（空）。差值可归因到插件。

## 3. 矩阵与耗时

| 轴 | 取值 |
|---|---|
| 臂 | a0 / a1 / a2 |
| 回归方式 | simultaneous / batched |
| KV 预算 | 60000 / 100000 |
| seed | **42 / 43**（默认 2 个，够做同号校验） |

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
| 试点（`ARMS="a1_rr a2_tlru" RESUME_MODES=simultaneous BUDGETS=60000 SEEDS="42"`） | 2 | 2 | **~7 min** |
| 快跑（默认：3 臂 × 2 模式 × 2 预算 × 2 seed） | 24 | **6** | **~1 h** |
| 全量（3 seed） | 36 | 6 | ~1.4 h |
| 不复用（`GROUP_BY_SERVER=0`，24 次启动） | 24 | 24 | ~1.5 h |

```bash
# 试点（7 分钟，先确认脚本/相位都对）
ARMS="a1_rr a2_tlru" RESUME_MODES=simultaneous BUDGETS=60000 SEEDS="42" bash run_matrix.sh

# 快跑（默认）
bash run_matrix.sh

# 全量
SEEDS="42 43 44" bash run_matrix.sh
```

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
  server_<seed>.log         # 含 [SimpleHybrid profile]
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
