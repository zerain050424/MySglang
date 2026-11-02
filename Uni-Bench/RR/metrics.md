# 预注册指标与判定阈值

## 1. 窗口定义

```
集中回归窗口 = task == "agentic_idle" 且 turn == pause_after_turn + 1（= 11）
             → 空闲组 8 条会话同时回归的那一批请求
恢复窗口   = task == "agentic_idle" 且 turn == 12
```

## 2. 主终点

| 指标 | 定义 | 为什么是主终点 |
|---|---|---|
| `burst_total_recompute` | 集中回归窗口内 `Σ (prompt − cached)` | 冷回归的真实代价（prefill token 量） |
| `burst_ttft_p99` | 集中回归窗口内 TTFT 的 99 分位 | 用户感知的尖峰 |
| `max_queue_ms` | 集中回归窗口内 `queue_time_ms` 最大值 | 队头阻塞（HOL blocking） |

## 3. 次终点（解释用）

| 指标 | 定义 |
|---|---|
| `hit_all` | 全量 `1 − Σrecompute/Σprompt` |
| `burst_hit` | burst 窗口的 `1 − Σrecompute/Σprompt` |
| `active_hit` | 活跃组（`agentic_active`）命中率 ← **必须检查，防止拿活跃组当燃料** |
| `burst_max_recompute` | burst 窗口单请求最大重算 |

## 4. 判定阈值（预先写定）

```
RR 有效 ⟺ (P1) burst_total_recompute 相对 base ≤ −15%
        且 (P2) burst_ttft_p99 相对 base ≤ −20%
        且 (P3) active_hit 相对 base 下降 ≤ 2 pt
阴性对照 ⟺ 在 ΣL/pool < 1.1 的格子里，|Δhit_all| ≤ 1 pt
```

## 5. 守卫（任一不满足则该格标注、不静默丢弃）

```
(S1) 回归跨度（burst 8 条请求的时间戳极差）—— 原始阈值 < 1.5 s；见修订
(S2) 请求失败 / OOM = 0
(S3) hit+miss = prompt 逐请求成立（驱动侧自动满足）
(S4) ΣL/pool ≥ 1.3 才有资格进入"压力格"
```

**修订记录（诚实条目）**：S1 原定 `< 1.5 s`，实测 5 个压力格为 2.7–11.5 s。
原因：8 条请求即使同时发出，服务端也需 8×(0.5–1.5 s) 串行 prefill，跨度下界 ~4 s。
→ 阈值修订为 `< 15 s`，并在 results.md 中保留原始数值与偏离标记。
该修订发生在**看到跨度数值之后**，因此"浪潮紧密度"不作为结论的支撑点。
