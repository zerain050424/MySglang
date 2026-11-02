# 负载设计：临界并发 × 稀疏活跃 × 回归浪潮

## 1. 理论目标

三个量：`n` 会话总数、`n_a` 同时活跃数、`m = pool/L` 缓存容量（以会话为单位）。
目标区间 `n_a < m < n`：能装下活跃的Agentic内容,但注入长空闲Agentic内容就会发生溢出,**必须驱逐空闲缓存才能保住活跃缓存**。

## 2. 

| 原则 | 旋钮 | 落点 |
|---|---|---|
| ① 临界并发 `n_a < m < n` | `n_a`、`n_i = n − n_a`、`L`、`pool` | 逻辑需求是物理容量的 1.4 到 1.9 倍,活跃请求只占容量的 70%,确保淘汰发生,但活跃会话可以正常写入|
| ② 长前缀对话先活跃再挂起 | 选择长上下文对话,在N轮后挂起| `L ≈ 5.5–8.5k` 能让冷 prefill达到 **1.1–1.7 s**|
| ③ 集中回归 | idle 组同一轮次、同一 `pause_s`、`arrival.step_s = 0.05` | 8 条idle对话同时回归|


## 4. 可行域（pool 36k / 73k，κ=0.5，L=6k）

| pool | m = pool/L | n_a 允许 | n 需满足 | ΣL/pool | ΣκL/pool |
|---|---|---|---|---|---|
| 35,975 | 6.0 | 2–5 | ≥ 7 | 1.4–1.9 | 0.70–0.95 |
| 73,121 | 12.2 | 4–11 | ≥ 13 | 1.4–1.9 | 0.70–0.95 |

## 5. 负载来源与切分

- 来源：**APIGenMT**（452 token/轮，48.8% 的轮有工具调用，L 可达 16k）
- `idle` 组：第 10 轮时链长 ∈ **[5.5k, 9.5k]**（长前缀、冷回归代价prefill时长高达1.1–1.7 s）
- `active` 组：第 10 轮时链长 ∈ **[1.5k, 2.8k]**（短链，持续写入，成本低）
- 两组都截断到 **12 轮**（10 轮 burst + 1 次 pause + 1 轮恢复）

选这个组合的原因：APIGenMT 每轮新增 ~200 token（不是 452），
所以"重前缀"只能在**早期就长**的会话里找；同时活跃组用短链，避免它们的增长吃掉池子。

## 6. 时间轴

```
t=0      n_a + n_i 条会话同时起跑（step 0.05 s），turn1..10，gap = 6 s
t≈60 s   idle 组进入 pause（60 s）；active 组继续写入（~200 tok/轮）
         窗口内插入 ~0.2–0.5×pool 的新内容 → 按 LRU 顺序吃掉空闲组
t≈120 s  idle 组 8 条同时回归（turn 11）← burst 窗口，测 TTFT / queue / prefill
t≈132 s  turn 12，观察恢复
```

## 7. tasks 配置模板

```yaml
tasks:
- name: agentic_active          # 活跃组：持续写入，是唯一的压力源
  data_path: <abs>/TaskData/burst/active_<N>_<name>.jsonl
  arrival: {mode: fixed, offset_s: 0.0, step_s: 0.05}
  gap_mode: fixed
  gap_s: 6.0
- name: agentic_idle            # 空闲组：第 10 轮后统一 pause
  data_path: <abs>/TaskData/burst/idle_<N>_<name>.jsonl
  arrival: {mode: fixed, offset_s: 0.0, step_s: 0.05}
  gap_mode: fixed
  gap_s: 6.0
  pause_after_turn: 10
  pause_s: 60
```

## 8. 策略（RR-EVICT）

```
每会话配额 quota_i = min(κ·L_i, L_i − max(512, 0.3·L_i))
1) 每个会话取当前最深叶子作为代表
2) d_i ≥ quota_i 的会话 **退出候选集**（硬准入）
3) 候选里选 _rr_last 最小者（最久没被服务）
4) 服务后 _rr_last = seq++（排到队尾）
5) 全部配额耗尽,回退 min(victim_key)（LRU）
```


