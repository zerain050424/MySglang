# 真实生产 trace 的轮间隔拟合（用于校准 P 的先验）

> 数据已下载到 `/root/traces/`；分析脚本 `exp/analyze_traces.py`；结果 `result/trace_gap_fit.csv`
> 2026-10-27

## 1. 数据源与可用性

| 文件 | 大小 | 字段 | 多轮关系 | 能用什么 |
|---|---|---|---|---|
| `qwen_traceA_blksz_16.jsonl`（阿里云百炼 to-C） | 54 MB | `chat_id / parent_chat_id / timestamp / turn / type / input_length / output_length / hash_ids` | ✅ `parent_chat_id` + `turn` | **轮间隔**、类型分布、长度分布 |
| `qwen_traceB_blksz_16.jsonl`（百炼 to-B API） | 92 MB | 同上 | ❌ 全部 `turn=1` | 到达过程（24 req/s）、长度 |
| `mooncake_trace.jsonl`（Mooncake arxiv） | 4.3 MB | `timestamp / input_length / output_length / hash_ids` | ❌ | 只有到达过程，且是**批式** |
| `cachewise-coding-traces/`（Claude Code 真实用户） | 仓库内 | `llm_call`/`tool_call` + ISO 时间戳 + cache token | ✅ `session_id` | **agentic 轮间隔** |

下载方式（Bailian 是 Git LFS，必须走 media 域名）：

```bash
curl -sL -o qwen_traceA_blksz_16.jsonl \
  https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/main/qwen_traceA_blksz_16.jsonl
curl -sL -o mooncake_trace.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl
```

## 2. 百炼 traceA 的轮间隔（n=19,957，43,058 条请求，时间窗 2h）

| type | n | p50 | p90 | p99 | **mu** | **sigma** | UniCache `TASK_CONFIG` |
|---|---|---|---|---|---|---|---|
| text | 15,297 | 106.7 s | 712.2 | 2212.3 | **4.78** | **1.28** | 4.770 / 1.279 |
| search | 3,763 | 118.7 | 777.1 | 2170.7 | **4.96** | **1.14** | 4.962 / 1.139 |
| file | 762 | 149.7 | 690.5 | 2264.4 | **5.08** | **1.06** | 5.077 / 1.064 |
| image | 135 | 75.0 | 353.2 | 1657.4 | **4.44** | **1.09** | 4.445 / 1.091 |
| ALL | 19,957 | 110.6 | 718.7 | 2207.4 | 4.82 | 1.25 | — |

**四位小数逐项吻合。** 说明 UniCache 里 text/image/search/file 的 `(mu, sigma)` 就是从这份
百炼 trace 拟合来的，不是拍脑袋。这四个常数可以信任。

## 3. agentic 那一组没有生产依据

`TASK_CONFIG` 里 `MULTI_TURN_AGENTIC = (1.81, 1.092)`，但：

- 百炼 traceA **没有 agentic 类型**（只有 text/image/search/file）；traceB 全单轮；
- AgentBank 原始数据（1.81/1.092 的来源）**没有时间戳**（只有 token ids），无法自己拟合；
- 手边唯一的真实 agentic 数据是 `cachewise-coding-traces`（Claude Code，94 会话 / 20,634 llm_call）：

| 量 | 实测（cachewise） | UniCache 假设 |
|---|---|---|
| 轮间隔中位 | 8.3 s | 6.1 s |
| p90 / p99 | 56.4 / 463 s | — |
| lognormal 拟合 | **mu=2.47, sigma=1.21** | 1.81 / 1.092 |

⇒ agentic 常数应改用 **(2.47, 1.21)**（或按 project 分别拟合）。

## 4. 换成实测参数后 P 差多少

`P(idle) = 1 − CDF_lognormal(idle; mu, sigma)`：

| idle | text（百炼实测） | agentic（UniCache） | agentic（ClaudeCode 实测） |
|---|---|---|---|
| 5 s | 0.993 | 0.573 | 0.762 |
| 10 s | 0.974 | 0.326 | 0.555 |
| 30 s | 0.859 | 0.073 | **0.221**（3.0×） |
| 90 s | 0.587 | 0.007 | **0.047**（6.8×） |
| 300 s | 0.235 | 0.0002 | **0.004**（20.7×） |

两件事：

1. **UniCache 的 agentic 常数把"暂停 90 秒"判得比真实轨迹严 6.8 倍**、300 秒严 20.7 倍。
   这正好解释了 R3 里 4 条 cold 会话被整链清零（P 判成 0.0034 却全部回来了）。
2. **同一个 `min()` 里 text 与 agentic 不可比**：idle=90s 时 text 是 0.587、agentic 是 0.047，
   差 **13 倍**。跨 workload 的淘汰顺序由这两组常数决定，而不是由实测命中效率决定 ——
   这是 alpha 机制补偿不过来的量级。

## 5. Mooncake trace 不能用于轮间隔

```
请求 23,608，时间窗 3.6e6 s；不同时间戳只有 1,180 个
每批请求数中位 20；批间隔 3053 s（p50/p90/p99 几乎相同）
```

它是**批式合成负载**（每批 ~20 条同时到达），没有 `chat_id`，也没有多轮关系。
只能用来做"到达过程/并发批"的压测形态，不能提供 `P` 需要的会话内间隔。

## 6. 对 S3（接 P）的直接结论

| 参数 | 取值来源 | 建议值 |
|---|---|---|
| `text / image / search / file` 的 `(mu, sigma)` | 百炼 traceA 实测 | 保持现有常数（已验证） |
| **`agentic` 的 `(mu, sigma)`** | cachewise 实测 | **改成 (2.47, 1.21)** |
| 会话级 `m_i` 的收缩先验 | 同上，按 `task_type` 取 | 同上 |

落地顺序（仍不改运行逻辑）：

```
P0'  把 TASK_CONFIG[MULTI_TURN_AGENTIC] 改为 (2.47, 1.21)  —— 一行
P1'  用 cachewise 的间隔序列作为 driver 的第三种到达模式（真实非合成）
P2'  在真实间隔模式下跑 W0(P≡1)/W1(P̂)/W2(oracle)，比 Brier + p99 + 命中率
```

`P0'` 会直接影响现有 `reuse_score` 的 agentic 项（同一常数被 `_reuse_score` 与 `p_reuse` 共用），
所以要在跑 S3 之前单独测一次"仅换常数"的三格对照，确认没有把 S2b 的收益吃掉。
