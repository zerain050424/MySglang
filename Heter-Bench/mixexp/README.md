
---

# mixexp * 混合负载实验套件

本实验套件（`exp3`）用于评估在四类异构负载混合且主导类变化的场景下，HeterCache（三队列 + 队内策略 + VD 仲裁）相比单策略（LRU / LFU / SLRU / T-LRU）的收益，以及各模块的消融贡献。

实验包含四个单类主导的测试场景（cell）。每个 cell 中四类负载同时在场，其中一类按 token 占优，用于分别检验不同压力源下多队列隔离的效果：

| cell | 主导类 | 考察目标 |
|---|---|---|
| `scan` | 一次性扫描流（apps） | 大流量、零复用冲刷下，可复用前缀能否保留 |
| `struct` | 结构复用（struct） | 跨会话共享的公共前缀是否被误驱逐 |
| `chat` | 多轮人工会话（chat） | 轮间间隔较长时，早期轮次能否保留到下次访问 |
| `agentic` | Agent 会话（agentic） | 长会话、短间隔、上下文快速增长时的会话内复用表现 |

---

## 零、负载份额口径

主导类定义为：**该类 prompt token 占四类总 token 的比例 ≥ 60%**（与会话条数无关）。由于不同负载的上下文长度差异极大，按会话数统计与按 token 统计的结果往往相反：

| 负载示例（实测量级） | 会话数 | 单条 token | token 占比 |
|---|---|---|---|
| apps 冲刷流 | 15 000 | ~400 | ~22% |
| agentic 长会话 | 96 | ~89 000 | ~35% |

上例中 `agentic` 的会话数仅为 `apps` 的 0.6%，但消耗了更多 token。因此实验采用以下口径控制组成：

- 数据构造阶段（`build_exp3_data.py`）按目标 token 占比反解各类会话条数，不按会话数量均分，结果写入 `manifest.json`；
- 运行前由压测端复核数据组成，运行后由 `analyze.py` 校验实测 `prompt_tokens` 占比与配置是否一致；
- 非主导类默认均分剩余份额（约 13.3%/类）。由于构造时需完整保留末尾会话，主导类实测占比通常在 60%~70% 之间。

---

## 一、四类负载

| 类 | 源 | 轮数结构 | 是否合成 | 跑前校验 |
|---|---|---|---|---|
| `apps` | AgentBank_apps 单轮 | 1 轮 | 真实会话用尽后按真实长度分布补合成冲刷会话（互不共享） | — |
| `struct` | 独立构造 | 1 轮 | 8 条前缀链 × 2000 token，链内共用前缀、私有正文唯一 | — |
| `chat` | ShareGPT `mt5` | 6–8 轮为主，20% 取 10–15 轮 | 否 | `min_turns: 5` |
| `agentic` | AgentBank_alfred | 30% 为 60+ 轮长尾，其余 6–10 轮 | 否 | `min_turns: 5` |

---

## 二、准备数据（三步，串行）

```bash
# 0) 原始 session_data.jsonl 需先由 tools/prepare_session_data.py 在实验机上生成
bash mixexp/prepare_data.sh          # 步 1 + 步 2（幂等；FORCE=1 重生成）

# 步 3：构造四个 cell（产物：TaskData/exp3/<cell>/tasks.yaml + 各类 jsonl + manifest.json）
python mixexp/tools/build_exp3_data.py --mix all --total-tokens 25e6
```

---

## 三、规模

规模分两处：**数据预算**给 `build_exp3_data.py --total-tokens`，**池容量**给 server 的
`MAX_TOTAL_TOKENS`。两者要同向放大，否则工作集压不满池子、各臂打平。

| 档位 | `--total-tokens`（造数据） | `MAX_TOTAL_TOKENS`（起 server） | `--max-inflight`（压测） | 用途 |
|---|---|---|---|---|
| 冒烟 | `3e6` | `60000` | `8` | 验证链路与四类负载并发 |
| 趋势 | `10e6` | `120000` | `16` | 观察策略趋势 |
| 结论 | `25e6` | `198000`（fp8 + mem-fraction 0.85） | `0`（不限） | 产出最终实验数据 |

```bash
# 冒烟档：先按 3e6 token 预算造 scan cell
python mixexp/tools/build_exp3_data.py --mix scan --total-tokens 3e6
```

---

## 四、实验臂（`arms_exp3.tsv`）

| arm | 策略 | 用途 |
|---|---|---|
| `lru` | lru | 基础基线：全局单一 LRU |
| `hc_full` | heter_cache | 完整方案：三队列 + 队内策略 + VD 仲裁 |
| `hc_norr` | heter_cache | 消融项：关闭队内复用分排序（历史口径为关闭会话轮询） |

> 说明：当前版本已移除配额机制（无 cap 与 reservation）与权重（`weight.weights` 已整体删除）。跨队列淘汰顺序由已兑现价值密度 VD 决定：优先驱逐 VD 最小的队列（占用不参与排序），其中 VD 是「未来 `horizon_s` 内每个驻留 token 的预期命中量」：各队列用自己的估计窗 `W_q = max(horizon_s, smoothing_k*T50_q)` 估命中速率，再投影回共享视野 `horizon_s`，因此三个队列的 VD 同量纲、可比较。VD=0（没有命中证据）时退化为按占用降序。

---

## 五、判定标准

`analyze.py` 会在汇总时自动校验以下硬性指标（任一项不通过则以非零状态码退出）：

| # | 判据 | 含义 |
|---|---|---|
| 1 | 总命中率 | `hc_full` 严格高于所有基线 |
| 2 | 分项下限 | `hc_full` 在最弱负载类别上的命中率不低于最优基线（避免牺牲单类换取总命中率） |
| 3 | 尾时延 | `hc_full` 的 `ttft_p99` 不超过基线的 1.10 倍 |
| 4 | 消融表现 | `hc_full` 不弱于任一 `hc_*` 消融臂 |
| 5 | 淘汰触发 | 总 `prompt_tokens` 超过缓存池容量，确保驱逐已实际发生 |

此外，需人工核对以下跨场景一致性指标：

**6. 份额一致性** —— 实测 token 占比与 `manifest.json` 的偏差控制在 ±2pp 以内。若出现明显偏差（通常由中途重启服务或修改 `--wait-scale` 改变到达节奏所致），该组结果需重新采集。

---

## 六、运行方式

臂的启停脚本（`run_exp3.sh` / `run_mixexp.sh`）已随旧驱动删除，现在是「一次 server、
一次 `scripts/bench_heap.py`」。换臂 = 重启 server + 换输出文件名，其余命令不变。

```bash
cd Heter-Bench
mkdir -p mixexp/result/exp3/chat

# 1) 起 server（hc_full 臂；arms_exp3.tsv 的 policy / policy_config 原样搬过来）
# 注意：下一行的 POLICY_CONFIG 是历史口径（rr / mt 已删）；同意义的新写法是
# '{"value":{"horizon_s":120.0,"floor_tokens":2048.0},"share":{"share_min":2},"reuse":{"enabled":true}}'
POLICY=heter_cache MAX_TOTAL_TOKENS=198000 \
  POLICY_CONFIG='{"value":{"horizon_s":120.0,"floor_tokens":2048.0},"rr":{"fair_rr":true,"kappa":0.5,"min_retain_tokens":512,"min_retain_frac":0.3},"share":{"share_min":2},"mt":{"enabled":true}}' \
  EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85" \
  LOG=mixexp/result/exp3/chat/server_full_hc_full.log \
  bash scripts/serve.sh          # 前台；另开终端等 curl -sf .../health 返回 200

# 2) 压测一个 cell（轮间空窗按 lognormal 真实等待）
"$PY" scripts/bench_heap.py \
  --tasks-config TaskData/exp3/chat/tasks.yaml \
  --base-url http://127.0.0.1:30000 \
  --arrival-scale 1.0 --wait-scale 1.0 \
  --output mixexp/result/exp3/chat/full_hc_full.json

# 3) 换臂：lru 只把 POLICY 改成 lru（不带 POLICY_CONFIG）；
#    hc_norr 用 arms_exp3.tsv 里那份配置（rr.fair_rr=false）。每换一臂都要重启 server。

# 4) 汇总（analyze 按 <scale>_<arm>[_rN].json 解析臂名，文件名别改成别的格式）
python mixexp/analyze.py --dir mixexp/result/exp3/chat --scale full --tokens 198000
```

结果文件按 cell 分目录存放，互不覆盖：

```
mixexp/result/exp3/<cell>/full_<arm>.json         每次运行的结果（--output 指定）
mixexp/result/exp3/<cell>/server_full_<arm>.log   server 日志（serve.sh 的 LOG=...）
```

并发口径：`bench_heap.py` 默认 `--max-inflight 0`（不限），排队交给服务端（体现在
`queue_time`）。要复现旧 driver 的「最多 N 个会话在场」，加 `--max-inflight N`。

旧的单比例组成（22/35/22/22）仍可用 `build_exp3_data.py` 不带 `--mix` 构造（legacy 模式，
同样生成 tasks.yaml），产物输出至 `mixexp/result/exp3/`；相位模式（`--mode drift`）与其
`configs/exp3/*.yaml` 随旧驱动一并删除。

---

## 七、运行观测指标

| 指标 | 位置 | 预期范围 | 异常处理 |
|---|---|---|---|
| 主导类 token 份额 | `manifest.json` / 结果 JSON | ≥60%，且与构造配置一致 | 偏差过大时检查到达节奏与 `lambda_rate` |
| `token usage` 峰值 | server 日志 | 0.60–0.90 | 长期 ≥0.95 则降低 `--total-tokens` 或扩大池容量；长期 ≤0.75 则提高 `--max-inflight` |
| 主导类请求数占比 | 结果 JSON `per_task` | 通常与 token 份额同向，但不必数值相等 | 受单次会话长度差异影响，两者差异较大属正常情况，以 token 占比为准 |
| 共用前缀存活 | `[HeterCache profile]` 中 `share≥2` | `struct` cell 明显高于其他 cell | 偏低时检查 `struct` 的 `prefix_tokens` 与变体数 |
| `chat` 命中率 | `per_task.ShareGPT` | 25–40% | 明显偏低时检查 `MULTI_TURN_MIN` 与数据轮数是否匹配 |
| `scan` cell 的 `apps` 命中率 | `per_task.AgentBank_apps` | 接近 0 | 若明显大于 0，排查合成冲刷会话是否误引入了重复前缀 |

---

## 八、排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `缺少输入数据：...nosys.mt5.jsonl` | 步 2 未执行 | 执行 `bash mixexp/prepare_data.sh` |
| `缺少数据文件` + 提示 prepare_data | 配置中的 `data_path` 尚未生成 | 同上，再执行 `--build-data` |
| `[chat] 主导类 chat 只能做到 55.3%` | 现有数据源总量不足以支撑该份额 | 降低 `--dominant-share` 或 `--total-tokens`，或扩充数据源 |
| `task 'ShareGPT': 4/10 条会话轮数 < 5` | 缓存了未经过滤的旧数据文件 | 执行 `FORCE=1 bash mixexp/prepare_data.sh` |
| `session_data.jsonl 期望 {session_id: session}` | 输入格式不符（如直接使用了原始对话文件） | 先通过 `tools/prepare_session_data.py` 转换 |
| `全部会话都被剔除：这看起来是单轮数据集` | 将单轮数据源（如 apps / Code）传入了多轮过滤器 | 过滤器仅适用于多轮数据源 |
| `phase 'x' needs N sessions from 'y', but only M are available` | 相位模式所需条数超出数据池容量 | 调低相位 `max_record`，或扩大数据源规模 |