# mixgrid · 混合配比网格实验（仿真器 + 真机双后端）

> 目标：回答一个 exp3 的「单类主导 cell」回答不了的问题——**当四类负载的 token 配比
> 连续变化时，HeterCache 相对 LRU 的增益是稳定存在、还是只在某个比例区间里才有？**
>
> 载体：同一份 `tasks.yaml` 同时喂 `sim`（CPU，跑生产 `mem_cache/` 代码）和真机
> （`scripts/serve.sh` + `scripts/bench_heap.py`）。网格扫全在仿真器上做，真机只跑锚点。
>
> 状态：设计稿 + 可跑工具（`mixgrid/tools/gen_mixgrid.py` / `run_sim.py`，已用合成
> fixture 冒烟）。数据与 GPU 在远程机（`/root/sg-unicache` 系）上，见 §7。

---

## 0. 一句话结论（设计意图）

沿用 wakman 的「配比网格 + 单参数翻转」骨架，但把两个前提换成本项目的：

| wakman（KV cache 论文） | 本实验 | 为什么必须换 |
| --- | --- | --- |
| 类型由请求自带标签（S/M/A 三分类） | 队列由 `HeterCacheStrategy` 状态机判定（`chat/agentic/struct/random` 四类） | 本项目不信任客户端标签；配比只能按**数据来源**给，落地按**状态机归类**回验 |
| 按会话数比例混合（`frac`） | 按 **prompt token 份额**混合 | 会话粒度差两个数量级（apps≈0.4k token/条、agentic≈89k token/条），按会话数配比会让「25% 的会话」变成「95% 的 token」 |
| 人造合成 trace | 四类真实/半真实池子（AgentBank、ShareGPT、自造 struct） | 要能对得上 exp3 已有结论 |

---

## 1. 要回答的三个问题（都可证伪）

| # | 问题 | 判据落在哪 |
| --- | --- | --- |
| Q1 | 配比连续变化时，`heter_cache` 相对 `lru` 的命中率增益是否有**稳定符号**？ | 20 个 mix 的增益中位数与胜率（§9 C2） |
| Q2 | 增益大小是否随某一类的份额单调变化（也就是「是不是只靠某一类吃饭」）？ | 增益 vs 份额的 Spearman 相关 + 分组均值（vertex / family / edge / interior） |
| Q3 | 离离线最优（OPT）还有多少空间，空间长在哪个配比上？ | `(heter−lru)/(opt−lru)` 覆盖率，锚点 mix（§9 C3） |

附带一个工程问题：**仿真器和真机在这套网格上是否同口径**（§9 C5 的一致性闸门）。

---

## 2. 为什么不用 exp3 的 cell 法

`mixexp/` 已有一套「四类各做一个 ≥60% 主导的 cell」，它回答的是「压力源分别是谁」。
本实验不重复它，两者关系：

| exp3（已有） | mixgrid（本实验） |
| --- | --- |
| 4 个 cell，单一类主导 | 20 个 mix，连续份额，含 3 类同时在场的内部点 |
| 每 cell 各造一份数据（`build_exp3_data.py`） | 四类各一份**池子**，mix 只改 `max_record` / `λ`，不复制数据 |
| 能说「某类主导时谁赢」 | 能说「份额推到哪个位置会翻转 / 增益衰减」 |

已有实测作为本实验的**先验效应量**（`mixexp/sim_engine_fcfs_full.md`，page_size=8、
capacity=198k、token 加权）：LRU 62.55% / heter 63.84%（+1.29pp）/ OPT 71.54%
（+9.00pp vs LRU）；分 cell 看 heter−LRU = scan +3.03、struct −0.08、chat +1.62、
agentic +0.62pp。**也就是说现在的增益是「靠 scan 撑起来的」，这正是 Q2 要查的事。**

---

## 3. 负载剖面（四个池子）

池子 = 一个类一份 `.jsonl`，所有 mix 共用；`mix` 只是 `max_record` + `lambda_rate` 的不同。

| 代号 | 池子（相对 `Heter-Bench/`） | 身份 | 期望队列 | 轮间 `mu/sigma`（lognormal，秒） |
| --- | --- | --- | --- | --- |
| **T** | `TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_apps_session_data.nosys.jsonl` | 单轮 API 调用，无复用（纯冲刷） | `random`（one-shot 带） | 0 / 0 |
| **S** | `TaskData/mixgrid/pools/struct.nosys.jsonl` | 自造：K 条链，链内共用 ~2000 token 前缀 | `struct` | 0 / 0 |
| **C** | `TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/ShareGPT_cleaned_valid_deduplicated_session_data.nosys.mt2.jsonl` | 多轮人工会话，回访间隔长（T50≈118s） | `chat` | 4.15 / 0.971 |
| **A** | `TaskData/AgentBankData/Qwen2.5-7B-Instruct/AgentBank_alfred_scale_5000_session_data.nosys.jsonl` | 多轮 agent，回访间隔短（T50≈6.1s） | `agentic` | 1.81 / 1.092 |

> `mt2` 的 `2` 来自 `MULTI_TURN_MIN`（当前代码常量为 2，历史上曾是 5）。
> 必须以 `bash mixexp/prepare_data.sh` 的**实际产物名**为准，别抄文档里的 `mt5`。

**队列名只是期望**：`VALUE_QUEUES = ("struct", "agentic", "chat")`，其余（单轮/无复用）
落 one-shot 带、在日志里叫 `random`。状态机按「是否复用 / 复用间隔 / 树分支」自己判，
所以每个 mix 都要用 `realized` 段回验（§6）。

---

## 4. 配比网格：20 个点

`share` 是 **prompt token 份额（%，和为 100）**，定义在 `mixgrid/configs/mixgrid.yaml`。

| 组 | id | T/S/C/A | 造这个点是为了 |
| --- | --- | --- | --- |
| vertex | `v_T100` | 100/0/0/0 | 纯冲刷地板：所有策略都应接近 0 命中 |
| vertex | `v_S100` | 0/100/0/0 | 纯结构复用天花板：`struct` 队列自主性 |
| vertex | `v_C100` | 0/0/100/0 | 纯长间隔多轮：TTL / share 判定的极限 |
| vertex | `v_A100` | 0/0/0/100 | 纯短间隔 agent：one-shot 带供给最少的工况 |
| family | `f_TS70_30` / `f_TS50_50` / `f_TS30_70` | 单轮族内 T↔S | **本项目 VD 最容易混的一对**（都单轮，只有「是否跨会话共享」不同） |
| family | `f_CA70_30` / `f_CA50_50` / `f_CA30_70` | 多轮族内 C↔A | 两者只差轮间间隔，检验 `reuse` 参数是否真的在起作用 |
| edge | `e_TC70_30` / `e_TC30_70` | 冲刷 vs chat | 跨族主导 |
| edge | `e_TA70_30` / `e_TA30_70` | 冲刷 vs agent | 跨族主导（最像生产混合） |
| edge | `e_SC70_30` / `e_SC30_70` | struct vs chat | 跨会话共享 vs 会话内共享 |
| edge | `e_SA70_30` / `e_SA30_70` | struct vs agent | 同上，短间隔 |
| interior | `i_2525` | 25/25/25/25 | 四类均衡（真机主锚点） |
| interior | `i_T40` | 40/30/15/15 | 单轮偏重但四类都在场 |

20 个点不是单纯形穷举（1% 粒度有 16 万种），而是「顶点 + 两条族内边 + 八条跨族边 + 两个内部点」，
目的是用最少的 run 把「谁赢」的区域边界画出来。

---

## 5. 三个不变量与预算反解

### 5.1 不变量（跨 mix 必须相同）

| 不变量 | 值 | 不守住会怎样 |
| --- | --- | --- |
| `token_rate` | 2000 tok/s（≈ exp3 的 24.5M/12000s） | 单位时间进缓存的 token 量不同 → 命中率混进了压力差 |
| `capacity` | 198k token（可扫，见 §6） | 池子大小不同 → 同上 |
| `seed` | 42（所有 mix 共用） | `max_record` 截断是「同一 shuffle 的前 N 条」→ 同 seed 才能让不同 mix 的子集互相**嵌套**，配比差异不会被「换了批会话」污染 |

### 5.2 预算与时长（`gen_mixgrid.py` 自动算）

```
B      = min(budget_cap, min_{i: s_i>0} pool_tokens_i / s_i )      # 总 prompt token
T      = B / token_rate                                            # 时长（各 mix 不同）
n_i    = round(s_i · B / mean_tokens_i)     λ_i = n_i / T          # max_record = n_i
```

`pool_tokens_i / s_i` 是「让第 i 类占 `s_i` 时池子允许的最大总预算」，取最小者 ⇒
**份额优先于规模**：任何一类都不会跑到没数据，宁可总 token 少一点。
`budget_cap = 8M` 是为了让真机单 run ≤ 70 分钟；仿真复核稳定性时可 `--budget-cap 25e6`。

量级参考（**以 `pool_stats.json` 实测为准**）：apps 池 ~6M token、struct 池 ~5M、
chat 池 ~3–10M（取决于 ShareGPT 保留会话数）、agentic 池极大（5000×~89k ≈ 4×10⁸）。
因此 **chat 池通常是最紧的那个约束**，会决定多数 mix 的 B。

`gen_mixgrid.py` 会把两个风险写进 `manifest.csv`：

- `low_sample`：某类会话数 < 15（低份额 agentic 档很常见：89k token/条，十几条就够占 20% token）；
- `份额漂移`：取整/截断后实际 token 份额与目标差 > 2pp。

**任何一条告警都不许静默忽略**：低样本档的方差要单独报，漂移档的解释要按实测份额写。

---

## 6. 参数轴与实验臂

### 6.1 轴

| 轴 | 档位 | 用在哪 |
| --- | --- | --- |
| mix | 20 点（§4） | 全网格 |
| capacity | `99k / 198k / 396k`（0.5×/1×/2×） | 网格（Q1/Q2 的稳健性）；**增益随容量衰减**是文献里的已知现象，必须扫 |
| `wait_scale` | `1.0` 主档；`0.25 / 4.0` | 只在 3 个锚点（`i_2525`、`e_TC30_70`、`e_TA30_70`）上跑，等价于 wakman 的「单参数翻转」：只改多轮回访间隔，看赢家会不会换 |

### 6.2 臂

| 臂 | 仿真器（`run_sim.py`） | 真机（`serve.sh`） | 备注 |
| --- | --- | --- | --- |
| `lru` | `--policy lru` | `POLICY=lru` | 基线 |
| `heter` | `--policy heter_cache --horizon-s 120 --smoothing-k 4 --floor-tokens 0` | `POLICY=heter_cache POLICY_CONFIG='{"value":{"horizon_s":120,"smoothing_k":4,"floor_tokens":0},"reuse":{"enabled":true}}'` | **参数必须显式钉死**：`floor_tokens` 的缺省值已从 2048 改成 0，别吃缺省 |
| `opt` | `--policy opt --use-map <path>` | 见 `mixexp/README.md` §二 | 只上锚点，不上网格 |
| `hc_norr`（消融） | — | `reuse.enabled=false` | **仿真器暂时跑不了**：`SimConfig` 只透传 `value` 组，没有 `reuse` 开关；要做就得先给 `SimConfig` 加字段（P4） |

`opt` 在真机侧还需要把 `"opt"` 加进 `RADIX_EVICTION_POLICY_CHOICES`
（`python/sglang/srt/arg_groups/choices.py:184`，否则参数解析就拒），这条 `mixexp/README.md` 已记。

### 6.3 为什么 OPT 只上锚点

仿真器里 `sim/opt_map.py::build_use_map` 吃的是**已有绝对时刻**的 trace（开环 `run()`），
而网格跑的是引擎闭环（`run_engine`）——准入按 `need = total − hit` 计预算，**不同臂的批组成不同 ⇒ 派发时刻不同**，
use map 的时间轴必须迭代重建。仓库里目前没有 `use_map_from_run`（只在那份远程脚本
`/root/sim_runs/sim_engine_cell.py` 里），所以本实验**不把 OPT 铺到 20×3 网格上**，
只在 3 个锚点用开环口径给上界。真机 OPT 走 `mixexp/README.md` §二 那套 capture → 重锚定 → 重放。

---

## 7. 数据准备

### 7.1 现状

数据不在本仓库：本地 `TaskData/*/` 只有准备脚本，`.jsonl` 与模型在远程机
（`REPRODUCE.md` 记的 AutoDL 流程；仿真复跑用的是 `/root/sg-unicache` + `/root/sl_old/bin/python`）。
远程目标当前都连不上（`autodl` / `autodl2` / `bitahub` / `seetacloud` 均 ECONNREFUSED / auth failed），
所以下面的命令要在能访问数据的那台机器上跑。

### 7.2 命令链（有数据但没池子的情况）

```bash
cd Heter-Bench

# 1) 原始 session_data.jsonl（远程已有则跳过）
bash TaskData/AgentBankData/data.sh
bash TaskData/SharedGPT-multi-turn/data.sh

# 2) 剥 Qwen system 段 + 剔掉轮数 < MULTI_TURN_MIN 的短会话（mt2）
bash mixexp/prepare_data.sh

# 3) struct 池：唯一自造负载，造一次即可（K 条链 × 2000 token 共享前缀）
python mixexp/tools/build_exp3_data.py --mix struct \
  --tokenizer /root/autodl-tmp/model/Qwen2.5-7B-Instruct
mkdir -p TaskData/mixgrid/pools
cp TaskData/exp3/struct/struct_exp3.nosys.jsonl TaskData/mixgrid/pools/struct.nosys.jsonl

# 4) 生成 20 份 tasks.yaml + manifest
python mixgrid/tools/gen_mixgrid.py
# → mixgrid/configs/mix/<mix>.yaml、manifest.csv、pool_stats.json
```

`gen_mixgrid.py` 首次会实测每个池子的「会话数 / 每条会话 prompt token 均值」（大文件
自动抽样，`--refresh` 重测）。**看三样东西再继续**：`manifest.csv` 的 `budget_tokens` /
`duration_s` / `low_sample` 列。

### 7.3 两个数据坑（会直接改结论）

1. **`AgentBank_alfred_scale_5000` 是复制扩量**：`Alfred_scale_up.py` 把原始会话复制到
   5000 条，并在首条消息前加 uuid 前缀。前缀匹配从 token 0 开始 ⇒ 各副本**不会**跨会话共享，
   缓存语义干净；但「5000 条」实际是「N 条原始会话的近似重复」，分布代表性弱于真 5000 条。
   报告里要写这一点；如果原始 `AgentBank_alfred.json` 会话数够用，优先用它。
2. **Qwen 公共 system 段必须剥**（`prepare_data.sh` 步 1）：它只有 24 token 但人人都有，
   不剥会把 apps 全钉成 `struct`（离线消融：random 召回 0.000 → 0.993，见 `mixexp/prepare_data.sh` 注释）。

---

## 8. 怎么跑

### 8.1 P0 冒烟（~10 分钟，先验证流水线）

```bash
cd Heter-Bench
python mixgrid/tools/gen_mixgrid.py --budget-cap 1e6 --out-dir /tmp/mixgrid_smoke
python mixgrid/tools/run_sim.py \
  --tasks-config /tmp/mixgrid_smoke/i_2525.yaml \
  --capacity 198000 --page-size 8 --policy lru --warmup-s 120 \
  --out /tmp/mixgrid_smoke/sim_i2525_lru.json
```

看三件事：`requests > 0`、`realized token share` 与目标差 < 2pp、
`realized queue occ` 里 `chat/agentic/struct` 都非零（全零说明分类没触发，先查 `mu/sigma` 与 `wait_scale`）。

> **不要用 `--wait-scale 0` 做正式实验**：轮间空窗归零会把多轮会话压成单轮
> （实测 `chat` 会话全被判成 `trans`），`chat` 队列直接空转。

### 8.2 P2 全网格（仿真器，CPU）

```bash
cd Heter-Bench
for mix in mixgrid/configs/mix/*.yaml; do
  id=$(basename "$mix" .yaml)
  for cap in 99000 198000 396000; do
    for arm in lru heter_cache; do
      python mixgrid/tools/run_sim.py --tasks-config "$mix" \
        --capacity $cap --page-size 8 --policy $arm --warmup-s 600 \
        --out "mixgrid/result/sim/${id}_cap${cap}_${arm}.json"
    done
  done
done
```

成本：`sim` 每 cell 26–50s（exp3 全量口径，2.3GB RSS 以内），120 个 run ≈ 2–3 小时单核；
可以按 mix 并行。

### 8.3 P3 真机锚点（GPU）

锚点 6 个 mix × 1 档容量 × 2 臂 = 12 个 run，单 run ≤ 70 分钟。每组三步：

```bash
cd Heter-Bench && mkdir -p mixgrid/result/real/i_2525

# ① 起 server（capacity / page_size 必须与仿真器一致）
POLICY=heter_cache MAX_TOTAL_TOKENS=198000 \
  POLICY_CONFIG='{"value":{"horizon_s":120,"smoothing_k":4,"floor_tokens":0},"reuse":{"enabled":true}}' \
  EXTRA_ARGS="--page-size 8 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85" \
  LOG=mixgrid/result/real/i_2525/server_heter.log \
  bash scripts/serve.sh

# ② 同一份 tasks.yaml 回放（与仿真器逐字相同）
"$PY" scripts/bench_heap.py \
  --tasks-config mixgrid/configs/mix/i_2525.yaml \
  --base-url http://127.0.0.1:30000 \
  --arrival-scale 1.0 --wait-scale 1.0 --max-inflight 0 --warmup-s 600 \
  --output mixgrid/result/real/i_2525/heter.json

# ③ lru 臂：POLICY=lru（去掉 POLICY_CONFIG），输出 lru.json
```

两条必须在起 server 时钉死的参数：

- **`--page-size 8`**：CUDA 平台的缺省是 **1**（`arg_groups/overrides.py::_page_size_default`），
  不显式传就会与仿真器的 `--page-size 8` 不同源，块级命中边界偏一个档。
  若坚持用 1，就**两侧都改成 1**并在报告里写明。
- **capacity 取真值**：`--max-total-tokens` 是名义值，实际可用槽位受 `mem-fraction-static`
  与分配器取整影响。起 server 后从日志里读 KV pool 的实际 token 数，用它（而不是名义值）
  填仿真器的 `--capacity`，并把两个数字都写进报告。

### 8.4 命名

```
mixgrid/result/sim/<mix>_cap<C>_<arm>.json                  + 可选 .jsonl（逐请求）
mixgrid/result/real/<mix>/<arm>.json / server_<arm>.log / <arm>.csv
```

两侧结果字段名刻意对齐：`hit_rate`、证据 `window.per_task`、`realized.*`。

---

## 9. 预注册判据（先写死，再跑）

| # | 判据 | 阈值 | 不过怎么办 |
| --- | --- | --- | --- |
| **C1 有效性** | 该 cell 的总 prompt token ≥ 5×capacity **且** `evicted_tokens > 0` | 硬门槛 | 该 cell 不进表（无淘汰 ⇒ 命中之差只是噪声） |
| **C2 主判据** | `heter − lru` 的 token 加权命中率**跨 20 mix 的中位数** ≥ **+1.5pp**，且胜率 ≥ 14/20 | 中位数 + 胜率 | 说明增益只在部分配比成立，按 Q2 定位区间，重写结论 |
| **C3 上界覆盖** | 锚点上 `(heter−lru)/(opt−lru)` 中位数 ≥ **25%** | 软但预先声明 | exp3 `scan` 已达 34%，低于 25% 说明只吃到边角 |
| **C4 公平** | 每个 mix 的「最弱类 per_task 命中率」不低于 `lru` 的对应值 **−0.5pp** | 硬门槛 | 靠饿死一类换总命中率不算赢（`awk` 口径见 `mixexp/analyze.py`） |
| **C5 一致性** | 锚点上 `|sim − real| ≤ 2.0pp`，且 6 个锚点的增益符号一致 | 硬门槛 | **先修保真，不许解读网格**；排查顺序见下 |

C5 不过时按此顺序查：`page_size` 是否两侧一致 → capacity 是否用了日志真值 →
统计窗（`--warmup-s`）是否一致 → `processor`/tokenizer 是否同模型 → `--max-inflight`
与仿真 `EngineConfig.max_prefill_tokens` 的批组成差异 → 真机是否有 chunked prefill。

C4 的实现细节：`per_type`/`session_type` **只有 `heter_cache` 臂有**（`SessionRegistry`
挂在该缓存内部；`lru` 臂的 `session_type` 恒为 `none`）。所以「最弱类」必须按
**`per_task`（数据来源）** 比，不能按 `per_type`。

---

## 10. 陷阱清单（都是本仓库踩过的）

| 陷阱 | 后果 | 规避 |
| --- | --- | --- |
| 用 `--wait-scale 0` 跑正式实验 | 多轮会话被压成单轮，`chat` 队空转 | 正式档固定 `wait_scale=1.0`；0 只用于冒烟 |
| `mt2`/`mt5` 后缀抄文档 | 文件不存在或用了未剔短会话的版本，短会话以 `trans` 身份抢 one-shot 带 | 以 `prepare_data.sh` 实际输出为准 |
| 不剥 Qwen system 段 | apps 被钉成 `struct`，`random` 召回 0 | 跑 `mixexp/prepare_data.sh` |
| 吃 `floor_tokens` 缺省值 | 缺省已 2048→0，不同分支结果不可比 | 臂参数一律显式写全 |
| 换 `seed` 做「重复实验」 | 等于换了一批会话，配比差异与数据差异混在一起 | seed 固定 42；重复实验用同一 seed + 不同 `--arrival-scale` 或多次 `rng` 偏移另开维度 |
| 用 `--max-total-tokens` 当 capacity | 与仿真器 `--capacity` 不同源，命中率不可比 | 从 server 日志读实际 KV pool token 数 |
| 只报总命中率 | one-shot 带把 T 类冲刷吸走后，结论会归因错 | 必报 `per_task` + `queue_occ` + `victim_sources` |
| 让 OPT 参与网格 | 闭环时间轴没迭代 → use map 全 `+∞`，OPT 静默退化成任意序，命中率会**低于正确值** | OPT 只在锚点、开环口径，且必须满足 `OPT ≥ 所有臂` 的自检 |
| 低样本档当等权证据 | 低份额 agentic 只有十几条会话，方差远大于别档 | `manifest.csv` 的 `low_sample` 列；统计上单独报或加权 |
| 改 `budget_cap`/`token_rate` 只补跑几个点 | 时长与压力变了，跨 mix 不可比 | 这两个参数属于「整批」参数，改了就整批重跑 |
| 真机跑完才发现 `page_size` 不一致 | 块级命中边界不同，命中率系统性偏移 | server 端显式 `--page-size 8`（CUDA 缺省是 **1**，不会自动对齐仿真器）；起 server 前把两侧值写进同一个变量/日志头 |

---

## 11. 阶段与预算

| 阶段 | 内容 | 载体 | 预算 |
| --- | --- | --- | --- |
| P0 | 冒烟：3 个 mix、1 档容量、`--budget-cap 1e6` | sim | 10 min CPU |
| P1 | 数据池 + `gen_mixgrid.py` + 检查 `manifest.csv` | 远程机 | 1–2 h（含大文件抽样） |
| P2 | 全网格 20×3×2 = 120 run（+ 3 锚点 × 2 档 `wait_scale` × 2 臂） | sim | 3–5 h CPU |
| P3 | 6 锚点 × 2 臂 = 12 run；capacity 校准 1 次 | GPU | 10–14 h（单 run ≤70 min） |
| P4 | 可选：`hc_norr` 消融（需给 `SimConfig` 加 `reuse` 字段）、`page_size` 8 vs 1、regret 账本、`--budget-cap 25e6` 复核 | sim + GPU | 另计 |

P2 出表之后**先看 C1/C2**；C2 不过就不要花 P3 的 GPU 时间，先把 Q2 的区间找出来。

---

## 12. 不做什么

- 不重跑 exp3 的单类主导 cell（已有的 4 个 cell 结论继续引用，不重测）。
- 不在网格上做 OPT（理由见 §6.3），也不把 `use_map` 当真机网格的必选项。
- 不做「路由器 / 负载均衡」方向（那是 `2026-wu` 那条线，不是淘汰策略）。
- 不引入合成发散负载（`sim.trace.synthetic_sessions`）：本文的合成只用于 struct 前缀链，
  其余三类都来自真实会话数据。

---

## 13. 文件清单

```
mixgrid/
  README.md                     ← 本文件（设计 + 运行手册 + 判据）
  configs/mixgrid.yaml          网格定义（20 个 mix、四个池子、三个不变量）
  configs/mix/<mix>.yaml        生成物：每个 mix 一份 tasks.yaml（sim 与真机共用）
  configs/mix/manifest.csv      生成物：B / T / 各类 n 与 λ / 低样本告警
  configs/mix/pool_stats.json   生成物：池子统计缓存
  tools/gen_mixgrid.py          份额 → tasks.yaml（含预算反解）
  tools/run_sim.py              仿真器后端（run_engine + warmup 对齐 + realized 回验）
  result/sim|real/…             结果（按 §8.4 命名）
```

两个工具的冒烟记录（合成 fixture，非真实池子）：

```
python mixgrid/tools/gen_mixgrid.py --grid /tmp/mixgrid_fx/grid.yaml --out-dir /tmp/mixgrid_fx/out
python mixgrid/tools/run_sim.py --tasks-config /tmp/mixgrid_fx/out/i_2525.yaml \
  --capacity 120 --page-size 1 --policy heter_cache --warmup-s 0 --wait-scale 0 \
  --out /tmp/mixgrid_fx/sim_heter.json
# → hit_rate(warmup 0s)=28.81%，realized token share 与目标 25% 差 ≤2.8pp
```
