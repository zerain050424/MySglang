# mixexp · 混合负载实验套件

评估四类异构负载混合、且主导类轮换的场景下，HeterCache（三队列 + 队内策略 + VD 仲裁）
相对单策略（LRU / LFU / SLRU / T-LRU）的收益与模块贡献。本文只覆盖**全量档**实验；
旧的冒烟 / 趋势档（`mix_quick.yaml` / `mix_mid.yaml`）与相位模式已删除。

三部分：

1. [实验负载](#一实验负载) —— 四个「单类主导」cell 的构成、环境与命令；
2. [OPT 上界](#二opt上界真实负载--真实推理框架) —— 用真实负载在真实推理框架里跑 OPT；
3. [regret rate](#三regret-rate淘汰后悔率) —— 淘汰决策的损失对账怎么开。

臂的启停脚本已随旧驱动删除：现在**一次 server、一次 `scripts/bench_heap.py`**。
换臂 = 重启 server + 换输出文件名，其余命令不变。

---

## 一、实验负载

### 1.1 主导口径与偏倚模式

四个 cell，每个 cell 里四类负载同时在场，其中一类按 **prompt token** 占优（≥60%），
用于分别检验不同压力源下多队列隔离的效果：

| cell | 主导类 | 考察目标（该 cell 制造的偏倚） |
|---|---|---|
| `scan` | apps（单轮冲刷） | 大流量、零复用冲刷下，可复用前缀能否保留 |
| `struct` | struct（跨会话共享前缀） | 公共前缀是否被误驱逐 |
| `chat` | chat（多轮人工会话） | 轮间间隔长（T50≈118s）时，早期轮次能否活到下次访问 |
| `agentic` | agentic（Agent 会话） | 长会话（单条 ≈89k token）、短间隔（T50≈6.1s）、上下文快速增长 |

口径三条（`build_exp3_data.py` 顶部有完整推导）：

- **主导 = 该类 prompt token / 四类 prompt token 总和 ≥ 60%**，与会话条数无关。
  不同负载的单条长度差两个数量级（apps ≈0.4k vs agentic ≈89k token），按会话数
  统计会得出相反的主导类；
- 配角均分剩余 `(1-share)/3`；构造时由配角实测 token 反解主导类条数
  （`dom_tokens = share/(1-share) × others_tokens`），主导池不足时反向缩配角，
  仍达不到阈值（容差 0.5pp）则报错退出；
- **偏倚在时间上是静止的**：`lambda_rate = 会话数 / --duration-s`（默认 12000s），
  四类同时收尾，token 组成不随时间漂移。多轮类在到达期结束后靠轮间 lognormal
  空窗延续（chat μ=4.15/σ=0.971，agentic μ=1.81/σ=1.092），拖长的只是尾巴，
  不改变组成。因此每个 cell 的偏倚是「结构性」的：四类同节奏到达，只有一类在
  token 上碾压，仲裁要回答的是「这一类该不该被削」。

实测份额（构造期）写在每个 cell 的 `manifest.json`；跑完由 `analyze.py` 复核
实测 `prompt_tokens` 占比与配置偏差 ≤ ±2pp。

### 1.2 四类负载的构成

| 类 | 源 | 轮数结构 | 会话数目标 | 合成规则 |
|---|---|---|---|---|
| `apps` | AgentBank_apps 单轮（nosys） | 1 轮 | 15000 | 真实会话用尽后，按真实长度分布采样补**合成冲刷会话**：body 为独立随机 token，互相无共享前缀，纯冲刷 |
| `struct` | 独立构造 | 1 轮 | 2600 | 8 条前缀变体链，每条链内会话共用同一段 **2000 token** 共享前缀 + 各自 64 token 私有正文；四类里唯一带跨会话共享 |
| `chat` | ShareGPT `*.nosys.mt5.jsonl` | 6–8 轮 80%，10–15 轮 20% | 800 | 不过滤直接用；`min_turns: 5` 跑前校验 |
| `agentic` | AgentBank_alfred `*.nosys.jsonl` | 6–10 轮 68.75%，60+ 轮 31.25% | 96 | 不过滤直接用；`min_turns: 5` 跑前校验 |

`mt5` 后缀的含义（`mixexp/prepare_data.sh` 头注释有消融数据）：多轮源里轮数 <5 的
会话终身判 TRANS，会挤占 one-shot 带压低 chat 命中率，所以构造前先剔掉；单轮源
（apps / Codeparrot）不做这一步。

### 1.3 数据准备（两步，串行；全量档做一次即可）

```bash
cd Heter-Bench

# 步 0（一次性）：原始 session_data.jsonl 由 tools/prepare_session_data.py 在实验机生成
# 步 1+2：剥 Qwen system 段 + 多轮源剔短会话（幂等；FORCE=1 重生成）
bash mixexp/prepare_data.sh

# 步 3：构造四个主导 cell（全量档 = 每 cell 25M token 预算）
#   产物：TaskData/exp3/<cell>/tasks.yaml + 四份 *_exp3.nosys.jsonl + manifest.json
python mixexp/tools/build_exp3_data.py --mix all --total-tokens 25e6
```

`--total-tokens` 只用于定**配角**的初始规模，主导类条数由份额反解，与它无直接关系；
四个 cell 都要的话一次 `--mix all` 造齐（各 cell 独立目录，互不覆盖）。
（legacy 单组成 22/35/22/22 仍可用 `--only` 构造，本文不再展开。）

### 1.4 环境配置与 token 池总量（四个 cell 相同）

规模分两处，必须同向放大，否则工作集压不满池子、各臂打平：

| 项 | 值 | 出处 |
|---|---|---|
| 数据预算 | `--total-tokens 25e6`（每 cell） | `build_exp3_data.py` |
| **token 池总量** | `MAX_TOTAL_TOKENS=198000` | server 的 `--max-total-tokens` |
| KV 精度 / 显存占比 | `--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85` | 支撑 198k 池 |
| 到达 / 空窗 | `--arrival-scale 1.0 --wait-scale 1.0`（真实节奏） | `bench_heap.py` |
| 并发上限 | `--max-inflight 0`（不限，排队交给服务端） | `bench_heap.py` |
| 预热窗 | `--warmup-s 1000` | 见 1.6 |

### 1.5 启动命令（按 cell）

臂配置（当前口径；`arms_exp3.tsv` 里的 `rr`/`mt` 配置是历史档案，重跑会报
TypeError，勿照抄）：

| arm | policy | POLICY_CONFIG |
|---|---|---|
| `lru` | `lru` | 不带 |
| `hc_full` | `heter_cache` | `{"reuse":{"enabled":true}}` |
| `hc_norr` | `heter_cache` | `{"reuse":{"enabled":false}}` |

`value`（horizon_s=120 / smoothing_k=4 / floor=2048）与 `share`（min=2 / TTL
120/180/300）用缺省值，显式写全也可。heter_cache 必须**同时**给
`--radix-eviction-policy heter_cache` 与 `--enable-heter-cache`（serve.sh 在
`POLICY=heter_cache` 时自动加后者），缺一不可。

每换一臂都要重启 server；文件名必须是 `<scale>_<arm>.json`（analyze 按此解析臂名）。

#### cell `chat`（其余三个 cell 只替换 `<cell>` 与输出目录）

```bash
cd Heter-Bench && mkdir -p mixexp/result/exp3/chat

# ── 臂 hc_full ─────────────────────────────────────────────
POLICY=heter_cache MAX_TOTAL_TOKENS=198000 \
  POLICY_CONFIG='{"reuse":{"enabled":true}}' \
  EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85" \
  LOG=mixexp/result/exp3/chat/server_full_hc_full.log \
  bash scripts/serve.sh          # 前台；另开终端等 /health 返回 200

"$PY" scripts/bench_heap.py \
  --tasks-config TaskData/exp3/chat/tasks.yaml \
  --base-url http://127.0.0.1:30000 \
  --arrival-scale 1.0 --wait-scale 1.0 --max-inflight 0 --warmup-s 1000 \
  --output mixexp/result/exp3/chat/full_hc_full.json

# ── 臂 lru：POLICY 改 lru、去掉 POLICY_CONFIG，输出名换 full_lru ──
# ── 臂 hc_norr：POLICY_CONFIG 改 {"reuse":{"enabled":false}}，输出 full_hc_norr ──

# 汇总
python mixexp/analyze.py --dir mixexp/result/exp3/chat --scale full --tokens 198000
```

#### cell `scan`

```bash
cd Heter-Bench && mkdir -p mixexp/result/exp3/scan
POLICY=heter_cache MAX_TOTAL_TOKENS=198000 \
  POLICY_CONFIG='{"reuse":{"enabled":true}}' \
  EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85" \
  LOG=mixexp/result/exp3/scan/server_full_hc_full.log \
  bash scripts/serve.sh
"$PY" scripts/bench_heap.py \
  --tasks-config TaskData/exp3/scan/tasks.yaml \
  --arrival-scale 1.0 --wait-scale 1.0 --max-inflight 0 --warmup-s 1000 \
  --output mixexp/result/exp3/scan/full_hc_full.json
python mixexp/analyze.py --dir mixexp/result/exp3/scan --scale full --tokens 198000
```

#### cell `struct`

```bash
cd Heter-Bench && mkdir -p mixexp/result/exp3/struct
POLICY=heter_cache MAX_TOTAL_TOKENS=198000 \
  POLICY_CONFIG='{"reuse":{"enabled":true}}' \
  EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85" \
  LOG=mixexp/result/exp3/struct/server_full_hc_full.log \
  bash scripts/serve.sh
"$PY" scripts/bench_heap.py \
  --tasks-config TaskData/exp3/struct/tasks.yaml \
  --arrival-scale 1.0 --wait-scale 1.0 --max-inflight 0 --warmup-s 1000 \
  --output mixexp/result/exp3/struct/full_hc_full.json
python mixexp/analyze.py --dir mixexp/result/exp3/struct --scale full --tokens 198000
```

#### cell `agentic`

```bash
cd Heter-Bench && mkdir -p mixexp/result/exp3/agentic
POLICY=heter_cache MAX_TOTAL_TOKENS=198000 \
  POLICY_CONFIG='{"reuse":{"enabled":true}}' \
  EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85" \
  LOG=mixexp/result/exp3/agentic/server_full_hc_full.log \
  bash scripts/serve.sh
"$PY" scripts/bench_heap.py \
  --tasks-config TaskData/exp3/agentic/tasks.yaml \
  --arrival-scale 1.0 --wait-scale 1.0 --max-inflight 0 --warmup-s 1000 \
  --output mixexp/result/exp3/agentic/full_hc_full.json
python mixexp/analyze.py --dir mixexp/result/exp3/agentic --scale full --tokens 198000
```

产物布局：

```
mixexp/result/exp3/<cell>/full_<arm>.json        结果（--output；同名 .csv 是逐请求明细）
mixexp/result/exp3/<cell>/server_full_<arm>.log  server 日志（serve.sh 的 LOG=...）
```

### 1.6 预热窗（`--warmup-s 1000` 的依据）

VD 的读数是时间常数为 `W_q = max(horizon, k*T50_q)` 的指数低通；`t < W_q` 时被
系统性低估 `1/(1-e^{-t/W_q})` 倍，且各队列 W_q 不同 → 低估倍数不同，这个**差分
偏差**正好落在「chat 还是 struct 先削」的仲裁边界上：

| 队列 | T50 | `W_q = max(120, 4·T50)` | t=100s 时低估 |
|---|---|---|---|
| chat | 117.9s | **471.6s** | **5.2×** |
| agentic | 6.1s | 120s | 1.8× |
| struct（无分布，回退 horizon） | — | 120s | 1.8× |

取 `--warmup-s 1000`（≈ 2·W_chat，收敛到 86%）。统计窗本身也要够长：
trace 总时长 ≥ 预热 + 2·W_max ≈ 1900s。`--duration-s 12000` 的全量 trace 满足。

结果 JSON 里 `warmup_s / warmup_requests / warmup_sessions` 与
`num_requests / num_sessions / prompt_tokens`（统计窗内）分列；CSV 每行有 `warmup`
布尔列。`analyze.py` 在 `hc_full` 预热 <1000s 时判据报 ✗——**不同预热的臂不能混排**。

### 1.7 判定与排错

`analyze.py` 汇总时校验（任一不通过非零退出）：① `hc_full` 总命中率严格高于所有
基线；② 最弱负载类别上的命中率不低于最优基线；③ `ttft_p99` ≤ 基线 ×1.10；
④ 不弱于任一消融臂；⑤ 总 prompt 超过池容量（驱逐确实发生）；⑥ 实测 token 份额
与 `manifest.json` 偏差 ≤ ±2pp。

| 现象 | 处理 |
|---|---|
| `缺少输入数据：...nosys.mt5.jsonl` | `bash mixexp/prepare_data.sh` |
| `[chat] 主导类 chat 只能做到 55.3%` | 数据源不足：降 `--dominant-share` / `--total-tokens`，或扩源 |
| `task 'ShareGPT': 4/10 条会话轮数 < 5` | 缓存了未过滤旧文件：`FORCE=1 bash mixexp/prepare_data.sh` |
| `session_data.jsonl 期望 {session_id: session}` | 用了原始对话文件：先过 `tools/prepare_session_data.py` |
| token usage 长期 ≥0.95 | 降 `--total-tokens` 或扩池；长期 ≤0.75 提高并发 |

---

## 二、OPT 上界（真实负载 + 真实推理框架）

OPT = Belady 离线最优：淘汰「下次使用时刻最远」的块（`evict_policy.OPTStrategy`）。
它的唯一输入是 use map（`{block_hash: [使用时刻]}`），生成规则与缓存层
`TreeNode.hash_value` 逐字一致（`sim/opt_map.py` 复用同一个 `get_storage_hash_str`）。
在真实 server 上跑 OPT 的流程 = **先采一次到达时刻，离线造 use map，再带 map 重放**。

### 2.0 前置改动（一次性）

`opt` 还不在服务端 CLI 选项里（策略工厂已注册，只差选择列表）：

```python
# python/sglang/srt/arg_groups/choices.py:184
RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu", "slru", "priority", "tlru", "heter_cache"]
#                                                                               ^ 加 "opt"
```

不加的话 `--radix-eviction-policy opt` 在参数解析阶段就被拒。

### 2.1 采一次到达时刻（capture run）

用任意便宜臂（建议 `lru`）在目标 cell 跑一次全量压测，逐请求 CSV 记下了每次
`(task, session_id, turn, ts)`：

```bash
POLICY=lru MAX_TOTAL_TOKENS=198000 \
  EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85" \
  LOG=mixexp/result/exp3/chat/server_full_lru.log \
  bash scripts/serve.sh
"$PY" scripts/bench_heap.py \
  --tasks-config TaskData/exp3/chat/tasks.yaml \
  --arrival-scale 1.0 --wait-scale 1.0 --max-inflight 0 --warmup-s 1000 \
  --output mixexp/result/exp3/chat/full_lru.json      # 同时写 full_lru.csv
```

为什么能这样采：OPT 的语义是「给定同一条到达流，最优淘汰序的上界」。第二次 run
用**同一份 tasks.yaml、同 seed、同 `--arrival-scale/--wait-scale`** 时，到达流由
`stable_seed(sid + task)` 与空窗 RNG 决定，与第一次逐字可复现；所以第一次采到的
时刻就是第二次 run 会发生的时刻。

### 2.2 造 use map（离线，不碰 GPU）

时间域必须对齐：server 的 `_now()` 是 `time.monotonic`（`radix_cache.py:328`），
而 CSV 里的 `ts` 是 `time.time()`。Linux 的 CLOCK_MONOTONIC 是系统级共享的，两者
只差一个常数偏移，在同一台机器、同一次内核启动周期内测一次即可（机器休眠/重启后重测）：

```bash
python -c 'import time; print(time.time() - time.monotonic())'   # 记为 OFFSET
```

拼接脚本（保存为 `mixexp/tools/build_use_map_from_bench.py`，要点如下；数据读取与
hash 规则全部复用现成模块，自己只写「时刻回填」这一层。注意 `Session`/`Turn` 是
frozen dataclass，改字段用 `dataclasses.replace`，不能直接赋值）：

```python
import csv, yaml
from dataclasses import replace
from sim.session_data import load_tasks          # sim/ 在仓库根
from sim.opt_map import build_use_map, write_use_map

CELL = "chat"
OFFSET = <2.2 里测到的值>
PAGESIZE = 1                                     # 必须等于起 server 时的 --page-size

tasks = yaml.safe_load(open(f"TaskData/exp3/{CELL}/tasks.yaml"))["tasks"]
sessions = load_tasks(tasks, base_dir=".")       # Session.turns[i].prompt 已是数据集 token ids

rows = {}
for r in csv.DictReader(open(f"mixexp/result/exp3/{CELL}/full_lru.csv")):
    rows[(r["task"], r["session_id"], int(r["turn"]))] = float(r["ts"])

fixed = []
for s in sessions:
    turns = []
    for idx, turn in enumerate(s.turns, 1):      # CSV 的 turn 列是 1 起的轮号
        ts = rows.get((s.task_name, s.session_id, idx))
        if ts is not None:
            turns.append(replace(turn, t=ts - OFFSET))   # 没采到的轮直接丢弃
    if turns:
        fixed.append(replace(s, turns=tuple(turns)))

write_use_map(f"TaskData/exp3/{CELL}/use_map_p{PAGESIZE}.json",
              build_use_map(fixed, page_size=PAGESIZE))
```

两个语义点（都是对的，不是遗漏）：

- use map 只含 **prompt** 块；模型自己生成的 response 永不复用，不在 map 里 →
  查询得 `+∞` → 最先被逐出，与仿真里「response 用唯一哨兵」同一语义；
- `build_use_map` 对不足一页的尾段直接丢弃，与缓存层 `page_aligned` 一致。

### 2.3 带 map 起服务并重放

```bash
POLICY=opt MAX_TOTAL_TOKENS=198000 \
  POLICY_CONFIG='{"use_map_path":"TaskData/exp3/chat/use_map_p1.json","block_size":1}' \
  EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85" \
  LOG=mixexp/result/exp3/chat/server_full_opt.log \
  bash scripts/serve.sh

"$PY" scripts/bench_heap.py \
  --tasks-config TaskData/exp3/chat/tasks.yaml \
  --arrival-scale 1.0 --wait-scale 1.0 --max-inflight 0 --warmup-s 1000 \
  --output mixexp/result/exp3/chat/full_opt.json
```

约束与判读：

- `block_size` 必须 == server 的 `--page-size`（`bind_page_size` 在构造时校验，
  不一致直接 ValueError——这是显式失败，不是静默错位）；
- **重放参数必须与 capture run 完全一致**（tasks.yaml / seed / arrival-scale /
  wait-scale）。改了任何一个，use map 里的时刻就不再是这次 run 会发生的时刻；
- `miss_warn`（默认 0.5）：use map 查不中比例过半时 server 日志告警一次。出现它
  说明三件事之一错了——page_size 不一致、时间域没对齐、重放参数漂移；
- **OPT 是上界**：它的命中率应当 ≥ 所有臂。若 OPT ≈ 或 < `hc_full`，几乎一定是
  use map 静默退化（如全 `+∞` 退化成任意序），不要把该数字当结果报；
- 开销：`_ensure_hash` 在淘汰时惰性补算父链 hash，`page_size=1` 时每页一次
  sha256、一个节点可达上千页。与其它臂公平对比就保持 `page_size=1`；要压开销就
  **四个臂统一** `--page-size 8`，并用 `block_size=8` 重建 use map——只改 OPT 一臂
  的 page_size 会让对比失真。

---

## 三、regret rate（淘汰后悔率）

regret 账本（`Heter-Bench/profile/regret_ledger.py`）量的是「每次逐出决策亏了多少」：
一次 `_pick_victim` + 级联 = 一个 stamp，`span` = 本次腾出 token；断点页（首个未命中
页）若在坟场里，这次 miss 就归因给那次决策；损失按 `span` 计。它有两个用途——
验收价值模型（`P_return` 的校准曲线 `pred->real`）和把重算税按决策源分账。

### 3.1 开启方式

账本挂在 **server 进程**的缓存上，用环境变量开（默认关，零开销）：

```bash
HETER_REGRET=1 \
HETER_REGRET_OUT=mixexp/result/exp3/chat/regret_full_hc_full.json \
POLICY=heter_cache MAX_TOTAL_TOKENS=198000 \
  POLICY_CONFIG='{"reuse":{"enabled":true}}' \
  EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85" \
  LOG=mixexp/result/exp3/chat/server_full_hc_full.log \
  bash scripts/serve.sh
```

- `HETER_REGRET=1`：装配账本（未设置时所有钩子短路，热路径零开销）；
- `HETER_REGRET_OUT=<path>`：报告 JSON 的落盘路径（原子覆盖，相对 server 启动
  目录 = 仓库根）；省略则只打日志不落盘；
- 只对 `heter_cache` 有意义：钩子都在 `HeterCacheRadixCache` 里，`lru`/`opt` 臂
  开了也没有数据来源；
- 与 `STRAT_LOG_LEVEL` 无关：regret 报告走自己的 info 行，不受 debug 级别限制。

### 3.2 读什么

报告随 `[HeterCache profile]` 每 30s 同拍输出（日志关键字 `[HeterCache regret]`），
JSON 每 tick 原子覆盖。报告里的 join/tax 与占用读数同一时刻，可直接按时间对齐：

```
[HeterCache regret] H=own(T50) requests=… tax=…% stamps=… evicted=… regret=… join=…%
  by source / by queue / by rule / by depth / by share   ← 五维分组，各带 regret% 与 censored%
  calibration pred->real: 0.36->0.40 0.53->1.00          ← 价值函数的校准曲线
```

判读顺序（先看采集质量，再看策略）：

1. **`join_rate`**：断点页能在坟场找到的比例。<50% 时渲染会带 WARN——此时
   `regret=0` 应当先怀疑采集（page_size / extra_key / 页对齐没对上），而不是策略好；
2. **`censored%`**：没到 H 的 stamp 是删失，不是「没回访」。chat 的 T50≈118s、
   p95≈965s，统计期不够长时 regret 会被系统性压低——这正是仿真（虚拟时钟可任意
   延长统计期）比真机更适合读 regret 曲线的原因；
3. **`tax`（recompute_pct）**：与 profile 的 hit 口径互补的全局重算税；
4. **by source / by queue**：哪个队列的账最亏（问题 1 的直接答案）；同队列下
   `rule` 列比较 `reuse` 与 LRU 兜底（问题 2 的直接答案）；
5. **by depth / by share**：浅页与高共享页的 regret 集中度——「浅的贵、共享的贵」
   若成立，regret 应集中在浅档与高 share 档；
6. **calibration**：各队列 lognormal 预测的 `P_return` vs 实测回访率。预测 0.9 的
   桶实测只有 0.2 ⇒ 队内价值函数错了，此时调仲裁公式没有意义。

### 3.3 边界

| 项 | 值 / 行为 | 备注 |
|---|---|---|
| 曲线横轴 | 30 / 120 / 300 / 600s + `own`（各队自己的 T50） | `own` 列是主读数 |
| 坟场保留 | 900s（`DEFAULT_RETAIN_S`，未暴露环境变量） | 想看更长的 H 先调大它 |
| 容量上限 | 200k 条，触顶丢最老（`dropped` 计数） | 稳态万级，正常触不到 |
| 开销 | 每请求 + 每淘汰决策各一次 blake2b；无逐页 hash | 不物化每页 hash 到节点 |
| 输出时机 | 随 profile tick（淘汰期） | run 收尾前最后一段无淘汰时，JSON 停在最后一次 tick；要终值就再看一眼日志末尾的 `[HeterCache regret]` |
