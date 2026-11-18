# mixexp · 混合负载实验
评估四类异构负载混合、且主导类轮换的场景下，HeterCache（三队列 + 队内策略 + VD 仲裁）
相对单策略（LRU / LFU / SLRU / T-LRU）的收益与模块贡献。
三部分：

1. [实验负载](#一实验负载)
2. [OPT 上界](#二opt上界真实负载--真实推理框架)
3. [regret rate](#三regret-rate淘汰后悔率)

> **最优配置速查**：TTFT 优先部署用 chain_lru 全局版（tail-stress p90 404ms、
> 四 cell 加权 66.25% 全胜 heter），heter 架构内最优为档 B 后悔门+全局定价
> （p90 581ms、加权 64.42%，默认关）——参数、数字与复现命令见
> [`best_configs.md`](best_configs.md)，全程推演见 [`chain_shape_report.md`](chain_shape_report.md)。
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

### 1.2 四类负载的构成

| 类 | 源 | 轮数结构 | 会话数目标 | 合成规则 |
|---|---|---|---|---|
| `apps` | AgentBank_apps 单轮 | 1 轮 | 15000 | 真实会话用尽后，按真实长度分布采样补**合成冲刷会话**：互相无共享前缀|
| `struct` | 独立构造 | 1 轮 | 2600 | 8 条前缀变体链，每条链内会话共用同一段 **2000 token** 共享前缀 + 各自 64 token 私有正文；四类里唯一带跨会话共享 |
| `chat` | ShareGPT `*.nosys.mt5.jsonl` | 6–8 轮 80%，10–15 轮 20% | 800 |仿真多轮对话|
| `agentic` | AgentBank_alfred `*.nosys.jsonl` | 6–10 轮 68.75%，60+ 轮 31.25% | 96 | 仿真Agentic会话|
### 1.3 数据准备

```bash
cd Heter-Bench

# 步 0（一次性）：原始 session_data.jsonl 由 tools/prepare_session_data.py 在实验机生成
# 步 1+2：剥 Qwen system 段 + 多轮源剔短会话（幂等；FORCE=1 重生成）
bash mixexp/prepare_data.sh

# 步 3：构造四个主导 cell（全量档 = 每 cell 25M token 预算）
#   产物：TaskData/exp3/<cell>/tasks.yaml + 四份 *_exp3.nosys.jsonl + manifest.json
python mixexp/tools/build_exp3_data.py --mix all --total-tokens 25e6
```

### 1.4 环境配置与 token 池总量
| 项 | 值 |
|---|---|
| 负载数据总量 | `--total-tokens 25e6` |
| **token 池总量** | `MAX_TOTAL_TOKENS=198000` | 
| KV 精度 / 显存占比 | `--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85` | 
| 到达 / 空窗 | `--arrival-scale 1.0 --wait-scale 1.0` |
| 并发上限 | `--max-inflight 0` | 

### 1.5 启动命令
| arm | policy | POLICY_CONFIG |
|---|---|---|
| `lru` | `lru` | 不带 |
| `hc_full` | `heter_cache` | `{"reuse":{"enabled":true}}` |
| `hc_norr` | `heter_cache` | `{"reuse":{"enabled":false}}` |

`value`（horizon_s=120 / smoothing_k=4 / floor=2048）与 `share`（min=2 / TTL
120/180/300）用缺省值，显式写全也可。heter_cache 必须**同时**给
`--radix-eviction-policy heter_cache` 与 `--enable-heter-cache`（serve.sh 在
`POLICY=heter_cache` 时自动加后者），缺一不可。

#### cell `chat`

```bash
cd Heter-Bench && mkdir -p mixexp/result/exp3/chat

# ── hc_full ─────────────────────────────────────────────
POLICY=heter_cache MAX_TOTAL_TOKENS=198000 \
  POLICY_CONFIG='{"reuse":{"enabled":true}}' \
  EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85" \
  LOG=mixexp/result/exp3/chat/server_full_hc_full.log \
  bash scripts/serve.sh        

"$PY" scripts/bench_heap.py \
  --tasks-config TaskData/exp3/chat/tasks.yaml \
  --base-url http://127.0.0.1:30000 \
  --arrival-scale 1.0 --wait-scale 1.0 --max-inflight 0 --warmup-s 1000 \
  --output mixexp/result/exp3/chat/full_hc_full.json

# ── lru：POLICY 改 lru、去掉 POLICY_CONFIG，输出名换 full_lru ──
# ── hc_norr：POLICY_CONFIG 改 {"reuse":{"enabled":false}}，输出 full_hc_norr ──

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

Output：

```
mixexp/result/exp3/<cell>/full_<arm>.json        结果（--output；同名 .csv 是逐请求明细）
mixexp/result/exp3/<cell>/server_full_<arm>.log  server 日志（serve.sh 的 LOG=...）
```
---

## 二、OPT 上界

OPT = Belady 离线最优：淘汰下次使用时刻最远的块（`evict_policy.OPTStrategy`）。
它的唯一输入是 use map（`{block_hash: [使用时刻]}`），生成规则与缓存层
`TreeNode.hash_value` 逐字一致（`sim/opt_map.py` 复用同一个 `get_storage_hash_str`）。
在真实 server 上跑 OPT 的流程 = **先采一次到达时刻，离线造 use map，再带 map 重放**。

### 2.0 前置改动（已落地）

`RADIX_EVICTION_POLICY_CHOICES`（`python/sglang/srt/arg_groups/choices.py:184`）已加入
`"opt"`，`--radix-eviction-policy opt` 能过参数解析；回归见
`test/manual/test_radix_eviction_policy_cli.py`。

旧版本缺这一项时，策略工厂明明注册了 `OPTStrategy`，服务却起不来
（`invalid choice: 'opt'`），整条流程断在解析阶段。

### 2.1 采一次到达时刻（capture run）

用任意实验（建议 `lru`）在目标 cell 跑一次全量压测，逐请求 CSV 记下每次
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

为什么能这样采：OPT 的语义是给定同一条到达流，最优淘汰序的上界。重放 run 用
**同一份 tasks.yaml、同 seed、同 `--arrival-scale/--wait-scale`** 时，到达流由
`stable_seed(sid + task)` 与空窗 RNG 决定，与第一次逐字可复现；所以第一次采到的
**相对延迟**就是重放 run 会发生的相对延迟（绝对时刻由 2.2 的重锚定对齐）。

### 2.2 采集成 use map（一条命令）

先选好重放起点 R：map 在 server 构造时被整体读入内存，所以**先 2.2 后 2.3**。

```bash
# 先定重放起点 R（epoch 秒）：现在 + 600s，给模型加载留余量
R=$(( $(date +%s) + 600 ))

python mixexp/tools/build_use_map_from_bench.py --cell chat --replay-start $R
```

缺省值即 `--csv mixexp/result/exp3/<cell>/full_lru.csv --tasks
TaskData/exp3/<cell>/tasks.yaml --out TaskData/exp3/<cell>/use_map_p1.json
--page-size 1`，可逐项覆盖。

脚本做三件事（数据读取与 hash 规则复用 `sim.session_data` / `sim.opt_map`，
自己只写「时刻回填」；`Session`/`Turn` 是 frozen dataclass，用
`dataclasses.replace` 回填）：

1. **时刻域换算**：CSV 的 `ts` 是 `time.time()`，server 的 `_now()` 是
   `time.monotonic`（`radix_cache.py:328`）；Linux 上系统级同源、差一个常数偏移，
   `--offset auto` 在构建时刻自测（要求采集与构建在同一次内核启动周期内，
   中间休眠/重启会在负时刻检查里显式炸出）；
2. **重锚定到重放时间轴（`--replay-start`）**：capture 与重放隔了几个小时的
   墙钟，而 server 的 `_now()` 是绝对 monotonic——map 里的时刻若留在 capture
   时间轴上，重放时所有块的下一次使用都在过去 → 全部查作 `+∞` → OPT 静默
   退化成任意序。所以脚本把相对延迟 `d_i = ts_i - C` 重锚到重放起点；常数级
   偏差（shell 延迟、首轮完成时刻近似起点）只移动 `now` 边界，不改块间次序；
3. **显式失败**：采集率 < `--min-coverage`（默认 0.9）、CSV 的 task 名对不上
   tasks.yaml（采的 cell 与给的 tasks.yaml 不是同一份负载）、换算出负时刻
   （offset 过期），都直接报错退出——这三者都意味着 map 不可信。

两个语义点：

- use map 只含 **prompt** 块；模型自己生成的 response 永不复用，不在 map 里 →
  查询得 `+∞` → 最先被逐出，与仿真里response 用唯一哨兵同一语义；
- `build_use_map` 对不足一页的尾段直接丢弃，与缓存层 `page_aligned` 一致。

### 2.3 带 map 起服务并重放

```bash
POLICY=opt MAX_TOTAL_TOKENS=198000 \
  POLICY_CONFIG='{"use_map_path":"TaskData/exp3/chat/use_map_p1.json","block_size":1}' \
  EXTRA_ARGS="--kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85" \
  LOG=mixexp/result/exp3/chat/server_full_opt.log \
  bash scripts/serve.sh

# 时间轴已锚定到 R：等模型加载完、到点再发射（模型加载慢就把 2.2 里的余量调大）
sleep $(( R - $(date +%s) ))
"$PY" scripts/bench_heap.py \
  --tasks-config TaskData/exp3/chat/tasks.yaml \
  --arrival-scale 1.0 --wait-scale 1.0 --max-inflight 0 --warmup-s 1000 \
  --output mixexp/result/exp3/chat/full_opt.json
```

换 cell / 换 seed / 换重放起点，都重跑一次 2.2（只差 `--replay-start`，离线、不碰 GPU）。

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

regret 账本（`Heter-Bench/profile/regret_ledger.py`）量的是每次逐出决策亏了多少：
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
