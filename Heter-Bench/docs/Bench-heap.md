# Heter-Bench 堆发射器设计与使用指南 (`Bench-heap.md`)

本文档详细剖析 Heter-Bench 的堆驱动真实墙钟 HTTP 压测发射器（[bench_heap.py](file:///home/rain/MySglang/Heter-Bench/scripts/bench_heap.py)），内容分为两部分：
1. **主流程深度解析**：系统架构、数据读入机制与异步请求调度发射流水线。
2. **使用指南与自动化脚本**：参数详解、结果指标解读以及开箱即用的自动化压测 Bash 脚本。

---

## 第一部分：压测发射器核心架构与主流程分析

### 1.1 核心设计理念：最小堆驱动与空窗解耦

在多轮对话（Chat）、智能体交互（Agentic）或代码补全等混合长文本场景中，传统的压测工具（如固定并发数的 wrk、Locust 或简化的闭环循环）存在一个严重缺陷：**客户端的轮间思考/等待时间（Think Time / Tool Wait）会占用并发名额**。如果客户端在等待 30 秒思考，连接槽位被占满，新会话就无法到达，导致真实到达率失真。

`bench_heap.py` 采用**事件驱动的全局最小堆调度器**：
- **空窗不占槽**：全局最小堆仅维护 `(due_ts, seq, PendingTurn)`。会话在轮间等待期间仅是堆中的一个时间戳节点，**完全不占用任何并发连接或运行协程**。
- **真实墙钟唤醒**：主循环通过 `asyncio.wait_for` 睡眠至堆顶到期时刻。只有到点的请求才创建轻量协程发射；流式响应接收完成后，立即计算下一轮到达时刻并重新入堆。
- **无客户端并发瓶颈**：连接层显式配置 `aiohttp.TCPConnector(limit=0)`，取消每主机 100 连接的默认限制，确保排队压力真实传导至服务端的 KV Cache 管理器与调度器。

```
                       tasks.yaml 配置
                             │
                             ▼
              [ 数据读入: load_sessions ]
                             │
                             ▼
        [ 混合到达流构建: build_hybrid_arrival_schedule ]
          ├── Task A: Poisson(λ=2.0) ──┐
          └── Task B: Poisson(λ=1.0) ──┴──► 按 (arrival_offset, counter) 升序
                                                      │
                                                      ▼
                                           [ 最小堆: ArrivalHeap ]
                                            (due_ts, seq, turn_1)
                                                      │
                                                      ▼
 ┌─────────────────────────────────────── HeapEmitter.run() 主循环 ───────────────────────────────────────┐
 │                                                                                                        │
 │   1. 槽位检查: _available_slots() 是否允许派发？若满或堆空 ──► 等待唤醒 (_wake.wait)                     │
 │   2. 时钟检查: wait_s = due_ts - now                                                                  │
 │      ├── wait_s > 0: 睡眠至到期时刻 (_sleep_until)，若期间有更早事件入堆可被提前唤醒                       │
 │      └── wait_s <= 0: 弹出当前到期轮次 pop_due(limit=slots)                                            │
 │   3. 并发发射: 派发协程 _fire() ──► 执行 _run_turn()                                                   │
 │                                                                                                        │
 └────────────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                      │
                         ┌────────────────────────────┴────────────────────────────┐
                         │                                                         │
                         ▼                                                         ▼
            [ 发送 HTTP: one_request ]                                [ 状态统计与流转: _run_turn ]
       - payload: 完整历史 input_ids                                 - 解析流式 SSE chunk 计算 TTFT
       - stream=True                                                 - 记录 Stats (按 warmup_s 切分)
       - 提取服务端 meta_info                                         - 若未达末轮:
         (cached_tokens, prompt_tokens,                                  wait_s = lognormvariate(μ, σ)
          queue_time_ms, prefill_ms)                                     入堆 (finished_ts + wait_s, turn+1)
                                                                         触发 _wake.set() 唤醒主循环
```

---

### 1.2 数据读入机制：如何载入与构造会话

数据读入由 `load_sessions`、`assert_turn_policy` 和 `build_hybrid_arrival_schedule` 协同完成：

#### 1. 会话文件解析 (`load_sessions`)
- **文件格式**：输入为 JSONL 文件，每行包含单个会话对象：`{"<session_id>": {"id": "...", "num_turns": N, "turn_1": {...}, "turn_2": {...}}}`。
- **长前缀保护与提前截断**：
  - 支持 `shuffle=False` 保持文件物理顺序。针对文档问答（如 MashQA）等数据集，相邻样本共享同一长文档前缀，若被全局打乱会导致前缀复用完全失效。
  - 当 `shuffle=False` 且指定了 `max_record` 时，读满数量即提前 `break`，避免将几十 GB 的全量数据一次性载入内存。

#### 2. 多轮策略校验 (`assert_turn_policy`)
- 在在线分类器（`SessionRegistry._classify`）中，轮数小于 `MULTI_TURN_MIN` 的会话会被判为单轮瞬态（`TRANS`），从而进入 One-shot 优先清理带。
- 为防止配置失误导致多轮会话因数据截断被误判为一次性流量，`assert_turn_policy` 会严格校验：若任务配置了 `min_turns`，输入数据集中任何 `num_turns < min_turns` 的样本都会直接抛出 `ValueError` 中断启动。

#### 3. 混合负载交错生成 (`build_hybrid_arrival_schedule`)
- 支持多任务异构混合（例如 Chat、Agentic、Struct 任务并行）：
  - 每个任务使用独立的随机种子 `seed + task_id`，保证各个任务内部打乱和到达时间的独立性与确定性。
  - **到达间隔**：支持两种模式：
    - `poisson`（默认）：按指数分布 `Exp(lambda_rate)` 累加到达时刻，泊松到达率之比直接决定各任务流量占比。
    - `fixed`：按确定性步长 `offset_s + step_s * i` 线性推进，用于严格复现固定间隔场景。
- 各任务首轮事件打上全局唯一递增的 `counter` 标签，合并后按 `(arrival_offset, counter)` 全局升序排列，生成全局到达时间表。

#### 4. Token 序列与模板历史一致性核验 (`_resolve_prompt_ids`)
- 多轮请求中，模型每轮实际输入是前文所有轮次对话的拼接。数据集预处理时已通过对应的 Chat Template 将历史渲染并 Tokenize 到了本轮的 `turn_i.input_ids` 中。
- `_resolve_prompt_ids` 在运行时提取该 ID 列表，并对比 `prev_input_ids + prev_response_ids` 是否为当前前缀：
  - 若为字面前缀（如 Qwen 风格），记录拼接成功；
  - 若模型在回复前插入了特殊角色头（如 Llama 风格），客户端简单拼接会与服务端模板不一致导致 Cache Miss。此时直接回退并采纳数据集原始预生成的完整 Token 序列，确保前缀命中不受客户端拼接差异干扰。

---

### 1.3 请求发送与调度流水线：如何高效发射与推进

#### 1. 最小堆维护 (`ArrivalHeap`)
堆元素定义为元组 `(due_ts, seq, PendingTurn)`：
- `due_ts`：到期的单调时钟时刻（秒）；
- `seq`：全局自增序号，保证在时间戳完全相同时，严格按入堆先后次序弹出，杜绝随机性；
- `PendingTurn`：包含任务元信息、会话数据和当前轮次索引 `turn_idx`。
- `pop_due(now, limit)`：原子化弹出所有到期且不超过 `limit`（当前剩余可用并发槽位）的轮次。

#### 2. 调度发射主循环 (`HeapEmitter.run`)
主调度循环是纯异步非阻塞设计：
1. **槽位检查**：如果设置了 `--max-inflight > 0` 且在飞请求已达上限，调用 `await self._wake.wait()` 挂起，等待任一请求完成；
2. **时钟睡眠**：获取堆顶的 `due_ts`，若未到点（`wait_s = due_ts - now > 0`），执行 `await asyncio.wait_for(self._wake.wait(), timeout=wait_s)`。如果在睡眠期间，某个先前请求完成并压入了**比当前堆顶更早到期**的新事件，`_wake` 事件会被触发，主循环立刻提前醒来并重新评估；
3. **并发分发**：调用 `pop_due(now=now, limit=slots)` 取得全部到期轮次，针对每个轮次递增 `_inflight` 并通过 `asyncio.create_task` 创建协程发射，主循环无需等待请求返回。

#### 3. 单轮请求执行与流式打点 (`one_request`)
- **流式接收（SSE）**：以 `stream=True` 向 SGLang 服务端的 `/generate` 端点发送请求。
- **TTFT 测量**：记录客户端发出请求到收到第一个 `data:` 数据块的时间差作为 TTFT。
- **端到端耗时与元数据提取**：接收完整响应后，提取服务端的 `meta_info`：
  - `prompt_tokens`：提示词 Token 数；
  - `cached_tokens`：命中的 KV Cache Token 数；
  - `completion_tokens`：生成的 Token 数；
  - `queue_time_ms`：服务端队列等待时间；
  - `prefill_ms`：纯预填充执行耗时（`prefill_finished_time - forward_entry_time`）。

#### 4. 多轮推进与随机等待 (`_push_next_turn`)
- 当请求完成且 `turn_idx < num_turns` 时：
  - 为该 `session_id` 派生独立稳定的伪随机数生成器 `random.Random(stable_seed_from_str(sid, base_seed))`；
  - 从任务配置的 `(mu, sigma)` 中采样对数正态分布等待时间：`wait_s = rng.lognormvariate(mu, sigma) * wait_scale`；
  - 下一轮的到期时刻计算为 `finished_ts + wait_s`，组装 `turn_idx + 1` 压入堆中，并调用 `self._wake.set()`。

#### 5. 统计窗与预热过滤 (`Stats`)
- **预热窗过滤**：为了消除系统刚启动时缓存从冷到热、以及算法价值估计窗未收敛带来的偏差，支持设置 `--warmup-s`。
- 请求完成时以当前墙钟时间判定：若完成时刻处于预热窗内，只计入 `warmup_*` 指标，不计入命中率和 TTFT 分位数统计；但明细数据仍完整写入 CSV（标记 `warmup=True`），便于对比冷启动过程。

---

## 第二部分：压测发射器使用指南与自动化脚本

### 2.1 环境准备

`bench_heap.py` 已实现解耦与按需导入：纯逻辑与单元测试不需要 `aiohttp`；但若要向真实推理服务发起 HTTP 压测，需确保运行环境具备异步网络与 YAML 解析库：

```bash
pip install aiohttp pyyaml
```

---

### 2.2 命令行参数完全速查

```bash
python Heter-Bench/scripts/bench_heap.py [OPTIONS]
```

| 参数名 | 默认值 | 类型 | 详细说明 |
|---|---|---|---|
| `--tasks-config` | **必填** | string | 任务配置文件路径（YAML 格式），定义各负载的数据集、到达率、轮间等待参数等 |
| `--base-url` | `http://127.0.0.1:30000` | string | SGLang Server 的 HTTP 基础地址 |
| `--arrival-scale` | `1.0` | float | 会话首轮到达时刻的缩放因子。`1.0`=原始泊松节奏，`0.0`=所有会话首轮立刻全并发到达 |
| `--wait-scale` | `1.0` | float | 轮间等待时间的缩放因子。`1.0`=按任务 lognormal 真实等待，`0.0`=会话上一轮结束后立即发送下一轮 |
| `--max-inflight` | `0` | int | 客户端同时在飞（Inflight）的请求数上限。`0` 表示不限（完全由服务端调度器排队） |
| `--warmup-s` | `0.0` | float | 预热时长（秒）。从发射器启动开始，此窗口内完成的请求不计入最终 summary 命中率和时延统计 |
| `--mu` | `None` | float | 全局覆盖任务中的轮间等待 lognormal 均值参数 $\mu$ |
| `--sigma` | `None` | float | 全局覆盖任务中的轮间等待 lognormal 标准差参数 $\sigma$ |
| `--seed` | `42` | int | 随机种子（用于会话轮间等待 RNG 派生） |
| `--timeout-s` | `1800.0` | float | 单个请求的 HTTP 超时时间（秒） |
| `--progress-every` | `200` | int | 每完成 N 个请求在终端打印一行实时进度日志。设为 `0` 关闭 |
| `--output` | `None` | string | 汇总 JSON 结果保存路径。若设置，会自动在同路径写入对应的 `.csv` 逐请求明细文件 |

---

### 2.3 配置文件结构 (`tasks.yaml`)

压测任务通过 YAML 文件进行组合，支持单任务或异构多任务混合：

```yaml
tasks:
  - name: "chat"                                # 任务标识
    data_path: "mixexp/TaskData/exp3/chat/session_data.mt2.jsonl" # 数据集路径（相对 Heter-Bench 根目录）
    task_id: 0                                  # 任务 ID（用于 RNG 派生隔离）
    seed: 42                                    # 独立随机种子
    lambda_rate: 1.0                            # 泊松到达速率 (会话数/秒)
    max_record: 200                             # 最大载入会话数 (-1 表示不限)
    min_turns: 2                                # 最小会话轮数校验阈值
    mu: 3.4                                     # 轮间等待 lognormal mu 参数
    sigma: 0.8                                  # 轮间等待 lognormal sigma 参数
    shuffle: true                               # 是否随机打乱会话顺序

  - name: "agentic"
    data_path: "mixexp/TaskData/exp3/agentic/session_data.mt2.jsonl"
    task_id: 1
    seed: 42
    lambda_rate: 0.5                            # 与 chat 构成 1.0 : 0.5 的到达比例
    max_record: 100
    mu: 1.8
    sigma: 1.0
    shuffle: true
```

---

### 2.4 自动化运行脚本 (`run_bench_heap.sh`)

为了将“启动推理服务 -> 等待健康检查就绪 -> 启动堆发射器压测 -> 测试完成退出清理”全流程自动化，项目提供了自动化脚本 [Heter-Bench/scripts/run_bench_heap.sh](file:///home/rain/MySglang/Heter-Bench/scripts/run_bench_heap.sh)。

#### 脚本功能特点：
1. **自动生命周期管理**：若对应端口未运行服务，脚本会自动调用 `serve.sh` 在后台启动 SGLang；压测结束或按 `Ctrl+C` 中断时，`trap` 机制会安全回收后台进程，不残留僵尸服务。
2. **服务就绪轮询**：每 2 秒检测一次 `/health` 端点，确认模型完全加载后再开始计时压测。
3. **灵活的环境变量传参**：支持通过环境变量无缝替换策略、显存大小、任务配置文件与压测参数。

#### 脚本完整源码：

```bash
#!/usr/bin/env bash
# run_bench_heap.sh: Heter-Bench 堆发射器端到端自动化运行脚本
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HETER_BENCH_DIR="${ROOT_DIR}/Heter-Bench"

# 1. 载入环境
if [ -f "${SCRIPT_DIR}/config.sh" ]; then source "${SCRIPT_DIR}/config.sh"; fi
if [ -f "${SCRIPT_DIR}/env.sh" ]; then source "${SCRIPT_DIR}/env.sh"; fi

PY="${PY:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
BASE_URL="http://${HOST}:${PORT}"

POLICY="${POLICY:-heter_cache}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-100000}"
POLICY_CONFIG="${POLICY_CONFIG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TASKS_CONFIG="${TASKS_CONFIG:-${HETER_BENCH_DIR}/configs/hybrid_tasks_config.yaml}"
OUTPUT="${OUTPUT:-${HETER_BENCH_DIR}/result/bench_heap_${POLICY}_$(date +%Y%m%d_%H%M%S).json}"

MAX_INFLIGHT="${MAX_INFLIGHT:-0}"
WAIT_SCALE="${WAIT_SCALE:-1.0}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
WARMUP_S="${WARMUP_S:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
TIMEOUT_S="${TIMEOUT_S:-1800}"

mkdir -p "$(dirname "${OUTPUT}")"
LOG_FILE="$(dirname "${OUTPUT}")/server_${POLICY}.log"
SERVER_PID=""

cleanup() {
    if [ -n "${SERVER_PID}" ]; then
        echo -e "\n[run_bench] 正在停止后台 Server (PID: ${SERVER_PID})..."
        kill -15 "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        echo "[run_bench] Server 已终止。"
    fi
}
trap cleanup EXIT INT TERM

is_server_ready() {
    curl -s --connect-timeout 2 "${BASE_URL}/health" >/dev/null 2>&1
}

# 2. 检查与启动 Server
if is_server_ready; then
    echo "[run_bench] 检测到已存在运行中的 Server (${BASE_URL})，直接复用。"
else
    echo "[run_bench] 启动后台 Server: POLICY=${POLICY}, MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}, PORT=${PORT}"
    export POLICY MAX_TOTAL_TOKENS PORT LOG="${LOG_FILE}"
    if [ -n "${POLICY_CONFIG}" ]; then export POLICY_CONFIG; fi
    if [ -n "${EXTRA_ARGS}" ]; then export EXTRA_ARGS; fi

    bash "${SCRIPT_DIR}/serve.sh" > "${LOG_FILE}" 2>&1 &
    SERVER_PID=$!

    echo "[run_bench] 等待 Server 初始化并就绪..."
    READY=0
    for ((i=1; i<=300; i++)); do
        if is_server_ready; then READY=1; break; fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[run_bench] 错误: Server 进程异常退出，请查看日志: ${LOG_FILE}"
            exit 1
        fi
        sleep 2
    done
    if [ "${READY}" -ne 1 ]; then
        echo "[run_bench] 错误: 等待 Server 超时，日志: ${LOG_FILE}"
        exit 1
    fi
    echo "[run_bench] Server 已成功就绪 (${BASE_URL})！"
fi

# 3. 运行堆压测发射器
echo "========================================================================"
echo "[run_bench] 开始执行 bench_heap 压测"
echo "  Tasks 配置    : ${TASKS_CONFIG}"
echo "  Base URL      : ${BASE_URL}"
echo "  Max Inflight  : ${MAX_INFLIGHT}"
echo "  Wait Scale    : ${WAIT_SCALE}"
echo "  Warmup (s)    : ${WARMUP_S}"
echo "  结果输出      : ${OUTPUT}"
echo "========================================================================"

"${PY}" "${SCRIPT_DIR}/bench_heap.py" \
    --tasks-config "${TASKS_CONFIG}" \
    --base-url "${BASE_URL}" \
    --max-inflight "${MAX_INFLIGHT}" \
    --wait-scale "${WAIT_SCALE}" \
    --arrival-scale "${ARRIVAL_SCALE}" \
    --warmup-s "${WARMUP_S}" \
    --progress-every "${PROGRESS_EVERY}" \
    --timeout-s "${TIMEOUT_S}" \
    --output "${OUTPUT}"

echo "========================================================================"
echo "[run_bench] 压测执行完成！"
echo "  JSON 摘要文件 : ${OUTPUT}"
CSV_OUTPUT="${OUTPUT%.json}.csv"
if [ -f "${CSV_OUTPUT}" ]; then
    echo "  CSV 逐行文件  : ${CSV_OUTPUT}"
fi
echo "========================================================================"
```

---

### 2.5 常用运行示例

#### 示例 1：默认混合负载基准测试（HeterCache 策略）
```bash
# 使用默认配置：heter_cache 策略，100k token 容量，混合负载
bash Heter-Bench/scripts/run_bench_heap.sh
```

#### 示例 2：运行 LRU 策略对比实验
```bash
POLICY=lru \
MAX_TOTAL_TOKENS=198000 \
TASKS_CONFIG=Heter-Bench/mixexp/TaskData/exp3/chat/tasks.yaml \
OUTPUT=Heter-Bench/result/chat_lru.json \
bash Heter-Bench/scripts/run_bench_heap.sh
```

#### 示例 3：压测 HeterCache 开启后悔弃权门（档 B 配置）与 1000 秒预热窗
```bash
POLICY=heter_cache \
POLICY_CONFIG='{"regret_gate":{"enabled":true,"k":0.6,"r_min":0.3,"priced_fallback":true,"floor_tokens":8192,"weight":0.02,"beta":1.0}}' \
MAX_TOTAL_TOKENS=198000 \
WARMUP_S=1000 \
TASKS_CONFIG=Heter-Bench/mixexp/TaskData/exp3/chat/tasks.yaml \
OUTPUT=Heter-Bench/result/chat_heter_b.json \
bash Heter-Bench/scripts/run_bench_heap.sh
```

#### 示例 4：纯客户端直接调用已有的 Server
如果 SGLang 服务已经在后台运行（端口 30000），可以直接执行 Python 脚本：
```bash
python3 Heter-Bench/scripts/bench_heap.py \
    --tasks-config Heter-Bench/mixexp/TaskData/exp3/chat/tasks.yaml \
    --base-url http://127.0.0.1:30000 \
    --arrival-scale 1.0 \
    --wait-scale 1.0 \
    --max-inflight 0 \
    --warmup-s 1000 \
    --output Heter-Bench/result/manual_run.json
```

---

### 2.6 输出结果解读

压测运行完毕后，会生成一个 JSON 汇总文件和一个同名的 CSV 明细文件。

#### 1. JSON 汇总核心字段
- `cache_hit_rate_pct`：统计窗口内的整体 Prefix Cache 命中率百分比。
- `ttft_p50_ms` / `ttft_p90_ms` / `ttft_p99_ms`：首 Token 延迟（Time to First Token）的 P50、P90、P99 分位数（毫秒）。
- `e2e_mean_ms` / `e2e_p99_ms`：端到端请求时延的平均值与 P99。
- `request_throughput`：扣除预热窗口后的有效吞吐量（请求数/秒）。
- `per_task`：各子任务（如 chat、agentic 等）各自独立的请求量、Token 量和命中率细分。
- `send_lag`：客户端内部调度延迟（实际发出时间与计划到期时间之差），用于校验压测机自身是否存在调度积压。

#### 2. CSV 明细字段（共 16 列）
每一行对应真实发起的一次 HTTP 轮次请求：
```
warmup,task,phase,session_id,turn,prior_tool_call,prompt_tokens,cached_tokens,recompute_tokens,ttft_ms,e2e_ms,gap_since_prev_end_s,send_lag_ms,queue_time_ms,prefill_ms,ts
```
- `warmup`：布尔值，标识该请求是否落在预热窗口内；
- `prior_tool_call`：上一轮是否存在工具调用；
- `recompute_tokens`：本次请求未命中的 Token 数量（`prompt_tokens - cached_tokens`）；
- `queue_time_ms`：服务端内部队列排队时长；
- `prefill_ms`：服务端纯计算预填充耗时；
- `gap_since_prev_end_s`：距离该会话上一轮完成经过的真实等待时长。
