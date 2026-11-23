# sim · HeterCache 离线前缀缓存仿真器

在 **CPU** 上还原 sglang 的 radix 前缀缓存与 HeterCache 策略（三队列 + one-shot 带 + 按已兑现价值密度 VD 跨队列淘汰 + 队内改写器）。树、标签、仲裁、账本全部直接 `import python/sglang/srt/mem_cache/`，本目录只做装配、生命周期管理与驱动调度。

## 目录结构

```
sim/
  __init__.py        # 导出常用入口（Simulator, SimConfig, EngineConfig, TaskSpec, ServiceModel, Session, Turn, load_tasks, SimResult）
  __main__.py        # CLI 入口（转调 sim.selfcheck.main）
  selfcheck.py       # 环境自检：装载 / 最小回路 / 虚拟时钟边界
  simulator.py       # 薄门面 Simulator（run_replay / run_closed_loop / run_engine / result()）
  core/              # bootstrap / clock / allocator / request / config / pricing / lifecycle (RequestLifecycle)
  workload/          # trace / session_data / service (ServiceModel) / arrival (TaskSpec, ArrivalPlan)
  drivers/           # common / policies (ReservePolicy, CommitPolicy) / budget / admission / replay / emitter / engine
  analysis/          # metrics / cost / opt_map / fidelity
```

## 快速开始

```bash
cd /home/rain/MySglang
python -m sim              # 环境自检：装载 / 最小回路 / 虚拟时钟边界
python -m sim --import-only
python -m sim --verbose    # 保留策略 profile 日志（默认 quiet）
```

从其它目录跑需要 `PYTHONPATH=/home/rain/MySglang`。

跑一条自己的 trace（不需要任何数据集）：

```python
from sim import SimConfig, Simulator, Session, Turn

sim = Simulator(SimConfig(capacity=4096, seed=7))
result = sim.run_replay([
    Session("apps_0", (Turn(t=0.0, prompt=list(range(400)), out=16),)),
    Session("ShareGPT-0", (Turn(t=30.0, prompt=list(range(400)) + [1], out=16),)),
])
print(result.render())          # 命中率 / 按会话类型 / victim 来源
print(result.fingerprint())     # 同 seed 复现基线
result.to_jsonl("out/run.jsonl")
```

直接读取 Heter-Bench 的 `session_data.jsonl`（与 `bench_heap.py` 同口径）：

```python
from sim import SimConfig, Simulator
from sim.workload.session_data import load_task

sessions = load_task(
    "Heter-Bench/TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/"
    "ShareGPT_cleaned_valid_deduplicated_session_data.jsonl",
    task_name="ShareGPT", max_record=200, shuffle=True, seed=42 + 2,
)
result = Simulator(SimConfig(capacity=100_000)).run_replay(sessions)
print(result.render())          # 含 per-task 命中率
```

## 为什么能在 CPU 上跑

只有两处重依赖挡路，且都不是仿真运行期需要的：

| 模块 | 谁导入、为什么 | 处理 |
|---|---|---|
| `sglang.srt.mem_cache.memory_pool` | `base_prefix_cache.py` 仅为其类型注解导入 `ReqToTokenPool`，却会把 `srt.configs` 和 `transformers` 拉起来 | 替身 `ReqToTokenPool`（实例化即报错） |
| `sglang.srt.observability.metrics_collector` | `base_prefix_cache.py` 中的符号仅在 `init_metrics_collector()` 里用 | 替身（调用即报错，`enable_metrics` 必须为 False） |

`sim/core/bootstrap.py` 的策略是**真模块优先**：实测能导入就不替换，仅当捕获 `ImportError` 时才装替身并记录原因（`python -m sim` 会打印）。环境完善后替身自动退场。

节点 `value` 用 CPU `torch.int64` tensor 承载（生产 `insert` 需要 `.clone()` 与切片，`match_prefix` 需要 `torch.cat`），内容无语义，只占长度。

## 核心组件与设计细节

### 分配器与请求生命周期（`sim/core/`）

`sim/core/allocator.py::TokenAllocator` 只响应缓存读取的四项接口：`size` / `device` / `available_size()` / `free_segment()`，外加驱动层用的 `alloc(n)` / `release(n)`。

- 不能用 `cache.evictable_size()` 冒充 `available_size()`：`evict()` 的承压判据需要包含已锁节点与在飞请求的真实剩余容量。
- `alloc` 超容量抛出 `OutOfTokens`，`free_segment` 超已分配直接报错，避免静默超配导致命中率虚高。
- 池账不变量（每一步后均校验）：`alloc.used == tree_tokens(cache) + lifecycle.inflight_tokens`。

单请求生命周期原语全部集中在 `sim/core/lifecycle.py::RequestLifecycle`（`prepare` / `match` / `lock` / `new_pending` / `admit` / `admit_locked` / `start` / `reserve_upto` / `commit_upto` / `finish` / `serve`）。策略配置构建由 `sim/core/config.py::build_policy_config` 负责，块代价前缀和计价由 `sim/core/pricing.py::PromptPricer` 负责。

### 单点虚拟时钟（`sim/core/clock.py`）

时间源由 `CacheInitParams.now` 单点持有（`VirtualClock`），由 `RadixCache._stamp()` 盖节点时间戳，再经 `HeterCacheStrategy.use_clock()` 推给占用账本、遥测、会话注册表与队内改写器。这保证了 TTL（120/180/300s）、`ONE_SHOT_IDLE_S`、复用窗口与承压校准窗口完全与虚拟时间同步。

### 工作负载与闭环到达规划（`sim/workload/`）

`sim/workload/arrival.py`（`TaskSpec`、`ArrivalPlan`）和 `sim/workload/service.py`（`ServiceModel`）将 `bench_heap.py` 的发射规则搬到虚拟时钟：

```python
from sim import EngineConfig, ServiceModel, SimConfig, Simulator, TaskSpec, load_tasks

sessions = load_tasks(cfg["tasks"], base_dir="Heter-Bench")
specs = [TaskSpec.from_config(t) for t in cfg["tasks"]]
service = ServiceModel.from_calibration("result/tailshot/calib_lru.json",
                                        decode_tokens_per_s=40.0)
result = Simulator(SimConfig(capacity=100_000)).run_closed_loop(
    sessions, specs, service=service, arrival_scale=1.0, wait_scale=1.0)
```

| 规则 | 实现 |
|---|---|
| 首轮到达 | 每 task 支持 Poisson、`arrival.mode=fixed` 或 `arrival.mode=segments`（分段非齐次泊松，支持静默段），整体乘 `arrival_scale` |
| 轮间空窗 | `lognormal(mu, sigma) * wait_scale`，每 session 一条独立 RNG，seed = `stable_seed_from_str(sid, seed + task_id*1000)` |
| 轮次时刻 | 闭环 `due = 本轮完成时刻 + 空窗` |
| 服务时长 | `prefill_base + prompt/prefill_rate + out/decode_rate`，可直读 `tools/calibrate.py` 产物 |

#### 构成漂移：`arrival.mode=segments`

```yaml
arrival:
  mode: segments
  segments:                       # ((until_s, lambda_rate), ...)，until_s 严格递增
    - {until_s: 300,  lambda_rate: 0.17}   # warmup
    - {until_s: 1500, lambda_rate: 0.56}   # P1 主导
    - {until_s: 2700, lambda_rate: 0.00}   # 静默段：这一段不注入
    - {until_s: 3900, lambda_rate: 0.10}
```

- 段内 `t += Exp(λ_k)`；跨出段尾的采样丢弃并截到段尾重抽；
- `λ=0` 是静默段（无到达），末段静默时会截断多余会话，否则沿用末段速率填满；
- 同 seed 逐位可复现（RNG 与 `poisson` 同源）。

### 边界处理

1. **response token**：优先保留数据集的 `Turn.response`。只有无 `response_ids` 的合成 trace 才回退到每请求唯一哨兵（`sim/core/request.py::OUTPUT_SENTINEL_BASE = 1 << 30`）。
2. **超大请求**：`SimConfig(oversize=...)` 默认 `"error"` 抛 `OutOfTokens`；设为 `"skip"` 则跳过落树、单独计入 `skipped_requests`，不进入命中率分母。

### 闭环并发驱动（`Simulator.run_closed_loop`）

`run_closed_loop(..., max_inflight=k)`：`0` 表示不限（`bench_heap` 默认），`1` 退化为单通道顺序队列。到点时调用 `lifecycle.start()`（只做 match、上锁、占槽，不写树），完成时调用 `lifecycle.finish()` 写入树并释放锁，完整复现同刻竞态、重复写入（`duplicate_insert_tokens`）与锁保护窗口。

### SGLang 引擎步调度（`Simulator.run_engine`）

`sim/drivers/engine.py`（配合 `budget.py`、`admission.py`、`policies.py`）模拟 SGLang Scheduler 单步批调度语义：

```python
from sim import EngineConfig, SimConfig, Simulator

result = Simulator(SimConfig(capacity=100_000)).run_engine(
    sessions, specs,
    scheduler=EngineConfig(
        max_prefill_tokens=8192,
        chunked_prefill_size=4096,   # -1 表示关闭 chunked prefill
        max_running_requests=64,
        schedule_policy="fcfs",     # 支持 "fcfs" / "lpm"
        progressive_cache=True,      # chunk 间与转 decode 时增量落树移锁
        lazy_decode_alloc=True,      # 按 chunk 与每 decode 步增量占槽并淘汰
    ),
    service=service,
)
print(result.summary["emit"])
```

`lazy_decode_alloc` 与 `progressive_cache` 分别抽象为 `ReservePolicy`（`EagerReserve` / `LazyReserve`）与 `CommitPolicy`（`CommitOnFinish` / `ProgressiveCommit`），主循环无开关条件分支；且每条完成记录直接携带 `RequestRecord.arrival_t`（到点时刻）供排队延迟统计读取。

### 上游 simulator 的时钟注入

`tools/sglang-simulator` 的离线模式不推进墙钟，通过在 `RadixCache.__init__` 挂 `C_RadixCacheClockHook` 将 `params.now` 指到 `StateManager.get_global_clock`（不覆盖显式传入的 `now`，如离线 `VirtualClock`）。验证见 `test/manual/test_sim_clock_injection.py`。

### OPT 上界：策略对比天花板

`eviction_policy="opt"` 在闭环下必须先运行在线基线冻结真实派发时间轴，再生成包含 response 页的 use map：

```python
from sim import EngineConfig, SimConfig, Simulator
from sim.analysis.opt_map import use_map_from_run, write_use_map

PAGE, CAP = 8, 198_000
run = Simulator(SimConfig(capacity=CAP, page_size=PAGE, eviction_policy="lru")).run_closed_loop(
    sessions, specs, service=service
)
write_use_map("/tmp/use_map.json", use_map_from_run(run, sessions, page_size=PAGE))
upper = Simulator(
    SimConfig(capacity=CAP, page_size=PAGE, eviction_policy="opt",
              opt_use_map="/tmp/use_map.json")
).run_replay(run.freeze_sessions(sessions))
print(upper.hit_rate)
```

### 保真校验与回归测试

`sim/analysis/fidelity.py` 在压力事件序列上逐请求校验树拓扑不变量（`abs_start` 链、`key/value` 等长、`evictable_leaves` 双向一致性）、`QueueOccupancyTracker` 与全树快照在 `reconcile` 后零误差，以及同 seed 双次运行指纹完全一致。

```bash
PYTHONPATH=$PWD:$PWD/python python -m pytest -q test/manual/test_sim_*.py
```

### 数据集读取约定（`sim/workload/session_data.py`）

| 字段 | 来源 | 依据 |
|---|---|---|
| `prompt` | `turn_i.input_ids` 整段历史 | emitter 直发 `input_ids` |
| `response` | `turn_i.response_ids` | 离线回放写入树中供下一轮匹配 |
| `out` | `len(turn_i.response_ids)` | 对齐 `max_new_tokens` |
| `tool_call` | 上一轮的 `tool_call`（首轮为 False） | 对齐 emitter payload |
| `task_name` | config 的 `name` | 用于 `per_task` 分组统计 |
