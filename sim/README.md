# sim · HeterCache 离线前缀缓存仿真器

在 **CPU** 上还原 sglang 的 radix 前缀缓存与 HeterCache 策略（三队列 + one-shot 带 +
按权重跨队列淘汰 + 队内改写器）。目标不是「另写一个缓存」，而是**让生产代码原样跑**：
树、标签、仲裁、账本全部 `import python/sglang/srt/mem_cache/`，本目录只做装配与驱动。

> 现状：M0~M3 已完成。仿真器已在同一个 run 里重放四类混合会话并产出策略口径的
> 指标（命中率 / victim 来源 / 队列占用）。保真对拍（M4）待做。

## 快速开始

```bash
cd /home/rain/MySglang
python -m sim              # M0~M3 自检：装载 / 回路 / 时钟 / 驱动
python -m sim --import-only
python -m sim --verbose    # 保留策略 profile 日志（默认 quiet）
```

从其它目录跑需要 `PYTHONPATH=/home/rain/MySglang`。

跑一条自己的 trace（不需要任何数据集）：

```python
from sim.runner import SimConfig, Simulator
from sim.trace import Session, Turn

sim = Simulator(SimConfig(capacity=4096, seed=7))
result = sim.run([
    Session("apps_0", (Turn(t=0.0, prompt=list(range(400)), out=16),)),
    Session("ShareGPT-0", (Turn(t=30.0, prompt=list(range(400)) + [1], out=16),)),
])
print(result.render())          # 命中率 / 按会话类型 / victim 来源
print(result.fingerprint())     # 同 seed 复现基线
result.to_jsonl("out/run.jsonl")
```

直接吃 Heter-Bench 的 `session_data.jsonl`（与 `bench_heap.py` 同口径）：

```python
from sim.runner import SimConfig, Simulator
from sim.session_data import load_task

sessions = load_task(
    "Heter-Bench/TaskData/SharedGPT-multi-turn/Qwen2.5-7B-Instruct/"
    "ShareGPT_cleaned_valid_deduplicated_session_data.jsonl",
    task_name="ShareGPT", max_record=200, shuffle=True, seed=42 + 2,
)
result = Simulator(SimConfig(capacity=100_000)).run(sessions)
print(result.render())          # 含 per-task 命中率
```

## 为什么能在 CPU 上跑

只有两处重依赖挡路，且都**不是运行时需要**的：

| 模块 | 谁导入、为什么 | 处理 |
|---|---|---|
| `sglang.srt.mem_cache.memory_pool` | `base_prefix_cache.py:23`，只服务 `:58` 的 `ReqToTokenPool` 注解，却把 `memory_pool → srt.configs → transformers` 拉起来 | 替身 `ReqToTokenPool`（实例化即报错） |
| `sglang.srt.observability.metrics_collector` | `base_prefix_cache.py:25-30`，四个符号只在 `init_metrics_collector()` 里用 | 替身（调用即报错，`enable_metrics` 必须 False） |

`sim/bootstrap.py` 的策略是**真模块优先**：实测能导入就不替换，只有 `ImportError` 才装
替身并记录原因（`python -m sim` 会打印）。所以环境修好后替身自动退场，不会出现
「以为在跑真代码，其实跑的是空壳」。

节点 `value` 用 CPU `torch.int64` tensor 承载（生产 `insert` 需要 `.clone()`/切片，
`match_prefix` 需要 `torch.cat`），内容无语义，只占长度。1M token ≈ 8MB。

## 里程碑

| # | 内容 | 状态 |
|---|---|---|
| **M0** | 装载层：CPU 上导入生产模块并跑通 match/insert/evict | ✅ 已完成（`bootstrap.py` + `__main__.py`） |
| M1 | 时钟单点化：`CacheInitParams.now` → `TreeNode` / `match` / `insert` / `QueueOccupancyTracker` / `CacheTelemetry` / `MultiAgenticEvict` 全部走同一虚拟时钟 | ✅ 已完成（`sim/clock.py`） |
| M2 | `TokenAllocator` + `create_simulated(eviction_policy=..., now=...)` | ✅ 已完成（`sim/allocator.py`） |
| M3 | 请求驱动：`SimReq` + `FakeReqToTokenPool` + `cache._adopt()` + 短缺即淘汰 + 指标 | ✅ 已完成（`sim/req.py`、`trace.py`、`runner.py`、`metrics.py`） |
| M4 | 保真对拍：树不变量 / 同 seed 复现 / TTL 边界 / （可选）与真 server 比命中率 | ✅ 已完成（`sim/fidelity.py` + `test/manual/test_sim_fidelity.py`） |
| R1 | 数据集接入：`session_data.jsonl → List[Session]` + per-task 指标 | ✅（`sim/session_data.py`） |
| R2 | 发射规则：闭环轮次 + 服务时长模型 + arrival/wait scale | ✅（`sim/schedule.py`） |
| R3 | 并发窗口：同刻 k 条在飞、锁持到完成、竞态可复现 | ✅（`Emitter(max_inflight=...)`） |
| R4 | response 唯一哨兵 + 超大请求显式处理 | ✅（`sim/req.py`、`SimConfig(oversize=...)`） |

### 分配器（M2）

`sim/allocator.py::TokenAllocator` 只回答缓存真正会问的四件事：`size` /
`device` / `available_size()` / `free_segment()`，外加驱动层用的 `alloc(n)`。

- **不能用 `cache.evictable_size()` 冒充 `available_size()`**：`evict()` 的承压判据要
  真实水位（含在飞请求与已锁节点），用可淘汰量会让承压窗口永不触发。
- `alloc` 超容量直接 `OutOfTokens`，`free_segment` 超已分配直接报错：驱动顺序错了
  应当当场炸，而不是静默超配让命中率虚高。
- 不变量（每一步都该成立，自检里逐请求校验）：
  `alloc.used == cache.evictable_size() + cache.protected_size()`

构造走 `HeterCacheRadixCache.create_simulated(...)`：`cls` 调用所以拿到的就是
HeterCache 后端，`eviction_policy_config` 原样展开成策略 kwargs，`now` 注入虚拟时钟。

### 为什么时间源必须单点（M1 已落地）

策略的 `HeterCacheStrategy(now=...)` 曾经能注入时钟，但节点时间戳写死在
`radix_cache.py`（`TreeNode.__init__`、`match`、`insert`）。注入假时钟后
`is_aged` / `is_one_shot` 会算成「假 now − 真时间戳」，结果是所有节点立刻老化或永不
老化——TTL（120/180/300s）、`ONE_SHOT_IDLE_S=90`、`reuse_window=90`、承压窗口 3s 全部
失真。现在时钟由 `CacheInitParams.now` 单点持有，`RadixCache._stamp()` 给每个新节点
盖时间戳，再经 `use_clock()` 推给账本 / 遥测 / 队内改写器。

### 闭环发射（R2）：`bench_heap` 的调度规则

`sim/schedule.py` 把 emitter 的时间轴搬到虚拟时钟：

```python
from sim.runner import SimConfig, Simulator
from sim.schedule import ServiceModel, TaskSpec

sessions = load_tasks(cfg["tasks"], base_dir="Heter-Bench")
specs = [TaskSpec.from_config(t) for t in cfg["tasks"]]
service = ServiceModel.from_calibration("result/tailshot/calib_lru.json",
                                        decode_tokens_per_s=40.0)
result = Simulator(SimConfig(capacity=100_000)).run_scheduled(
    sessions, specs, service=service, arrival_scale=1.0, wait_scale=1.0)
```

| 规则 | 实现 |
|---|---|
| 首轮到达 | 每 task 独立 Poisson（`expovariate(λ)` 累加）或 `arrival.mode=fixed`（`offset + step*i`），乘 `arrival_scale` |
| 轮间空窗 | `lognormal(mu, sigma) * wait_scale`，每 session 一条独立 RNG，seed = `stable_seed_from_str(sid, seed + task_id*1000)`（与 emitter 同式，金标对拍过） |
| 轮次时刻 | **闭环**：`due = 本轮完成时刻 + 空窗`，而不是 trace 里写死的 t |
| 服务时长 | `prefill_base + prompt/prefill_rate + out/decode_rate`，`prefill_*` 直读 `tools/calibrate.py` 产物 |

为什么必须有服务时长模型：虚拟时钟里没有「读流完成」，没有它 `finished_ts` 就等于
`开始时刻`，闭环退化成 0 延迟——`wait_scale`/lognormal 都对，但整条时间轴偏快，
老化 TTL 与 one-shot 豁免会在错的时刻触发。

### 边界处理（R4）

两处会静默改变树形，所以显式建模：

1. **response token 每请求唯一**（`sim/req.py::OUTPUT_SENTINEL_BASE`）。
   若所有请求共用一串 token，同一个 prompt 的重放会命中上一次的 response 段，
   树节点数/占用/`duplicate` 统计全偏低。也不能用数据集的 `response_ids`：
   下一轮的 prompt 后缀正是它，用了反而会命中生产里必然 miss 的那一截
   （生产存的是模型自己生成的 token）。
2. **超大请求**：`SimConfig(oversize=...)`
   - `"error"`（默认）：抛 `OutOfTokens`，带 rid/total/capacity——逼调用方别用
     小池子跑大会话；
   - `"skip"`：跳过落树、记 `skipped=True` 与 `summary["skipped_requests"]`，
     命中量照记（请求确实被服务了），池账与锁都干净。

### 并发窗口（R3）：同刻 k 条在飞

`run_scheduled(..., max_inflight=k)`：`0` = 不限（`bench_heap` 的默认口径），
`1` = 单通道顺序队列。默认是 `0`。

并发语义的关键是**把 begin 与 end 拆开**：

| | begin（到点） | end（完成） |
|---|---|---|
| 命中 | `match_prefix`（此时别人还没写树） | — |
| 锁 | `inc_lock_ref(命中节点)` | `dec_lock_ref` |
| 槽位 | `evict(shortfall) → alloc(need)` | 归还重复段 |
| 树 | **不写** | `_adopt(prompt+output)` |

由此得到的行为差异（也是三种模型最容易被忽略的分歧）：

* **竞态（racing）**：同刻到点、前缀相同的两条请求都 miss，然后各自插入——
  后者的 `need` 里有 64 token 是重复写入（`emit.duplicate_insert_tokens`）。
  `max_inflight=1` 时后到者会命中（`hits=[0,64,64]` vs 并发 `[0,0,0]`）。
* **锁窗口**：在飞期间命中的前缀 `lock_ref>0`、不在 `evictable_leaves`，
  别的请求逼出的淘汰动不了它（`test_sim_concurrency.py` 有专门的用例）。
* **池账**：在飞未落树的 KV 让不变量变成
  `alloc.used == tree_tokens + inflight_tokens`（M4 的等树版本只在无在飞时成立）。

代价（写清楚，避免过度解读）：写树发生在**完成时刻**，生产在 chunked prefill 下
是分块边写，所以本模型让前缀可见的时间略晚；这会使并发命中略低、`duplicate`
略高。方向是保守的（不虚高命中率）。

### 数据集接入（R1）：`session_data.jsonl → List[Session]`

`sim/session_data.py` 逐字段对齐 `bench_heap.py` 的读取口径：

| 字段 | 来源 | 依据 |
|---|---|---|
| `prompt` | `turn_i.input_ids` 整段历史 | emitter 直发 `input_ids`（`bench_heap:_resolve_prompt_ids` 返回值恒等数据集） |
| `out` | `len(turn_i.response_ids)` | emitter 只发 `max_new_tokens` |
| `tool_call` | **上一轮**的 `tool_call`（第 1 轮为 False） | emitter payload：`sess[f"turn_{idx-1}"].get("tool_call") if idx > 1 else None` |
| `task_name` | tasks_config 的 `name` | emitter 的 `per_task` 口径 |

打乱/截断/轮数过滤也同口径：`load_task(..., shuffle=True, seed=seed+task_id, max_record=..., min_turns=...)`。

由此得到一条推论：第 k+1 轮的 prompt 后缀是**数据集的** response，而生产树里存的是
**模型实际生成的** token，两者一般不同 ⇒ 那一截必然 miss。仿真器把 response 写成
每请求唯一的哨兵 id，正是为了让这件事也成立。

指标同时给出两套分组：`per_type`（session_type，策略视角）与 `per_task`
（tasks_config 的 name，emitter 视角）。

### 请求驱动（M3）

单轮请求六步，顺序对齐线上（`schedule_batch` 的 match → 挂锁 → 分配/写入 → release）：

1. `clock.advance_to(t)` —— 时间只由 trace 推进；
2. `match_prefix` —— 内含 `observe_request`（分类证据）；
3. `inc_lock_ref(匹配节点)` —— 在飞请求保护已匹配前缀；
4. 短缺即 `evict` → `alloc(need)` —— 对齐 allocator 的 shortfall 回调；
5. `_adopt(prompt+output, split_prompt=True)` —— 内含 `touch` + `decode_start` + 分裂；
6. `dec_lock_ref`。

三个必须知道的建模决定：

- **第 3 步不能省**。没有锁，第 4 步可能把刚命中的前缀淘汰掉、插入时再重建，
  命中率与 victim 来源都会偏离线上（线上在飞请求的节点不在 `evictable_leaves`）。
- **请求行不建模**。仿真只把树当占用主体，所以 `req.kv.cache_protected_len` 被设为
  整条长度，令 `_adopt` 收尾的「释放重复槽位」段为空。这样池账恒等于树内 token，
  而 `_adopt` 的语义（`touch` / `decode_start` / prompt 边界分裂）一字不改。
- **单请求超池容量直接报错**（`OutOfTokens` 带 rid/need/capacity）：顺序模型不模拟
  retraction（生产会拆 chunk 或重算），这是 trace/容量不匹配的信号，不是缓存行为。

指标里 `victim_sources` / `queue_hit` / `queue_evicted` / `sched_occ` 直接读生产对象
（`eviction_strategy.profiler` 与 `occupancy`），只有逐请求命中率是新算的。

### 保真对拍（M4）

`sim/fidelity.py` 跑 80 请求的压力 trace（分裂 / 跨会话共享 / 一次性 / 老化四类模式
轮流出现），每一步后全树校验三件事：

1. **拓扑不变量**：`abs_start` 链（子 = 父偏移 + 父段长）、`len(key) == len(value)`、
   `evictable_leaves` 与树形双向一致（无子节点且未上锁 ⇔ 在集合里）。分裂路径算错
   不会报错，只会让命中位置偏移——所以逐请求校验而不是只查末态。
2. **账本对拍**：`occupancy.occ`（增量维护）vs 全树快照（`total - one_shot`）。
   增量记账**有意**允许两次校准之间漂移，所以只要求 reconcile 之后逐键相等；
   reconcile 之前的漂移作为数据报出来（实测 struct +64 / chat +16 / agentic 0），
   它是对的：节点被升级成 one-shot 或改了标签时，增量侧不会回冲。
3. **复现性**：同 seed 两次 run 的逐请求指纹（含淘汰量与水位）必须相同。

接进 pytest：

```bash
PYTHONPATH=$PWD/python python -m pytest -q test/manual/test_sim_fidelity.py
```

**不包掉**：与真 server 的命中率对拍。那需要 GPU 起服务 + 同一份 token 化 trace，
属于搬运回仓库后的实验步骤（Simulator 的 `result.to_jsonl()` 与 Heter-Bench 的
逐请求口径一致，可直接拿来比对）。

## 建模规则（driver 已强制）

1. `SimReq.origin_input_ids` 只放 **prompt**，不含 output（否则分类器 prompt_len 偏大）。
2. 每请求 `rid` 唯一（`CacheTelemetry.on_hit` 按 rid 去重，重复则命中计数恒 0）；
   driver 用 `session_id#序号` 自动生成。
3. `match_prefix` 必须在 `_adopt` 之前（出生标签取 registry 里的 session_type）。
4. 每次 run 前 `SessionRegistry.reset_global()` 再建 cache（进程级单例）；
   `Simulator.prepare()` 负责这一步。
5. 分配量 = `len(page_aligned(prompt+output)) - hit`，不要用 insert 返回值推断
   （分裂是净零）。
6. `page_size=1` 起步（>1 会截断 prompt 尾 token）。
7. 会话账本的时钟也必须同源：`SessionRegistry.use_clock()` 由
   `HeterCacheStrategy.use_clock()` 从缓存层推下去。否则 `last_active` 是墙钟、
   `is_one_shot` 的时间豁免永远不成立，one-shot 带静默失效。

自检不变量（自检里已生效）：`alloc.used == cache.evictable_size() + cache.protected_size()`。

## 边界（写清楚，避免以后误用）

- **顺序模型**：不模拟并发 racing / chunked prefill / retraction / page sharding /
  HiCache 多级 / 多副本；在飞窗口只有「当前这一条请求」。
- 单条请求超过池容量会直接报错（见上文），不会静默降级。
- 因此与真 server 对拍时只期望「同量级」，不期望逐 token 相同。
- 不复制策略：本目录出现第二份仲裁或标签逻辑就是 bug。
