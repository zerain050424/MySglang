# sim · HeterCache 离线前缀缓存仿真器

在 **CPU** 上还原 sglang 的 radix 前缀缓存与 HeterCache 策略（三队列 + one-shot 带 +
按权重跨队列淘汰 + 队内改写器）。目标不是「另写一个缓存」，而是**让生产代码原样跑**：
树、标签、仲裁、账本全部 `import python/sglang/srt/mem_cache/`，本目录只做装配与驱动。

> 现状：仓库正在被并行修改。本目录**不改 `python/sglang/`**；等改动收敛后再统一搬运
> 回仓库测试。当前只实现逻辑。

## 快速开始

```bash
cd /home/rain/MySglang
python -m sim              # M0 自检：装载 + 最小 match/insert/evict 回路
python -m sim --import-only
```

从其它目录跑需要 `PYTHONPATH=/home/rain/MySglang`。

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
| M3 | 请求驱动：`SimReq` + `FakeReqToTokenPool` + `cache._adopt()` + 短缺即淘汰 + 指标 | 待做 |
| M4 | 保真对拍：树不变量 / 同 seed 复现 / TTL 边界 / （可选）与真 server 比命中率 | 待做 |

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

## 建模规则（M3 会强制遵守）

1. `SimReq.origin_input_ids` 只放 **prompt**，不含 output（否则分类器 prompt_len 偏大）。
2. 每请求 `rid` 唯一（`CacheTelemetry.on_hit` 按 rid 去重，重复则命中计数恒 0）。
3. `match_prefix` 必须在 `_adopt` 之前（出生标签取 registry 里的 session_type）。
4. 每次 run 前 `SessionRegistry.reset_global()` 再建 cache（进程级单例）。
5. 分配量 = `len(page_aligned(key)) - hit`，不要用 insert 返回值推断（分裂是净零）。
6. `page_size=1` 起步（>1 会截断 prompt 尾 token）。

自检不变量（自检里已生效）：`alloc.used == cache.evictable_size() + cache.protected_size()`。

## 边界（写清楚，避免以后误用）

- **顺序模型**：不模拟并发 racing / 在飞请求 `lock_ref` / chunked prefill / retraction /
  page sharding / HiCache 多级 / 多副本。
- 因此与真 server 对拍时只期望「同量级」，不期望逐 token 相同。
- 不复制策略：本目录出现第二份仲裁或标签逻辑就是 bug。
