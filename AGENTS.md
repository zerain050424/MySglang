# Agent 工作守则（本仓库）

给在这个仓库里做重命名、解耦、删代码的 agent 用。以下每条都来自实际事故或实际验证，
不是通用建议。

## 1. 开工前先确认工作区不是你以为的样子

用户会在同一个工作区并行提交。我曾三次在「按记忆中的 HEAD」操作时踩空：

- `c8d061c 精简代码` 删掉了 `HeterCacheStrategy.get_priority`，我却在几轮之后才发现类无法实例化；
- 同一天里 `heter_utils.py` 被并行改动两次；
- 我「恢复」一个方法后，用户又删了一次。

规则：

```bash
git log --oneline -3 && git status --short
```

必须看，而且要看**未提交改动是不是自己造成的**。`git diff <file>` 里出现不认识的删除时，
先读那个提交的意图（`git show <commit>`），再决定是恢复还是顺着改，不要直接 checkout 覆盖。

## 2. 不要用裸字符串锚点切割大文件

本轮三次误删都源于 `s.index("某段文本")`：

| 事故 | 原因 | 后果 |
|---|---|---|
| 删 `get_priority` | `s.index` 匹配到另一个同名方法 | 目标方法没删对 |
| 删 `_per_queue_tag_token_count` | 结束锚点用了文件末尾才出现的注释 | 连带切掉 420 行核心方法 |
| 同上（第一次） | 结束锚点写成 `def sweep_profile` | 切掉整段 `SessionRegistry` |

规则：

1. 锚点必须唯一。删除一段方法时，用**下一个方法的 `def` 行**作结束锚点，并在删前
   `assert start < end`、`assert 下一个方法名 in s[end:end+200]`。
2. 优先用 `edit` 工具的精确匹配，而不是脚本切片。
3. **删完立刻对照行数**：

   ```bash
   git diff --stat
   ```

   预期删 60 行却显示 426 行，就是切多了。
4. 逐个确认关键方法还在：

   ```bash
   for m in inc_lock_ref _match_prefix_helper _insert_helper _delete_leaf; do
     rg -c "def $m" python/sglang/srt/mem_cache/radix_cache.py
   done
   ```

## 3. 这个代码库里「删掉一个方法」常见的静默退化

删错不会报错，只会让行为悄悄变样。已知的五种：

| 删除对象 | 静默后果 | 发现方式 |
|---|---|---|
| 策略上的 `is_one_shot` | `radix_cache._is_one_shot` 的 `getattr` 回退成 `is_decode`，profile 分母与实际淘汰口径分叉 | 读 `_is_one_shot` 实现；改动后跑 `test_heter_cache_abs_offset` 的 one-shot 用例 |
| 策略的 `victim_overrides` | 缓存层两个 observer 元组恒空 → SLRU 共享数账本停更 → `share_of` 恒 0 → 所有节点被判 one-shot | `rg "node_observers"` 看装配链是否完整 |
| 策略的 `needs_capacity` | 驱动层 `getattr(..., False)` 拿到 False → `capacity=0` → `effective_targets` 全 0 → 配额退化成「每队都超上限」 | 起服务看 `[HeterCache profile]` 的 target 是否全 0 |
| 缓存层的 `_per_queue_token_count` | 配额账本失去唯一校准源，`_occ` 只会随增量缓慢漂移 | 它被 `_observe_occupancy` 调用，不是日志 |
| 缓存层的 `_record_insert` 钩子 | 新建节点不进账本，配额排名恒空、只能走随机兜底 | `test_evict_loop` 会出现 `quota:fallback` 满屏 |

**归纳**：凡是靠 `getattr(obj, "能力", 默认值)` 探测的能力，删掉实现等于**静默返回默认值**。
改这类代码时，先 `rg "getattr\(.*strategy"` 把所有探测点列出来。

## 4. 三条不能动的口径不变量

1. **`record_evict` 必须早于 `free_segment`，且早于同轮下一次 `_pick_victim`。**
   配额账本在同一次淘汰循环里被反复读取，扣减晚了会按虚高水位点名。写在
   `HeterCacheRadixCache._release_node` 的文档里。
2. **`SessionProbe.on_charge` 必须在 `d` 截断之前回调。** 截断后 `d <= L` 恒成立，
   `d > L + slack` 的告警分支永远进不去。
3. **观测快照一次遍历、两家消费。** `_observe_occupancy` 先问两个时钟，再取一次
   `_per_queue_token_count`，同时喂校准与渲染。不要拆成两次调用，那会让树遍历翻倍。

## 5. 本机验证工具箱

### 5.1 环境限制

base conda 的 `huggingface-hub==1.33.0` 与 `transformers` 要求的 `<1.0` 冲突，
**任何 `import torch` + 完整 sglang 导入链的脚本都跑不起来**。受影响：
`test_heter_cache_abs_offset.py`、`test_force_agentic.py`、`test_cache_reset_isolation.py`。

能直接跑的：

```bash
PYTHONPATH=$PWD/python python -m pytest -q \
  test/manual/test_quota_selector.py test/manual/test_slru_one_shot.py
PYTHONPATH=$PWD/python python test/manual/test_evict_loop.py     # ALL PASSED 才算过
PYTHONPATH=$PWD/python python test/manual/test_session_type_classifier.py
```

### 5.2 不依赖 torch 的验证手法

`test_quota_selector.py` 与 `test_evict_loop.py` 用「stub 包 + 按路径加载」绕开 torch：

```python
for pkg in ("sglang", "sglang.srt"):
    m = types.ModuleType(pkg); m.__path__ = []; sys.modules[pkg] = m
m = types.ModuleType("sglang.srt.mem_cache"); m.__path__ = ["python/sglang/srt/mem_cache"]
sys.modules["sglang.srt.mem_cache"] = m
```

**新增模块必须同步加进装载列表**，否则报 `No module named ...`。本轮为此改过两次：
`queue_occupancy`、`heter_bench_bridge`。

纯逻辑验证（不建真缓存）也能做：`HeterCacheRadixCache.__new__(cls)` 加假协作者，
即可测 `_release_node` 的调用序列、`_observe_occupancy` 的时钟调度。

### 5.3 假树测试的陷阱

给假缓存 stub `_per_queue_token_count` 返回 `{}` 时，驱动层新增的 reconcile 会用它把
测试种下的 `_occ` 清零 → 配额排名为空 → 断言全部走兜底。桩要回显账本：

```python
return {q: {"total": v, "one_shot": 0.0} for q, v in self.occupancy.occ.items()}
```

### 5.4 CRLF 会让 `bash -n` 误报

仓库 `core.autocrlf=true`，工作区脚本是 CRLF，而 `.gitattributes` 规定 `*.sh eol=lf`。
`bash -n` 会对 `do\r` 报语法错。验证前先转：

```bash
tr -d '\r' < script.sh > /tmp/s.sh && bash -n /tmp/s.sh
```

### 5.5 改完先 grep 旧名归零，但要划清范围

```bash
rg -n "旧符号" python/ test/ Heter-Bench/        # 目标范围应归零
```

域外不要动：上游 sglang 的通用英文（`memory_pool.py`、`test/registered/**`）、
Heter-Bench 的数据集层术语（`TaskData/`、`wtl/replay.py` 的 `meta["workload"]`）。
Heter-Bench 文档里的历史数字与旧口径说明保留原文，只在需要时加注「数据源已下线」。

## 6. 提交习惯

- **一批一个 commit**，能在中间停下来验证的边界才切批；跨批的中间态若不可运行，要在
  commit message 里写明，或者合并成一批。
- message 里写**实测数字**而不是形容词：「record_evict 1011ns → 611ns（-40%）」比
  「性能提升」有用。
- 等价性改动必须附对拍结论：本轮的「500 步随机 insert/evict 后 `_occ` 逐键一致」、
  「`_release_node` 8 步调用序列与改前相同」都写进了提交说明。
- 删代码后看一眼 `git status` 是否只包含本次目标文件。

## 7. 这个仓库里已确立的几条设计结论

避免重复讨论：

- `HeterCacheStrategy.get_priority` **保持继承基类 + 显式抛 `NotImplementedError`**：
  HeterCache 没有跨队列可比的节点分数，退回任何默认序都是假语义。
- 配额**必填**（`QuotaParams` 空 caps 直接 `ValueError`），不存在「未启用配额」的降级路径。
- 全局复用分（`_reuse_score` / `victim_key`）已整体删除，不要以任何形式复活。
- 观测三层已分离：`QueueOccupancyTracker`（账本）、`CacheTelemetry`（时钟+打点）、
  驱动层 `_observe_occupancy`（编排）。策略只做判据与裁决。
- 策略无关的队列占用日志（`OccSampler` / `QueueOcc` / `TagOcc`）已删除，
  `[HeterCache profile]` 是队列占用的唯一出口。
- 观测实现全部在 `Heter-Bench/`，sglang 侧只留 `heter_bench_bridge.py` 这一个桥
  （只做定位与导入；Heter-Bench 缺席时启动即报错，不留空实现）：
  `Heter-Bench/log/heter_logger.py`、`profile/{heter_profiler,cache_telemetry,node_label_logger}.py`。
  新增观测代码请放这里，不要放进 `python/sglang/srt/mem_cache/`。
