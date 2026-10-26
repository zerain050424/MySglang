# step · `d_i` 的实施步骤

> 配套：`Design1.md`（设计）、`README.md`（实验）。
> 本文只写"改哪里、按什么顺序、怎么判死"。分两层：**L1 只记账**，**L2 换选择键**。

---

## 0. 总览

| 层 | 内容 | 行为变化 | 净增行数 |
|---|---|---|---|
| **L1** | 节点记 `session_id` + 会话状态表 + 缺口记账 | 无 | ~80（`uni_utils.py` +75，其余 +5） |
| **L2** | 选择键从"全局最小分数"改为"会话缺口加权最小" | 有 | ~45 |

L1 可独立合并、独立验收；L2 在 L1 的账目对得上之后再动。

---

## L1 · 只记账

### 1.1 改动清单

| 文件 | 位置 | 改动 | 行数 |
|---|---|---|---|
| `mem_cache/base_prefix_cache.py` | `InsertParams` (77) | 加 `session_id: Optional[str] = None` | +1 |
| `mem_cache/radix_cache.py` | `TreeNode.__init__` (263) | 加 `self.session_id = None` | +1 |
| | `_split_node` (831) | 前缀半边继承 `session_id`，位置同 `task_type` / `is_decode` | +1 |
| | `_insert_helper` (879) | 签名加 `session_id`；新建节点处赋值（与 927 行 `task_type` 同处） | +2 |
| | `insert` (480) | 从 `params.session_id` 透传到 `_insert_helper` | +1 |
| | `__init__`（≈351，与 `evictable_leaves` 同处） | `self.session_registry = SessionRegistry()` | +1 |
| | `_adopt` (503) | ① 传 `session_id=req.session_id`；② 写会话状态：`L_i`、gap 样本、`d_i = 0`、`last_active` | +12 |
| | `_delete_leaf` (959) | 记账勾子：`d_{owner} += len(node.value)`，`node_count -= 1` | +3 |
| `mem_cache/uni_utils.py` | 文件末尾（现 114 行） | `SessionState` + `SessionRegistry`（`touch` / `charge` / `sweep` / `summary`） | +75 |
| `mem_cache/evict_policy.py` | `profile_summary`（≈300） | 输出 `Σd`、`p50/max d_i` | +5 |

### 1.2 数据结构

放在 `uni_utils.py` 末尾，与 `WORKLOAD_LABELS` / `classify_simple_hybrid_task_type` 同一模块：

```
SessionState        # 一条会话的账
    session_id      str
    L               int      最近一次请求的 prompt 长度
    d               int      当前缺口（token），请求时归零、淘汰时累加
    node_count      int      仍驻留的节点数，归零即回收
    last_active     float    monotonic
    gaps            list     轮间隔样本（按 rid 去重）

SessionRegistry     # 表 + 四个方法
    touch(req, prompt_len, ts)   _adopt 调用
    charge(node)                 _delete_leaf 调用
    sweep(now)                   回收
    summary()                    给 profile 日志
```

`charge(node)` 只读 `len(node.value)` 与 `node.session_id`，按鸭子类型写，不 import `radix_cache`，因此 `uni_utils` 与 `radix_cache` / `evict_policy` 之间不会成环。

### 1.3 两个挂点的理由

- **`_delete_leaf`**：节点消亡的唯一原语（`_evict_one` 的 one-shot 连删、`release_radix_session` 都经过它）。挂这里不漏账；挂 `_evict_one` 会漏掉非淘汰路径的释放。
- **`_adopt`**：唯一同时持有 `req` 与 `prefix_len` 的插入入口（`cache_finished_req` 与 `cache_unfinished_req` 都走它）。

### 1.3.1 为什么放 `uni_utils.py`

- 与 `WORKLOAD_LABELS` / `alpha_key_of` 同属 UniCache 迁移层的常量与状态，命名口径一致；
- 该模块现在只有 `math` / `enum` / `time` / `typing` 依赖，加状态表不引入新依赖，也不反向依赖树；
- 对上游的 diff 变成"一个已有文件尾部追加一块"，不新增文件名，rebase 冲突面最小。

**必须是实例而不是模块级单例**：`SessionRegistry()` 在 `RadixCache.__init__` 里创建，挂 `self.session_registry`。模块级 dict 会在多 cache 实例（benchmark、单元测试、HiCache 分支）之间串账。

### 1.4 `d_i = 0` 成立的依据

会话自己的请求写入后，链完整驻留；缺口只由后续淘汰产生。因此

```
d_i 只在两处变化：请求写入时归零、片段被删时累加。
```

不需要用 `cached_tokens` 反解，也不需要处理共享前缀的部分命中。

### 1.5 三个边角

| 边角 | 处理 |
|---|---|
| chunked prefill 让同一请求多次进入 `_adopt` | 按 `rid` 去重后才记 gap 样本；`d_i = 0` 照常 |
| 共享前缀的分裂只留一个 owner | 记最后写入者；共享节点只有变成叶子时才可淘汰，此时它只服务一条链 |
| `session_id is None` | 不进会话表，`d` 不计，选择键保持现状（灰度共存） |

---

## L2 · 换选择键

### 2.1 改动清单

| 文件 | 位置 | 改动 |
|---|---|---|
| `mem_cache/simple_hybrid_radix_cache.py` | `evict` (20)，第 47–52 行 `min(leaves, key=strategy.victim_key)` | 改为会话级选择（见 2.2） |
| | `_evict_one` (56) | 加护栏过滤与放宽路径 |
| `mem_cache/evict_policy.py` | `SimpleHybridStrategy` | 加 `session_victim_key(state)`、`P_i` 四档量化与 200ms 节流；**现有 `victim_key` / `_reuse_score` / `is_one_shot` 不动** |
| `arg_groups/kv_cache_hook.py` | (≈375) | `--radix-eviction-policy-config` 校验 `fair_drr` / `kappa` / `min_retain` / `pin_ttl` |
| `arg_groups/choices.py` | (184) | 不新增 policy 名，用 config 开关 |

### 2.2 选择规则

```
刷新 P_i（若到刷新周期）
循环 直到 num_evicted ≥ num_tokens:
    candidates = 可淘汰叶子中，其所属会话满足 d_i < min(κ·L_i, L_i − min_retain) 的部分
    为空 → 放宽到 d_i < κ·L_i；仍为空 → 放宽到全部叶子
    victim = argmin over candidates of ( P_i · d_i , −L_i )
    释放 victim，d_{owner} += len(victim.value)
```

band-0（one-shot）行为不变。

---

## 3. 实施顺序与判死标准

| 步 | 内容 | 判据 | 不过则 |
|---|---|---|---|
| **S1** | 合 L1，跑 R3 | 日志 `d_i` 与 CSV `prompt − cached` 逐轮一致；`Σd_i ≤ 淘汰总量`；TTFT / 命中率与基线**零差异** | L1 记账有漏，回 `_delete_leaf` / `_adopt` 查 |
| **S2** | 开 L2，`w = 1`（纯轮询） | `max_i d_i ≈ 淘汰总量/N + 最大片段`（本配置 ≈ 0.7k）；`HHI(组) < 0.5`；命中率不变 | 选择逻辑或所有权归属错 |
| **S3** | 开 `P_i`（视界加权） | `p99 resume TTFT` 不劣于 S2；`max_i d_i` ≤ `κ·L_i` 恒成立；记录 Brier 校准度 | 权重口径错，回 `P_i` 定义 |

S1 是纯观测改动，验收标准是**零回归**；S2、S3 才允许指标变化。

---

---

## 4. 不做的事

- 不动 `_reuse_score` / `is_one_shot` / 两带结构 / alpha 控制器。
- 不动 `TaskData`、`bench_unicache.py`、`exp/` 下的实验脚本。
- 不引入 `cost()` 折算（见 `Design1.md` §9）。
- 不做跨 workload 仲裁。

---

## 5. S1 执行结果（2026-10-26）

### 落地内容

| 文件 | 改动 | 行数 |
|---|---|---|
| `mem_cache/uni_utils.py` | `SessionState` + `SessionRegistry`（`touch` / `attach` / `charge` / `sweep` / `summary`） | **+123** |
| `mem_cache/radix_cache.py` | `TreeNode.session_id`；9 处接线（import、`__init__`、`insert`、`_adopt`、`_split_node`、`_insert_helper`、`_delete_leaf`、`evict`） | +26 |
| `mem_cache/evict_policy.py` | `profile_summary` / `maybe_update_alpha` 加会话摘要 | +10 |
| `mem_cache/simple_hybrid_radix_cache.py` | profile 调用传第三份口径 | +3 |

`base_prefix_cache.py` 无需改动：`InsertParams.session_id` 字段本来就有。

### 零回归（判据达标）

同一配置（R3）、同一 seed，与基线对比：

| 指标 | 基线 | S1 | Δ |
|---|---|---|---|
| 请求数 / 错误 | 336 / 0 | 336 / 0 | 0 |
| 全局命中率 | 88.11% | 88.10% | −0.01 |
| TTFT p50 / p90 / p99 | 54.09 / 75.51 / 195.16 ms | 53.11 / 72.93 / 180.32 ms | 墙钟抖动，非因果 |
| 级联指标（wipe / HHI 组 / cold 命中） | 4 / 1.00 / 0.012 | 4 / 1.00 / 0.012 | 不变 |

### 账目自洽

```
[SimpleHybrid profile]  各次
  sessions=17  d_sum=0      d_p50=0  d_max=0    d_nonzero=0
  sessions=23  d_sum=5200   d_p50=52 d_max=692  d_nonzero=12
  sessions=34  d_sum=12208  d_p50=0  d_max=2695 d_nonzero=12
```

对账结果：

| 检查 | 结果 |
|---|---|
| `d_nonzero == 12` | ✅ 精确等于 12 条 victim；stirrer（text）`evicted=0` ⇒ `d=0` |
| `d_sum ≤ agentic evicted` | ✅ 12,208 ≤ 14,339（差额是各自请求后归零的部分） |
| `d_max` 与最长链同量级 | ⚠️ 2,695 vs `max(L26 + response26) = 2,569`，**高 5%** |
| `session_id=None` 不入表 | ✅ 单测通过 |
| chunked prefill 同 rid 不重复记 gap | ✅ 单测通过 |

### 未决项 → 已定位并修正（同日）

**现象**：`d_max = 2695` 而 `max(L26 + response26) = 2569`。

**定位**（`SGLANG_SESSION_TRACE_SID` 打开后逐次 charge 打印 `nid/abs_start/len`）：

```
阶段1  nid=24/54/86/125/…   abs=229,305,379,535,695,…   稀疏、跨越
阶段2  nid=36/73/75/105/…   abs=229,302,305,377,…       完整连续平铺 [229,1754)
```

阶段 2 是一条完整链（逐段首尾相接），阶段 1 是**同一 token 区间上的另一代节点**。
57 次 charge 的 token 求和 2355，去重覆盖仅 1729 ⇒ **26% 重复计账**。

**根因**：同一会话在树里可以同时存在**活路径与孤儿分支**。会话被部分清空后，
下一轮请求从断点插入时匹配不到断点以上的旧节点，于是在断点处另开分支、把断点以下
重新插一遍；旧的那截没有删，成为同级孤儿，但仍挂着该 `session_id`，淘汰时照样计入 `d`。

**修正**：`d` 的语义是当前缺口，上限即链长，超出部分直接截断：

```
st.d += num_tokens
if st.L and st.d > st.L:
    st.d = st.L
```

调试日志在截断前打印，保留原始超额供后续定位。

**修正后**（R3 同一配置）：

| 指标 | 修正前 | 修正后 |
|---|---|---|
| `d_max` | 2695 | **1978**（≤ 最长链 2552，等于某会话被整链清空时的 L） |
| `d_sum`（末次 profile） | 12208 | **10911** |
| 全局命中率 | 88.10% | 88.08%（基线 88.11%） |
| TTFT p50 / p99 | 53.11 / 180.32 | 53.83 / 204.90（抖动区间 180–205） |

**精确版（未实施）**：给节点加 `gen` 代际戳 —— `touch` 时 `session.gen += 1`；
`_insert_helper` / `_match_prefix_helper` 在刷新 `last_access_time` 处顺带 `node.gen = gen`；
`charge` 只在 `node.gen == session.gen` 时计账。孤儿分支被淘汰时不计入 `d`，
不需要截断兜底。等 L2 需要更精确的 `d` 时再上。

**遗留观测**：孤儿分支本身是真实的内存浪费（占着 KV 但不在任何活路径上），
它们是叶子、会被正常回收，但会推迟热数据的回收时机。属淘汰策略问题，不在 L1 范围。
