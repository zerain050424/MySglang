# Autoresearch：HeterCache 相对 LRU 的可复现、可解释收益

## 目标与声明边界

优化 HeterCache 的淘汰策略，使它在**预先登记、可复现的异构负载测试集**中，相对同条件 LRU 提高可复用请求的 KV token 命中率，并用淘汰决策证据解释收益来自哪里。

不承诺、也不得宣称对任意可能负载都胜过 LRU：任意负载没有有限的可验证边界，且在无淘汰/纯单次访问等场景，LRU 可能与 HeterCache 持平或更好。结论只对本提示词登记并实际跑完的负载、容量、模型与硬件范围成立。禁止为了找一个看起来有利的指标而事后换主指标。

## 启动前硬门槛（任何代码修改或远端实验之前）

1. 检查 `git status`、当前分支和最近提交。若工作区不干净、目标文件含有非本实验改动，立即停下：不得 stash、reset、checkout、覆盖、自动提交或回滚用户改动。实验必须在独立、干净的 worktree/分支上从双方确认的基线开始。
2. 用户未提交改动已按其指示提交为基线 `e51973b`（main），实验分支 `autoresearch/heter-vs-lru-1111` 从该基线切出。基线之后的历史即实验自身的历史，回滚只回滚实验提交。
3. 核对 `.auto/measure.sh`、`.auto/checks.sh`、`.auto/remote_eval.py` 与实际目录、数据、端口、环境变量和目标机器；先做无改动 smoke test，不能进入优化循环。
4. 先冻结并记录测试矩阵：工作负载配置、数据文件校验和、容量、模型/服务参数、策略开关、seed、指标计算代码、硬件及软件版本。冻结后不得改 trace、生成器、标签、容量或统计口径来改善结果；确有测量错误时先记录原因、修正并重跑完整配对基线。
5. 远端认证只通过 Pi-Tramp/安全凭据配置（凭据存 `~/.heter_remote_env`，在仓库外）；密码、令牌不得写入 prompt、脚本、日志或提交。

## 预登记负载范围

只使用仓库中**已存在且能复现**的数据，并在第一次基线前固定具体文件和参数。至少覆盖：

- 其他类型会话受到 one-shot 扫描流搅拌，含低、中、高淘汰压力；
- 多级队列之间存在仲裁和淘汰，选择最优的淘汰方式；
- burst/pause 或负载阶段变化的场景；
- 中性反例：无淘汰压力、同质复用或近似全 one-shot，检查 HeterCache 不造成明显退化（最后验证）。

每个负载家族至少选 3 个容量档（低压、临界压力、高压）。不同策略必须使用完全相同的请求序列、容量、服务参数和模型。LRU 与 HeterCache 配对运行，运行前清空缓存；记录运行顺序并交错策略顺序，避免温度/时钟漂移偏袒某一臂。只把配置中明确标记的可复用 task 纳入主指标，禁止从 session_id 前缀猜类别。

## 指标与成功判据

### 主指标

`reusable_token_hit_pct`：对配置中预先标记的可复用 task，计算 `100 × Σcached_tokens / Σprompt_tokens`。

TTFT：命中率带来的性能提升报告 p10/p50/p90。

### 次指标 / 护栏

- 全体 `overall_token_hit_pct`；每个队列的 token 命中率，避免扫描流掩盖可复用流退化。
- 可复用 task 的 TTFT p95/p99、整体 TTFT p95、吞吐。
- 缓存正确性、错误数、显存/容量、淘汰与队列 profile；指标必须来自真实结果文件，不能手工编造。

### 成功条件（须全部满足）

1. holdout 配对实验的主指标宏平均增益至少 **+2.0 个百分点**，且配对差的 95% bootstrap 置信区间下界 **> +1.0 个百分点**。
2. 每个预登记压力负载家族的平均配对差均不为负；任何单个压力 cell 相对 LRU 不得退化超过 **1.0 个百分点**。中性反例可持平，但不得退化超过 1.0 个百分点。
3. 整体 token 命中率不得退化超过 1.0 个百分点；可复用 task 的 TTFT p95/p99 不得退化超过 5%。若无法可靠测得该护栏，报告"未验证"，不得声称无代价。
4. 至少提供一条可复核的机制链：观测到的负载特征 → 策略信号/配额 → 实际 victim 选择变化 → 命中率变化。profile/逐请求数据必须支持解释；单纯相关性或某一 seed 的好看数字不算。
5. 正确性检查全部通过，且最终收益在未参与调参的 holdout seeds 上成立。

若主指标提升小于噪声、置信区间跨零、只在一个 workload family 成立，或护栏失败，结果判为未达成；保留最佳已验证版本并如实报告，不得把次指标改成新的主指标继续宣称成功。

## How to Run

`./.auto/measure.sh <cell 参数>` → 在远端双 4090 实验机上跑 LRU/HeterCache 配对基准，输出 `METRIC name=value` 行（reusable_token_hit_pct 等详见 `.auto/remote_eval.py` 头注释）。凭据从 `~/.heter_remote_env` 读取，不在仓库内。

## Files in Scope

- `python/sglang/srt/mem_cache/evict_policy.py` — HeterCacheStrategy：one-shot 判定、victim_key、配额（QuotaParams）、候选挑选。
- `python/sglang/srt/mem_cache/victim_override.py` — 各队列候选改写器（RR/SLRU/MT）与账本。
- `python/sglang/srt/mem_cache/uni_utils.py` / `heter_utils.py` — 策略参数与共享数据模型（新增参数只在这里）。
- `python/sglang/srt/mem_cache/heter_cache_radix_cache.py` — 仅当策略接口需要时做最小适配，不动淘汰循环大框架。
- `Heter-Bench/mixexp/arms_exp3.tsv` — 实验臂清单（新增候选臂配置属实验代码）。
- `Heter-Bench/mixexp/analyze.py`、`.auto/remote_eval.py`、`.auto/measure.sh` — 指标解析与测量编排。
- 对应单测（语义变更时同步更新，不得弱化断言）。

## Off Limits

- LRU 基线、Radix 树结构、淘汰循环大框架（`if one_shot: ... elif leaves:`）、请求语义。
- 数据集、trace、任务分类、请求顺序、seed、容量、计分公式、错误处理——不得为提分而改。
- 按 session ID、固定 key 长度、特定 trace 数值、seed 或实验目录做特判；不得吞掉失败请求、只报告有利子集。
- 复活已删除的全局复用分（`_reuse_score` / `victim_key`）；不引入无证据复杂度。
- 顺手重构无关模块、添加依赖、降低已有测试断言。
- 仓库不变量：`record_evict` 早于 `free_segment` 与同轮下一次 `_pick_victim`；`SessionProbe.on_charge` 在截断前回调；`_observe_occupancy` 一次遍历两家消费；`getattr(..., default)` 探测的能力不得静默丢失。新增观测代码放 `Heter-Bench/`，sglang 侧只留桥接。

## Constraints

- `.auto/checks.sh` 必须通过。
- 双 4090：只跑两个独立单卡实验（`CUDA_VISIBLE_DEVICES` 显式绑定），配对比较在同一张卡上完成；**禁止分布式推理**（TP>1/torchrun/FSDP 等）。
- 每轮一个明确假设，先写预期机制与可证伪信号；先语义检查再跑配对基准。
- 不做随机调参：先 profile 找瓶颈，再提结构性假设。

## 退出条件

满足成功条件并在 holdout 上复现后立即停止。未达成时有界停止：最多 30 次 iteration 或 8 小时墙钟（先到为准）；连续 8 次无超出噪声的改善提前停止。`maxIterations=30`（见 `.auto/config.json`）。启动门槛不满足、凭据暴露、多卡推理迹象、指标口径无法复现 → 立即停止并报告。

## What's Been Tried

| # | cell / 改动 | 结果 | 判定 |
|---|---|---|---|
| 1 | smoke（链路验证） | 双臂跑通，0 错误 | keep（非优化） |
| 2 | fast_mid 基线：hc_full 缺省 vs LRU | 可复用 57.48 vs 84.75（-27.3pp），TTFT p99 801 vs 119ms | keep（基线） |
| 3 | quick_wt1 基线（真实节奏、分类正确）：hc_full vs LRU | 59.90 vs 84.10（-24.2pp）——均匀税复现，排除分类伪影解释 | keep（基线） |
| 4 | fast_mid 归因消融：hc_demandcap（caps 按需求对齐） | 54.31，比 hc_full 还差 → 证伪「cap 错配主导」 | keep（消融） |

### 已定位机制（双重现）
1. **均匀税**：hc 下所有多轮 task 每个 turn 桶均匀掉 ~20pp；ShareGPT t1（跨会话共享）
   39.7→7.5、codeparrot 68→36 —— 被杀的是跨会话共享 + 会话早期内容。
2. **M1 早期轮死亡螺旋**：前 4 轮节点（turn_count<MULTI_TURN_MIN=5）标 QTRANS/one-shot，
   活不过创建者下次重读；victim_src one_shot=157k(quick)/993k(mid)。
3. **M2 cap 硬上限**：quota:agentic=445k(quick)/4009k(mid)，同时 chat 队列大量闲置、
   targets 合计 < pool —— 有空闲却强拆主类。
4. **测量协议教训**：wait 压缩会把 chat 轮间隔压到 GAP_SPLIT_S=15s 以下 → 会话误分类为
   agentic；正式 cell 一律 wait=1.0；exp3 压缩档已移除。
5. **公平性教训**：sl_old venv 的 sglang 指向 /root/sg-unicache，必须
   PYTHONPATH=实验 clone 才能测到策略改动（已修）。

### 下一轮（8h 预算耗尽，留给后续会话）
F1（once-read 离开 one-shot 带）与 F2（cap 仅跨队列饥饿时强制），完整设计与
记账分析见 `.auto/ideas.md`。LRU 参照结果可复用（同代码同 seed，--skip-run lru）。
