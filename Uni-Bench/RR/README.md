# RR：临界并发下的会话级保留（硬配额 + 朴素轮询）

本目录沉淀 **RR-EVICT** 策略的负载设计、预注册参数、指标定义与全部实测结果。

```
RR/
  README.md          本文件：索引与复现步骤
  design.md          负载设计：三原则 → 参数、四个可行性约束、时间轴、模板公式
  parameters.md      预注册参数：每个格子的全部旋钮取值与派生量（跑之前写定）
  metrics.md         预注册指标：定义 + 判定阈值（跑之前写定）
  results.md         实测结果（你正在编辑的版本）
  kappa_sweep.md     κ 0.1–0.8 全扫描记录（含违约数与容量模型校验）
  rolling_design.md  两两轮转空闲负载（12 会话 / 2 idle）设计 + 结构性限制 + 实测
  ttft.md            TTFT 口径、各实验实测、来源分解（排队 vs prefill）
  stair_design.md    台阶式空闲（pause 15/30/45/60 s）：唯一同时有冷度梯度与真实级联的构型
  scripts/           池构造、矩阵驱动、汇总脚本
  configs/           各格的 tasks yaml（绝对路径，可直接用）
  results/           每格的逐请求 CSV、汇总 JSON、summary.csv / summary_by_cell.csv
  figs/              6 张图（分组对比 / TTFT CDF / 保头分布 / 逐轮时间线 / κ 扫描 / 甜点区）
  patches/fair_rr.diff  策略实现（evict_policy.py + uni_utils.py）
```

## 图

| 图 | 内容 |
|---|---|
| `figs/fig1_grouped.png` | 5 个压力格的 burst 总重算 / TTFT p99 / max queue，base vs RR |
| `figs/fig2_ttft_cdf.png` | 4 个池种子合并的 burst TTFT ECDF |
| `figs/fig3_retained.png` | 每条空闲会话在回归时保住了多少前缀（核心机理） |
| `figs/fig4_timeline.png` | 空闲组逐轮命中，pause 前后 |
| `figs/fig5_kappa.png` | κ ∈ {0.3, 0.5, 0.7} 扫描 |
| `figs/fig6_sweetspot.png` | 溢出比 vs 收益（含 73k 阴性对照） |
| `figs/fig7_rolling.png` | 两两轮转空闲：逐轮回归命中 + 聚合重算 |
| `figs/fig8_ttft.png` | TTFT 均值分解（queue/prefill/other）+ TTFT vs 重算量 |
| `figs/fig9_serial.png` | 回归浪潮的串行服务 Gantt（排队 vs prefill 逐请求） |
| `figs/fig10_stair.png` | 台阶式空闲：冷度梯度下的保留率与回归代价（论文主图候选） |
| `figs/fig11_k3_vs_base.png` | κ=0.3 vs 基线：逐对保留/重算 + TTFT 的 queue/prefill 分解 |

## 一句话结论

在 `ΣL/pool ≈ 1.5–1.9` 的临界区，把「配额」从**排序时的过滤条件**改成**候选集的准入条件**，
配合最朴素的轮询，使 burst 期重算 −13~−31%、TTFT p99 −11~−52%，且活跃组命中不降。
无压力区（`ΣL/pool < 1.1`）数值等价，不产生副作用。

## 复现

```bash
# 1. 实现：fair_rr 参数 + APIGenMT 分类
#    patches/fair_rr.diff 是相对 HEAD(ffdb47b) 的 diff，当前工作区已包含这些改动；
#    只有在干净工作区（git stash / 新 clone）上 `git apply` 才有意义。
#    注意：该 diff 里也包含 commit 之后对注释的格式改动。

# 2. 构造负载（APIGenMT，长链 idle + 短链 active）
/root/sl_old/bin/python RR/scripts/build_burst_pool.py \
    --seed 20251027 --n-active 3 --n-idle 8 --pause-s 60 --name s1n3

# 3. 跑一格（示例：RR）
POOL=burst_s1n3 POLICY=rr WAIT_SCALE=1.0 SEED=42 \
  POLICY_CFG='{"fair_rr":true}' TAG=burst_s1n3_rr \
  bash tailshot/run_large.sh

# 4. 汇总
/root/sl_old/bin/python RR/scripts/summarize_burst.py
```

> `run_large.sh` 在 `tailshot/` 下（实验驱动，未随本目录迁移）。
> `configs/*.yaml` 使用绝对 `data_path`，可直接复制到 `tailshot/large/` 使用。
