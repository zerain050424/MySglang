# 最优配置速查（TTFT 优先 / heter 保守改进）

> 【长尾优化专题 · 现行最优推荐】汇总 TTFT 优先部署配置（chain_lru 全局版）与 heter 保守改进档位（档 A/B）
> 时间：2026-11-18 · 分支 W1（`aba2b66`）· 全部数字实测于 tail-stress
> （cap=198k、page=8、engine FCFS、1700 请求）与 exp3 四 cell（同
> `exp3_cells_ttl300.md` 口径）。详细过程见 `chain_shape_report.md`。

## 一、chain_lru 全局版（TTFT 优先部署的推荐配置）

```json
{ "floor_tokens": 8192, "weight": 0.02, "gate": "multiturn", "tail_first": true }
```

用法：`--radix-eviction-policy chain_lru`（仿真里 `SimConfig(eviction_policy="chain_lru",
policy_config={...})`；gate="multiturn" 是唯一推荐档——rerode 漏第一轮、none 在真实负载养死锚）。

| 指标 | tail-stress | 对照 |
|---|---|---|
| hit | **80.05%** | LRU 77.63 / heter 73.4 / OPT 84.84 |
| TTFT p50 / **p90** / p95 / p99 (ms) | 135 / **404** / 745 / 1458 | LRU p90 920（−56%）；OPT 184 |
| warm 整链丢失 | **23.8%** | LRU 52.2%（死循环消除，回到首轮天然基线） |
| warm mean miss | 3089 tok | LRU 3875（−20%，总 miss 是降的） |

四 cell（原始 hit_rate）：scan 36.09 / struct 82.68 / **chat 62.61** / agentic 83.12，
**加权 66.25%**（LRU 62.54 / heter 64.24，全胜 heter +2.01pp；距 OPT 5.29pp）。
floor 6144~9216 × weight 0.02~0.03 是平台，非单点运气。

复现：

```bash
# tail-stress（任意机器，sim 环境）
PYTHONPATH=$PWD:$PWD/python ~/venvs/sim/bin/python \
  Heter-Bench/mixexp/tools/build_tail_stress.py \
  --arms lru,chain_lru,opt --floor 8192 --weight 0.02 --gate multiturn
# 四 cell（实验机 /root/sg-unicache）
for c in scan struct chat agentic; do
  CELL=$c PAGE=8 CAP=198000 ARMS=chain_lru CHAIN_FLOOR=8192 CHAIN_WEIGHT=0.02 \
  CHAIN_GATE=multiturn OUT_DIR=/root/sim_runs/engine_chain_mt/$c \
  /root/sl_old/bin/python -u /root/sim_runs/sim_engine_cell_v2.py
done
```

## 二、heter_cache 档 B（heter 架构内的最优，保守改进开关，默认关）

```json
{
  "regret_gate": {
    "enabled": true,
    "k": 0.6, "r_min": 0.3, "window_s": 120,
    "priced_fallback": true,
    "floor_tokens": 8192, "weight": 0.02, "beta": 1.0
  }
}
```

机制：R_q（实测后悔密度）做弃权门打断 VD 引向锚队的正反馈；全弃权后由
`PricedFallback` 全树叶子定价兜底（β=1 把 R 归因加权乘进锚罚，β=2 过重、
会伤 cold）。

| 指标 | tail-stress | 对照 |
|---|---|---|
| hit | **78.4%** | heter 73.4（+5.0pp）/ LRU 77.63 |
| TTFT p50 / **p90** / p95 / p99 (ms) | 137 / **581** / 993 / 2070 | heter p90 1646（**−64%**）；LRU 920（−37%） |
| warm 整链丢失 | 49.1% | heter 76.9%（正反馈断掉，但仍高于 chain_lru） |

四 cell：scan 33.76 / struct 82.41 / **chat 58.71** / agentic 82.28，
加权 64.42%（heter 64.24）——**无劣化**，chat +0.64pp。
有效参数域：k∈{0.4~0.8} × r_min=0.3（p90 611~652 平台）；k=0.5/r_min=0.1 过敏会翻车。

复现：

```bash
# tail-stress
PYTHONPATH=$PWD:$PWD/python ~/venvs/sim/bin/python \
  Heter-Bench/mixexp/tools/build_tail_stress.py \
  --arms lru,heter_rgate --rg-k 0.6 --rg-rmin 0.3 --pf --pf-beta 1.0
# 四 cell
for c in scan struct chat agentic; do
  CELL=$c PAGE=8 CAP=198000 ARMS=heter_rgate RG_K=0.6 RG_RMIN=0.3 RG_PF=1 \
  OUT_DIR=/root/sim_runs/engine_rgate_pf/$c \
  /root/sl_old/bin/python -u /root/sim_runs/sim_engine_cell_v2.py
done
```

## 三、SLA 乘数（类别公平旋钮，2026-11-18 实验）

`sla: {queue: multiplier}` 乘在锚罚上（`penalty = weight × sla_q × 缺口`），
用于「低频大块损失类被均值定价低估」的公平修正。**有效域很窄**：

| 场景 | sla 配置 | 结果 |
|---|---|---|
| tail-stress（chat 占 18.8%，少数类） | chat=2.0, agentic=0.5 | **warm p90 1060→860（−19%）**，整体 p90 391/p99 1033，hot/cold/agentic 无损 ✔ |
| chat cell（chat 占 60%，主导类） | chat=2.0 | **chat hit 62.6→54.6（−8pp）❌** |
| chat cell | agentic=0.5 单独 | 62.29，无损 |
| agentic cell（主导类降权） | agentic=0.5 | 82.78 vs 83.12（−0.34）❌ |
| struct cell | 同上 | 82.86 vs 82.68（+0.18）✔ |

**边界条件：SLA 乘数只能给少数类用。** 给主导类加保护 = 主削减来源
「只进不出」，容量被挤占、削减集中到该类自己的 fresh/深段。死锚理论已证伪
（session_ttl 扫 3 档数字逐位不变）；但借此补上了 chain_lru 的「会话终结
释放锚」出口（ttl 判据与 heter 同口径，当前负载下未触发、无害）。

## 四、已证否 / 禁忌（换配置前先看）

* **深度代价加权**（cost-oracle gap ≤ 0）：不值做，勿复活；
* **gate="none" 只在人造负载（tail-stress）最优**，真实混合负载四 cell 全线跌；
* **三件套同开**（弃权门 + 队内 ChainEvict + 定价兜底）：p90 1239 vs 581，
  互相打架；队内 ChainEvict 单独用是无效组件（四 cell ±0.1pp）；
* **β>1**：chat 锚罚过重、削减挤到别的池（cold hit 0.851 受损）；
* **k=0.5 / r_min=0.1**：全弃权过敏，随机兜底乱削（p90 1218、p95 4677）。

## 五、边界与未验

* chain_lru 的 chunked prefill 场景、`weight` 量纲校准（0.02 s/token 比真实
  prefill 速率高两个量级，换硬件要重调）、drift3、真机 bench 端到端；
* 档 A/B 与 chain_lru 均默认关，默认行为逐位不变（84 个回归测试覆盖）；
* 本地无 exp3 数据，四 cell 数字来自实验机 bitahub `/root/sim_runs/`
  （`engine_chain_mt/`、`engine_rgate/`）。
