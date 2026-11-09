# Heter-Bench 文档索引

说明型和报告型文档集中在这里。**各目录的自述 README 不搬**——它们回答"这个目录怎么用"，
跟着代码走更有用：`W-TinyLRU/README.md`、`classify/README.md`、`exp/README.md`、
`RR_VS_TLRU/README.md`。

## 说明

| 文件 | 内容 | 原位置 |
|---|---|---|
| [REPRODUCE.md](REPRODUCE.md) | 服务器复现流程（AutoDL + Qwen2.5-7B-Instruct）：环境、数据、起服务、跑实验 | `Heter-Bench/REPRODUCE.md` |
| [benchmark.md](benchmark.md) | 完整 benchmark 指南：参数语义、矩阵跑法、指标口径、陷阱清单 | `Heter-Bench/benchmark.markdown` |

## 实验报告

| 文件 | 内容 | 原位置 |
|---|---|---|
| [reports/rr_vs_tlru.md](reports/rr_vs_tlru.md) | RR-EVICT vs T-LRU：burst/pause 场景（小流量挤掉长挂起会话后集中回归） | `RR_VS_TLRU/对比结果.md` |
| [reports/lru_vs_hetercache.md](reports/lru_vs_hetercache.md) | LRU vs HeterCache：搅拌压力下固定重访间隔、只扫池子周转速率 | `tailshot/LRU_VS_HETERCACHE.md` |
| [reports/wtinylru_result.md](reports/wtinylru_result.md) | 两带 SLRU 淘汰：设计思路 / 负载设计 / 对照组 / 实验结果 | `W-TinyLRU/result.md` |
| [reports/classify_min_share64.md](reports/classify_min_share64.md) | `min_share=64` 的会话分类三件套指标（最终判定 / 逐轮精度 / token 暴露） | `classify/SHARE64_SUMMARY.md` |
| [reports/classify_system_prompt_ablation.md](reports/classify_system_prompt_ablation.md) | 去掉公共 Qwen system prompt 后重跑，与抬高 `min_share` 的等价性验证 | `classify/SYSTEM_PROMPT_ABLATION.md` |
| [reports/design1.md](reports/design1.md) | Design1：计算感知的公平轮询淘汰（DRR + 视界加权） | `exp/Design1.md` |
| [reports/exp_steps.md](reports/exp_steps.md) | `d_i` 的实施步骤（L1 只记账 / L2 换选择键） | `exp/step.md` |

## 约定

- 新增**说明**（怎么跑、参数什么意思）→ 放 `docs/` 顶层；
- 新增**实验报告**（带结论和数据）→ 放 `docs/reports/`，文件名用 `主体_变量.md`；
- 目录自述 → 写在那个目录自己的 `README.md` 里，不要上提到这里。
