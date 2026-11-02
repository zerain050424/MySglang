# 脚本说明

| 文件 | 作用 | 原始调用 |
|---|---|---|
| `build_burst_pool.py` | 从 APIGenMT 切出 idle/active 两个池并生成 tasks yaml | `--seed <s> --n-active <3|4|8> --n-idle 8 --pause-s 60 --name <tag>` |
| `run_matrix.sh` | 全量矩阵（κ 扫描 / n_a 扫描 / 73k 阴性对照 / n_a=4 / 3 个池种子） | `bash RR/scripts/run_matrix.sh` |
| `run_extra.sh` | 补跑因命名冲突丢失的两格（thrash_n8、κ=0.3） | `bash RR/scripts/run_extra.sh` |
| `summarize_burst.py` | 汇总 `result/burst/*.csv` 的 burst 窗口指标 | `/root/sl_old/bin/python RR/scripts/summarize_burst.py` |

依赖 `tailshot/run_large.sh`（起 server → 跑 bench → 停 server），
它支持 `POLICY_CFG`（策略参数 JSON）与 `MEM_FRAC`（`--mem-fraction-static`）。
