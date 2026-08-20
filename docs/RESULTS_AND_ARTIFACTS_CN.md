# 实验结果、来源与论文工件

本文档集中说明实验口径、冻结 Test 结果、结果来源以及论文表图位置。环境安装、数据预处理、训练、Test 和 checkpoint 复评命令统一见 [`FULL_PIPELINE_CN.md`](FULL_PIPELINE_CN.md)。

## 1. 论文模型命名

| 论文名称 | 内部键 | State | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` / `Base` | × | × |
| DCRNN + State | `dssn_sasr` / `State` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` / `FA-DPR` | × | ✓ |
| STaR-GNN | `full` / `Full` | ✓ | ✓ |
| STGCN | 独立基线 | × | × |

## 2. 选择协议与公平性

最终设置只根据 Validation 确定；参数固定后才执行 common-46 Test。Test 不参与参数选择、early stopping 或组件取舍。四个消融单元共享数据、Pearson 图、DCRNN hidden size、decoder、优化器和训练协议。

重新执行冻结 checkpoint 时，`verify_pretrained.sh --re-evaluate` 会审计 checkpoint、图、common-46 索引、关闭 teacher forcing 的 Test 推理以及冻结指标差异。

## 3. 两套 MAE 口径必须分开

本仓库同时保留两种合法但用途不同的 MAE 定义，**不得在同一比较列中混用**。

### 3.1 统一实验 / aggregate-demand 口径

用于 STGCN、DCRNN、STaR-GNN 的统一实验、消融、168 h 逐日分析和运行解释。先在每个样本-小时将 DMA A--J 的真实值和预测值分别求和，再对总需求序列统一计算 MAE、MAPE、RMSE 和 NSE。

STaR-GNN 的 common-46 aggregate-demand MAE 为：

- 24 h：`4.360841`
- 168 h：`4.919812`

对应文件：`paper/tables/literature/table_internal_common46.*`，以及 `paper/tables/test_all_models_common46.*`。

### 3.2 MSCMNet publisher-compatible 口径

用于与 Que et al. (2024) 补充材料 S1-1--S1-8 中已发表的 GRU、LSTM、MSNet、MSCMNet 结果直接比较。该补充材料的 `total` 行采用混合口径：

- `total MAE`：A--J 十个 DMA 的 MAE 之和；
- `total MAPE/RMSE/NSE`：在 A--J 求和后的小时总需求序列上计算。

STaR-GNN 在这一 publisher-compatible 口径下的 total MAE 为：

- 24 h：`9.424199`
- 168 h：`12.233590`

`build_paper_tables.py` 会自动验证两套口径：MAPE/RMSE/NSE 在内部表与文献表中必须一致，而 MAE 必须保持不同定义；同时使用 `mscmnet_paper_metrics.yaml` 中的 DMA A--J 数据验证 MSCMNet_W/WM 的 published total MAE 等于 A--J DMA MAE 之和（允许原表三位小数舍入误差）。

## 4. common-46 冻结 Test：统一实验口径

| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | STGCN | 5.850690 | 2.424526 | 7.904592 | 0.960590 |
| 24 h | DCRNN | 5.356315 | 2.212928 | 6.848257 | 0.970419 |
| 24 h | **STaR-GNN** | **4.360841** | **1.804574** | **5.534656** | **0.980679** |
| 168 h | STGCN | 8.574033 | 3.575848 | 10.305691 | 0.933337 |
| 168 h | DCRNN | 7.734838 | 3.248413 | 9.817428 | 0.939504 |
| 168 h | **STaR-GNN** | **4.919812** | **2.013774** | **6.160881** | **0.976176** |

完整消融结果见 `paper/tables/test_ablation_common46.*`。FA-DPR 相对 DCRNN 的 168 h MAPE 是注册比较中的唯一例外，完整方向关系为 31/32，不删除该结果。

## 5. 与 MSCMNet 原文同口径的完整文献对比

下表用于正文中的跨模型总体比较。GRU、LSTM、MSNet 和 MSCMNet 系列为 Que et al. (2024) Supplementary Tables S1-1--S1-8 的 reported 结果；DCRNN、STGCN 和 STaR-GNN 在相同 common-46 样本上按 publisher-compatible 口径重新计算。

| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | GRU | 16.314 | 3.100 | 10.194 | 0.916 |
| 24 h | LSTM | 17.698 | 2.900 | 9.711 | 0.920 |
| 24 h | MSNet | 15.537 | 3.200 | 9.526 | 0.929 |
| 24 h | MSCMNet_WM | 14.790 | 2.700 | 7.924 | 0.957 |
| 24 h | MSCMNet_M | 14.912 | 2.800 | 8.111 | 0.954 |
| 24 h | MSCMNet_W | 14.471 | 2.600 | 7.586 | 0.959 |
| 24 h | DCRNN | 11.917 | 2.213 | 6.848 | 0.970 |
| 24 h | STGCN | 12.358 | 2.425 | 7.905 | 0.961 |
| 24 h | **STaR-GNN** | **9.424** | **1.805** | **5.535** | **0.981** |
| 168 h | GRU | 18.305 | 3.100 | 11.353 | 0.918 |
| 168 h | LSTM | 18.678 | 2.900 | 11.031 | 0.922 |
| 168 h | MSNet | 15.908 | 3.200 | 9.698 | 0.930 |
| 168 h | MSCMNet_WM | 15.290 | 2.700 | 8.097 | 0.957 |
| 168 h | MSCMNet_M | 15.405 | 2.800 | 8.395 | 0.953 |
| 168 h | MSCMNet_W | 14.950 | 2.600 | 7.756 | 0.960 |
| 168 h | DCRNN | 16.801 | 3.248 | 9.817 | 0.940 |
| 168 h | STGCN | 14.569 | 3.576 | 10.306 | 0.933 |
| 168 h | **STaR-GNN** | **12.234** | **2.014** | **6.161** | **0.976** |

自动生成文件：`paper/tables/literature/table_literature_comparison_common46.*`。为兼容旧路径，`table_comparison_common46.*` 保留为同一张文献对比表的别名。

**注意：**不得用 aggregate-demand MAE `4.360841/4.919812` 与 MSCMNet 补充材料中的 `14.471/14.950` 直接计算相对提升。

## 6. 指标与工件从哪里产生

冻结最佳 checkpoint 重新推理：

```text
scripts/reproduce/reproduce.py
  -> scripts/innovation/evaluate_star_dcrnn.py
  -> run_star_checkpoint_evaluation()
  -> prepare_dcrnn_test_data()
  -> protocol_indices["common_46"]
```

一次评估同时生成：

- `metrics_aggregate_total_common_46.csv`：统一 aggregate-demand 四指标；
- `metrics_common_46.csv`：A--J DMA 指标及 publisher-compatible `total` 行；
- `predictions.npz`：冻结预测与 common-46 索引；
- `test_summary.json`：checkpoint/图哈希、样本数、teacher forcing 与 Test 隔离信息。

## 7. 论文表格与图件

主要表格：

| 文件 | 用途 |
|---|---|
| `paper/tables/literature/table_internal_common46.*` | STGCN/DCRNN/STaR-GNN 统一 aggregate-demand 对比 |
| `paper/tables/literature/table_literature_comparison_common46.*` | GRU/LSTM/MSNet/MSCMNet + DCRNN/STGCN/STaR-GNN publisher-compatible 文献对比 |
| `paper/tables/literature/table_ablation_common46.*` | DCRNN/State/FA-DPR/STaR-GNN 消融 |
| `paper/tables/test_dma_metrics_long.csv` | DMA A--J 四指标 |
| `paper/tables/test_day1_day7_metrics.csv` | 168 h 连续七日分析 |
| `paper/tables/pearson_*.csv` | Pearson 功能图工件 |

图件口径固定如下：

- `paper/figures/test_overall_24h.*`、`test_overall_168h.*`：**publisher-compatible 9 模型总体文献对比**，与本页第 5 节完全一致；
- `test_ablation_24h.*`、`test_ablation_168h.*`：aggregate-demand 消融；
- `test_day1_day7_models.*`、`test_day1_day7_ablation.*`：aggregate-demand 逐日分析；
- `test_dma_mae_24h.*`、`test_dma_mae_168h.*`：各 DMA 自身 MAE；
- `pearson_correlation_heatmap.*`：训练期 Pearson 功能关联图。

`build_detailed_test_artifacts.py` 先生成内部统一实验图；随后 `build_literature_figures.py` 只覆盖 `test_overall_24h/168h`，因此不会污染消融、逐日和 DMA 图的评价口径。

## 8. 一键生成与审计

冻结 checkpoint 和预测已安装时：

```bash
bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

只重新生成论文表和全部图件：

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

python scripts/reproduce/build_detailed_test_artifacts.py

python scripts/reproduce/build_literature_figures.py \
  --table paper/tables/literature/table_literature_comparison_common46.csv \
  --output paper/figures
```

成功时应依次看到：

```text
Metric convention audit: PASS
Publisher-compatible figure audit: PASS
```

正文和补充材料应直接使用自动生成的 CSV/Markdown/PNG/PDF，不得手工改表形成预期排序。
