# 实验结果、来源与论文工件

本文档说明论文最终采用的评价口径、模型比较、消融结果、DMA 明细及自动生成工件。环境安装、数据预处理、训练、Test 和 checkpoint 复评命令见 [`FULL_PIPELINE_CN.md`](FULL_PIPELINE_CN.md)。

## 1. 论文模型命名

| 论文名称 | 内部键 | SAS-Norm | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` / `Base` | × | × |
| DCRNN + SAS-Norm | `dssn_sasr` / `State` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` / `FA-DPR` | × | ✓ |
| STaR-GNN | `full` / `Full` | ✓ | ✓ |
| STGCN | 独立图时空基线 | × | × |

最终参数只根据 Validation 确定；参数固定后才执行 common-46 Test。Test 不参与参数选择、early stopping 或组件取舍。所有最终预测均关闭 teacher forcing。

## 2. 论文最终指标口径

论文正文中的**总体模型比较和消融实验统一采用 MSCMNet publisher-compatible `total` 口径**，与 Que et al. (2024) Supplementary Tables S1-1--S1-8 保持一致：

- `total MAE`：A--J 十个 DMA 的 MAE 之和；
- `total MAPE/RMSE/NSE`：在 A--J 求和后的小时总需求序列上计算。

因此，STaR-GNN 正文主结果的 total MAE 为：

- 24 h：`9.424199`
- 168 h：`12.233590`

仓库仍保留 aggregate-demand MAE `4.360841/4.919812`，仅用于运行解释、逐日诊断和可复现审计，不用于与 GRU/LSTM/MSNet/MSCMNet 的正文横向 MAE 比较。

`build_paper_tables.py` 会自动检查：

1. MSCMNet_W/WM 的 published total MAE 是否等于 A--J DMA MAE 之和（允许三位小数舍入误差）；
2. DCRNN/STGCN/STaR-GNN 在两套口径下 MAPE、RMSE、NSE 是否一致；
3. publisher-compatible MAE 是否保持与 aggregate-demand MAE 分离；
4. publisher-compatible 消融关系是否严格为 `30/32`。

## 3. 总体模型比较：9 个模型

下表是论文主模型对比表。GRU、LSTM、MSNet 和 MSCMNet 系列来自 Que et al. (2024) reported 结果；DCRNN、STGCN 和 STaR-GNN 在相同 common-46 Test 上按相同 publisher-compatible 口径计算。

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

自动生成文件：`paper/tables/literature/table_literature_comparison_common46.*`。

## 4. 消融实验：同一 publisher-compatible 口径

消融表只保留图相关基线和 STaR-GNN 两个核心模块，所有数值与第 3 节使用同一口径。

| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | STGCN | 12.358 | 2.425 | 7.905 | 0.961 |
| 24 h | DCRNN | 11.917 | 2.213 | 6.848 | 0.970 |
| 24 h | DCRNN + SAS-Norm | 10.468 | 2.010 | 6.134 | 0.976 |
| 24 h | DCRNN + FA-DPR | 11.238 | 1.945 | 6.079 | 0.977 |
| 24 h | **STaR-GNN** | **9.424** | **1.805** | **5.535** | **0.981** |
| 168 h | STGCN | 14.569 | 3.576 | 10.306 | 0.933 |
| 168 h | DCRNN | 16.801 | 3.248 | 9.817 | 0.940 |
| 168 h | **DCRNN + SAS-Norm** | **12.208** | 2.102 | 6.468 | 0.974 |
| 168 h | DCRNN + FA-DPR | 14.086 | 3.278 | 9.332 | 0.945 |
| 168 h | **STaR-GNN** | 12.234 | **2.014** | **6.161** | **0.976** |

publisher-compatible 消融自动验收为 **30/32**。两个真实例外必须透明保留：

1. FA-DPR 的 168 h MAPE（3.277716%）略高于 DCRNN（3.248413%）；
2. 168 h 下 SAS-Norm-only 的 sum-of-DMA MAE 为 `12.207835`，略低于完整 STaR-GNN 的 `12.233590`，差异约 0.21%。

因此正文不应写“完整模型在两个预测范围四项指标上均最优”，而应说明完整模型在 24 h 四项指标均最优，在 168 h 的 MAPE/RMSE/NSE 最优，而 SAS-Norm-only 在 sum-of-DMA MAE 上略低。

自动生成文件：`paper/tables/literature/table_ablation_common46.*`；自动验收：`table_ablation_audit.json`。

## 5. STaR-GNN 的 DMA-level 结果

逐 DMA 表不再做跨 DMA 聚合，而是直接报告各 DMA 自身 MAE、MAPE、RMSE 和 NSE。

| Horizon | DMA | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | A | 0.806 | 12.413 | 1.229 | 0.643 |
| 24 h | B | 0.264 | 2.775 | 0.367 | 0.850 |
| 24 h | C | 0.161 | 5.398 | 0.209 | 0.935 |
| 24 h | D | 2.017 | 6.278 | 2.513 | 0.853 |
| 24 h | E | 1.258 | 1.557 | 1.593 | 0.988 |
| 24 h | F | 0.626 | 5.889 | 0.787 | 0.670 |
| 24 h | G | 0.810 | 3.103 | 1.028 | 0.940 |
| 24 h | H | 1.255 | 5.379 | 1.780 | 0.929 |
| 24 h | I | 1.045 | 4.285 | 1.316 | 0.766 |
| 24 h | J | 1.183 | 5.205 | 1.483 | 0.844 |
| 168 h | A | 0.859 | 12.445 | 1.248 | 0.625 |
| 168 h | B | 0.337 | 3.587 | 0.439 | 0.785 |
| 168 h | C | 0.205 | 6.647 | 0.269 | 0.892 |
| 168 h | D | 2.302 | 6.970 | 2.878 | 0.806 |
| 168 h | E | 2.091 | 2.536 | 2.600 | 0.967 |
| 168 h | F | 0.695 | 6.453 | 0.864 | 0.588 |
| 168 h | G | 1.155 | 4.340 | 1.469 | 0.878 |
| 168 h | H | 1.534 | 6.314 | 1.971 | 0.914 |
| 168 h | I | 1.454 | 5.873 | 1.813 | 0.558 |
| 168 h | J | 1.601 | 6.874 | 2.018 | 0.714 |

自动生成文件：`paper/tables/literature/table_star_gnn_dma_common46.*`。

需要注意，DMA 间需求规模不同，因此不能只依据绝对 MAE 判断哪个 DMA “更容易预测”；MAPE、RMSE 和 NSE 应联合解释区域异质性。

## 6. 图件与正文用途

论文主图固定为：

- `paper/figures/test_overall_24h.*`、`test_overall_168h.*`：9 模型 publisher-compatible 总体比较；
- `paper/figures/test_ablation_24h.*`、`test_ablation_168h.*`：STGCN/DCRNN/SAS-Norm/FA-DPR/STaR-GNN publisher-compatible 消融；
- `paper/figures/test_star_gnn_dma_metrics.*`：STaR-GNN 在 DMA A--J 上的 24 h/168 h MAE、MAPE、RMSE、NSE 四面板图。

以下保留为补充分析/可复现工件：

- `test_day1_day7_models.*`、`test_day1_day7_ablation.*`：168 h 逐日 aggregate-demand 分析；
- `test_dma_mae_24h.*`、`test_dma_mae_168h.*`：历史多模型 DMA-level MAE 诊断；
- `pearson_correlation_heatmap.*`：训练期 Pearson 功能关联图。

## 7. 指标与工件来源

冻结最佳 checkpoint 重新推理会同时生成：

- `metrics_aggregate_total_common_46.csv`：aggregate-demand 诊断指标；
- `metrics_common_46.csv`：A--J DMA 指标及 publisher-compatible `total` 行；
- `predictions.npz`：冻结预测与 common-46 索引；
- `test_summary.json`：checkpoint/图哈希、样本数、teacher forcing 和 Test 隔离信息。

## 8. 一键重新生成与验证

```bash
bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

如果不重新推理，仅重建论文表和图：

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

python scripts/reproduce/build_detailed_test_artifacts.py

python scripts/reproduce/build_literature_figures.py \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --ablation-table paper/tables/literature/table_ablation_common46.csv \
  --dma-table paper/tables/literature/table_star_gnn_dma_common46.csv \
  --output paper/figures
```

成功时应至少看到：

```text
Metric convention audit: PASS
Publisher-compatible ablation audit: 30/32 PASS
Publisher-compatible figure audit: PASS
```

正文与补充材料应直接使用自动生成的表和图，不手工改数值形成预期排序。
