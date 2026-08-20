# 实验结果、来源与论文工件

本文档是当前仓库 **manuscript-facing 结果的主入口**。环境、数据预处理、构图、训练和冻结 checkpoint 验证见 [`FULL_PIPELINE_CN.md`](FULL_PIPELINE_CN.md)；最终 Figure 1--5 的生成命令与故障排查见 [`PLOTTING_CN.md`](PLOTTING_CN.md)。

## 1. 论文模型命名

| 论文名称 | 内部键 | SAS-Norm | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` / `Base` | × | × |
| DCRNN + SAS-Norm | `dssn_sasr` / `State` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` / `FA-DPR` | × | ✓ |
| STaR-GNN | `full` / `Full` | ✓ | ✓ |
| STGCN | 独立图时空基线 | × | × |

`State`、`Full`、`Base` 只作为源码/冻结工件兼容标签存在；论文正文统一使用 **SAS-Norm、FA-DPR、STaR-GNN、DCRNN**。

最终参数只根据 Validation 确定；参数冻结后才执行 common-46 Test。Test 不参与参数选择、early stopping 或模块取舍，所有最终预测均关闭 teacher forcing。

## 2. 最终指标口径

论文正文中的总体比较和消融统一采用与 Que et al. (2024) Supplementary Tables S1-1--S1-8 对齐的 publisher-compatible `total` 口径：

- `total MAE`：DMA A--J 十个 DMA 的 MAE 之和；
- `total MAPE/RMSE/NSE`：在 A--J 求和后的小时总需求序列上计算。

因此 STaR-GNN 的 manuscript-facing total MAE 为：

- 24 h：`9.424199`
- 168 h：`12.233590`

仓库仍保留 aggregate-demand MAE `4.360841/4.919812`，但只用于内部运行诊断、旧逐日分析和可复现审计，**不用于正文跨模型 MAE 比较**。

详细定义与自动验收见 [`../paper/tables/literature/METRIC_CONVENTIONS.md`](../paper/tables/literature/METRIC_CONVENTIONS.md)。

## 3. Table 1：总体模型比较

GRU、LSTM、MSNet 与 MSCMNet variants 来自 Que et al. (2024) 的 reported results；DCRNN、STGCN 与 STaR-GNN 为当前 common-46 Test 的 publisher-compatible 复评结果。

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

自动生成文件：

- `paper/tables/literature/table_literature_comparison_common46.csv`
- `paper/tables/literature/table_literature_comparison_common46.md`

相对 MAE 改善范围：

- 24 h：相较 GRU/LSTM/MSNet/MSCMNet variants 约降低 34.9%--46.7%；相较 DCRNN/STGCN 约降低 20.9%--23.7%；
- 168 h：相较 GRU/LSTM/MSNet/MSCMNet variants 约降低 18.2%--34.5%；相较 DCRNN/STGCN 约降低 16.0%--27.2%。

注意：reported literature baselines 与 common-46 复评基线必须在正文和 Figure 1 中明确区分来源。

## 4. Table 2：publisher-compatible 消融

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

publisher-compatible 消融验收为 **30/32**。两个真实例外：

1. FA-DPR 168 h MAPE（3.277716%）略高于 DCRNN（3.248413%）；
2. SAS-Norm-only 168 h sum-of-DMA MAE 为 `12.207835`，略低于 Full 的 `12.233590`（约 0.21%）。

因此正文的安全表达是：

> 完整模型在 24 h 四项指标上均最优；在 168 h 的 MAPE、RMSE 和 NSE 上最优，而 SAS-Norm-only 在 sum-of-DMA MAE 上略低于完整模型。

自动生成文件：

- `paper/tables/literature/table_ablation_common46.*`
- `paper/tables/literature/table_ablation_audit.json`

## 5. Table 3：DMA-level 结果

逐 DMA 结果直接报告 A--J 各自的 MAE、MAPE、RMSE 和 NSE，不进行跨 DMA 聚合。最终表：

- `paper/tables/literature/table_star_gnn_dma_common46.csv`
- `paper/tables/literature/table_star_gnn_dma_common46.md`

DMA 间需求规模不同，因此不能只依据绝对 MAE 判断某个 DMA “更容易/更困难”；应联合 MAPE、RMSE、NSE 和需求规模解释空间异质性。

Figure 4 进一步比较 STaR-GNN 与 DCRNN/STGCN 的 DMA-level MAE：10 DMA × 2 horizons × 2 baselines 共 40 个比较全部为正改善，范围约 1.26%--61.20%。这支持“跨 DMA 一致正改善”，但不支持“所有 DMA 均同等幅度大幅改善”。

## 6. 最终 Figure 1--5

正文图已经从旧的绝对柱状图体系收敛为五张回答不同科学问题的主图：

### Figure 1 — Relative performance improvement

文件：`paper/figures/manuscript_fig1_relative_improvement.*`

- Panel (a)：MAE/MAPE/RMSE 相对降低率；
- Panel (b)：绝对 NSE gain；
- Que et al. (2024) 的 reported models 用 `†` 标记；DCRNN/STGCN 为 common-46 复评。

### Figure 2 — Day 1--Day 7 long-horizon behavior

文件：`paper/figures/manuscript_fig2_day1_day7_publisher_mae.*`

- Panel (a)：DCRNN/STGCN/STaR-GNN 的 Day 1--Day 7 publisher-compatible MAE 与 bootstrap 95% CI；
- Panel (b)：DCRNN、SAS-Norm-only、FA-DPR-only、STaR-GNN 相对 Day 1 的 MAE 变化率。

Day 7 相对 Day 1：

- DCRNN：`+38.25%`
- FA-DPR：`+11.93%`
- SAS-Norm：`+2.64%`
- STaR-GNN：`+1.70%`

正确解释：SAS-Norm 是低长时域 MAE 漂移的主要贡献模块；FA-DPR 单独也降低了 DCRNN 的退化速度；Full 保持稳定 MAE，并在 168 h MAPE/RMSE/NSE 上取得最佳综合结果。

### Figure 3 — ECDF across 46 test origins

文件：`paper/figures/manuscript_fig3_origin_ecdf.*`

正文只比较 DCRNN、STGCN、STaR-GNN，避免把样本稳健性与消融问题混在一起。

STaR-GNN paired win rates：

- 24 h vs DCRNN：45/46（97.8%）
- 24 h vs STGCN：45/46（97.8%）
- 168 h vs DCRNN：46/46（100.0%）
- 168 h vs STGCN：40/46（87.0%）

### Figure 4 — DMA-level spatial consistency

文件：`paper/figures/manuscript_fig4_dma_mae_improvement.*`

展示 DMA A--J 在 24 h/168 h 下相对 DCRNN/STGCN 的 MAE reduction，回答总体改善是否由少量区域驱动。

### Figure 5 — Representative 168 h trajectory

文件：`paper/figures/manuscript_fig5_representative_168h_trajectory.*`

案例预先按规则选择：从 46 个 common origins 中选 STaR-GNN publisher-compatible 168 h MAE 最接近中位数的 origin，避免按视觉效果 cherry-picking。

当前选中 origin：

- STGCN MAE：14.653
- DCRNN MAE：15.517
- STaR-GNN MAE：12.182

注意：Figure 5 下 panel 是 aggregate-demand trajectory 的逐小时 absolute error，与用于样本选择的 publisher-compatible sum-of-DMA MAE 不是同一统计量。

最终图设计边界见 [`MANUSCRIPT_FIGURES_FINAL_CN.md`](MANUSCRIPT_FIGURES_FINAL_CN.md)，caption 见 [`../paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md`](../paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md)。

## 7. 正文与 Supplementary 分工

### 正文

- Table 1：9 模型总体比较；
- Table 2：publisher-compatible 消融；
- Table 3：STaR-GNN DMA-level 性能；
- Figure 1--5：相对优势、长时域行为、origin 稳健性、DMA 空间一致性、代表性周轨迹。

### Supplementary / 内部诊断

以下旧图仍保留，但不再承担正文主要结论：

- `test_overall_24h.*` / `test_overall_168h.*`
- `test_ablation_24h.*` / `test_ablation_168h.*`
- `test_star_gnn_dma_metrics.*`
- `test_day1_day7_models.*` / `test_day1_day7_ablation.*`
- `test_dma_mae_24h.*` / `test_dma_mae_168h.*`
- `pearson_correlation_heatmap.*`（可按论文结构放方法/数据分析或 Supplementary）

其中旧 `test_day1_day7_*` 基于 aggregate-demand 诊断口径，不能替代最终 Figure 2 的 publisher-compatible Day 1--Day 7 分析。

## 8. 一键重建正文表格

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout
```

成功时应看到：

```text
Metric convention audit: PASS
Publisher-compatible ablation audit: 30/32 PASS
```

如需旧的 aggregate-demand/DMA/Pearson 诊断工件，可执行：

```bash
python scripts/reproduce/build_detailed_test_artifacts.py
```

该脚本生成的 `paper/reports/TEST_RESULTS_CN.md` 属于**内部诊断层**，不是最终 manuscript-facing 结果入口。

## 9. 两阶段生成最终正文图

第一阶段：

```bash
python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820
```

成功标志：

```text
Manuscript scientific-figure audit: PASS
```

第二阶段：

```bash
python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures
```

成功标志：

```text
Refined manuscript Figure 2 and Figure 3: PASS
```

第二阶段会覆盖第一阶段的最终 Figure 2/3，并生成：

- `fig2_day1_day7_degradation.csv`
- `fig3_origin_win_rates.csv`
- `manuscript_empirical_figure_audit.json`

完整作图教程见 [`PLOTTING_CN.md`](PLOTTING_CN.md)。

## 10. 审计原则

- 不手工修改自动生成的数值来形成预期排序；
- reported literature results 与当前 common-46 复评结果明确区分来源；
- publisher-compatible MAE 与 aggregate-demand MAE 始终分开；
- 168 h SAS-Norm-only MAE 略低于 Full 的例外必须透明保留；
- Figure 5 的代表样本由固定中位误差规则选择，不依据图形外观；
- 所有正文主图同时保留 CSV/JSON 审计工件，以便论文写作和审稿复核。