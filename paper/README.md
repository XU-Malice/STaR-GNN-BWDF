# 论文表格、图件与审计工件

`paper/` 保存 Journal of Hydrology 稿件使用的正文表格、正文 Figure 1--5、Supplementary/内部诊断图、Figure captions 以及可追溯 CSV/JSON。

> **正文数值请以 `tables/literature/` 的 publisher-compatible 表和 `tables/manuscript/` 的图件审计数据为准。** 旧 `test_*` 表图包含 aggregate-demand 诊断口径，不应与正文 total MAE 混用。

## 1. 正文 Table 1--3

### Table 1 — Overall comparison

```text
tables/literature/table_literature_comparison_common46.csv
tables/literature/table_literature_comparison_common46.md
```

包含 GRU、LSTM、MSNet、MSCMNet variants、DCRNN、STGCN、STaR-GNN 的 24 h/168 h MAE、MAPE、RMSE、NSE。

### Table 2 — Ablation and graph-model comparison

```text
tables/literature/table_ablation_common46.csv
tables/literature/table_ablation_common46.md
tables/literature/table_ablation_audit.json
```

模型：STGCN、DCRNN、DCRNN + SAS-Norm、DCRNN + FA-DPR、STaR-GNN。publisher-compatible 消融为 30/32。

### Table 3 — DMA-level STaR-GNN performance

```text
tables/literature/table_star_gnn_dma_common46.csv
tables/literature/table_star_gnn_dma_common46.md
```

报告 DMA A--J 在两个 horizon 下的 MAE、MAPE、RMSE、NSE。

指标口径见 [`tables/literature/METRIC_CONVENTIONS.md`](tables/literature/METRIC_CONVENTIONS.md)。

## 2. 正文 Figure 1--5

每张图均同时保留 PNG 预览和 PDF 矢量版本：

```text
figures/manuscript_fig1_relative_improvement.{png,pdf}
figures/manuscript_fig2_day1_day7_publisher_mae.{png,pdf}
figures/manuscript_fig3_origin_ecdf.{png,pdf}
figures/manuscript_fig4_dma_mae_improvement.{png,pdf}
figures/manuscript_fig5_representative_168h_trajectory.{png,pdf}
```

科学问题分别为：

1. 跨模型、跨指标相对优势；
2. 168 h Day 1--Day 7 长时域误差行为及组件贡献；
3. 46 个 common origins 上的样本稳健性；
4. DMA A--J 上改善的空间一致性；
5. 预先固定中位误差规则选择的代表性 168 h 预测轨迹。

最终 Figure captions：[`captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md`](captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md)。

## 3. Figure 1--5 审计数据

```text
tables/manuscript/
```

其中保存：

- Figure 1 的相对 MAE/MAPE/RMSE reduction 与 NSE gain；
- Figure 2 的 Day 1--Day 7 publisher-compatible MAE、bootstrap 95% CI 与相对 Day 1 退化率；
- Figure 3 的逐 origin publisher-compatible MAE 与 paired win rates；
- Figure 4 的 DMA-level MAE reduction；
- Figure 5 的代表样本选择规则、origin 索引、模型误差和 168 h trajectory；
- `manuscript_empirical_figure_audit.json`：最终实证解释 guardrails。

具体字段见 [`tables/manuscript/README.md`](tables/manuscript/README.md)。

## 4. 两阶段生成最终正文图

先确保 `tables/literature/table_literature_comparison_common46.csv` 与冻结预测已存在。

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

第二阶段：

```bash
python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures
```

第二阶段会覆盖第一阶段生成的 Figure 2 和 Figure 3，使其成为当前 manuscript 最终布局。

完整教程见 [`../docs/PLOTTING_CN.md`](../docs/PLOTTING_CN.md)。

## 5. Supplementary / 内部诊断图

以下文件保留用于补充材料、运行审计或历史诊断，不再作为正文主要证据：

```text
figures/test_overall_24h.*
figures/test_overall_168h.*
figures/test_ablation_24h.*
figures/test_ablation_168h.*
figures/test_star_gnn_dma_metrics.*
figures/test_day1_day7_models.*
figures/test_day1_day7_ablation.*
figures/test_dma_mae_24h.*
figures/test_dma_mae_168h.*
figures/pearson_correlation_heatmap.*
```

其中旧 `test_day1_day7_*` 使用 aggregate-demand 诊断定义；最终 Figure 2 使用 publisher-compatible sum-of-DMA day-wise MAE，二者不能混用。

## 6. 旧内部报告

`reports/TEST_RESULTS_CN.md` 来源于 `build_detailed_test_artifacts.py`，主要服务 aggregate-demand、旧逐日与 Pearson 诊断。它不是当前正文总结果的权威入口。

当前 manuscript-facing 结果请看：

- [`../docs/RESULTS_AND_ARTIFACTS_CN.md`](../docs/RESULTS_AND_ARTIFACTS_CN.md)
- [`../docs/MANUSCRIPT_FIGURES_FINAL_CN.md`](../docs/MANUSCRIPT_FIGURES_FINAL_CN.md)
- [`tables/literature/METRIC_CONVENTIONS.md`](tables/literature/METRIC_CONVENTIONS.md)

所有自动生成材料只用于 Test 后透明报告与审计，不用于 Test 后选参或修改模型。