# 论文表格、图件与审计工件

`paper/` 保存 Journal of Hydrology 稿件使用的正文表格、Figure 1--5、caption 以及可追溯 CSV/JSON。

> **正文 total 结果以 `tables/literature/` 的 publisher-compatible 表和 `tables/manuscript/` 的图件审计为准。** 旧 `test_*` 表图包含 aggregate-demand 诊断口径，不与正文 total MAE 混用。

## Table 1 — Overall comparison

```text
tables/literature/table_literature_comparison_common46.csv
tables/literature/table_literature_comparison_common46.md
```

包含 GRU、LSTM、MSNet、MSCMNet variants、DCRNN、STGCN、STaR-GNN。

## Table 2 — Factorial ablation

```text
tables/literature/table_ablation_common46.csv
tables/literature/table_ablation_common46.md
tables/literature/table_ablation_audit.json
```

**只包含：** DCRNN、DCRNN + SAS-Norm、DCRNN + FA-DPR、STaR-GNN。STGCN 是独立 graph baseline，不属于消融。

168 h publisher-compatible MAE：SAS-Norm-only `12.208`，STaR-GNN `12.234`；差约 `0.21%`。最终 manuscript audit 使用 7-origin moving-block bootstrap 限制对这一细小点估计差异的解释。

## Table 3 — DMA-level STaR-GNN performance

```text
tables/literature/table_star_gnn_dma_common46.csv
tables/literature/table_star_gnn_dma_common46.md
```

报告 DMA A--J 在 24 h/168 h 下的 MAE、MAPE、RMSE、NSE。

指标口径见 [`tables/literature/METRIC_CONVENTIONS.md`](tables/literature/METRIC_CONVENTIONS.md)。

## Figure 1--5

```text
figures/manuscript_fig1_relative_improvement.{png,pdf}
figures/manuscript_fig2_day1_day7_publisher_mae.{png,pdf}
figures/manuscript_fig3_origin_ecdf.{png,pdf}
figures/manuscript_fig4_dma_mae_improvement.{png,pdf}
figures/manuscript_fig5_representative_168h_trajectory.{png,pdf}
```

- Figure 1：overall relative improvement；
- Figure 2：**four-model factorial ablation only, no STGCN**；
- Figure 3：DCRNN/STGCN/STaR-GNN origin-level robustness；
- Figure 4：DMA-level improvement versus DCRNN/STGCN；
- Figure 5：pre-specified median-error representative 168 h trajectory。

图注：[`captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md`](captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md)。

## Figure audit data

最终 Figure 2/3 的关键审计文件：

```text
tables/manuscript/fig2_ablation_daywise_reduction_vs_dcrnn.csv
tables/manuscript/fig2_day1_day7_degradation.csv
tables/manuscript/fig2_full_vs_sas_block_bootstrap.json
tables/manuscript/fig3_origin_win_rates.csv
tables/manuscript/manuscript_empirical_figure_audit.json
```

Figure 4/5：

```text
tables/manuscript/fig4_dma_mae_improvement.csv
tables/manuscript/fig5_representative_168h_selection.json
```

## Final regeneration

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820

python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures \
  --block-bootstrap-iterations 50000 \
  --block-bootstrap-length 7 \
  --block-bootstrap-seed 20260820
```

正文表格统一显示 3 位小数；CSV/JSON 保留完整精度。最终 PNG 为 300 dpi，PDF 为矢量格式。
