# Journal of Hydrology 投稿版表格、图件与审计工件

`paper/` 同时保留两类资产：

1. **submission-facing artifacts**：真正用于论文正文/Supplementary 的最终表图；
2. **legacy / diagnostic artifacts**：旧 `manuscript_fig1...5`、`test_*` 等历史复现和内部诊断图。

> **投稿版权威入口：** `tables/submission/`、`figures/submission/`、`figures/supplementary/` 和 `tables/manuscript/submission/`。

实验设计见 [`../docs/EXPERIMENT_DESIGN_FINAL_CN.md`](../docs/EXPERIMENT_DESIGN_FINAL_CN.md)，作图流程见 [`../docs/PLOTTING_CN.md`](../docs/PLOTTING_CN.md)。

---

## Main Table 1 — Overall forecasting performance

显示版：

```text
tables/submission/table1_overall_performance.md
```

全精度来源：

```text
tables/literature/table_literature_comparison_common46.csv
```

包含 GRU†、LSTM†、MSNet†、MSCMNet variants†、DCRNN、STGCN、STaR-GNN。

† 为 Que et al. (2024) reported results；DCRNN、STGCN、STaR-GNN 为 common-46 协议下重新评价。

---

## Main Table 2 — Factorial ablation

显示版：

```text
tables/submission/table2_factorial_ablation.md
```

全精度来源：

```text
tables/literature/table_ablation_common46.csv
tables/literature/table_ablation_audit.json
```

**严格只有四个模型：**

```text
DCRNN
DCRNN + SAS-Norm
DCRNN + FA-DPR
STaR-GNN
```

STGCN 是独立 graph baseline，不属于消融。

168 h publisher-compatible MAE：SAS-Norm-only `12.208`，STaR-GNN `12.234`；差约 `0.21%`。该点估计差异由 Main Fig. 1 的 ordered moving-block evidence 限定解释；完整模型在 168 h 的 MAPE、RMSE、NSE 以及 aggregate-demand MAE 上更优。

---

## Supplementary Table S1 — DMA-level metrics

显示版：

```text
tables/submission/tableS1_dma_metrics.md
```

全精度来源：

```text
tables/literature/table_star_gnn_dma_common46.csv
```

正文不再用一张 20-row 表重复 Main Fig. 2b 的空间结论。

---

# Main result figures

最终正文只保留三张核心结果图。

## Main Figure 1 — Ablation mechanism and lead-time stability

```text
figures/submission/main_fig1_ablation_leadtime.pdf
figures/submission/main_fig1_ablation_leadtime.svg
figures/submission/main_fig1_ablation_leadtime.png
```

- Panel a：四个 factorial variants 的 Day 1--Day 7 absolute publisher-compatible MAE + moving-block 95% CI；
- Panel b：相对自身 Day 1 的 lead-time degradation，并直接标 Day 7 端点。

回答：**SAS-Norm 与 FA-DPR 分别如何影响周尺度准确性与 lead-time 稳定性？**

## Main Figure 2 — Temporal and spatial robustness

```text
figures/submission/main_fig2_temporal_spatial_robustness.pdf
figures/submission/main_fig2_temporal_spatial_robustness.svg
figures/submission/main_fig2_temporal_spatial_robustness.png
```

- Panel a：46 common origins 上的 paired MAE improvement + moving-block mean CI + win count；
- Panel b：10 DMA × 2 horizons × 2 graph baselines 的 sequential-blue improvement heatmap。

回答：**平均优势是否只来自少数有利日期或少数 DMA？**

## Main Figure 3 — Week-ahead demand dynamics

```text
figures/submission/main_fig3_week_ahead_dynamics.pdf
figures/submission/main_fig3_week_ahead_dynamics.svg
figures/submission/main_fig3_week_ahead_dynamics.png
```

- Panel a：46 origins × 7 days 的 diurnal aggregate-demand absolute-error profile；
- Panel b：预先规定 median-error rule 选出的 representative 168 h trajectory；
- Panel c：同一 representative origin 的 aggregate-demand hourly absolute error。

回答：**统计优势在真实一周需求轨迹中具体表现为什么？**

水量单位为 `L s⁻¹`。

---

# Supplementary figures

## Figure S1 — Relative improvement over all baselines

```text
figures/supplementary/supp_figS1_relative_improvement.*
```

原总体 relative-improvement heatmap 降级为 Supplementary，并改为单向 sequential-blue 编码。

## Figure S2 — Per-origin ECDF

```text
figures/supplementary/supp_figS2_origin_ecdf.*
```

保留 DCRNN / STGCN / STaR-GNN 的 per-origin ECDF，作为 Main Fig. 2a paired-difference analysis 的 distributional reassurance。

---

# Submission figure audit data

```text
tables/manuscript/submission/
```

包含：

```text
main_fig1_daywise_block_ci.csv
main_fig1_day7_degradation.csv
main_fig2_origin_paired_improvement.csv
main_fig2_origin_paired_summary.csv
main_fig2_dma_improvement.csv
main_fig3_diurnal_aggregate_error.csv
main_fig3_representative_trajectory.csv
main_fig3_representative_selection.json
submission_figure_audit.json
```

这些文件是图中数值的独立可复算来源。

---

# Metric conventions

正文总体和消融遵循 publisher-compatible 口径：

- total MAE = DMA A--J MAE 之和；
- total MAPE/RMSE/NSE = 在 A--J 小时总需求轨迹上计算；
- aggregate-demand MAE 仅作为系统总需求轨迹的内部/补充诊断。

完整说明见：

```text
tables/literature/METRIC_CONVENTIONS.md
```

---

# Canonical regeneration

先生成全精度审计表：

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout
```

再生成投稿显示表：

```bash
python scripts/reproduce/render_submission_tables.py
```

最后一次性生成所有投稿图：

```bash
python scripts/reproduce/render_submission_figures.py \
  --release results/paper/frozen_v1 \
  --block-length 7 \
  --bootstrap-iterations 50000 \
  --bootstrap-seed 20260821
```

最终图统一输出 PDF、editable SVG 和 300 dpi PNG。

---

# Legacy / diagnostic figures

根 `figures/` 下的：

```text
manuscript_fig1_...
...
manuscript_fig5_...
test_*
pearson_correlation_heatmap.*
```

继续保留用于历史复现、内部诊断和审计，但**不再是最终投稿图权威入口**。
