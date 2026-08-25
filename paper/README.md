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

包含 GRU、LSTM、MSNet、MSCMNet variants、DCRNN、STGCN、STaR-GNN。前六个时序/多尺度模型取自 Que et al. (2024) 的报告值；DCRNN、STGCN、STaR-GNN 为 common-46 协议下重新评价。

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

168 h total MAE：SAS-Norm-only `12.208`，STaR-GNN `12.234`；差约 `0.21%`。该点估计差异由 Main Fig. 2 的 ordered moving-block evidence 限定解释；完整模型在 168 h 的 MAPE、RMSE、NSE 以及 aggregate-demand MAE 上更优。

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

表中并列给出 DCRNN、STGCN、STaR-GNN 在 10 个 DMA、两个预测时域和四个指标上的绝对结果，用于支撑 Main Fig. 3b 与 Supplementary Fig. S1。

---

# Main result figures

最终正文保留四张核心结果图，按“总体优势—组件作用—稳健性—实际轨迹”展开。

## Main Figure 1 — Overall four-metric performance

```text
figures/submission/main_fig1_overall_performance.pdf
figures/submission/main_fig1_overall_performance.svg
figures/submission/main_fig1_overall_performance.png
```

- Panel a：STaR-GNN 相对六个 published models 与两个 graph baselines 的 MAE/MAPE/RMSE 降幅；
- Panel b：对应的 NSE 绝对提升（$\Delta$NSE）。

回答：**STaR-GNN 是否在 24 h 与 168 h、四个互补指标上都保持总体优势？**

## Main Figure 2 — Four-metric ablation and lead-time stability

```text
figures/submission/main_fig2_ablation_leadtime.pdf
figures/submission/main_fig2_ablation_leadtime.svg
figures/submission/main_fig2_ablation_leadtime.png
```

- 四个 panel 分别给出 MAE、MAPE、RMSE、NSE；
- SAS-Norm、FA-DPR、Full 均相对 DCRNN 做 paired improvement；同一天内用水平错位的点和置信区间避免 SAS-Norm 与 Full 重叠。

回答：**SAS-Norm 与 FA-DPR 分别如何影响周尺度准确性与 lead-time stability？**

## Main Figure 3 — Four-metric temporal and spatial robustness

```text
figures/submission/main_fig3_temporal_spatial_robustness.pdf
figures/submission/main_fig3_temporal_spatial_robustness.svg
figures/submission/main_fig3_temporal_spatial_robustness.png
```

- Panel a：46 个 common origins 上四指标的 mean improvement 与 win count；
- Panel b：10 个 DMA 上四指标的 mean improvement 与 win count。

回答：**平均优势是否只来自少数有利日期或少数 DMA？**

## Main Figure 4 — Week-ahead demand dynamics

```text
figures/submission/main_fig4_week_ahead_dynamics.pdf
figures/submission/main_fig4_week_ahead_dynamics.svg
figures/submission/main_fig4_week_ahead_dynamics.png
```

- Panel a：46 origins × 7 days 的 diurnal aggregate-demand absolute-error profile；
- Panel b：预先规定 median-total-MAE rule 选出的 representative 168 h trajectory；
- Panel c：同一 representative origin 的 aggregate-demand hourly absolute error。

回答：**统计优势在真实一周需求轨迹中具体表现为什么？**

水量单位为 `L s⁻¹`。

---

# Supplementary figures

## Supplementary Figure S1 — Detailed DMA improvements

```text
figures/supplementary/supp_figS1_dma_improvement.*
```

逐 DMA 展示 MAE、MAPE、RMSE 的相对降幅和 NSE 的绝对增益；发散配色如实保留 168 h / DMA G / STGCN 下 RMSE 与 NSE 的两个例外。

## Supplementary Figure S2 — Per-origin ECDF

```text
figures/supplementary/supp_figS2_origin_ecdf.*
```

保留 DCRNN / STGCN / STaR-GNN 的 per-origin total-MAE ECDF，作为 Main Fig. 3a 的 distributional reassurance。

---

# Submission figure audit data

```text
tables/manuscript/submission/
```

包含：

```text
main_fig2_daywise_paired_improvement.csv
main_fig3_origin_paired_improvement.csv
main_fig3_origin_paired_summary.csv
main_fig3_dma_improvement.csv
main_fig4_diurnal_aggregate_error.csv
main_fig4_representative_trajectory.csv
main_fig4_representative_selection.json
submission_figure_audit.json
```

这些文件是图中数值的独立可复算来源。

---

# Metric conventions

正文总体和消融遵循统一口径：

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
