# Journal of Hydrology 投稿版表格、图件与审计工件

`paper/` 同时保留两类资产：

1. **submission-facing artifacts**：真正用于论文正文/Supplementary 的最终表图；
2. **legacy / diagnostic artifacts**：旧 `manuscript_fig1...5`、`test_*` 等历史复现和内部诊断图。

> **投稿版权威入口：** `tables/submission/`、`figures/submission/` 和 `tables/manuscript/submission/`。

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

168 h total MAE：SAS-Norm-only `12.208`，STaR-GNN `12.234`；差约 `0.21%`。该点估计差异由 Main Fig. 4 的 ordered moving-block evidence 限定解释；完整模型在 168 h 的 MAPE、RMSE、NSE 以及 aggregate-demand MAE 上更优。

---

## Supplementary Table S1 — DMA-level metrics

显示版：

```text
tables/submission/tableS1_dma_metrics.md
```

全精度来源：

```text
tables/literature/table_temporal_models_dma.csv
tables/literature/table_graph_models_dma_common46.csv
tables/literature/table_all_models_dma.csv
```

表中并列给出全部九种模型在 10 个 DMA、两个预测时域和四个指标上的绝对结果，用于支撑 Main Fig. 2 和 Main Fig. 3。时序模型原始 MAPE 小数已统一转换为百分数。

## Supplementary Table S2 — DMA-level local margins

显示版：

```text
tables/submission/tableS2_dma_local_margin.md
```

对每个“预测时域–DMA–指标”组合，表中独立给出最强非 STaR-GNN 竞争模型及 Fig. 3 使用的有符号幅度。所有负值均被保留。

## Supplementary Table S3 — Forecast-origin robustness

```text
tables/submission/tableS3_forecast_origin_robustness.md
tables/submission/tableS3_forecast_origin_robustness.csv
```

仅对同协议 DCRNN、STGCN、STaR-GNN 报告 46-origin 四指标改善、moving-block CI、胜出数及观测定义的高波动窗口统计。A 类 published models 不具有逐起点输出，未被推断或补齐。

---

# Main result figures

最终正文保留六张核心结果图，按“总体优势—跨 DMA 分布—局部竞争边界—组件作用—逐起点稳健性—实际轨迹”展开。

## Main Figure 1 — Overall four-metric performance

```text
figures/submission/main_fig1_overall_performance.pdf
figures/submission/main_fig1_overall_performance.svg
figures/submission/main_fig1_overall_performance.png
```

- Panel a：STaR-GNN 相对六个时序模型与两个图模型的 MAE/MAPE/RMSE 降幅；
- Panel b：对应的 NSE 绝对提升（$\Delta$NSE）。

回答：**STaR-GNN 是否在 24 h 与 168 h、四个互补指标上都保持总体优势？**

## Main Figure 2 — DMA-level performance breadth

```text
figures/submission/main_fig2_dma_performance.pdf
figures/submission/main_fig2_dma_performance.svg
figures/submission/main_fig2_dma_performance.png
```

- 四个 panel：STaR-GNN 相对八种基线的 MAE、MAPE、RMSE、NSE 逐 DMA 有符号改善分布；
- 小点保留全部十个 DMA，大点与线段为中位数和四分位距，圆/方形区分 24 h 与 168 h。

回答：**相对不同模型族的改善是否广泛分布于 DMA，而非由少数分区驱动？**

## Main Figure 3 — DMA-specific local competitive margin

```text
figures/submission/main_fig3_dma_local_margin.pdf
figures/submission/main_fig3_dma_local_margin.svg
figures/submission/main_fig3_dma_local_margin.png
```

- 四个 panel：STaR-GNN 相对每个 DMA–指标–预测时域内最强竞争者的有符号幅度；
- 连接 24 h 与 168 h，保留局部负值和 DMA 身份。

回答：**系统级优势在不同 DMA 中如何保持、收窄或反转？**

## Main Figure 4 — Four-metric ablation and lead-time stability

```text
figures/submission/main_fig4_ablation_leadtime.pdf
figures/submission/main_fig4_ablation_leadtime.svg
figures/submission/main_fig4_ablation_leadtime.png
```

- 四个 panel 分别给出 MAE、MAPE、RMSE、NSE；
- SAS-Norm、FA-DPR、Full 均相对 DCRNN 做 paired improvement；同一天内用水平错位的点和置信区间避免 SAS-Norm 与 Full 重叠。

回答：**SAS-Norm 与 FA-DPR 分别如何影响周尺度准确性与 lead-time stability？**

## Main Figure 5 — Forecast-origin and difficult-window robustness

```text
figures/submission/main_fig5_origin_robustness.pdf
figures/submission/main_fig5_origin_robustness.svg
figures/submission/main_fig5_origin_robustness.png
```

- Panels a–c：46 个共同起点上的四指标 paired effects 与 ordered moving-block 95% CI；
- Panel d：仅由观测 normalized mean absolute ramp 定义的高波动窗口胜出数。

回答：**平均优势是否跨预测起点成立，并在困难需水条件下保持？**

## Main Figure 6 — Week-ahead demand dynamics

```text
figures/submission/main_fig6_week_ahead_dynamics.pdf
figures/submission/main_fig6_week_ahead_dynamics.svg
figures/submission/main_fig6_week_ahead_dynamics.png
```

- Panel a：全部测试窗口 × 7 days 的 diurnal aggregate-demand absolute-error profile；
- Panel b：预先规定 median-total-MAE rule 选出的 representative 168 h trajectory；
- Panel c：同一 representative origin 的 aggregate-demand hourly absolute error。

回答：**统计优势在真实一周需求轨迹中具体表现为什么？**

水量单位为 `L s⁻¹`。

---

# Submission figure audit data

```text
tables/manuscript/submission/
```

包含：

```text
main_fig2_dma_ranks.csv
main_fig2_dma_pairwise_improvement.csv
main_fig3_dma_strongest_competitor.csv
main_fig4_daywise_paired_improvement.csv
main_fig5_origin_paired_improvement.csv
main_fig5_origin_summary.csv
main_fig6_diurnal_aggregate_error.csv
main_fig6_representative_trajectory.csv
main_fig6_representative_selection.json
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
