# Manuscript Figures 初始设计记录（已被最终方案替代）

> **状态：历史设计稿，不再作为当前正文图表方案。**
>
> 当前 Journal of Hydrology 稿件的最终 Figure 1--5、Table 1--3、实证解释边界和两阶段制图流程，请以 [`MANUSCRIPT_FIGURES_FINAL_CN.md`](MANUSCRIPT_FIGURES_FINAL_CN.md) 为准。

本文件仅用于保留设计演化记录，避免后续误把早期“总体/消融绝对柱状图”重新当作正文主方案。

## 当前唯一有效的正文表图

正文三张表：

1. **Table 1 — Overall comparison**：`paper/tables/literature/table_literature_comparison_common46.*`
2. **Table 2 — Factorial ablation**：`paper/tables/literature/table_ablation_common46.*`
3. **Table 3 — DMA-level STaR-GNN performance**：`paper/tables/literature/table_star_gnn_dma_common46.*`

Table 2 只包含 DCRNN、DCRNN + SAS-Norm、DCRNN + FA-DPR、STaR-GNN。STGCN 是独立 graph baseline，不进入消融。

正文五张图：

1. **Figure 1 — Relative performance improvement**：`paper/figures/manuscript_fig1_relative_improvement.*`
2. **Figure 2 — Factorial ablation and Day 1--Day 7 long-horizon behavior**：`paper/figures/manuscript_fig2_day1_day7_publisher_mae.*`
3. **Figure 3 — ECDF across 46 common origins**：`paper/figures/manuscript_fig3_origin_ecdf.*`
4. **Figure 4 — DMA-level spatial consistency**：`paper/figures/manuscript_fig4_dma_mae_improvement.*`
5. **Figure 5 — Representative 168 h trajectory**：`paper/figures/manuscript_fig5_representative_168h_trajectory.*`

最终证据链：

> **总体精度 → 模块贡献与长时域稳定性 → 测试起点稳健性 → 跨 DMA 空间一致性 → 代表性一周预测行为**

## 旧方案为什么被替代

- `test_overall_*`、`test_ablation_*`、`test_star_gnn_dma_metrics.*` 的绝对值信息已由 Table 1--3 更精确地承担；
- 旧 `test_day1_day7_*` 使用 aggregate-demand 诊断口径，不能替代当前 Figure 2 的 publisher-compatible 分析；
- 早期把 STGCN 与组件消融混在同一图表，不利于区分“外部 baseline 比较”和“factorial module ablation”。

## 当前解释边界

- manuscript-facing factorial cell audit = **30/32**；
- 旧 `31/32` 只属于 aggregate-demand internal hierarchy；
- FA-DPR 168 h MAPE 略差于 DCRNN；
- SAS-Norm-only 168 h publisher-compatible MAE `12.207835`，STaR-GNN `12.233590`，差约 `0.21%`；
- 7-origin moving-block bootstrap 对 Full−SAS 均值差给出的 95% CI 跨过 0，因此该差异不作为稳定性能差距解读。

当前作图命令见 [`PLOTTING_CN.md`](PLOTTING_CN.md)，最终结果见 [`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md)。
