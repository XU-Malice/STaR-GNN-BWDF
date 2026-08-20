# Manuscript Figures 初始设计记录（已被最终方案替代）

> **状态：历史设计稿，不再作为当前正文图表方案。**
>
> 当前 Journal of Hydrology 稿件的最终 Figure 1--5、Table 1--3、实证解释边界以及两阶段制图流程，请以 [`MANUSCRIPT_FIGURES_FINAL_CN.md`](MANUSCRIPT_FIGURES_FINAL_CN.md) 为准。

本文件保留仅用于记录图表设计从“总体/消融绝对柱状图”向“科学问题驱动的主图体系”演化的过程，避免后续误把旧方案重新当作最终版本。

## 当前唯一有效的正文图表体系

正文三张表：

1. **Table 1 — Overall comparison**：`paper/tables/literature/table_literature_comparison_common46.*`
2. **Table 2 — Ablation and graph-model comparison**：`paper/tables/literature/table_ablation_common46.*`
3. **Table 3 — DMA-level STaR-GNN performance**：`paper/tables/literature/table_star_gnn_dma_common46.*`

正文五张图：

1. **Figure 1 — Relative performance improvement**：`paper/figures/manuscript_fig1_relative_improvement.*`
2. **Figure 2 — Day 1--Day 7 long-horizon behavior**：`paper/figures/manuscript_fig2_day1_day7_publisher_mae.*`
3. **Figure 3 — ECDF across 46 common origins**：`paper/figures/manuscript_fig3_origin_ecdf.*`
4. **Figure 4 — DMA-level spatial consistency**：`paper/figures/manuscript_fig4_dma_mae_improvement.*`
5. **Figure 5 — Representative 168 h trajectory**：`paper/figures/manuscript_fig5_representative_168h_trajectory.*`

最终证据链为：

> **总体精度 → 模块贡献与长时域稳定性 → 测试起点稳健性 → 跨 DMA 空间一致性 → 代表性一周预测行为**

## 为什么旧绝对柱状图不再作为正文主图

`test_overall_*`、`test_ablation_*`、`test_star_gnn_dma_metrics.*` 等图仍保留在仓库中，但精确绝对值已经由 Table 1--3 承担。正文主图需要回答额外的科学问题，而不是把表格重新画一遍，因此这些旧图降为 Supplementary/内部参考。

旧 `test_day1_day7_*` 图采用 aggregate-demand 诊断口径，也不能替代当前 Figure 2 的 publisher-compatible Day 1--Day 7 分析。

## 当前作图入口

完整两阶段制图命令、输入/输出、审计文件、成功标志与故障处理见：

- [`PLOTTING_CN.md`](PLOTTING_CN.md)
- [`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md)

最终 Figure captions：

- [`../paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md`](../paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md)

## 重要结果解释边界

当前 publisher-compatible 消融为 **30/32**，不是旧 aggregate-demand 诊断中的 31/32。特别是：

- FA-DPR 168 h MAPE 略差于 DCRNN；
- SAS-Norm-only 的 168 h sum-of-DMA MAE（12.207835）略低于 Full（12.233590）。

因此不能再沿用旧设计稿中“Full 在两个 horizon 所有指标均严格最优”的表达。