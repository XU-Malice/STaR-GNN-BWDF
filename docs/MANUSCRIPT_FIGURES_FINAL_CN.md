# 上一版五图结果方案（历史记录，已被替代）

> **状态：LEGACY / SUPERSEDED。** 本文件记录 2026-08-20 完成的上一版 Journal of Hydrology 五张结果图方案，用于审计和历史复现。它已经被新的 claim-driven 投稿设计替代，**不得再作为当前投稿图权威说明**。

当前权威实验与图表设计：

- [`EXPERIMENT_DESIGN_FINAL_CN.md`](EXPERIMENT_DESIGN_FINAL_CN.md) — 最终 Results evidence architecture；
- [`PLOTTING_CN.md`](PLOTTING_CN.md) — canonical submission renderer 教程；
- [`../paper/README.md`](../paper/README.md) — 当前投稿表图路径；
- [`../paper/captions/SUBMISSION_RESULT_FIGURE_CAPTIONS.md`](../paper/captions/SUBMISSION_RESULT_FIGURE_CAPTIONS.md) — 当前投稿 caption。

## 历史版本曾采用的五张结果图

```text
manuscript_fig1_relative_improvement.*
manuscript_fig2_day1_day7_publisher_mae.*
manuscript_fig3_origin_ecdf.*
manuscript_fig4_dma_mae_improvement.*
manuscript_fig5_representative_168h_trajectory.*
```

这些图仍保留在 `paper/figures/`，因为它们对应已审计的历史结果表示，并可用于 Supplementary/diagnostic 对照；但当前正文已经重新组织为：

```text
Main Table 1
  overall predictive performance
        ↓
Main Table 2 + Main Fig. 1
  factorial ablation + lead-time stability
        ↓
Main Fig. 2
  temporal + spatial robustness
        ↓
Main Fig. 3
  population-to-instance week-ahead dynamics
```

新的投稿图位于：

```text
paper/figures/submission/
paper/figures/supplementary/
```

## 为什么替代上一版五图结构

上一版五张图在数值和科学事实上没有错误，但按更高标准的 claim-driven evidence architecture 复审后存在三个信息组织问题：

1. relative-improvement heatmap 与 Table 1 都主要证明“总体性能更好”，独立 inference gain 有限；
2. ECDF 与 DMA heatmap 分成两张主图，但它们可以共同回答“优势是否跨时间和空间稳定”这一更强的 Results-level question；
3. representative trajectory 只给单个 instance，缺少全体测试样本上的 population-level 日内误差证据。

因此最终设计做了以下升级：

- 原 relative-improvement 图 → **Supplementary Fig. S1**；
- 原 ECDF → **Supplementary Fig. S2**；
- paired origin-level improvement + DMA heatmap → **Main Fig. 2**；
- population diurnal error + representative trajectory + local error → **Main Fig. 3**；
- ablation 图重新设计为 absolute Day-wise MAE + lead-time degradation → **Main Fig. 1**。

## 历史结果仍然有效

替换的是**证据组织和视觉表达**，不是冻结 prediction 或指标数值。以下结果仍保持不变：

- STaR-GNN publisher-compatible MAE：24 h `9.424199`，168 h `12.233590`；
- 四模型 factorial ablation，STGCN 不属于消融；
- 168 h SAS-Norm-only publisher MAE `12.207835`，与 Full 相差约 0.21%；
- Day 7 relative-to-Day 1：DCRNN约 `+38.25%`、FA-DPR约 `+11.93%`、SAS-Norm约 `+2.64%`、STaR-GNN约 `+1.70%`；
- paired win counts：`45/46, 45/46, 46/46, 40/46`；
- 40/40 DMA-horizon-baseline MAE comparisons 为正；
- representative origin 继续使用预先固定的 median-error proximity rule。

旧图不得被用于制造与新投稿图不同的指标口径或结论；两套图只是对同一冻结结果采用不同的信息架构。
