# common-46 内部 Test 诊断报告（aggregate-demand 口径）

> **重要：本文件不是当前 Journal of Hydrology 稿件的 manuscript-facing total 结果入口。**
>
> 本报告由 legacy `build_detailed_test_artifacts.py` 流程产生，主要用于 aggregate-demand、旧逐日、DMA 和 Pearson 工件的内部诊断。正文 total MAE、最终 publisher-compatible 消融和 Figure 1--5 请以以下文件为准：
>
> - [`../../docs/RESULTS_AND_ARTIFACTS_CN.md`](../../docs/RESULTS_AND_ARTIFACTS_CN.md)
> - [`../tables/literature/METRIC_CONVENTIONS.md`](../tables/literature/METRIC_CONVENTIONS.md)
> - [`../../docs/MANUSCRIPT_FIGURES_FINAL_CN.md`](../../docs/MANUSCRIPT_FIGURES_FINAL_CN.md)
> - [`../../docs/PLOTTING_CN.md`](../../docs/PLOTTING_CN.md)

## 1. 为什么这里的 MAE 与正文不同

本报告中的总体 MAE 使用：

\[
MAE_{agg}=MAE\left(\sum_i \hat y_i,\sum_i y_i\right),
\]

即先将十个 DMA 的需求求和，再对总需求序列计算 MAE。

正文跨模型 total MAE 使用：

\[
MAE_{publisher}=\sum_i MAE_i,
\]

即分别计算 DMA A--J 的 MAE 后求和。

因此两个数值都可以正确复现，但回答的问题不同，不能互换。

STaR-GNN：

| Horizon | Internal aggregate-demand MAE | Manuscript publisher-compatible MAE |
|---|---:|---:|
| 24 h | 4.360841 | **9.424199** |
| 168 h | 4.919812 | **12.233590** |

## 2. 内部 aggregate-demand 总体诊断

下表仅用于复现旧运行诊断。`State` 是 SAS-Norm-only 的内部兼容标签，`Full` 是 STaR-GNN 的内部兼容标签。

| Horizon | Internal model label | MAE | MAPE (%) | RMSE | NSE |
|---|---|---:|---:|---:|---:|
| 24 h | STGCN | 5.850690 | 2.424526 | 7.904592 | 0.960590 |
| 24 h | DCRNN | 5.356315 | 2.212928 | 6.848257 | 0.970419 |
| 24 h | State / SAS-Norm | 4.895320 | 2.010448 | 6.133886 | 0.976269 |
| 24 h | FA-DPR | 4.739336 | 1.944550 | 6.079036 | 0.976691 |
| 24 h | Full / STaR-GNN | 4.360841 | 1.804574 | 5.534656 | 0.980679 |
| 168 h | STGCN | 8.574033 | 3.575848 | 10.305691 | 0.933337 |
| 168 h | DCRNN | 7.734838 | 3.248413 | 9.817428 | 0.939504 |
| 168 h | State / SAS-Norm | 5.122511 | 2.102380 | 6.468312 | 0.973739 |
| 168 h | FA-DPR | 7.578056 | 3.277716 | 9.332415 | 0.945334 |
| 168 h | Full / STaR-GNN | 4.919812 | 2.013774 | 6.160881 | 0.976176 |

这里的 MAPE/RMSE/NSE 本身是在 aggregate-demand series 上计算，因此与 publisher-compatible 总体表中的对应三项一致；主要需要防止混淆的是 **MAE**。

## 3. 旧 31/32 与最终 30/32 的区别

旧 aggregate-demand 层级的 factorial relation 记录为 31/32，其唯一明显例外是 FA-DPR 168 h MAPE 略差于 DCRNN。

但当前论文消融采用 publisher-compatible sum-of-DMA MAE 后，还存在第二个真实例外：

- SAS-Norm-only 168 h MAE：`12.207835`
- Full STaR-GNN 168 h MAE：`12.233590`

因此最终 manuscript-facing 消融是 **30/32**，而不是 31/32。

正文应写：完整模型在 24 h 四项指标上均最优；168 h 下 MAPE、RMSE、NSE 最优，而 SAS-Norm-only 的 publisher-compatible MAE 略低于 Full。

## 4. Legacy Day 1--Day 7 诊断

旧 `test_day1_day7_*` 表图把 168 h 预测拆成七个 24 h 区间，并在十 DMA **总需求序列**上重算 MAE/MAPE/RMSE/NSE。因此它们属于 aggregate-demand diagnostics。

当前正文 Figure 2 已改为 publisher-compatible daily sum-of-DMA MAE，并增加 46 origins 的 bootstrap 95% CI。请使用：

```text
paper/figures/manuscript_fig2_day1_day7_publisher_mae.*
paper/tables/manuscript/fig2_day1_day7_publisher_mae_ci.csv
paper/tables/manuscript/fig2_day1_day7_degradation.csv
```

最终 Day 7 相对 Day 1：

- DCRNN：+38.25%
- FA-DPR：+11.93%
- SAS-Norm：+2.64%
- STaR-GNN：+1.70%

## 5. DMA 与 Pearson 工件

Legacy detailed-artifact 流程仍用于生成：

- `tables/test_dma_metrics_long.csv`
- `tables/test_dma_*_wide.*`
- `tables/pearson_*`
- `figures/test_dma_mae_*`
- `figures/pearson_correlation_heatmap.*`

这些工件仍有审计价值，但正文 DMA 相对改善使用最终 Figure 4：

```text
paper/figures/manuscript_fig4_dma_mae_improvement.*
paper/tables/manuscript/fig4_dma_mae_improvement.csv
```

当前 40/40 个 DMA-horizon-baseline MAE 比较均为正改善。

## 6. 当前正文 Figure 1--5

```text
figures/manuscript_fig1_relative_improvement.*
figures/manuscript_fig2_day1_day7_publisher_mae.*
figures/manuscript_fig3_origin_ecdf.*
figures/manuscript_fig4_dma_mae_improvement.*
figures/manuscript_fig5_representative_168h_trajectory.*
```

对应审计位于：

```text
tables/manuscript/
```

Figure 3 当前 paired win rates：

- 24 h vs DCRNN：45/46
- 24 h vs STGCN：45/46
- 168 h vs DCRNN：46/46
- 168 h vs STGCN：40/46

Figure 5 代表样本按 STaR-GNN publisher-compatible 168 h MAE 最接近中位数的预先固定规则选择，而不是按视觉效果挑选。

## 7. 当前文档权威顺序

若本文件与其他结果文档产生疑问，按以下顺序判断：

1. `paper/tables/literature/METRIC_CONVENTIONS.md`
2. `paper/tables/literature/table_*`
3. `paper/tables/manuscript/*.csv/json`
4. `docs/RESULTS_AND_ARTIFACTS_CN.md`
5. `docs/MANUSCRIPT_FIGURES_FINAL_CN.md`
6. 本 legacy internal report

本报告保留是为了让历史 aggregate-demand 诊断仍可追溯，而不是作为正文最终结果。