# Journal of Hydrology 最终结果图表方案（基于冻结 common-46 实证审计）

本文档在 `MANUSCRIPT_FIGURES_CN.md` 的初始设计基础上，根据冻结 common-46 结果的实际统计表现进一步收敛正文图表。原则不变：**表格负责精确数值，图件负责揭示相对优势、长时域行为、样本稳健性、跨 DMA 一致性和代表性预测行为。**

## 1. 正文表格保持不变

### Table 1. Overall comparison of competing models

使用 `paper/tables/literature/table_literature_comparison_common46.*`。

报告 24 h/168 h 下 GRU、LSTM、MSNet、MSCMNet_WM、MSCMNet_M、MSCMNet_W、DCRNN、STGCN、STaR-GNN 的 MAE、MAPE、RMSE、NSE。跨论文比较采用 publisher-compatible total：MAE 为 DMA A--J MAE 之和；MAPE/RMSE/NSE 在小时级总需求序列上计算。

### Table 2. Ablation and graph-model comparison

使用 `paper/tables/literature/table_ablation_common46.*`。

保留 STGCN、DCRNN、DCRNN + SAS-Norm、DCRNN + FA-DPR、STaR-GNN 的精确结果。尤其保留 168 h 中 SAS-Norm-only 的 sum-of-DMA MAE 略低于 Full 的真实例外，避免图形化结果造成过度概括。

### Table 3. DMA-level performance of STaR-GNN

使用 `paper/tables/literature/table_star_gnn_dma_common46.*`。

报告 DMA A--J 在 24 h/168 h 下的 MAE、MAPE、RMSE、NSE，作为绝对空间表现的精确结果。

## 2. Figure 1：跨模型、跨指标的相对性能优势

文件：`manuscript_fig1_relative_improvement.*`

- Panel (a)：MAE/MAPE/RMSE 相对降低率；
- Panel (b)：绝对 NSE 增益。

Figure 1 不重复 Table 1 的绝对值，而回答：STaR-GNN 的优势是否同时存在于不同 baseline、不同误差指标和两个预测时域？

实证结果显示，所有已纳入的 baseline 在 24 h 和 168 h 的 MAE、MAPE、RMSE 上相对 STaR-GNN 均为正改善，NSE 也均提高。对于 Que et al. (2024) 的 GRU/LSTM/MSNet/MSCMNet variants，图中必须用 `†` 明确标记为 reported results；DCRNN 和 STGCN 为 common-46 下重新评估结果，不能混称为统一条件重训。

## 3. Figure 2：168 h 长时域行为——最终采用双 panel

文件：`manuscript_fig2_day1_day7_publisher_mae.*`

### Panel (a)：绝对 long-horizon MAE

只比较 DCRNN、STGCN、STaR-GNN，并保留 46 个 common test origins 的 bootstrap 95% CI。

目的：回答 STaR-GNN 在 Day 1--Day 7 上是否持续保持较低绝对预测误差。

### Panel (b)：相对 Day 1 的误差变化率

比较 DCRNN、DCRNN + SAS-Norm、DCRNN + FA-DPR、STaR-GNN：

\[
\Delta_d=\frac{MAE_d-MAE_{Day1}}{MAE_{Day1}}\times100\%.
\]

冻结结果显示，Day 7 相对 Day 1 的 MAE 变化约为：

- DCRNN：+38.2%；
- DCRNN + FA-DPR：+11.9%；
- DCRNN + SAS-Norm：+2.6%；
- STaR-GNN：+1.7%。

因此，正文不应写“FA-DPR 单独消除了长时域误差累积”。更准确的机制解释是：**SAS-Norm 是抑制 168 h MAE 漂移的主要贡献模块；FA-DPR 单独也能降低 DCRNN 的长时域退化速度；二者结合后，Full 保持了接近无漂移的 MAE 行为，同时在 168 h 的 MAPE、RMSE 和 NSE 上取得最佳综合结果。**

同样不能写“Full 在所有 Day 和所有指标上均优于 SAS-Norm-only”，因为这一点不符合冻结结果。

## 4. Figure 3：46 个测试起点上的样本稳健性

文件：`manuscript_fig3_origin_ecdf.*`

最终正文 ECDF **只比较 DCRNN、STGCN、STaR-GNN**。消融模块不再放入 ECDF，以免把“样本稳健性”和“模块消融”两个科学问题混在同一图中。

冻结结果的 paired win rate：

- 24 h：STaR-GNN 相比 DCRNN 为 45/46（97.8%），相比 STGCN 为 45/46（97.8%）；
- 168 h：STaR-GNN 相比 DCRNN 为 46/46（100.0%），相比 STGCN 为 40/46（87.0%）。

因此，Figure 3 可以支持：总体平均性能改善并非由少数有利测试窗口驱动；特别是在 168 h 下，STaR-GNN 在全部 46 个 common test origins 上均优于 DCRNN。

不要用该图宣称 Full 在 168 h 的每个 origin 都优于 SAS-Norm-only；实际 paired win rate 仅为 19/46，这一关系应留在 Table 2 与 Figure 2(b) 中解释。

## 5. Figure 4：跨 DMA 空间一致性

文件：`manuscript_fig4_dma_mae_improvement.*`

保持 DMA A--J × 4 个比较条件的 MAE reduction heatmap：

- 24 h vs DCRNN；
- 24 h vs STGCN；
- 168 h vs DCRNN；
- 168 h vs STGCN。

冻结结果中 10 DMA × 2 horizon × 2 baseline 共 40 个比较全部为正改善，范围约 1.26%--61.20%。因此这张图是正文强证据：整体优势具有跨 DMA 一致性，并非由少数 DMA 拉动。

同时必须指出改善幅度存在空间异质性；例如部分 DMA 的提升较小。正文宜写“consistent positive improvements across all DMAs”而不是“uniformly large improvements”。

## 6. Figure 5：非 cherry-picked 的 168 h 代表性轨迹

文件：`manuscript_fig5_representative_168h_trajectory.*`

选择规则固定为：在 46 个 common test origins 中，选择 STaR-GNN publisher-compatible 168 h MAE 最接近其中位数的起点。

当前选中 origin 的 publisher-compatible MAE：

- STGCN：14.653；
- DCRNN：15.517；
- STaR-GNN：12.182。

即便采用中位误差样本而非最好样本，STaR-GNN 仍明显优于两个图基线。该图只用于 qualitative evidence：展示一周内日内峰谷、跨日水平变化和长时域轨迹漂移，不替代 Table 1 的定量总体结论。

注意：下 panel 的逐小时 absolute error 基于 aggregate-demand trajectory，与 publisher-compatible sum-of-DMA MAE 是不同统计量，图注必须明确这一点。

## 7. 正文 Results 的最终证据链

建议顺序：

1. **Overall predictive accuracy**：Table 1 + Figure 1；
2. **Ablation and component contributions**：Table 2 + Figure 2；
3. **Robustness across forecast origins**：Figure 3；
4. **Spatial consistency across DMAs**：Table 3 + Figure 4；
5. **Representative weekly forecasting behavior**：Figure 5。

这形成一条从“平均精度”到“机制”再到“样本稳健性、空间稳健性、定性行为”的完整证据链，既符合 Journal of Hydrology 对工程可靠性和空间异质性的关注，也符合高水平深度学习/时空预测论文将 benchmark table、ablation、forecast-horizon analysis 和 qualitative trajectory 分开组织的实验习惯。

## 8. Supplementary

以下已有绝对指标图建议放 Supplementary，而不再承担正文主要结论：

- `test_overall_24h.*` / `test_overall_168h.*`；
- `test_ablation_24h.*` / `test_ablation_168h.*`；
- `test_star_gnn_dma_metrics.*`；
- 旧的 aggregate-demand Day 1--Day 7 诊断图。

雷达图、3D 图、装饰性气泡图不采用，因为它们不能增加可验证证据，且会降低精确比较的可读性。

## 9. 生成顺序

第一阶段从冻结结果生成审计工件和基础主图：

```bash
python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820
```

第二阶段根据实证审计生成最终的 Figure 2 和 Figure 3：

```bash
python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures
```

第二阶段会覆盖 Figure 2/3 的基础版本，并额外生成：

- `fig2_day1_day7_degradation.csv`；
- `fig3_origin_win_rates.csv`；
- `manuscript_empirical_figure_audit.json`。

这些审计文件用于支撑 Results 正文和后续复核。