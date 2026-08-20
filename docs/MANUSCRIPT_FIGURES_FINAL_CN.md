# Journal of Hydrology 最终结果图表方案

本文档固定 STaR-GNN 投稿 **Journal of Hydrology** 时的最终结果组织。原则是：**表格负责精确数值，图件负责回答科学问题；消融与外部 baseline 分开；不通过选择性四舍五入、截断坐标或删结果制造排序。**

## 1. Table 1：总体模型比较

文件：`paper/tables/literature/table_literature_comparison_common46.*`

包含 9 个模型：GRU、LSTM、MSNet、MSCMNet_WM、MSCMNet_M、MSCMNet_W、DCRNN、STGCN、STaR-GNN。

- GRU/LSTM/MSNet/MSCMNet variants：Que et al. (2024) reported results；
- DCRNN/STGCN/STaR-GNN：common-46 下重新评估；
- 跨论文比较采用 publisher-compatible total：MAE 为 DMA A--J MAE 之和，MAPE/RMSE/NSE 在小时级总需求序列上计算。

STaR-GNN：

| Horizon | MAE | MAPE (%) | RMSE | NSE |
|---|---:|---:|---:|---:|
| 24 h | 9.424 | 1.805 | 5.535 | 0.981 |
| 168 h | 12.234 | 2.014 | 6.161 | 0.976 |

## 2. Table 2：四模型 factorial ablation

文件：`paper/tables/literature/table_ablation_common46.*`

**只包含：**

1. DCRNN；
2. DCRNN + SAS-Norm；
3. DCRNN + FA-DPR；
4. STaR-GNN。

STGCN 是独立图时空 baseline，不属于组件消融，不能出现在 Table 2。

### 24 h

| Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---:|---:|---:|---:|
| DCRNN | 11.917 | 2.213 | 6.848 | 0.970 |
| DCRNN + SAS-Norm | 10.468 | 2.010 | 6.134 | 0.976 |
| DCRNN + FA-DPR | 11.238 | 1.945 | 6.079 | 0.977 |
| **STaR-GNN** | **9.424** | **1.805** | **5.535** | **0.981** |

### 168 h

| Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---:|---:|---:|---:|
| DCRNN | 16.801 | 3.248 | 9.817 | 0.940 |
| **DCRNN + SAS-Norm** | **12.208** | 2.102 | 6.468 | 0.974 |
| DCRNN + FA-DPR | 14.086 | 3.278 | 9.332 | 0.945 |
| STaR-GNN | 12.234 | **2.014** | **6.161** | **0.976** |

168 h 的 SAS-Norm-only 与 STaR-GNN publisher-compatible MAE 仅相差 0.025755（约 0.21%）。由于 168 h forecast origins 每 24 h 启动一次并强烈重叠，最终审计使用 7-origin moving-block bootstrap；`Full - SAS` 均值差的 95% CI 跨过 0。因此正文应写“MAE 点估计近似持平”，而不是写成 Full 明显退化或明显优于 SAS-Norm。

## 3. Figure 1：跨模型相对优势

文件：`manuscript_fig1_relative_improvement.*`

- Panel (a)：MAE/MAPE/RMSE 相对降低率；
- Panel (b)：NSE 绝对增益。

Figure 1 回答：STaR-GNN 的优势是否跨不同 baseline、指标和预测时域持续存在。

MAE 相对提升：

- 24 h vs sequence/multiscale reported models：34.9%--46.7%；
- 24 h vs DCRNN/STGCN：20.9%--23.7%；
- 168 h vs sequence/multiscale reported models：18.2%--34.5%；
- 168 h vs DCRNN/STGCN：16.0%--27.2%。

## 4. Figure 2：四模型消融与 168 h 长时域行为

文件：`manuscript_fig2_day1_day7_publisher_mae.*`

Figure 2 **不包含 STGCN**。

### Panel (a)：Day-wise MAE reduction relative to DCRNN

比较：

- DCRNN + SAS-Norm；
- DCRNN + FA-DPR；
- STaR-GNN。

纵轴：

\[
\Delta MAE_d=\frac{MAE_{DCRNN,d}-MAE_{model,d}}{MAE_{DCRNN,d}}\times100\%.
\]

该 panel 直接回答“每个模块在不同 forecast lead 上带来了多少改善”，避免把 0.02 左右的绝对 MAE 点估计差异放大成主视觉。

### Panel (b)：MAE change relative to Day 1

比较全部四个 factorial variants。

\[
G_d=\frac{MAE_d-MAE_{Day1}}{MAE_{Day1}}\times100\%.
\]

Day 7 相对 Day 1：

- DCRNN：+38.25%；
- DCRNN + FA-DPR：+11.93%；
- DCRNN + SAS-Norm：+2.64%；
- STaR-GNN：+1.70%。

该结果支持：SAS-Norm 是降低长时域绝对误差的主要贡献模块；FA-DPR 可减缓误差随 lead time 增长；两者结合后 Full 获得最小的 Day-1-to-Day-7 相对退化，并在 168 h 的 MAPE、RMSE、NSE 和 aggregate-demand MAE 上表现更好。

## 5. Figure 3：46 个测试起点上的稳健性

文件：`manuscript_fig3_origin_ecdf.*`

只比较 DCRNN、STGCN、STaR-GNN。这里回答的是 baseline robustness，不是消融。

paired win rates：

- 24 h vs DCRNN：45/46；
- 24 h vs STGCN：45/46；
- 168 h vs DCRNN：46/46；
- 168 h vs STGCN：40/46。

## 6. Table 3 + Figure 4：DMA 空间一致性

Table 3：`table_star_gnn_dma_common46.*`

Figure 4：`manuscript_fig4_dma_mae_improvement.*`

10 DMA × 2 horizons × 2 graph baselines = 40 个 MAE comparisons，全部为正改善，范围约 1.26%--61.20%。正文宜写“consistent positive improvements across all DMAs with heterogeneous magnitudes”，不要写“uniformly large improvements”。

## 7. Figure 5：代表性 168 h 轨迹

文件：`manuscript_fig5_representative_168h_trajectory.*`

使用预先固定的 median-error proximity 规则，而不是挑最好看的样本。当前 common index = 70：

- STGCN：14.653；
- DCRNN：15.517；
- STaR-GNN：12.182。

下 panel 的 hourly absolute error 是 aggregate-demand diagnostic，与 publisher-compatible sum-of-DMA MAE 不同，图注必须明确。

## 8. 论文 Results 的最终证据链

1. **Overall predictive accuracy**：Table 1 + Figure 1；
2. **Ablation and long-horizon behavior**：Table 2 + Figure 2；
3. **Robustness across forecast origins**：Figure 3；
4. **Spatial consistency across DMAs**：Table 3 + Figure 4；
5. **Representative weekly forecasting behavior**：Figure 5。

这种组织避免把 baseline comparison、factorial ablation、样本稳健性和空间异质性混在一张图里，更适合 Journal of Hydrology 的工程与水文叙事。

## 9. 生成命令

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
  --figure-dir paper/figures \
  --block-bootstrap-iterations 50000 \
  --block-bootstrap-length 7 \
  --block-bootstrap-seed 20260820
```

第二阶段额外生成：

- `fig2_ablation_daywise_reduction_vs_dcrnn.csv`；
- `fig2_day1_day7_degradation.csv`；
- `fig2_full_vs_sas_block_bootstrap.json`；
- `fig3_origin_win_rates.csv`；
- `manuscript_empirical_figure_audit.json`。

最终 PNG 使用 300 dpi；PDF 保留矢量格式。正文表格统一采用 3 位小数，审计 CSV/JSON 保留完整精度。
