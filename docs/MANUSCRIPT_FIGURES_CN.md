# Journal of Hydrology 论文图表方案（STaR-GNN）

本文档固定正文结果部分的图表分工。目标不是把同一组指标重复画成不同图，而是遵循高水平时空预测与水务预测论文常见的证据组织原则：**表格负责精确数值，图件负责揭示相对优势、长时域行为、样本稳健性、跨 DMA 一致性与代表性预测轨迹。**

## 1. 正文表格

### Table 1. Overall comparison of competing models

文件：`paper/tables/literature/table_literature_comparison_common46.*`

模型：GRU、LSTM、MSNet、MSCMNet_WM、MSCMNet_M、MSCMNet_W、DCRNN、STGCN、STaR-GNN。

指标：24 h/168 h 的 MAE、MAPE、RMSE、NSE。

口径：与 Que et al. (2024) 补充材料一致；total MAE 为 A--J DMA MAE 之和，MAPE/RMSE/NSE 在总需求序列上计算。

### Table 2. Ablation and graph-model comparison

文件：`paper/tables/literature/table_ablation_common46.*`

模型：STGCN、DCRNN、DCRNN + SAS-Norm、DCRNN + FA-DPR、STaR-GNN。

用途：给出消融实验的精确结果，不用图形掩盖 168 h 中 SAS-Norm-only MAE 略低于 Full 的真实例外。

### Table 3. DMA-level performance of STaR-GNN

文件：`paper/tables/literature/table_star_gnn_dma_common46.*`

内容：DMA A--J 在 24 h/168 h 下的 MAE、MAPE、RMSE、NSE。

用途：报告空间异质性与每个 DMA 的绝对表现。

## 2. 正文主图

所有主图由：

```bash
python scripts/reproduce/build_manuscript_results_figures.py
```

自动生成。每张图同时生成 CSV/JSON 审计工件到 `paper/tables/manuscript/`。

### Figure 1. Relative performance improvement over competing models

输出：

- `paper/figures/manuscript_fig1_relative_improvement.png`
- `paper/figures/manuscript_fig1_relative_improvement.pdf`

设计：

- 左侧：8 个 baseline × 6 个误差项的相对改善热力图；
- 右侧：24 h/168 h 的绝对 NSE 增益。

误差指标定义：

\[
\Delta E=\frac{E_{baseline}-E_{STaR}}{E_{baseline}}\times100\%.
\]

NSE 不做百分比改善，报告：

\[
\Delta NSE=NSE_{STaR}-NSE_{baseline}.
\]

科学问题：STaR-GNN 的优势是否同时存在于不同 baseline、不同误差指标和两个预测时域，而非仅由某一个指标驱动？

正文作用：替代普通 9 模型柱状图；精确绝对值继续由 Table 1 承担。

### Figure 2. Day-1--Day-7 error evolution under the 168 h task

输出：

- `paper/figures/manuscript_fig2_day1_day7_publisher_mae.png`
- `paper/figures/manuscript_fig2_day1_day7_publisher_mae.pdf`

曲线：STGCN、DCRNN、DCRNN + SAS-Norm、DCRNN + FA-DPR、STaR-GNN。

每一天按 24 h 切分 168 h 预测，对每个 common-46 起点先计算 A--J DMA MAE 之和，然后计算 46 个起点的均值及非参数 bootstrap 95% CI。

科学问题：随着预测提前期从 Day 1 增加到 Day 7，误差如何演化？STaR-GNN 是否能够减缓长时域误差累积？

该图与 FA-DPR 的研究动机直接对应，是 168 h 结果解释的核心主图。

### Figure 3. Error distribution across 46 common test origins

输出：

- `paper/figures/manuscript_fig3_origin_ecdf.png`
- `paper/figures/manuscript_fig3_origin_ecdf.pdf`

形式：24 h 与 168 h 两个 panel 的 ECDF。

比较：STGCN、DCRNN、DCRNN + SAS-Norm、DCRNN + FA-DPR、STaR-GNN。

每个测试起点的误差定义为该起点 A--J DMA MAE 之和。

科学问题：总体平均性能优势是否稳定存在于大多数测试起点，而不是由少数有利样本驱动？

解读：在相同累计概率下，曲线越靠左表示误差分布整体越小。

### Figure 4. DMA-level spatial consistency of improvements

输出：

- `paper/figures/manuscript_fig4_dma_mae_improvement.png`
- `paper/figures/manuscript_fig4_dma_mae_improvement.pdf`

形式：DMA A--J × 4 个比较条件的相对 MAE 改善热力图。

四列：

- 24 h vs DCRNN；
- 24 h vs STGCN；
- 168 h vs DCRNN；
- 168 h vs STGCN。

科学问题：总体性能提升是否具有跨 DMA 一致性，还是主要由少数区域贡献？

正文作用：与 Table 3 的绝对 DMA 指标互补；Table 3 给绝对性能，Figure 4 给相对空间优势。

### Figure 5. Representative 168 h forecast trajectory

输出：

- `paper/figures/manuscript_fig5_representative_168h_trajectory.png`
- `paper/figures/manuscript_fig5_representative_168h_trajectory.pdf`

上 panel：Observed、STGCN、DCRNN、STaR-GNN 的 168 h 总需求轨迹。

下 panel：三个模型相对于真实总需求的逐小时绝对误差。

案例选择规则预先固定：

> 在 common-46 的 46 个测试起点中，选择 STaR-GNN publisher-compatible 168 h MAE 最接近其中位数的起点。

该规则只依据误差中位数距离，不依据图形外观，避免 cherry-picking。

科学问题：模型是否能够在完整一周中持续跟踪日内峰谷、跨日水平变化与周期结构，并避免随预测距离增加产生明显轨迹漂移？

## 3. Supplementary figures

以下已有图件建议保留，但不作为正文主图：

- `test_overall_24h.*` / `test_overall_168h.*`：绝对指标柱状图；
- `test_ablation_24h.*` / `test_ablation_168h.*`：绝对消融指标柱状图；
- `test_star_gnn_dma_metrics.*`：STaR-GNN 各 DMA 的绝对四指标；
- `test_day1_day7_models.*` / `test_day1_day7_ablation.*`：旧 aggregate-demand 诊断图；
- `pearson_correlation_heatmap.*`：方法/数据分析部分的功能关联图。

## 4. 图表选择原则

- **表格**：模型数量多、指标多、需要精确读取数值时使用；
- **热力图**：模型 × 指标或 DMA × 条件的二维比较，突出一致性与相对改善；
- **折线图**：预测 horizon、Day 1--Day 7 等有序维度，表达误差演化；
- **ECDF**：表达 46 个测试起点的完整误差分布，避免只报告均值；
- **时间序列轨迹图**：定性展示峰谷、周期、漂移和长时域行为；
- **普通柱状图**：仅保留为 Supplementary 的绝对指标参考，不承担正文的主要科学结论。

不采用雷达图、3D 图和装饰性气泡图，因为这类图难以提供额外可验证证据，且不利于精确比较。

## 5. 学术组织依据

该组织方式吸收两类文献的实验表达思路：

1. 时空预测/深度学习工作通常同时报告精确 benchmark 表，并针对长预测范围、预测轨迹和消融行为提供专门分析，而不是只重复画绝对指标柱状图；例如 Graph WaveNet (IJCAI 2019) 将长时序建模作为核心问题并使用多预测范围结果验证方法。
2. 多 DMA 水需求研究不仅报告 total demand，还强调 DMA-level 表现、跨区域稳定性和代表性预测场景；Que et al. (Water Research X, 2024) 同时分析单 DMA、total demand 和不同预测任务。Journal of Hydrology 的多时间尺度水需求预测工作也通过多个 forecasting scales 验证模型在不同预测尺度上的稳定优势。

因此，本文最终的 Results 证据链固定为：

> **总体精度（Tables 1--2 + Fig. 1） → 长时域稳定性（Fig. 2） → 样本稳健性（Fig. 3） → 跨 DMA 空间一致性（Table 3 + Fig. 4） → 代表性一周轨迹（Fig. 5）。**
