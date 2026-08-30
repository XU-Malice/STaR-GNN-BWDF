# Journal of Hydrology 投稿版作图教程

本教程只描述最终投稿版权威图表。实验逻辑见 [`EXPERIMENT_DESIGN_FINAL_CN.md`](EXPERIMENT_DESIGN_FINAL_CN.md)。旧 `manuscript_fig1...5` 与 `test_*` 图仅用于历史复现/诊断。

## 1. 指标口径

正文总体比较和 factorial ablation 使用：

\[
MAE_{total}=\sum_{i=A}^{J}MAE_i.
\]

MAPE、RMSE、NSE 在 A--J 的小时总需求序列上计算。内部系统总需求诊断另用：

\[
MAE_{agg}=MAE\left(\sum_i\hat y_i,\sum_i y_i\right).
\]

两种 MAE 回答不同问题，不得混合排序。

## 2. 重建全精度表与投稿显示表

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

python scripts/reproduce/render_submission_tables.py \
  --source-dir paper/tables/literature \
  --output-dir paper/tables/submission \
  --release results/paper/frozen_v1
```

生成：

```text
paper/tables/submission/table1_overall_performance.md
paper/tables/submission/table2_factorial_ablation.md
paper/tables/submission/tableS1_dma_metrics.md
paper/tables/submission/tableS2_dma_local_margin.md
paper/tables/submission/tableS3_forecast_origin_robustness.{md,csv}
```

Table 1 通过表注说明 Que et al. (2024) 数值与本文流程结果的来源，不使用符号。Table S1 并列报告全部九种模型的逐 DMA 四指标；时序模型 MAPE 已统一转换为百分数。

## 3. 一次性生成最终图

```bash
MPLCONFIGDIR=/tmp/star_gnn_mpl python scripts/reproduce/render_submission_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --dma-table paper/tables/literature/table_all_models_dma.csv \
  --main-output paper/figures/submission \
  --audit-output paper/tables/manuscript/submission \
  --block-length 7 \
  --bootstrap-iterations 50000 \
  --bootstrap-seed 20260821
```

成功标志：

```text
Submission figure renderer: PASS
Main figures:
  Main Fig. 1 — overall four-metric performance
  Main Fig. 2 — cross-DMA pairwise improvement distributions
  Main Fig. 3 — 24 h absolute DMA performance
  Main Fig. 4 — 168 h absolute DMA performance
  Main Fig. 5 — four-metric ablation and lead-time stability
  Main Fig. 6 — forecast-origin and difficult-window robustness
  Main Fig. 7 — week-ahead demand dynamics
```

每张图同时输出 PDF、editable SVG 和 300 dpi PNG。

## 4. 图件逻辑

### Main Figure 1

Panel a 为 STaR-GNN 相对六个时序模型与 DCRNN/STGCN 的 MAE、MAPE、RMSE 降幅；Panel b 为 NSE 绝对增益。模型名不带额外符号。

### Main Figure 2

四个 panel 分别为 MAE、MAPE、RMSE、NSE。纵轴为八种基线，横轴为 STaR-GNN 相对每个基线的有符号逐 DMA 改善。小点保留十个 DMA，大点与线段分别为中位数和四分位距；圆和方形区分 24 h 与 168 h。

### Main Figure 3

2 × 2 分面分别呈现 MAE、MAPE、RMSE 和 NSE。每个 DMA 使用蓝色方点和灰色空心圆直接比较 STaR-GNN 与该位置上的最优基线 24 h 绝对值，并以竖向线段连接；局部最优基线占优时，空心圆及连线改为橙色。MAPE 和 NSE 使用明确标示的聚焦纵轴。

### Main Figure 4

沿用 Main Fig. 3 的 2 × 2 分面、配对点和颜色编码，但单独呈现 168 h 结果。相同指标与 24 h 图共用纵轴范围，以支持跨预测时域的直接比较。

### Main Figure 5

四个 panel 分别为 MAE、MAPE、RMSE、NSE。SAS-Norm-only、FA-DPR-only 与 Full 均相对 DCRNN 做逐日 paired improvement；误差指标为相对降幅，NSE 为绝对增益。

同一 forecast day 内三个模型使用轻微水平错位，并固定 color、marker、linestyle。这样 SAS-Norm 与 STaR-GNN 即使数值接近也能辨认；误差棒为 ordered seven-window moving-block 95% CI。

### Main Figure 6

Panels a–c 展示同协议 DCRNN/STGCN/STaR-GNN 的 46-origin 四指标 paired effects 与 ordered seven-origin moving-block 95% CI；Panel d 使用仅由观测需求定义的 normalized mean absolute ramp，汇总最高四分位窗口中的胜出数。

### Main Figure 7

采用 scale-to-instance 结构：全部测试窗口 × 7 days 的日内 aggregate-demand error profile、预先规定 median-total-MAE rule 的 representative 168 h trajectory，以及同一窗口的 hourly absolute error。

## 5. 统一视觉系统

`scripts/reproduce/manuscript_plot_style.py` 集中管理：

```text
STaR-GNN              #0F4D92  deep blue, hero
DCRNN                  #5C5C5C  dark gray
STGCN                  #A6A6A6  light gray
DCRNN + SAS-Norm       #8FB6D5  soft blue
DCRNN + FA-DPR         #9A86B8  muted violet
Observed               #1F1F1F  near-black
```

同一模型跨图固定颜色、线型和 marker；主图按约 7.4 in 双栏宽度直接设计；图内不放长标题，完整解释进入 caption。

## 6. 最终检查

```bash
python scripts/reproduce/audit_public_repository.py \
  --require-frozen \
  --require-paper-artifacts
```

投稿前必须确认：

- 最终表图中没有额外符号和内部工程措辞；
- PDF/SVG/PNG 均可打开，SVG 文字可编辑；
- Main Fig. 2 的基线轴、改善轴、预测时域与分布统计含义完整；
- Main Figs. 3–4 的 DMA 轴、独立绝对指标尺度、最优基线定义和橙色例外完整；
- Main Fig. 5 的三条 variant series 可分辨；
- caption、审计 CSV/JSON 与图中数值一致。
