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
```

Table 1 使用行分组和表注区分 Que et al. (2024) 报告值与 common-46 graph-model 复评，不使用符号。Table S1 并列报告 DCRNN、STGCN、STaR-GNN 的逐 DMA 四指标。

## 3. 一次性生成最终图

```bash
MPLCONFIGDIR=/tmp/star_gnn_mpl python scripts/reproduce/render_submission_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --main-output paper/figures/submission \
  --supp-output paper/figures/supplementary \
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
  Main Fig. 2 — four-metric ablation and lead-time stability
  Main Fig. 3 — four-metric temporal and spatial robustness
  Main Fig. 4 — week-ahead demand dynamics
Supplementary figures:
  Fig. S1 — detailed four-metric DMA improvements
  Fig. S2 — per-origin total-MAE ECDF
```

每张图同时输出 PDF、editable SVG 和 300 dpi PNG。

## 4. 图件逻辑

### Main Figure 1

Panel a 为 STaR-GNN 相对六个 published reference models 与 DCRNN/STGCN 的 MAE、MAPE、RMSE 降幅；Panel b 为 NSE 绝对增益。模型名不带额外符号。

### Main Figure 2

四个 panel 分别为 MAE、MAPE、RMSE、NSE。SAS-Norm-only、FA-DPR-only 与 Full 均相对 DCRNN 做逐日 paired improvement；误差指标为相对降幅，NSE 为绝对增益。

同一 forecast day 内三个模型使用轻微水平错位，并固定 color、marker、linestyle。这样 SAS-Norm 与 STaR-GNN 即使数值接近也能辨认；误差棒为 ordered seven-origin moving-block 95% CI。

### Main Figure 3

Panel a 汇总 46 个 common forecast origins，Panel b 汇总 10 个 DMA。颜色编码 improved-comparison rate，单元格同时给 mean improvement 与 wins/comparisons。逐 DMA 共 160 个比较，158 个改善；两个例外必须保留。

### Main Figure 4

采用 scale-to-instance 结构：46 origins × 7 days 的日内 aggregate-demand error profile、预先规定 median-total-MAE rule 的 representative 168 h trajectory，以及同一 origin 的 hourly absolute error。

### Supplementary Figure S1

逐 DMA 展示四指标 improvement。由于存在两个真实负值，使用以 0 为中心的发散配色：蓝色改善、红色退化。

### Supplementary Figure S2

DCRNN、STGCN、STaR-GNN 的 24 h / 168 h per-origin total-MAE ECDF，作为 Main Fig. 3a 的分布证据。

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
- Main Fig. 2 的三条 variant series 可分辨；
- Main Fig. 3b 与 Supplementary Fig. S1 如实显示两个非改善比较；
- caption、审计 CSV/JSON 与图中数值一致。
