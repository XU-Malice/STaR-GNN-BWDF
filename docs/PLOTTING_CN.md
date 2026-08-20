# Journal of Hydrology 最终论文作图教程

本文档只说明最终正文 Figure 1--5 的生成与检查。最终结果见 [`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md)，图表设计见 [`MANUSCRIPT_FIGURES_FINAL_CN.md`](MANUSCRIPT_FIGURES_FINAL_CN.md)。

## 1. 两套 MAE 必须分开

正文总体比较、消融、Figure 1--3 使用 publisher-compatible MAE：

\[
MAE_{publisher}=\sum_{i=A}^{J} MAE_i.
\]

内部 aggregate-demand MAE：

\[
MAE_{agg}=MAE\left(\sum_i\hat y_i,\sum_i y_i\right).
\]

STaR-GNN：

| Horizon | aggregate-demand MAE | publisher-compatible MAE |
|---|---:|---:|
| 24 h | 4.360841 | 9.424199 |
| 168 h | 4.919812 | 12.233590 |

不要把两套 MAE 放在同一横向比较中。

## 2. 先重建论文表格

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout
```

成功标志：

```text
Metric convention audit: PASS
Factorial ablation model-set audit: PASS (4 models, no STGCN)
Publisher-compatible factorial cell audit: 30/32 PASS
```

检查：

```bash
cat paper/tables/literature/table_ablation_common46.csv
cat paper/tables/literature/table_ablation_audit.json
```

Table 2 必须只有：

```text
DCRNN
DCRNN + SAS-Norm
DCRNN + FA-DPR
STaR-GNN
```

如果出现 STGCN，说明仍在使用旧生成器或旧分支。

## 3. 生成总体/消融/DMA 绝对指标图

```bash
python scripts/reproduce/build_literature_figures.py \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --ablation-table paper/tables/literature/table_ablation_common46.csv \
  --dma-table paper/tables/literature/table_star_gnn_dma_common46.csv \
  --output paper/figures
```

成功标志：

```text
Overall figure audit: PASS
Factorial ablation figure audit: PASS (4 models, no STGCN)
```

这些绝对柱状图主要作为 supplementary/reference；正文优先使用下一节 Figure 1--5。

## 4. 第一阶段：生成 Figure 1--5 的基础审计工件

```bash
python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820
```

该阶段生成：

- Figure 1 relative improvement；
- Figure 2/3 基础数据；
- Figure 4 DMA improvement；
- Figure 5 representative 168 h trajectory；
- `paper/tables/manuscript/` 下的可复算 CSV/JSON。

第一阶段 Figure 2/3 会被第二阶段覆盖；不要在第二阶段前把它们视为最终投稿图。

## 5. 第二阶段：生成最终 Figure 2 与 Figure 3

```bash
python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures \
  --block-bootstrap-iterations 50000 \
  --block-bootstrap-length 7 \
  --block-bootstrap-seed 20260820
```

成功标志：

```text
Refined manuscript Figure 2 and Figure 3: PASS
Figure 2 factorial-model audit: PASS (4 models, no STGCN)
```

该阶段会生成/覆盖：

```text
paper/figures/manuscript_fig2_day1_day7_publisher_mae.png
paper/figures/manuscript_fig2_day1_day7_publisher_mae.pdf
paper/figures/manuscript_fig3_origin_ecdf.png
paper/figures/manuscript_fig3_origin_ecdf.pdf
```

并生成：

```text
paper/tables/manuscript/fig2_ablation_daywise_reduction_vs_dcrnn.csv
paper/tables/manuscript/fig2_day1_day7_degradation.csv
paper/tables/manuscript/fig2_full_vs_sas_block_bootstrap.json
paper/tables/manuscript/fig3_origin_win_rates.csv
paper/tables/manuscript/manuscript_empirical_figure_audit.json
```

## 6. 最终 Figure 2 的科学问题

### Panel (a)

比较：

```text
DCRNN + SAS-Norm
DCRNN + FA-DPR
STaR-GNN
```

纵轴：相对 DCRNN 的 day-wise MAE reduction：

\[
\frac{MAE_{DCRNN,d}-MAE_{model,d}}{MAE_{DCRNN,d}}\times100\%.
\]

DCRNN 本身作为 0% reference，不需要再画一条重复的 baseline 曲线。

### Panel (b)

比较四个 factorial variants 的 Day-1-relative MAE change：

\[
\frac{MAE_d-MAE_{Day1}}{MAE_{Day1}}\times100\%.
\]

Day 7：

```text
DCRNN                 +38.25%
DCRNN + FA-DPR        +11.93%
DCRNN + SAS-Norm       +2.64%
STaR-GNN               +1.70%
```

因此 Figure 2 强调的是模块贡献和 lead-time robustness，而不是放大 168 h overall MAE 的 0.21% 点估计差异。

## 7. 为什么使用 7-origin moving-block bootstrap

168 h 预测每 24 h 启动一次，相邻 forecast origins 共享大量预测时段，不能简单把 46 origins 当作相互独立。

`refine_manuscript_results_figures.py` 对 Full−SAS 的 168 h per-origin publisher MAE 差使用 ordered moving-block bootstrap：

- block length = 7 origins；
- circular moving blocks；
- iterations = 50,000；
- seed = 20260820。

当前均值差约：

```text
Full - SAS ≈ +0.025755
```

相当于约 +0.21%，bootstrap 95% CI 跨过 0。该审计用于限制论文措辞：不把 0.21% 解释为稳定显著差异。

## 8. Figure 3 检查

Figure 3 只比较：

```text
DCRNN
STGCN
STaR-GNN
```

这是 baseline robustness，不是消融。

预期 paired win rates：

```text
24 h  vs DCRNN: 45/46
24 h  vs STGCN: 45/46
168 h vs DCRNN: 46/46
168 h vs STGCN: 40/46
```

## 9. Figure 4 检查

```bash
cat paper/tables/manuscript/fig4_dma_mae_improvement.csv
```

应有 40 rows，且全部 `MAE_reduction_pct > 0`。

范围约：

```text
1.26% -- 61.20%
```

不要写成“各 DMA 均有同等幅度提升”。

## 10. Figure 5 检查

```bash
cat paper/tables/manuscript/fig5_representative_168h_selection.json
```

应为：

```text
selected_common_index = 70
STGCN     ≈ 14.653
DCRNN     ≈ 15.517
STaR-GNN  ≈ 12.182
```

选择规则是 STaR-GNN per-origin publisher MAE 最接近 46 origins 中位数，不依赖视觉效果。

## 11. 投稿图件格式

正文图统一：

- PDF 矢量；
- PNG 300 dpi；
- 白底；
- 不使用 3D、渐变、雷达图；
- 线型和 marker 同时区分模型，兼顾灰度打印；
- 坐标轴写清单位/统计量；
- 不通过截断 y 轴夸大 0.21% 差异；
- 表格正文统一 3 位小数，CSV/JSON 审计保留完整精度。

## 12. 最终一次性检查

```bash
python -m py_compile \
  scripts/reproduce/build_paper_tables.py \
  scripts/reproduce/build_literature_figures.py \
  scripts/reproduce/build_manuscript_results_figures.py \
  scripts/reproduce/refine_manuscript_results_figures.py

python -m pytest tests/test_paper_artifacts.py -q
```

然后：

```bash
git status --short
```

只在确认最终 PNG/PDF、CSV/JSON 与脚本均已重新生成后再提交。
