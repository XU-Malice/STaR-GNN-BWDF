# Journal of Hydrology 投稿版作图教程

本教程只描述最终投稿版权威图表。实验逻辑见 [`EXPERIMENT_DESIGN_FINAL_CN.md`](EXPERIMENT_DESIGN_FINAL_CN.md)。

> **权威规则：**旧 `manuscript_fig1...5` 与 `test_*` 图继续保留用于历史复现/诊断；真正用于投稿的图只由 `render_submission_figures.py` 生成。

---

## 1. 两套 MAE 必须分开

正文总体比较和 factorial ablation 使用 publisher-compatible MAE：

\[
MAE_{publisher}=\sum_{i=A}^{J}MAE_i.
\]

内部系统总需求诊断：

\[
MAE_{agg}=MAE\left(\sum_i\hat y_i,\sum_i y_i\right).
\]

两者都合法，但回答不同问题，不得混合排序。

---

## 2. 先重建全精度审计表

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout
```

预期：

```text
Metric convention audit: PASS
Factorial ablation model-set audit: PASS (4 models, no STGCN)
Publisher-compatible factorial cell audit: 30/32 PASS
```

原始 CSV/JSON 保留全精度。

---

## 3. 生成投稿显示表

```bash
python scripts/reproduce/render_submission_tables.py \
  --source-dir paper/tables/literature \
  --output-dir paper/tables/submission
```

生成：

```text
paper/tables/submission/table1_overall_performance.md
paper/tables/submission/table2_factorial_ablation.md
paper/tables/submission/tableS1_dma_metrics.md
```

正文表统一 3 位小数；Table 1 用 `†` 区分 Que et al. (2024) 已发表结果和 common-46 重新评价的 graph models。

---

## 4. 一次性生成最终投稿图

```bash
python scripts/reproduce/render_submission_figures.py \
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
  Main Fig. 1 — ablation and lead-time stability
  Main Fig. 2 — temporal and spatial robustness
  Main Fig. 3 — week-ahead demand dynamics
Supplementary figures:
  Fig. S1 — relative improvement over all baselines
  Fig. S2 — per-origin ECDF
```

每张图同时输出：

```text
PDF  — 投稿/排版用矢量图
SVG  — 可编辑文字矢量图
PNG  — 300 dpi 预览
```

---

## 5. Main Figure 1

### Panel a — Absolute day-wise publisher-compatible MAE

四个 factorial variants：

```text
DCRNN
DCRNN + SAS-Norm
DCRNN + FA-DPR
STaR-GNN
```

展示 Day 1--Day 7 绝对 MAE，并使用 ordered 7-origin moving-block bootstrap 95% CI。

### Panel b — Lead-time degradation

\[
100\times\frac{MAE_d-MAE_{Day1}}{MAE_{Day1}}.
\]

只在 Day 7 标关键端点。历史冻结结果应接近：

```text
DCRNN                 +38.25%
DCRNN + FA-DPR        +11.93%
DCRNN + SAS-Norm       +2.64%
STaR-GNN               +1.70%
```

Main Fig. 1 的目的不是隐藏 168 h `12.208 vs 12.234`，而是把 absolute accuracy 与 lead-time stability 分成两个不同 inferential roles。

---

## 6. Main Figure 2

### Panel a — Paired forecast-origin improvement

对同一 forecast origin：

\[
\Delta MAE_s=MAE_{baseline,s}-MAE_{STaR,s}.
\]

正值表示 STaR-GNN 更好。四组比较：

```text
24 h vs DCRNN
24 h vs STGCN
168 h vs DCRNN
168 h vs STGCN
```

每组画 46 个 paired differences、moving-block mean 95% CI，并标 win count。预期：

```text
45/46
45/46
46/46
40/46
```

### Panel b — DMA robustness

10 DMA × 2 horizons × 2 graph baselines = 40 cells。

所有单元格应 `MAE_reduction_pct > 0`，范围约 `1.26%--61.20%`。使用单向 sequential blue heatmap；不要使用正负 diverging colormap。

---

## 7. Main Figure 3

采用 scale-to-instance 结构。

### Panel a

全部 46 origins × 7 forecast days 折叠成 24 h 日内周期，比较 DCRNN、STGCN、STaR-GNN 的 aggregate-demand absolute error，单位为 `L s⁻¹`，CI 使用 moving-block bootstrap。

### Panel b

median-error proximity rule 选出的 representative 168 h aggregate-demand trajectory。

### Panel c

同一 origin 的 hourly aggregate-demand absolute error。

需求单位来自 BWDF 原始 net inflow 定义：`L/s`。

---

## 8. Supplementary figures

### Figure S1

原 relative-improvement heatmap。因为所有 improvement 均为正，使用 sequential blue，不再使用 `RdBu_r`。Published reference models 与 re-evaluated graph baselines 用分隔线区分。

### Figure S2

DCRNN、STGCN、STaR-GNN 的 24 h/168 h per-origin ECDF，作为 Main Fig. 2a paired analysis 的 distributional reassurance。

---

## 9. 统一视觉系统

由：

```text
scripts/reproduce/manuscript_plot_style.py
```

集中管理。

最终颜色：

```text
STaR-GNN              #0F4D92  deep blue, hero
DCRNN                  #5C5C5C  dark gray
STGCN                  #A6A6A6  light gray
DCRNN + SAS-Norm       #8FB6D5  soft blue
DCRNN + FA-DPR         #9A86B8  muted violet
Observed               #1F1F1F  near-black
```

规则：

- 同一模型在所有 panel 中固定颜色、线型和 marker；
- baseline 降低视觉权重，hero method 最突出；
- top/right spines 关闭；
- legend 无边框；
- 主图宽度按约 7.4 in（约 188 mm）直接设计；
- 图内不放长标题，完整描述放 caption；
- 不通过截断坐标或选择性小数位夸大差异。

---

## 10. 审计数据

新的投稿图审计文件在：

```text
paper/tables/manuscript/submission/
```

包括：

```text
main_fig1_daywise_block_ci.csv
main_fig1_day7_degradation.csv
main_fig2_origin_paired_improvement.csv
main_fig2_origin_paired_summary.csv
main_fig2_dma_improvement.csv
main_fig3_diurnal_aggregate_error.csv
main_fig3_representative_trajectory.csv
main_fig3_representative_selection.json
submission_figure_audit.json
```

不要只看 PNG；关键结论必须能从 CSV/JSON 独立复算。

---

## 11. 最终 QA

```bash
python -m py_compile \
  scripts/reproduce/manuscript_plot_style.py \
  scripts/reproduce/render_submission_tables.py \
  scripts/reproduce/render_submission_figures.py

python -m pytest \
  tests/test_paper_release.py \
  tests/test_paper_artifacts.py \
  -q

git diff --check
```

重建源码校验：

```bash
python scripts/reproduce/regenerate_source_checksums.py
bash scripts/reproduce/verify_source.sh
```

只有在图、表、CSV/JSON 与源码审计全部通过后再提交。
