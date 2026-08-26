# Legacy manuscript figure audit artifacts

本文件记录早期两阶段作图流程，仅用于历史追溯，不再定义投稿图号。当前 Journal of Hydrology 正文 Figure 1--5 的权威审计工件位于 [`submission/`](submission/)，图件与数据对应关系见 [`../../README.md`](../../README.md)。以下旧 Figure 1--5 名称均指 `paper/figures/manuscript_*` 历史图，不应与当前 `paper/figures/submission/` 混用。

## Stage 1

```bash
python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820
```

## Stage 2

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

## Figure 1

`fig1_relative_improvement.csv`

- STaR-GNN 相对 8 个 baseline 的 MAE/MAPE/RMSE reduction；
- NSE 为 absolute gain；
- `source` 区分 Que et al. (2024) reported results 与 common-46 re-evaluated results。

## Figure 2

Stage 1 基础文件：

```text
fig2_day1_day7_publisher_mae_ci.csv
fig2_day1_day7_publisher_mae_metadata.json
```

Stage 2 最终审计：

```text
fig2_ablation_daywise_reduction_vs_dcrnn.csv
fig2_day1_day7_degradation.csv
fig2_full_vs_sas_block_bootstrap.json
```

Figure 2 是**纯 factorial ablation**，不包含 STGCN。

- Panel (a)：SAS-Norm / FA-DPR / STaR-GNN 相对 DCRNN 的 day-wise publisher-compatible MAE reduction；
- Panel (b)：DCRNN / SAS-Norm / FA-DPR / STaR-GNN 相对各自 Day 1 的 MAE change。

Day 7 vs Day 1：

```text
DCRNN                 +38.25%
DCRNN + FA-DPR        +11.93%
DCRNN + SAS-Norm       +2.64%
STaR-GNN               +1.70%
```

`fig2_full_vs_sas_block_bootstrap.json` 使用 7-origin moving-block bootstrap 审计 168 h Full−SAS publisher-MAE 差。当前点估计约 +0.025755（约 +0.21%），95% CI 跨过 0，因此不解释为稳定性能差异。

## Figure 3

```text
fig3_origin_publisher_mae.csv
fig3_origin_win_rates.csv
```

正文 Figure 3 只比较 DCRNN、STGCN、STaR-GNN；这是 baseline robustness，不是消融。

当前 paired win rates：

```text
24 h  vs DCRNN: 45/46
24 h  vs STGCN: 45/46
168 h vs DCRNN: 46/46
168 h vs STGCN: 40/46
```

## Figure 4

`fig4_dma_mae_improvement.csv`

40 个 DMA-horizon-baseline MAE comparisons 全部为正，范围约 1.26%--61.20%。

## Figure 5

```text
fig5_representative_168h_selection.json
fig5_representative_168h_trajectory.csv
```

代表样本按预先固定的 median-error proximity 规则选择；当前 `common_index=70`。

## Final audit

`manuscript_empirical_figure_audit.json` 汇总：

- Figure 2 四模型消融集合；
- Day-7 degradation；
- Full-vs-SAS moving-block bootstrap；
- Figure 3 paired win rates；
- interpretation guardrails。

CSV/JSON 保留完整精度；正文表格统一显示 3 位小数。
