# Manuscript Figure 1--5 审计工件

本目录保存最终正文 Figure 1--5 的可追溯 CSV/JSON。**它们由两阶段制图流程共同生成，不是全部只由一个脚本产生。**

## 1. 生成顺序

### Stage 1：从冻结预测生成基础审计数据与 Figure 1--5

```bash
python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820
```

成功标志：

```text
Manuscript scientific-figure audit: PASS
```

### Stage 2：根据 Stage 1 审计结果生成最终 Figure 2/3

```bash
python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures
```

成功标志：

```text
Refined manuscript Figure 2 and Figure 3: PASS
```

Stage 2 会覆盖 `paper/figures/` 中 Stage 1 的 Figure 2 和 Figure 3，并新增长时域退化率、paired win rate 和最终解释 guardrail 审计。

## 2. 文件说明

### Figure 1

- `fig1_relative_improvement.csv`
  - STaR-GNN 相对 8 个 baseline 的 MAE/MAPE/RMSE reduction；
  - NSE 使用绝对 gain，不做百分比改善；
  - `source` 字段区分 Que et al. (2024) reported results 与 common-46 re-evaluated baselines。

### Figure 2

- `fig2_day1_day7_publisher_mae_ci.csv`
  - 168 h 按 7 个 24 h day 切分；
  - 每个 origin/day 先计算 DMA A--J 的 MAE 再求和；
  - 保存 46 个 common origins 的均值和 bootstrap 95% CI。
- `fig2_day1_day7_publisher_mae_metadata.json`
  - 指标定义、bootstrap iterations 和 seed。
- `fig2_day1_day7_degradation.csv`
  - **Stage 2 生成**；
  - 保存各模型相对 Day 1 的 day-wise MAE 变化率。

当前 Day 7 vs Day 1：DCRNN +38.25%、FA-DPR +11.93%、SAS-Norm +2.64%、STaR-GNN +1.70%。

### Figure 3

- `fig3_origin_publisher_mae.csv`
  - 46 个 common origins 的逐 origin publisher-compatible MAE；
  - 包含 STGCN、DCRNN、SAS-Norm、FA-DPR、STaR-GNN 原始审计数据。
- `fig3_origin_win_rates.csv`
  - **Stage 2 生成**；
  - 正文 Figure 3 最终只比较 STaR-GNN vs DCRNN/STGCN 的 paired win rates。

当前 win rates：

- 24 h vs DCRNN：45/46（97.8%）
- 24 h vs STGCN：45/46（97.8%）
- 168 h vs DCRNN：46/46（100.0%）
- 168 h vs STGCN：40/46（87.0%）

### Figure 4

- `fig4_dma_mae_improvement.csv`
  - DMA A--J × 24/168 h × DCRNN/STGCN；
  - 保存 STaR-GNN 的 DMA-level MAE reduction。

当前 40/40 个比较均为正改善，但幅度存在空间异质性。

### Figure 5

- `fig5_representative_168h_selection.json`
  - 记录预先固定的样本选择规则；
  - 选择 STaR-GNN publisher-compatible 168 h MAE 最接近 46 origins 中位数的样本；
  - 保存 selected common index、median error 和模型误差。
- `fig5_representative_168h_trajectory.csv`
  - 保存该 origin 的 observed/predicted aggregate-demand 168 h trajectory；
  - Figure 5 下 panel 的逐小时 absolute error 也是 aggregate-demand 轨迹误差。

注意：Figure 5 的选择指标是 publisher-compatible sum-of-DMA MAE，而图中的 trajectory/error 是 aggregate demand；两种统计量用途不同。

### 最终解释 guardrail

- `manuscript_empirical_figure_audit.json`
  - **Stage 2 生成**；
  - 固定 Day 1--Day 7 退化率与 paired win rates；
  - 明确 168 h 下不应声称 Full 在 MAE 上严格优于 SAS-Norm-only。

## 3. 与正文表格的关系

本目录服务 Figure 1--5。正文 Table 1--3 位于：

```text
paper/tables/literature/
```

其中 publisher-compatible 口径见：

- `paper/tables/literature/METRIC_CONVENTIONS.md`

不要使用本目录的图件审计数据去重新定义 Table 1--3 的指标口径。

## 4. 禁止手工修改排序

这些 CSV/JSON 是论文结果的审计层。若发现图形或结论与数据不一致，应修改制图代码或正文解释，而不是手工修改 CSV/JSON 数值来形成预期排序。

完整作图教程：[`../../../docs/PLOTTING_CN.md`](../../../docs/PLOTTING_CN.md)。