# Journal of Hydrology 最终论文作图教程

本文档说明如何从冻结 common-46 Test 工件重新生成当前正文 **Figure 1--5**，以及如何检查每张图对应的 CSV/JSON 证据。这里的目标不是“把结果画出来”而已，而是保证：

1. 图的指标口径与正文 Table 1--3 一致；
2. reported literature results 与 common-46 复评结果来源明确；
3. Day 1--Day 7、ECDF、DMA improvement 和代表样本选择均可复算；
4. 不通过手工挑样本或改 CSV 形成预期结论。

最终图表设计见 [`MANUSCRIPT_FIGURES_FINAL_CN.md`](MANUSCRIPT_FIGURES_FINAL_CN.md)，最终结果与数值见 [`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md)。

---

## 1. 先明确：正文图使用哪套 MAE

仓库存在两种 MAE，必须区分。

### 1.1 Publisher-compatible MAE：正文使用

\[
MAE_{publisher}=\sum_{i=A}^{J}MAE_i.
\]

即先分别计算 DMA A--J 的 MAE，再将十个 DMA MAE 求和。

正文总体比较、消融、Figure 1、Figure 2、Figure 3 的主统计量均使用该定义。

STaR-GNN：

- 24 h：`9.424199`
- 168 h：`12.233590`

### 1.2 Aggregate-demand MAE：内部诊断

\[
MAE_{agg}=MAE\left(\sum_i \hat y_i,\sum_i y_i\right).
\]

STaR-GNN：

- 24 h：`4.360841`
- 168 h：`4.919812`

这套值用于旧的总需求运行诊断、部分 legacy `test_*` 图和 Figure 5 轨迹下 panel 的逐小时 aggregate-demand absolute error，不用于正文跨模型 total MAE 排名。

详细定义见 [`../paper/tables/literature/METRIC_CONVENTIONS.md`](../paper/tables/literature/METRIC_CONVENTIONS.md)。

---

## 2. 作图前的必要输入

在仓库根目录执行。默认假设：

```text
results/paper/frozen_v1/
paper/tables/literature/table_literature_comparison_common46.csv
```

已经存在。

冻结目录至少需要 DCRNN、STGCN、SAS-Norm-only、FA-DPR-only、Full 的 common-46 predictions/evaluation 工件。可以先检查：

```bash
find results/paper/frozen_v1 -name predictions.npz | sort
```

以及：

```bash
ls -lh paper/tables/literature/table_literature_comparison_common46.csv
```

如果正文表格尚未重建，先执行：

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout
```

期望看到：

```text
Metric convention audit: PASS
Publisher-compatible ablation audit: 30/32 PASS
```

`30/32` 是真实最终结果，不是错误。两个例外为：

1. FA-DPR 168 h MAPE 略差于 DCRNN；
2. SAS-Norm-only 168 h publisher-compatible MAE 略低于 Full。

---

## 3. 为什么最终作图分成两个阶段

当前正文图不是单脚本一次性“画完即用”。原因是 Figure 2 和 Figure 3 在看到完整冻结统计后进行了科学问题层面的收敛：

- Figure 2 最终需要把“绝对长时域精度”和“相对 Day 1 的误差退化”分成两个 panel；
- Figure 3 最终只保留 DCRNN、STGCN、STaR-GNN，避免把 origin-level robustness 与消融问题混在同一张 ECDF 中。

因此：

```text
Stage 1: build_manuscript_results_figures.py
    ↓
生成基础 Figure 1--5 + 完整审计 CSV/JSON
    ↓
Stage 2: refine_manuscript_results_figures.py
    ↓
覆盖最终 Figure 2/3 + 生成 degradation / win-rate / guardrail 审计
```

**最终提交论文时必须完成 Stage 2。**

---

## 4. Stage 1：生成基础 Figure 1--5 与审计数据

执行：

```bash
python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820
```

成功时应看到类似：

```text
Manuscript figures: .../paper/figures
Manuscript audit tables: .../paper/tables/manuscript
Manuscript scientific-figure audit: PASS
```

### 4.1 为什么 bootstrap seed 固定

Figure 2 的 Day 1--Day 7 95% CI 使用 nonparametric bootstrap。固定：

```text
iterations = 5000
seed = 20260820
```

是为了让不同服务器和不同时间重新生成时 CI 可复现，而不是每次图形略有变化。

### 4.2 Stage 1 生成的正文图

```text
paper/figures/manuscript_fig1_relative_improvement.png
paper/figures/manuscript_fig1_relative_improvement.pdf
paper/figures/manuscript_fig2_day1_day7_publisher_mae.png
paper/figures/manuscript_fig2_day1_day7_publisher_mae.pdf
paper/figures/manuscript_fig3_origin_ecdf.png
paper/figures/manuscript_fig3_origin_ecdf.pdf
paper/figures/manuscript_fig4_dma_mae_improvement.png
paper/figures/manuscript_fig4_dma_mae_improvement.pdf
paper/figures/manuscript_fig5_representative_168h_trajectory.png
paper/figures/manuscript_fig5_representative_168h_trajectory.pdf
```

其中 Figure 2/3 还是基础版，下一阶段会覆盖。

---

## 5. Stage 2：生成最终 Figure 2 和 Figure 3

执行：

```bash
python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures
```

成功时应看到：

```text
Refined manuscript Figure 2 and Figure 3: PASS
```

并打印 Day 7 相对 Day 1 的 MAE 变化，以及 paired win rates。

当前冻结结果应为：

```text
Day 7 MAE change relative to Day 1:
  DCRNN: +38.25%
  DCRNN + SAS-Norm: +2.64%
  DCRNN + FA-DPR: +11.93%
  STaR-GNN: +1.70%

Paired STaR-GNN win rates across common origins:
  24h vs DCRNN: 45/46 (97.8%)
  24h vs STGCN: 45/46 (97.8%)
  168h vs DCRNN: 46/46 (100.0%)
  168h vs STGCN: 40/46 (87.0%)
```

若这些值发生明显变化，应先检查输入 predictions、common indices、metric convention 或代码版本，而不是直接接受新图。

---

## 6. Figure 1：Relative performance improvement

输出：

```text
manuscript_fig1_relative_improvement.{png,pdf}
```

### 6.1 Panel (a)：误差指标相对降低率

对 MAE/MAPE/RMSE：

\[
\Delta E=\frac{E_{baseline}-E_{STaR}}{E_{baseline}}\times100\%.
\]

正值表示 STaR-GNN 误差更低。

### 6.2 Panel (b)：NSE gain

NSE 不做百分比改善，而使用：

\[
\Delta NSE=NSE_{STaR}-NSE_{baseline}.
\]

### 6.3 来源标记

审计文件：

```text
paper/tables/manuscript/fig1_relative_improvement.csv
```

其中 `source` 字段必须区分：

- `reported_Que_et_al_2024`
- `common_46_re_evaluated`

GRU/LSTM/MSNet/MSCMNet variants 在图中用 `†` 标记为 reported results。不能把这些文献值描述为本仓库统一条件重训。

### 6.4 快速检查

```bash
column -s, -t < paper/tables/manuscript/fig1_relative_improvement.csv | head -20
```

所有纳入 baseline 的 MAE/MAPE/RMSE reduction 应为正，NSE gain 也应为正。

---

## 7. Figure 2：168 h Day 1--Day 7 长时域行为

输出：

```text
manuscript_fig2_day1_day7_publisher_mae.{png,pdf}
```

### 7.1 Panel (a)：绝对 publisher-compatible MAE

只比较：

- DCRNN
- STGCN
- STaR-GNN

对每个 common origin、每个预测日：

1. 截取对应 24 h；
2. 分别计算 A--J 的 day-wise MAE；
3. 十个 DMA MAE 求和，得到 publisher-compatible daily MAE；
4. 对 46 个 origins 求均值；
5. bootstrap 得到 95% CI。

审计文件：

```text
fig2_day1_day7_publisher_mae_ci.csv
fig2_day1_day7_publisher_mae_metadata.json
```

### 7.2 Panel (b)：相对 Day 1 的误差变化率

\[
\Delta_d=\frac{MAE_d-MAE_{Day1}}{MAE_{Day1}}\times100\%.
\]

比较：

- DCRNN
- DCRNN + SAS-Norm
- DCRNN + FA-DPR
- STaR-GNN

审计文件：

```text
fig2_day1_day7_degradation.csv
```

### 7.3 正确的论文解释

数据支持：

- DCRNN 随 horizon 延长出现明显 MAE 漂移；
- FA-DPR-only 降低了退化速度；
- SAS-Norm-only 把长时域 MAE 漂移压到很低；
- Full 的 Day 1--Day 7 变化也接近稳定，并在 168 h MAPE/RMSE/NSE 上给出最佳综合结果。

**不能写：**

- “FA-DPR 单独消除了长时域误差累积”；
- “Full 在 Day 1--Day 7 每一天都严格优于 SAS-Norm-only”；
- “Full 在 168 h 所有指标严格最优”。

---

## 8. Figure 3：46 个 common origins 的 ECDF

输出：

```text
manuscript_fig3_origin_ecdf.{png,pdf}
```

正文最终只显示：

- DCRNN
- STGCN
- STaR-GNN

每个 origin 的 x 值：

\[
MAE_s^{publisher}=\sum_i MAE_{s,i}.
\]

曲线越靠左，表示在相同累计概率下误差更低。

原始审计：

```text
fig3_origin_publisher_mae.csv
```

paired win rates：

```text
fig3_origin_win_rates.csv
```

当前必须复现：

- 24 h vs DCRNN：45/46
- 24 h vs STGCN：45/46
- 168 h vs DCRNN：46/46
- 168 h vs STGCN：40/46

这张图支持“平均改善不是少数有利测试窗口造成的”。特别是 168 h 下，STaR-GNN 在全部 46 个 origins 上均低于 DCRNN。

不要用 Figure 3 宣称 Full 对 SAS-Norm-only 也是 46/46；实际 168 h paired win rate 不是如此，消融关系应由 Table 2 与 Figure 2 解释。

---

## 9. Figure 4：DMA-level spatial consistency

输出：

```text
manuscript_fig4_dma_mae_improvement.{png,pdf}
```

对 DMA `i` 和 baseline `b`：

\[
\Delta MAE_{i,b}=\frac{MAE_{i,b}-MAE_{i,STaR}}{MAE_{i,b}}\times100\%.
\]

列：

- 24 h vs DCRNN
- 24 h vs STGCN
- 168 h vs DCRNN
- 168 h vs STGCN

行：DMA A--J。

审计：

```text
fig4_dma_mae_improvement.csv
```

当前 40/40 个比较都应为正，范围约 1.26%--61.20%。

正确表述是：

> consistent positive improvements across all DMAs, with heterogeneous improvement magnitudes

而不是“所有 DMA 均同等幅度大幅提升”。

---

## 10. Figure 5：代表性 168 h trajectory

输出：

```text
manuscript_fig5_representative_168h_trajectory.{png,pdf}
```

### 10.1 固定选择规则

从 46 个 common origins 中选择：

\[
s^*=\arg\min_s\left|MAE_s-\operatorname{median}(MAE)\right|.
\]

其中 MAE 是 STaR-GNN 的 publisher-compatible 168 h MAE。

这避免按“图看起来最好”挑样本。

选择审计：

```text
fig5_representative_168h_selection.json
```

当前应为：

```text
selected_origin_position_zero_based = 42
selected_common_index = 70
STGCN MAE = 14.653121...
DCRNN MAE = 15.516927...
STaR-GNN MAE = 12.182450...
```

trajectory 数据：

```text
fig5_representative_168h_trajectory.csv
```

### 10.2 图中上下 panel 的统计量不同

- 上 panel：A--J 求和后的 aggregate-demand observed/predicted trajectory；
- 下 panel：aggregate-demand 的逐小时 absolute error；
- 样本选择：publisher-compatible sum-of-DMA MAE。

三者用途不同，因此 caption 必须明确。

---

## 11. 最终审计文件

Stage 2 最后生成：

```text
paper/tables/manuscript/manuscript_empirical_figure_audit.json
```

可直接查看：

```bash
cat paper/tables/manuscript/manuscript_empirical_figure_audit.json
```

它固定两类解释 guardrails：

1. Figure 2 的 Day-7-vs-Day-1 退化率；
2. Figure 3 的 paired win rates。

特别明确：**SAS-Norm 是低长时域 MAE 的主要贡献者，不允许把 Full 描述成 168 h MAE 上对 SAS-Norm-only 的严格全面支配。**

---

## 12. 如何确认最终图确实是最新版本

执行：

```bash
ls -lh --time-style=long-iso paper/figures/manuscript_fig*
```

Figure 2 和 Figure 3 的时间应不早于 Stage 2 执行时间。

检查全部最终文件：

```bash
find paper/figures -maxdepth 1 -type f -name 'manuscript_fig*' | sort
find paper/tables/manuscript -maxdepth 1 -type f | sort
```

应有 5 张图 × PNG/PDF，共 10 个图文件，以及对应 CSV/JSON/README 审计工件。

---

## 13. 只重新画 Figure 2 和 Figure 3

如果 Stage 1 的审计 CSV/JSON 已存在，且没有修改冻结 predictions 或 Table 1，可以只执行：

```bash
python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures
```

这不会重新训练，也不会重新读取模型 checkpoint；只从现有审计数据重新构建最终 Fig. 2/3。

如果改变了冻结预测、common indices 或 overall table，则必须先重新运行 Stage 1。

---

## 14. 旧图应该怎么用

仓库仍保留：

```text
test_overall_24h.*
test_overall_168h.*
test_ablation_24h.*
test_ablation_168h.*
test_star_gnn_dma_metrics.*
test_day1_day7_models.*
test_day1_day7_ablation.*
test_dma_mae_24h.*
test_dma_mae_168h.*
pearson_correlation_heatmap.*
```

当前建议：

- absolute overall/ablation/DMA bar figures → Supplementary / 参考；
- legacy Day 1--Day 7 aggregate-demand figures → 内部诊断；
- Pearson heatmap → 方法/数据分析或 Supplementary，视论文版面安排；
- 正文结果主图只使用 `manuscript_fig1...5`。

旧图没有“错”，但回答的科学问题和指标层级与最终正文不同。

---

## 15. 常见问题

### Q1：为什么我看到 STaR-GNN MAE 是 4.36，而正文写 9.424？

因为你看的 4.36 是 aggregate-demand MAE；正文 9.424 是 publisher-compatible sum-of-DMA MAE。两者都可复现，但用途不同。

### Q2：为什么消融不是 31/32？

31/32 是旧 aggregate-demand 层级的诊断关系。最终 publisher-compatible 消融是 30/32，因为还存在“168 h SAS-Norm-only MAE 略低于 Full”这一真实例外。

### Q3：为什么 Figure 2 的数值和旧 `test_day1_day7_models.*` 不一样？

旧图按 aggregate-demand 重算；最终 Figure 2 按 publisher-compatible daily sum-of-DMA MAE 计算，并在 46 origins 上给出 bootstrap CI。

### Q4：为什么 Figure 3 不画 SAS-Norm 和 FA-DPR？

最终正文 Figure 3 的问题是“相对主要图基线的 origin-level 稳健性”。消融组件已经由 Table 2 和 Figure 2 解释，再放入 ECDF 会混合两个科学问题。

### Q5：Figure 5 为什么不挑 STaR-GNN 最好的样本？

为了避免 cherry-picking。当前采用固定的中位误差邻近规则。

### Q6：能不能手工换 Figure 5 的 origin，让图更漂亮？

不能在看完图后按视觉效果更换。如果论文确有新的预先定义选择原则，必须先写明规则，再根据规则重新选择并记录审计 JSON。

### Q7：bootstrap CI 每次为什么不同？

使用本教程固定的 `--bootstrap-seed 20260820` 和 `--bootstrap-iterations 5000`。若仍不同，检查 NumPy/代码版本和输入预测是否一致。

### Q8：脚本提示缺少 predictions.npz？

先执行冻结 checkpoint 验证：

```bash
bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

然后确认 `results/paper/frozen_v1/` 中对应 evaluation 目录和 predictions 存在。

---

## 16. 推荐的最终提交前检查

```bash
python -m py_compile \
  scripts/reproduce/build_manuscript_results_figures.py \
  scripts/reproduce/refine_manuscript_results_figures.py

python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820

python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures

cat paper/tables/manuscript/manuscript_empirical_figure_audit.json
ls -lh paper/figures/manuscript_fig*
git status --short
```

如果两个脚本均 PASS、audit 数值与冻结结果一致、5 张图全部存在，再将 PNG/PDF 与 `paper/tables/manuscript/` 一起提交。

---

## 17. 文档权威层级

当不同历史文件存在表述差异时，按以下顺序判断：

1. `paper/tables/literature/METRIC_CONVENTIONS.md` — 指标定义；
2. `paper/tables/literature/table_*` — 正文精确数值；
3. `paper/tables/manuscript/*.csv/json` — Figure 1--5 审计；
4. `docs/MANUSCRIPT_FIGURES_FINAL_CN.md` — 最终图表科学问题与解释边界；
5. 本文 `PLOTTING_CN.md` — 最终作图操作流程；
6. legacy `test_*` 图表与 `paper/reports/TEST_RESULTS_CN.md` — 内部诊断/历史工件。

任何正文数字和结论都应能够沿这条链追溯回冻结 common-46 预测。