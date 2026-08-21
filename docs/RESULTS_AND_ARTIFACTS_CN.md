# 实验结果、来源与投稿工件

本文档记录当前冻结结果的**数值事实、来源边界与投稿工件位置**。最终实验问题和 Results 证据链见 [`EXPERIMENT_DESIGN_FINAL_CN.md`](EXPERIMENT_DESIGN_FINAL_CN.md)；投稿作图见 [`PLOTTING_CN.md`](PLOTTING_CN.md)。

> **投稿版权威入口：** `paper/tables/submission/`、`paper/figures/submission/`、`paper/figures/supplementary/` 和 `paper/tables/manuscript/submission/`。旧 `manuscript_fig1...5` / `test_*` 工件仅用于历史复现和内部诊断。

## 1. 模型与消融定义

| 论文名称 | 内部键 | SAS-Norm | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` / `Base` | × | × |
| DCRNN + SAS-Norm | `dssn_sasr` / `State` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` / `FA-DPR` | × | ✓ |
| STaR-GNN | `full` / `Full` | ✓ | ✓ |

**STGCN 是独立 graph baseline，不属于 factorial ablation。**

四个消融变体共享同一数据、Pearson 功能图、DCRNN backbone、decoder、训练和 Test 协议，仅改变 SAS-Norm / FA-DPR 是否启用。

## 2. 正文指标口径

总体比较和消融采用 publisher-compatible 口径：

- `total MAE` = DMA A--J 十个 DMA-level MAE 之和；
- `total MAPE/RMSE/NSE` = 在 A--J 小时需求先求和后的系统总需求序列上计算；
- primary Test = `common_46`；
- Test `teacher forcing=0`；
- Test target 不参与训练、early stopping 或模块选择。

内部同时保留：

\[
MAE_{agg}=MAE\left(\sum_i\hat y_i,\sum_i y_i\right),
\]

作为 aggregate-demand trajectory diagnostic，不用于正文横向 total MAE 排序。

STaR-GNN：

| Horizon | aggregate-demand MAE | publisher-compatible MAE |
|---|---:|---:|
| 24 h | 4.360841 | 9.424199 |
| 168 h | 4.919812 | 12.233590 |

## 3. Main Table 1 — Overall forecasting performance

显示版：

```text
paper/tables/submission/table1_overall_performance.md
```

全精度来源：

```text
paper/tables/literature/table_literature_comparison_common46.csv
```

| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | GRU† | 16.314 | 3.100 | 10.194 | 0.916 |
| 24 h | LSTM† | 17.698 | 2.900 | 9.711 | 0.920 |
| 24 h | MSNet† | 15.537 | 3.200 | 9.526 | 0.929 |
| 24 h | MSCMNet-WM† | 14.790 | 2.700 | 7.924 | 0.957 |
| 24 h | MSCMNet-M† | 14.912 | 2.800 | 8.111 | 0.954 |
| 24 h | MSCMNet-W† | 14.471 | 2.600 | 7.586 | 0.959 |
| 24 h | DCRNN | 11.917 | 2.213 | 6.848 | 0.970 |
| 24 h | STGCN | 12.358 | 2.425 | 7.905 | 0.961 |
| 24 h | **STaR-GNN** | **9.424** | **1.805** | **5.535** | **0.981** |
| 168 h | GRU† | 18.305 | 3.100 | 11.353 | 0.918 |
| 168 h | LSTM† | 18.678 | 2.900 | 11.031 | 0.922 |
| 168 h | MSNet† | 15.908 | 3.200 | 9.698 | 0.930 |
| 168 h | MSCMNet-WM† | 15.290 | 2.700 | 8.097 | 0.957 |
| 168 h | MSCMNet-M† | 15.405 | 2.800 | 8.395 | 0.953 |
| 168 h | MSCMNet-W† | 14.950 | 2.600 | 7.756 | 0.960 |
| 168 h | DCRNN | 16.801 | 3.248 | 9.817 | 0.940 |
| 168 h | STGCN | 14.569 | 3.576 | 10.306 | 0.933 |
| 168 h | **STaR-GNN** | **12.234** | **2.014** | **6.161** | **0.976** |

† 为 Que et al. (2024) reported results；DCRNN、STGCN 和 STaR-GNN 为 common-46 复评，两类来源不得描述为完全相同代码条件下重训。

MAE 提升：

- 24 h vs sequence/multiscale reported models：约 34.9%--46.7%；
- 24 h vs DCRNN/STGCN：约 20.9%--23.7%；
- 168 h vs sequence/multiscale reported models：约 18.2%--34.5%；
- 168 h vs DCRNN/STGCN：约 16.0%--27.2%。

## 4. Main Table 2 — Factorial ablation

显示版：

```text
paper/tables/submission/table2_factorial_ablation.md
```

全精度来源：

```text
paper/tables/literature/table_ablation_common46.csv
paper/tables/literature/table_ablation_audit.json
```

| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | DCRNN | 11.917 | 2.213 | 6.848 | 0.970 |
| 24 h | DCRNN + SAS-Norm | 10.468 | 2.010 | 6.134 | 0.976 |
| 24 h | DCRNN + FA-DPR | 11.238 | 1.945 | 6.079 | 0.977 |
| 24 h | **STaR-GNN** | **9.424** | **1.805** | **5.535** | **0.981** |
| 168 h | DCRNN | 16.801 | 3.248 | 9.817 | 0.940 |
| 168 h | **DCRNN + SAS-Norm** | **12.208** | 2.102 | 6.468 | 0.974 |
| 168 h | DCRNN + FA-DPR | 14.086 | 3.278 | 9.332 | 0.945 |
| 168 h | STaR-GNN | 12.234 | **2.014** | **6.161** | **0.976** |

### 168 h Full vs SAS-Norm 的解释边界

- publisher-compatible MAE：SAS `12.207835` < Full `12.233590`，差 `0.025755`（约 `0.21%`）；
- aggregate-demand MAE：Full `4.919812` < SAS `5.122511`；
- Full 的 168 h MAPE、RMSE、NSE 更优；
- ordered 7-origin moving-block bootstrap 对 Full−SAS publisher-MAE 均值差给出的 95% CI 跨过 0。

因此正文应写为：**两者在 168 h sum-of-DMA MAE 的点估计近似持平；SAS-Norm 是绝对 MAE 改善的主要贡献模块，而完整模型在相对误差、较大误差、整体拟合和 lead-time stability 上更均衡。**

不要写“STaR-GNN 168 h MAE 明显不如 SAS-Norm”，也不要为了排序修改或选择性四舍五入数值。

## 5. Main Figure 1 — Ablation mechanism and lead-time stability

路径：

```text
paper/figures/submission/main_fig1_ablation_leadtime.*
```

### Panel a

四个 factorial variants 的 Day 1--Day 7 absolute publisher-compatible MAE，使用 ordered 7-origin moving-block bootstrap 95% CI。

### Panel b

相对各自 Day 1 的 MAE change：

- DCRNN：Day 7 约 `+38.25%`；
- DCRNN + FA-DPR：约 `+11.93%`；
- DCRNN + SAS-Norm：约 `+2.64%`；
- STaR-GNN：约 `+1.70%`。

核心 inference：**SAS-Norm 是周尺度绝对 MAE 改善的主要来源；FA-DPR 更明显地抑制随预测提前期增加的误差累积；完整模型形成最稳定的 Day-1-to-Day-7 误差演化。**

## 6. Main Figure 2 — Temporal and spatial robustness

路径：

```text
paper/figures/submission/main_fig2_temporal_spatial_robustness.*
```

### Panel a — paired forecast-origin improvement

同一 forecast origin 直接计算：

\[
\Delta MAE_s=MAE_{baseline,s}-MAE_{STaR,s}.
\]

预期 win counts：

- 24 h vs DCRNN：45/46；
- 24 h vs STGCN：45/46；
- 168 h vs DCRNN：46/46；
- 168 h vs STGCN：40/46。

### Panel b — DMA robustness

10 DMA × 2 horizons × 2 graph baselines = 40 comparisons，全部为正 MAE reduction，范围约 `1.26%--61.20%`。

核心 inference：**平均改善并非由少数有利日期或少数 DMA 驱动；改善方向在时间和空间上均一致，但幅度具有空间异质性。**

## 7. Main Figure 3 — Week-ahead demand dynamics

路径：

```text
paper/figures/submission/main_fig3_week_ahead_dynamics.*
```

采用 population-to-instance 证据链：

- Panel a：46 origins × 7 forecast days 折叠到 24 h 日内周期后的 aggregate-demand absolute-error profile + moving-block CI；
- Panel b：median-error proximity rule 选出的 representative 168 h aggregate-demand trajectory；
- Panel c：同一 origin 的 hourly absolute error。

代表性 origin 仍为预先规定规则选择，不依据视觉效果挑选；历史审计对应 common index 70，publisher-compatible MAE 约为 STGCN `14.653`、DCRNN `15.517`、STaR-GNN `12.182`。

需求和 aggregate-demand absolute error 单位：`L s⁻¹`。

## 8. Supplementary evidence

### Table S1 — DMA-level detailed metrics

```text
paper/tables/submission/tableS1_dma_metrics.md
```

### Figure S1 — Relative improvement over all baselines

```text
paper/figures/supplementary/supp_figS1_relative_improvement.*
```

用于概括不同 model families 和 horizons 上的 relative improvement，不再承担主文独立 claim。

### Figure S2 — Per-origin ECDF

```text
paper/figures/supplementary/supp_figS2_origin_ecdf.*
```

作为 Main Fig. 2a paired-difference analysis 的 distributional reassurance。

## 9. Submission figure audit

```text
paper/tables/manuscript/submission/
```

关键工件：

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

主结论必须能够由这些 CSV/JSON 独立复算，而不是只依赖 PNG。

## 10. 冻结论文设置

```yaml
learning_rate: 0.0003
weight_decay: 0.0
cl_decay_steps: 500
state_loss_weight: 0.03
max_epochs: 100
seed: 0
```

共同设置：10 DMA；history `672 h`；horizons `24/168 h`；stride `24 h`；hidden `32`；1 recurrent layer；diffusion `K=2`；batch `16`；patience `15`；Test teacher forcing `0`。

## 11. Pearson 功能图

- 仅使用训练期 demand；
- Pearson positive functional dependency；
- negative correlation clip to zero；zero diagonal；
- no threshold / Top-K；no adjacency self-loop；
- random-walk normalization；
- 24 h 与 168 h 共用同一静态图。

## 12. Canonical generation

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

python scripts/reproduce/render_submission_tables.py

python scripts/reproduce/render_submission_figures.py \
  --release results/paper/frozen_v1 \
  --block-length 7 \
  --bootstrap-iterations 50000 \
  --bootstrap-seed 20260821
```

成功时：

```text
Metric convention audit: PASS
Factorial ablation model-set audit: PASS (4 models, no STGCN)
Publisher-compatible factorial cell audit: 30/32 PASS
Submission tables: PASS
Submission figure renderer: PASS
```

旧 `paper/figures/manuscript_fig1...5`、`test_*` 和 `MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md` 继续保留用于历史复现/诊断，但不再是投稿版权威工件。
