# STaR-GNN-BWDF 文档索引

本目录同时包含方法、最终实验设计、结果工件、完整复现流程和历史图表记录。当前 Journal of Hydrology 稿件以 **claim-driven submission architecture** 为准，不再把旧 Figure 1--5 作为投稿版权威图件。

## 1. 新读者推荐顺序

1. [`../README.md`](../README.md) — 仓库入口与快速验证；
2. [`METHOD_CN.md`](METHOD_CN.md) — SAS-Norm、FA-DPR、Pearson 功能图与源码对应；
3. [`EXPERIMENT_DESIGN_FINAL_CN.md`](EXPERIMENT_DESIGN_FINAL_CN.md) — **最终实验问题、主表/主图和 Results 证据链**；
4. [`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md) — 上一版结果审计记录，仅作历史追溯；
5. [`PLOTTING_CN.md`](PLOTTING_CN.md) — 从冻结预测一次性生成 submission tables 与 Main Fig. 1--6；
6. [`FULL_PIPELINE_CN.md`](FULL_PIPELINE_CN.md) — 从环境、数据、图、训练到 Test/clean-room 的完整流程；
7. [`RELEASE_CN.md`](RELEASE_CN.md) — GitHub Release、冻结资产和独立验收。

## 2. 当前 manuscript-facing 权威入口

### 实验设计

[`EXPERIMENT_DESIGN_FINAL_CN.md`](EXPERIMENT_DESIGN_FINAL_CN.md)

正文结果结构：

```text
Table 1 + Main Fig. 1 → overall four-metric performance
Main Fig. 2 + Table S1 → all-model DMA-level performance breadth
Main Fig. 3 + Table S1 → strongest local competitor and horizon transition
Table 2 + Main Fig. 4 → factorial ablation + lead-time stability
Main Fig. 5 → forecast-origin and difficult-window robustness
Main Fig. 6 → week-ahead demand dynamics, implications, limitations
```

### 指标定义

[`../paper/tables/literature/METRIC_CONVENTIONS.md`](../paper/tables/literature/METRIC_CONVENTIONS.md)

正文 total MAE 采用 DMA A--J MAE 求和；MAPE/RMSE/NSE 在小时总需求序列上计算。STaR-GNN total MAE：24 h `9.424199`、168 h `12.233590`。

### 投稿显示表

```text
paper/tables/submission/table1_overall_performance.md
paper/tables/submission/table2_factorial_ablation.md
paper/tables/submission/tableS1_dma_metrics.md
paper/tables/submission/tableS2_dma_local_margin.md
paper/tables/submission/tableS3_forecast_origin_robustness.{md,csv}
```

源 CSV 保留完整精度，显示表统一 3 位小数。

### 投稿图

```text
paper/figures/submission/main_fig1_overall_performance.*
paper/figures/submission/main_fig2_dma_performance.*
paper/figures/submission/main_fig3_dma_local_margin.*
paper/figures/submission/main_fig4_ablation_leadtime.*
paper/figures/submission/main_fig5_origin_robustness.*
paper/figures/submission/main_fig6_week_ahead_dynamics.*
```

### Figure captions

[`../paper/captions/SUBMISSION_RESULT_FIGURE_CAPTIONS.md`](../paper/captions/SUBMISSION_RESULT_FIGURE_CAPTIONS.md)

## 3. 文档状态说明

| 文件 | 当前用途 | manuscript-facing |
|---|---|:---:|
| `METHOD_CN.md` | 最终方法名与源码对应 | ✓ |
| `EXPERIMENT_DESIGN_FINAL_CN.md` | **最终实验与证据链** | ✓ |
| `RESULTS_AND_ARTIFACTS_CN.md` | 上一版结果/工件审计记录 | 否 |
| `PLOTTING_CN.md` | 最终投稿作图教程 | ✓ |
| `FULL_PIPELINE_CN.md` | 完整复现与代码流转 | 部分 |
| `RELEASE_CN.md` | 发布与 clean-room | 工程文档 |
| `MANUSCRIPT_FIGURES_FINAL_CN.md` | 上一版五图设计，保留用于历史对照 | 否 |
| `MANUSCRIPT_FIGURES_CN.md` | 更早期图表设计 | 否 |

`FULL_PIPELINE_CN.md` 中的内部兼容标签：

- `State` / `dssn_sasr` → **SAS-Norm**
- `FA-DPR` / `fa_dpr` → **FA-DPR**
- `Full` / `full` → **STaR-GNN**
- `Base` / `backbone` → **DCRNN**

## 4. 两套 MAE 不要混用

### 正文 total MAE

```text
MAE_total = sum(DMA A--J MAE)
```

STaR-GNN：24 h `9.424199`；168 h `12.233590`。

### 内部 aggregate-demand MAE

```text
MAE_agg = MAE(sum prediction, sum target)
```

STaR-GNN：24 h `4.360841`；168 h `4.919812`。

后者用于系统总需求轨迹诊断和 legacy `test_*` 工件，不用于正文跨模型 total MAE 排序。

## 5. 消融解释边界

factorial ablation 为 **30/32**。必须透明保留：

- FA-DPR 168 h MAPE 略差于 DCRNN；
- SAS-Norm-only 168 h MAE `12.207835` 略低于 STaR-GNN `12.233590`。

这两个边界不被隐藏；Main Fig. 4 用 ordered moving-block evidence 说明 Full 与 SAS 的 168 h MAE 点估计接近，同时揭示 SAS-Norm 与 FA-DPR 对四指标和 lead-time stability 的不同作用。

## 6. 当前 Results 证据链

```text
Overall capability
  Table 1 + Main Fig. 1
      ↓
Spatial breadth across DMAs
  Main Figs. 2--3 + Table S1
      ↓
Component mechanism
  Table 2 + Main Fig. 4
      ↓
Forecast-origin and difficult-condition robustness
  Main Fig. 5 + Table S3
      ↓
Population-to-instance week-ahead behavior
  Main Fig. 6
```

每个 subsection 采用：

> Claim → quantitative evidence → comparison → local interpretation → boundary → next question

而不是连续使用 “Table X shows / Figure Y shows”。

## 7. Legacy / 内部诊断入口

以下内容继续保留用于历史复现和补充分析，但不再是投稿版权威图表：

- `paper/reports/TEST_RESULTS_CN.md`
- `paper/tables/test_*`
- `paper/figures/test_*`
- `paper/figures/manuscript_fig1...5`
- `paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md`

最终投稿必须优先查看 `paper/tables/submission/`、`paper/figures/submission/` 与 `paper/captions/SUBMISSION_RESULT_FIGURE_CAPTIONS.md`。
