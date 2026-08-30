# STaR-GNN：多 DMA 小时需水预测

[English](README_EN.md)｜[文档索引](docs/README.md)｜[方法说明](docs/METHOD_CN.md)｜[最终实验设计](docs/EXPERIMENT_DESIGN_FINAL_CN.md)｜[结果与工件](docs/RESULTS_AND_ARTIFACTS_CN.md)｜[投稿作图教程](docs/PLOTTING_CN.md)

本仓库是 STaR-GNN 的独立可复现实现，用于 10 个分区计量区域（DMA）的 24 h day-ahead 与 168 h week-ahead 联合需水预测。仓库包含 split-aware 数据预处理、仅训练期 Pearson 功能图、DCRNN/STGCN baseline、SAS-Norm 与 FA-DPR factorial ablation、冻结 checkpoint、common-46 Test 复评，以及 Journal of Hydrology 稿件所用的投稿表图和审计工件。

> **Manuscript-facing 结果统一采用 total 口径。** 内部 aggregate-demand MAE 与正文 total MAE 均保留，但不得混用。详见 [`paper/tables/literature/METRIC_CONVENTIONS.md`](paper/tables/literature/METRIC_CONVENTIONS.md)。

## 1. 正式 factorial ablation

| Model | Internal variant | SAS-Norm | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` / `Base` | × | × |
| DCRNN + SAS-Norm | `dssn_sasr` / `State` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` / `FA-DPR` | × | ✓ |
| STaR-GNN | `full` / `Full` | ✓ | ✓ |

**STGCN 是独立图时空 baseline，不属于消融。**

## 2. Manuscript metric convention

与 Que et al. (2024) supplementary total 一致：

- total MAE = DMA A--J 十个 DMA-level MAE 之和；
- total MAPE/RMSE/NSE = 在 A--J 小时总需求序列上计算；
- primary Test = `common_46`；
- Test `teacher forcing=0`；
- Test target 不参与训练、early stopping 或模块选择。

STaR-GNN：

| Horizon | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---:|---:|---:|---:|
| 24 h | **9.424** | **1.805** | **5.535** | **0.981** |
| 168 h | **12.234** | **2.014** | **6.161** | **0.976** |

内部 aggregate-demand MAE 为 `4.360841 / 4.919812`，仅用于系统总需求轨迹和运行诊断。

## 3. 四模型消融结果

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

168 h total MAE 中，SAS-Norm-only 与 STaR-GNN 仅差 `0.025755`（约 `0.21%`）。相邻 week-ahead forecast origins 高度重叠，因此采用 ordered 7-origin moving-block bootstrap 限定解释；该差异的均值区间跨过 0，不被表述为稳定优劣关系。

## 4. Journal of Hydrology 投稿版权威证据链

正文采用 **2 张主表 + 7 张主结果图**：

```text
Main Table 1 + Main Fig. 1  Overall four-metric performance
        ↓
Main Figs. 2–4 + Tables S1–S2  DMA-level breadth and local absolute performance
        ↓
Main Table 2 + Main Fig. 5   Factorial ablation + lead-time stability
        ↓
Main Fig. 6                  Forecast-origin + difficult-window robustness
        ↓
Main Fig. 7                  Population-to-instance week-ahead dynamics
```

### Main Figure 1 — Overall four-metric performance

- 相对六个时序模型与 DCRNN、STGCN 的 MAE/MAPE/RMSE 降幅；
- 对应的 NSE 绝对增益，24 h 与 168 h 并列。

### Main Figure 2 — DMA-level performance breadth

- STaR-GNN 相对八种基线的四指标逐 DMA 有符号改善分布；
- 保留全部 DMA 点，并用中位数和四分位距概括跨 DMA 覆盖。

### Main Figure 3 — 24 h DMA-level absolute performance

- 四个独立子图分别呈现 MAE、MAPE、RMSE 和 NSE；
- 每个 DMA 直接比较 STaR-GNN 与局部最优基线的 24 h 绝对指标。

### Main Figure 4 — 168 h DMA-level absolute performance

- 与 Main Fig. 3 采用相同分面和颜色编码；
- 独立坐标突出周尺度下 DMA A、E、G 及 I–NSE 的局部例外。

### Main Figure 5 — Four-metric ablation and lead-time stability

- MAE、MAPE、RMSE、NSE 的逐日 paired improvement 与 moving-block 95% CI；
- 同日水平错位、marker 和 linestyle 共同解决 SAS-Norm 与 STaR-GNN 的视觉重合。

### Main Figure 6 — Forecast-origin and difficult-window robustness

- 同协议 DCRNN、STGCN、STaR-GNN 的 46-origin 四指标 paired effects；
- ordered 7-origin moving-block 95% CI；
- 仅由观测 normalized mean absolute ramp 定义的高波动四分位窗口。

### Main Figure 7 — Week-ahead demand dynamics

- 全部测试窗口 × 7 forecast days 的日内 aggregate-demand error profile；
- 预先固定 median-error rule 选出的 representative 168 h trajectory；
- 对应 hourly absolute error。需求单位为 `L s⁻¹`。

Supplementary Table S1 给出全部九种模型的 DMA A--J 详细四指标；Supplementary Table S2 给出 Figs. 3–4 中逐 DMA 局部最强竞争者及精确差异；Supplementary Table S3 给出 Fig. 6 的逐起点稳健性统计。

最终设计见 [`docs/EXPERIMENT_DESIGN_FINAL_CN.md`](docs/EXPERIMENT_DESIGN_FINAL_CN.md)。

## 5. Submission artifact paths

```text
paper/tables/submission/
  table1_overall_performance.md
  table2_factorial_ablation.md
  tableS1_dma_metrics.md
  tableS2_dma_local_margin.md
  tableS3_forecast_origin_robustness.{md,csv}

paper/figures/submission/
  main_fig1_overall_performance.{pdf,svg,png}
  main_fig2_dma_performance.{pdf,svg,png}
  main_fig3_dma_absolute_24h.{pdf,svg,png}
  main_fig4_dma_absolute_168h.{pdf,svg,png}
  main_fig5_ablation_leadtime.{pdf,svg,png}
  main_fig6_origin_robustness.{pdf,svg,png}
  main_fig7_week_ahead_dynamics.{pdf,svg,png}
```

图中 STaR-GNN 使用深蓝 `#0F4D92` 作为唯一 hero color；DCRNN/STGCN 使用灰度 baseline，SAS-Norm/FA-DPR 使用低饱和 variant colors。统一样式由 `scripts/reproduce/manuscript_plot_style.py` 管理。

## 6. 数据、图与冻结设置

- 数据期：2021-01-01 至 2023-03-05，小时分辨率；
- train 截止：2022-12-15 23:00；Test 自 2022-12-16；
- 10 DMA；history = 672 h；horizons = 24/168 h；stride = 24 h；
- Pearson 图仅使用训练期需求；negative correlation clip to zero；zero diagonal；no threshold / Top-K；
- random-walk normalization；static graph；24/168 h 共用；diffusion step `K=2`；
- hidden=32；1 recurrent layer；batch=16；patience=15；
- learning rate `3e-4`；weight decay `0`；CL decay `500`；state loss weight `0.03`；seed `0`。

## 7. 一键验证与生成投稿工件

```bash
conda env create -f environment.yml
conda activate star-gnn-bwdf
python -m pip install -e .

bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

只重建投稿表图：

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

最终图同时输出 PDF、editable SVG 和 300 dpi PNG。详细教程见 [`docs/PLOTTING_CN.md`](docs/PLOTTING_CN.md)。

## 8. 结果来源边界

总体比较中的 GRU、LSTM、MSNet 与 MSCMNet variants 为 Que et al. (2024) reported results；DCRNN、STGCN、STaR-GNN 为 common-46 复评。仓库不会将两类来源表述为完全相同代码条件下重训。

旧 `paper/figures/manuscript_fig1...5`、`paper/figures/test_*` 和旧 captions 继续保留用于历史复现/诊断，但不再是投稿版权威入口。
