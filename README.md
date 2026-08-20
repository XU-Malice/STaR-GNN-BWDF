# STaR-GNN：多 DMA 小时需水预测

[English](README_EN.md)｜[文档索引](docs/README.md)｜[完整复现教程](docs/FULL_PIPELINE_CN.md)｜[方法说明](docs/METHOD_CN.md)｜[最终结果](docs/RESULTS_AND_ARTIFACTS_CN.md)｜[最终图表方案](docs/MANUSCRIPT_FIGURES_FINAL_CN.md)

本仓库是 STaR-GNN 的独立可复现实现，用于 10 个分区计量区域（DMA）的 24 h day-ahead 与 168 h week-ahead 联合需水预测。仓库包含 split-aware 数据预处理、仅训练期 Pearson 功能图、DCRNN/STGCN baseline、SAS-Norm 与 FA-DPR factorial ablation、冻结 checkpoint、common-46 Test 复评，以及 Journal of Hydrology 稿件所用表格、图件和审计工件。

> **Manuscript-facing 结果统一采用 publisher-compatible total 口径。** 内部 aggregate-demand MAE 与正文 total MAE 均保留，但不得混用。详细定义见 [`paper/tables/literature/METRIC_CONVENTIONS.md`](paper/tables/literature/METRIC_CONVENTIONS.md)。

## 1. 模型与消融

正式 factorial ablation 只有四个模型：

| Model | Internal variant | SAS-Norm | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` / `Base` | × | × |
| DCRNN + SAS-Norm | `dssn_sasr` / `State` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` / `FA-DPR` | × | ✓ |
| STaR-GNN | `full` / `Full` | ✓ | ✓ |

**STGCN 是独立图时空 baseline，不属于消融。**

## 2. 最终指标口径

与 Que et al. (2024) supplementary total 一致：

- total MAE = DMA A--J 十个 DMA-level MAE 之和；
- total MAPE/RMSE/NSE = 在 A--J 小时需求求和后的总需求序列上计算；
- main Test = `common_46`；
- Test `teacher forcing=0`；
- Test target 不参与训练、early stopping 或模块选择。

STaR-GNN manuscript-facing 结果：

| Horizon | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---:|---:|---:|---:|
| 24 h | **9.424** | **1.805** | **5.535** | **0.981** |
| 168 h | **12.234** | **2.014** | **6.161** | **0.976** |

内部 aggregate-demand MAE 为 `4.360841 / 4.919812`，仅用于总需求轨迹和运行解释。

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

168 h 的 publisher-compatible MAE 中，SAS-Norm-only 与 STaR-GNN 仅差 `0.025755`（约 `0.21%`）。这不是预测结果变化，而是 aggregate-demand MAE 与 sum-of-DMA MAE 两种合法聚合定义造成的细微排序差异。由于相邻 168 h forecast origins 高度重叠，最终审计使用 7-origin moving-block bootstrap；Full−SAS 的均值差 95% CI 跨过 0。因此仓库和论文均不把这一 0.21% 点估计解释为稳定性能差异。

## 4. 论文主要结果图

- **Figure 1**：STaR-GNN 相对各 baseline 的 MAE/MAPE/RMSE reduction 与 NSE gain；
- **Figure 2**：四模型 factorial ablation 的 Day 1--Day 7 module contribution 与 long-horizon degradation，**不含 STGCN**；
- **Figure 3**：DCRNN/STGCN/STaR-GNN 在 46 common origins 上的 ECDF robustness；
- **Figure 4**：STaR-GNN 相对 DCRNN/STGCN 的 DMA-level MAE improvement；
- **Figure 5**：按预先固定 median-error rule 选取的 representative 168 h trajectory。

详见 [`docs/MANUSCRIPT_FIGURES_FINAL_CN.md`](docs/MANUSCRIPT_FIGURES_FINAL_CN.md)。

## 5. 数据与图协议

- 数据期：2021-01-01 至 2023-03-05，小时分辨率；
- 训练截止：2022-12-15 23:00；Test 自 2022-12-16 开始；
- 10 DMA；history = 672 h；forecast horizons = 24/168 h；stride = 24 h；
- Pearson 图仅由训练期需求构建；
- negative correlation clip to zero；zero diagonal；no threshold / Top-K；
- random-walk normalization；static graph；24 h/168 h 共用；
- diffusion step `K=2`。

## 6. 冻结论文设置

```yaml
learning_rate: 0.0003
weight_decay: 0.0
cl_decay_steps: 500
state_loss_weight: 0.03
max_epochs: 100
seed: 0
```

模型公共设置包括 hidden=32、1 recurrent layer、batch=16、early stopping patience=15。

## 7. 推荐验证顺序

```bash
conda env create -f environment.yml
conda activate star-gnn-bwdf
python -m pip install -e .

bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

重新生成 manuscript-facing 表格与图件：

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

python scripts/reproduce/build_literature_figures.py \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --ablation-table paper/tables/literature/table_ablation_common46.csv \
  --dma-table paper/tables/literature/table_star_gnn_dma_common46.csv \
  --output paper/figures

python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820

python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures \
  --block-bootstrap-iterations 50000 \
  --block-bootstrap-length 7 \
  --block-bootstrap-seed 20260820
```

最终成功标志包括：

```text
Metric convention audit: PASS
Factorial ablation model-set audit: PASS (4 models, no STGCN)
Publisher-compatible factorial cell audit: 30/32 PASS
Refined manuscript Figure 2 and Figure 3: PASS
Figure 2 factorial-model audit: PASS (4 models, no STGCN)
```

## 8. 结果来源边界

总体比较中的 GRU、LSTM、MSNet 与 MSCMNet variants 为 Que et al. (2024) reported results；DCRNN、STGCN、STaR-GNN 为 common-46 复评。仓库不会把两类来源错误描述为完全相同代码条件下重训。

## 9. 更多文档

- [`docs/RESULTS_AND_ARTIFACTS_CN.md`](docs/RESULTS_AND_ARTIFACTS_CN.md)：最终结果与解释；
- [`docs/RESULT_CONSISTENCY_AUDIT_CN.md`](docs/RESULT_CONSISTENCY_AUDIT_CN.md)：代码/历史/指标交叉审计；
- [`docs/FULL_PIPELINE_CN.md`](docs/FULL_PIPELINE_CN.md)：完整复现；
- [`docs/METHOD_CN.md`](docs/METHOD_CN.md)：方法与源码映射；
- [`docs/PLOTTING_CN.md`](docs/PLOTTING_CN.md)：表图生成与检查。

原始数据不随仓库重新分发；获取方式见 `data/README.md`。
