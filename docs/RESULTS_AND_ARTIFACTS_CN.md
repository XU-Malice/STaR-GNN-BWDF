# 实验结果、来源与论文工件

本文档是当前仓库 **manuscript-facing 结果的唯一主入口**。环境、数据预处理、构图、训练和冻结 checkpoint 验证见 [`FULL_PIPELINE_CN.md`](FULL_PIPELINE_CN.md)；最终 Figure 1--5 设计见 [`MANUSCRIPT_FIGURES_FINAL_CN.md`](MANUSCRIPT_FIGURES_FINAL_CN.md)。

## 1. 模型与消融定义

| 论文名称 | 内部键 | SAS-Norm | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` / `Base` | × | × |
| DCRNN + SAS-Norm | `dssn_sasr` / `State` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` / `FA-DPR` | × | ✓ |
| STaR-GNN | `full` / `Full` | ✓ | ✓ |

**STGCN 是独立图时空 baseline，不属于消融。**

四个 factorial variants 共享同一数据、Pearson 图、DCRNN backbone、decoder、训练协议和 Test 协议，只改变 SAS-Norm / FA-DPR 是否启用。

## 2. 最终评价口径

正文总体比较与消融统一采用与 Que et al. (2024) supplementary total 一致的 publisher-compatible 口径：

- `total MAE` = DMA A--J 十个 DMA-level MAE 之和；
- `total MAPE/RMSE/NSE` = 在 A--J 小时需求先求和后得到的总需求序列上计算；
- 主 Test = `common_46`；
- Test `teacher forcing=0`；
- Test target 不参与训练、early stopping 或模块选择。

仓库同时保留 aggregate-demand MAE：

\[
MAE_{agg}=MAE\left(\sum_i\hat y_i,\sum_i y_i\right),
\]

仅用于运行解释和内部诊断。不要把它与 publisher-compatible MAE 混在同一横向比较中。

STaR-GNN：

| Horizon | aggregate-demand MAE | publisher-compatible MAE |
|---|---:|---:|
| 24 h | 4.360841 | 9.424199 |
| 168 h | 4.919812 | 12.233590 |

## 3. Table 1：总体模型比较

文件：`paper/tables/literature/table_literature_comparison_common46.*`

| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | GRU | 16.314 | 3.100 | 10.194 | 0.916 |
| 24 h | LSTM | 17.698 | 2.900 | 9.711 | 0.920 |
| 24 h | MSNet | 15.537 | 3.200 | 9.526 | 0.929 |
| 24 h | MSCMNet_WM | 14.790 | 2.700 | 7.924 | 0.957 |
| 24 h | MSCMNet_M | 14.912 | 2.800 | 8.111 | 0.954 |
| 24 h | MSCMNet_W | 14.471 | 2.600 | 7.586 | 0.959 |
| 24 h | DCRNN | 11.917 | 2.213 | 6.848 | 0.970 |
| 24 h | STGCN | 12.358 | 2.425 | 7.905 | 0.961 |
| 24 h | **STaR-GNN** | **9.424** | **1.805** | **5.535** | **0.981** |
| 168 h | GRU | 18.305 | 3.100 | 11.353 | 0.918 |
| 168 h | LSTM | 18.678 | 2.900 | 11.031 | 0.922 |
| 168 h | MSNet | 15.908 | 3.200 | 9.698 | 0.930 |
| 168 h | MSCMNet_WM | 15.290 | 2.700 | 8.097 | 0.957 |
| 168 h | MSCMNet_M | 15.405 | 2.800 | 8.395 | 0.953 |
| 168 h | MSCMNet_W | 14.950 | 2.600 | 7.756 | 0.960 |
| 168 h | DCRNN | 16.801 | 3.248 | 9.817 | 0.940 |
| 168 h | STGCN | 14.569 | 3.576 | 10.306 | 0.933 |
| 168 h | **STaR-GNN** | **12.234** | **2.014** | **6.161** | **0.976** |

### 总体 MAE 提升

- 24 h vs GRU/LSTM/MSNet/MSCMNet variants：34.9%--46.7%；
- 24 h vs DCRNN/STGCN：20.9%--23.7%；
- 168 h vs GRU/LSTM/MSNet/MSCMNet variants：18.2%--34.5%；
- 168 h vs DCRNN/STGCN：16.0%--27.2%。

**来源边界：** GRU/LSTM/MSNet/MSCMNet variants 为 Que et al. (2024) reported results；DCRNN、STGCN、STaR-GNN 为本仓库 common-46 复评。不能写成 9 个模型全部在完全相同训练条件下重训。

## 4. Table 2：factorial ablation

文件：`paper/tables/literature/table_ablation_common46.*`

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

### 168 h MAE 如何解释

SAS-Norm-only 的 publisher-compatible MAE 为 `12.207835`，STaR-GNN 为 `12.233590`，差值 `0.025755`，约为 SAS-Norm 的 `0.21%`。

这不是预测结果被改动，而是 aggregate-demand MAE 与 sum-of-DMA MAE 两种聚合定义的排序差异：

- aggregate-demand MAE：STaR-GNN `4.919812` < SAS-Norm `5.122511`；
- publisher-compatible MAE：SAS-Norm `12.207835` < STaR-GNN `12.233590`。

由于 168 h forecast origins 每隔 24 h 启动并强烈重叠，最终 Figure 2 审计使用 **7-origin moving-block bootstrap** 而不是把 46 origins 当作独立样本。Full−SAS 的均值差 CI 跨过 0，因此正文应表述为：

> 在 168 h publisher-compatible MAE 上，SAS-Norm-only 与完整 STaR-GNN 的点估计近似持平；完整模型同时取得更低的 MAPE、RMSE、更高的 NSE，以及更低的 aggregate-demand MAE 和更小的 Day-1-to-Day-7 相对误差增长。

不要写“STaR-GNN 在 168 h MAE 明显不如 SAS-Norm”，也不要为了排序修改或选择性四舍五入数值。

## 5. Figure 2：模块贡献与长时域行为

最终 Figure 2 不含 STGCN。

### Panel (a)

报告 SAS-Norm、FA-DPR、Full 相对 DCRNN 的 Day-wise publisher-compatible MAE reduction。

### Panel (b)

报告四个 factorial variants 相对各自 Day 1 的 MAE change。

Day 7 相对 Day 1：

- DCRNN：+38.25%；
- DCRNN + FA-DPR：+11.93%；
- DCRNN + SAS-Norm：+2.64%；
- STaR-GNN：+1.70%。

该图用于支持“长时域稳定性”与“模块互补性”，而不是放大 0.21% 的整体 MAE 点估计差异。

## 6. Figure 3：测试起点稳健性

只比较 DCRNN、STGCN、STaR-GNN：

- 24 h vs DCRNN：45/46；
- 24 h vs STGCN：45/46；
- 168 h vs DCRNN：46/46；
- 168 h vs STGCN：40/46。

这是 baseline robustness analysis，不是消融。

## 7. DMA-level 结果

Table 3：`paper/tables/literature/table_star_gnn_dma_common46.*`

Figure 4：`paper/figures/manuscript_fig4_dma_mae_improvement.*`

10 DMA × 2 horizons × 2 graph baselines 的 40 个 MAE comparisons 全部为正，提升范围约 1.26%--61.20%。改善幅度具有空间异质性。

## 8. Representative 168 h trajectory

Figure 5 使用固定 median-error proximity 规则选择 common index 70，而不是挑选最优样本：

- STGCN：14.653；
- DCRNN：15.517；
- STaR-GNN：12.182。

图中 aggregate-demand hourly error 与 publisher-compatible sum-of-DMA MAE 属于不同统计量，必须在 caption 中区分。

## 9. 冻结论文设置

`results/paper/frozen_v1/MANIFEST.json` 与 paper config：

```yaml
learning_rate: 0.0003
weight_decay: 0.0
cl_decay_steps: 500
state_loss_weight: 0.03
max_epochs: 100
seed: 0
```

共同设置还包括：

- 10 DMA；
- history = 672 h；
- horizons = 24/168 h；
- stride = 24 h；
- hidden = 32；
- RNN layers = 1；
- diffusion step K = 2；
- batch = 16；
- early stopping patience = 15；
- Test teacher forcing = 0。

## 10. Pearson 功能图

- 仅训练期 demand；
- Pearson positive functional dependency；
- negative correlation clip to zero；
- zero diagonal；
- no threshold / Top-K；
- no self-loop in adjacency；
- random-walk normalization；
- 24 h 与 168 h 共用同一静态图。

## 11. 最终自动生成与验证

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

成功时至少应看到：

```text
Metric convention audit: PASS
Factorial ablation model-set audit: PASS (4 models, no STGCN)
Publisher-compatible factorial cell audit: 30/32 PASS
Refined manuscript Figure 2 and Figure 3: PASS
Figure 2 factorial-model audit: PASS (4 models, no STGCN)
```

最终正文表格统一使用 3 位小数；审计 CSV/JSON 保留完整精度。
