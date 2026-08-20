# STaR-GNN 结果、消融与仓库一致性审计

本文件用于固定 2026-08-20 对当前 GitHub 仓库的交叉核对结果。审计依据包括：冻结 common-46 结果、逐 DMA/逐 origin 审计文件、当前评价代码、论文制表/制图脚本、冻结 MANIFEST、数据/图/训练配置，以及此前 aggregate-demand 结果记录。

## 1. 最重要的结论

### 1.1 正式消融实验只有四个模型

正式组件消融必须严格定义为：

1. DCRNN：`backbone` / `Base`；
2. DCRNN + SAS-Norm：`dssn_sasr` / `State`；
3. DCRNN + FA-DPR：`fa_dpr` / `FA-DPR`；
4. STaR-GNN：`full` / `Full`。

STGCN 是独立图时空 baseline，不属于 factorial ablation。当前 `tests/test_paper_artifacts.py` 和 `build_detailed_test_artifacts.py` 中的 `ABLATION_MODELS=(DCRNN, State, FA-DPR, Full)` 与这一设计一致。

当前仓库中 `build_paper_tables.py`、`build_literature_figures.py`、`MANUSCRIPT_FIGURES_FINAL_CN.md` 等后期 manuscript-facing 文件把 STGCN 混入了 ablation table/figure，这属于**实验组织错误**，不是冻结结果错误。后续应修正生成器，而不是只手工删表格行。

### 1.2 168 h 的“Full 是否全面超过 SAS-Norm”取决于 MAE 定义

仓库存在两套合法但用途不同的 MAE：

- aggregate-demand MAE：先把十个 DMA 的需求相加，再计算 MAE；
- publisher-compatible MAE：先分别计算 A--J 的 DMA MAE，再将十个 MAE 求和。

冻结预测没有因为指标口径切换而改变。

#### aggregate-demand MAE（旧内部/运行诊断）

| Model | 168 h MAE |
|---|---:|
| DCRNN + SAS-Norm | 5.1225108922 |
| STaR-GNN | **4.9198118610** |

因此在旧的 aggregate-demand 体系中，STaR-GNN 的 MAE 确实优于 SAS-Norm-only；结合 MAPE、RMSE、NSE，当时的 `Full vs State = 4/4` 是正确的。

#### publisher-compatible sum-of-DMA MAE（当前正文跨论文口径）

| Model | 168 h MAE |
|---|---:|
| DCRNN + SAS-Norm | **12.2078351150** |
| STaR-GNN | 12.2335903993 |

差值为 `+0.0257552843`，即 Full 的点估计约高 `0.21097%`。

该结果已通过两条独立路径核对：

1. `metrics_common_46.csv` 的 `total` 行；
2. `paper/tables/test_dma_mae_wide_168h.csv` 中 A--J 十个 DMA MAE 重新求和。

二者完全一致，因此这不是 `build_paper_tables.py` 手工写错数字。

### 1.3 这个 0.21% 差异不能写成有意义的“Full 更差”

基于 `fig3_origin_publisher_mae.csv` 中 46 个相同 common test origins 的配对结果：

- SAS-Norm-only 平均 publisher MAE：`12.2078351151`；
- STaR-GNN 平均 publisher MAE：`12.2335903994`；
- 平均配对差 `Full - SAS-Norm`：`+0.0257552843`；
- Full 更低：19/46 origins；SAS-Norm 更低：27/46 origins；
- 200,000 次配对 bootstrap（seed `20260820`）的均值差 95% CI：约 `[-0.0564, 0.1081]`；
- paired t-test：`p≈0.5466`；
- Wilcoxon signed-rank：`p≈0.5957`。

因此最严谨的 manuscript 解释应是：

> 在 168 h 的 publisher-compatible sum-of-DMA MAE 上，SAS-Norm-only 的点估计比完整模型低约 0.21%，但该差异在 46 个配对测试起点上没有显示出统计上的稳定优势；STaR-GNN 同时在 aggregate-demand MAE、MAPE、RMSE 和 NSE 上优于 SAS-Norm-only。

不要写“STaR-GNN 在 168 h MAE 明显不如 SAS-Norm”，也不要为了保持排序而把 `12.233590` 改成更小的数。

## 2. 最终四模型 publisher-compatible 消融表

### 24 h

| Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---:|---:|---:|---:|
| DCRNN | 11.917304 | 2.212928 | 6.848257 | 0.970419 |
| DCRNN + SAS-Norm | 10.467994 | 2.010448 | 6.133886 | 0.976269 |
| DCRNN + FA-DPR | 11.238099 | 1.944550 | 6.079036 | 0.976691 |
| STaR-GNN | **9.424199** | **1.804574** | **5.534656** | **0.980679** |

### 168 h

| Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---:|---:|---:|---:|
| DCRNN | 16.800744 | 3.248413 | 9.817428 | 0.939504 |
| DCRNN + SAS-Norm | **12.207835** | 2.102380 | 6.468312 | 0.973739 |
| DCRNN + FA-DPR | 14.085994 | 3.277716 | 9.332415 | 0.945334 |
| STaR-GNN | 12.233590 | **2.013774** | **6.160881** | **0.976176** |

STGCN 不出现在这张表中；其结果只属于总体 baseline comparison。

## 3. 31/32 与 30/32 为什么同时存在

旧 `test_hierarchy.json` 和早期 README 使用 aggregate-demand MAE，因此：

- Full vs SAS-Norm 在 168 h 是 4/4；
- 唯一例外是 FA-DPR 168 h MAPE 略差于 DCRNN；
- 对应旧内部关系为 31/32。

2026-08-20 后，manuscript-facing ablation 为与 Que et al. (2024) 补充表保持可比而采用 publisher-compatible sum-of-DMA MAE。此时额外出现 Full 168 h MAE 点估计略高于 SAS-Norm-only，因此按逐格方向计数变为 30/32。

这两个数字不能混称为同一个验收：

- `31/32`：legacy/internal aggregate-demand hierarchy；
- `30/32`：manuscript-facing publisher-compatible factorial cell audit。

## 4. Table 1 / Figure 1 审计

九模型总体比较的核心数值与 `configs/evaluation/mscmnet_literature_totals.yaml`、DCRNN/STGCN/STaR-GNN common-46 结果以及 `fig1_relative_improvement.csv` 对齐。

STaR-GNN publisher-compatible 总体结果：

| Horizon | MAE | MAPE (%) | RMSE | NSE |
|---|---:|---:|---:|---:|
| 24 h | 9.424199 | 1.804574 | 5.534656 | 0.980679 |
| 168 h | 12.233590 | 2.013774 | 6.160881 | 0.976176 |

MAE 相对提升重新计算后：

- 24 h vs GRU/LSTM/MSNet/MSCMNet variants：约 34.88%--46.75%；
- 24 h vs DCRNN/STGCN：约 20.92%--23.74%；
- 168 h vs GRU/LSTM/MSNet/MSCMNet variants：约 18.17%--34.50%；
- 168 h vs DCRNN/STGCN：约 16.03%--27.18%。

`fig1_relative_improvement.csv` 与上述计算一致。

注意：GRU/LSTM/MSNet/MSCMNet variants 是 Que et al. (2024) reported results；DCRNN、STGCN、STaR-GNN 是当前 common-46 复评。不得写成九个模型全部在同一代码条件下重训。

## 5. Table 3 / Figure 4 审计

`table_star_gnn_dma_common46.csv` 的 A--J 十个 MAE 求和可精确恢复 STaR-GNN 的 publisher-compatible total MAE：

- 24 h：`9.4241992052`；
- 168 h：`12.2335903993`。

`fig4_dma_mae_improvement.csv` 的 40 个比较（10 DMA × 2 horizons × DCRNN/STGCN）全部为正，范围约 `1.2611%--61.1981%`。因此“所有 DMA 均获得正 MAE 改善”这一结论可以保留，但不能写成“改善幅度均匀或都很大”。

## 6. Figure 2 / Figure 3 / Figure 5 审计

### Figure 2

当前 day-wise publisher-compatible MAE 的七日均值可恢复整体 168 h publisher MAE：

- SAS-Norm-only 七日均值：`12.207835115`；
- STaR-GNN 七日均值：`12.233590399`。

Day 7 相对 Day 1：

- DCRNN：`+38.245%`；
- DCRNN + SAS-Norm：`+2.643%`；
- DCRNN + FA-DPR：`+11.934%`；
- STaR-GNN：`+1.698%`。

所以 Full 的长时域相对稳定性最好，但其 168 h sum-of-DMA MAE 与 SAS-Norm-only 是近似持平关系。

最终 Figure 2 若用于“ablation and component contributions”，两个 panel 都应只使用四个 factorial variants；STGCN 应从该消融图中移除。

### Figure 3

当前 final ECDF 只比较 DCRNN、STGCN、STaR-GNN，这是 baseline robustness analysis，不是消融，因此 STGCN 在这里合理。

配对胜率：

- 24 h STaR-GNN vs DCRNN：45/46；
- 24 h STaR-GNN vs STGCN：45/46；
- 168 h STaR-GNN vs DCRNN：46/46；
- 168 h STaR-GNN vs STGCN：40/46。

### Figure 5

代表性 168 h origin 使用预先规定的 median-error proximity 规则，当前选择 common index `70`：

- STGCN publisher MAE：14.653121；
- DCRNN：15.516927；
- STaR-GNN：12.182450。

该选择与视觉外观无关，可以保留。下 panel 的 aggregate-demand hourly absolute error 必须继续与 publisher-compatible MAE 明确区分。

## 7. 数据、图和训练协议审计

### 数据

当前 `paper_split.yaml` 与冻结协议一致：

- 全期：2021-01-01 00:00 至 2023-03-05 23:00；19,056 h；
- 训练：至 2022-12-15 23:00；17,136 h；
- Test：自 2022-12-16；1,920 h；
- 10 DMA；
- history 672 h；
- horizons 24/168 h；
- stride 24 h；
- common-46 = 46 origins。

### Pearson 图

当前 `pearson_static.yaml` / `dcrnn.yaml` 一致：

- training-only Pearson；
- `corr_threshold=null`；
- negative correlations clipped to zero；
- adjacency diagonal zero / no self-loop；
- static graph；
- random-walk normalization；
- 10 nodes；
- diffusion step `K=2`。

### 模型和训练

当前 paper configs / frozen MANIFEST 一致：

- hidden=32；layers=1；
- batch=16；
- lr=0.0003；weight decay=0；
- inverse-sigmoid scheduled sampling；`cl_decay_steps=500`；
- state loss weight=0.03；
- max epochs=100；early stopping patience=15；
- seed=0；
- Test teacher forcing=0；
- checkpoint selection = validation first / test once；
- `test_targets_used_for_training_or_selection=false`。

FA-DPR frozen config：24 h daily token，attention dim=16，1 head，dropout=0，gate bias=-2，future calendar conditioning=true。

## 8. 当前仓库需要修正的文件

### 必须修正生成逻辑

1. `scripts/reproduce/build_paper_tables.py`
   - `PUBLISHER_ABLATION_MODELS` 删除 STGCN；
   - `table_ablation_common46.*` 只输出四个 factorial variants；
   - metric-convention note 不再把 STGCN 写进 ablation。

2. `scripts/reproduce/build_literature_figures.py`
   - `ABLATION_MODELS` 删除 STGCN；
   - ablation 图标题不再写 “ablation and graph-model comparison”。

3. `scripts/reproduce/build_manuscript_results_figures.py`
   - Figure 2 的 ablation/day-wise 统计单独使用四模型集合，避免 STGCN 被带入消融审计。

4. `scripts/reproduce/refine_manuscript_results_figures.py`
   - Figure 2 两个 panel 均按四模型 factorial ablation 呈现；
   - Figure 3 仍保留 DCRNN/STGCN/STaR-GNN；
   - 增加 Full vs SAS-Norm 168 h 的 paired difference / CI guardrail，避免以后把 0.21% 写成稳定退化。

5. `build_detailed_test_artifacts.py::_write_report()`
   - 明确 legacy `31/32` 是 aggregate-demand internal audit；
   - 不得覆盖 manuscript-facing `30/32` 定义。

### 必须同步文档/生成物

- `README.md` / `README_EN.md`；
- `docs/RESULTS_AND_ARTIFACTS_CN.md`；
- `docs/MANUSCRIPT_FIGURES_FINAL_CN.md`；
- `docs/PLOTTING_CN.md`；
- `docs/FULL_PIPELINE_CN.md` 中 31/32 与最终表图入口；
- `paper/README.md`；
- `paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md`；
- `paper/tables/literature/table_ablation_common46.csv/.md`；
- `paper/tables/literature/METRIC_CONVENTIONS.md`；
- `paper/tables/manuscript/README.md`；
- 最终 Figure 2 PNG/PDF。

## 9. 不应修改的真实数据

以下值已经通过代码与派生工件相互核对，不能为了形成漂亮排序而手工修改：

- STaR-GNN publisher MAE：9.424199 / 12.233590；
- SAS-Norm-only 168 h publisher MAE：12.207835；
- STaR-GNN aggregate-demand MAE：4.360841 / 4.919812；
- 168 h Full vs SAS-Norm publisher MAE 的 0.21% 点估计反转；
- FA-DPR 168 h MAPE = 3.277716%，略差于 DCRNN 3.248413%；
- Figure 3 paired win rates；
- Figure 4 的 40/40 positive DMA comparisons；
- Figure 5 common index 70 的 median-rule selection。

修正目标是**恢复实验定义和解释的一致性**，不是改变冻结预测或选择性删除不完美结果。
