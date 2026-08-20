# STaR-GNN 结果、消融与仓库一致性最终审计

本文件固定 2026-08-20 对当前冻结结果、评价代码、论文表图生成器、数据/图/训练配置及历史记录的最终交叉核对结论。对应修正已落实在分支 `audit/fix-ablation-metric-consistency`。

## 1. 正式消融定义

factorial ablation 严格只有四个模型：

1. DCRNN：`backbone` / `Base`；
2. DCRNN + SAS-Norm：`dssn_sasr` / `State`；
3. DCRNN + FA-DPR：`fa_dpr` / `FA-DPR`；
4. STaR-GNN：`full` / `Full`。

STGCN 是独立图时空 baseline，不属于消融。

本分支已从以下 manuscript-facing 位置删除 STGCN ablation 身份：

- `scripts/reproduce/build_paper_tables.py`；
- `scripts/reproduce/build_literature_figures.py`；
- `paper/tables/literature/table_ablation_common46.*`；
- `docs/RESULTS_AND_ARTIFACTS_CN.md`；
- `docs/MANUSCRIPT_FIGURES_FINAL_CN.md`；
- `paper/README.md`；
- Figure 2 最终设计与 caption。

Figure 3、Figure 4、Figure 5 中仍可出现 STGCN，因为这些图回答 baseline robustness / spatial comparison / representative trajectory，而不是 factorial ablation。

## 2. 两套 MAE 与历史结果为什么不矛盾

仓库保留两种合法但用途不同的 MAE。

### 2.1 Internal aggregate-demand MAE

\[
MAE_{agg}=MAE\left(\sum_i\hat y_i,\sum_i y_i\right).
\]

168 h：

| Model | MAE |
|---|---:|
| DCRNN + SAS-Norm | 5.1225108922 |
| **STaR-GNN** | **4.9198118610** |

因此旧 aggregate-demand 结果中 `Full vs SAS-Norm = 4/4` 是正确的。

### 2.2 Manuscript publisher-compatible MAE

\[
MAE_{publisher}=\sum_{i=A}^{J}MAE_i.
\]

168 h：

| Model | MAE |
|---|---:|
| **DCRNN + SAS-Norm** | **12.2078351150** |
| STaR-GNN | 12.2335903993 |

差值：

```text
Full - SAS = +0.0257552843
relative difference ≈ +0.21097%
```

这个值已由两条独立路径复核：

1. 冻结 `metrics_common_46.csv` 的 `total` 行；
2. A--J 十个 DMA MAE 重新求和。

所以冻结预测没有变化，变化的是 MAE 聚合定义。

## 3. 168 h Full vs SAS-Norm：最终统计解释

46 个 common origins 是按 24 h 步长连续启动的 168 h 预测，相邻样本强烈重叠，因此不宜把 46 origins 简单视为相互独立。

最终 manuscript audit 使用 **ordered circular moving-block bootstrap**：

```text
n_origins = 46
block_length = 7 origins
iterations = 50,000
seed = 20260820
```

结果：

```text
Full mean MAE = 12.2335903994
SAS mean MAE  = 12.2078351151
Full - SAS    = +0.0257552843
95% block-bootstrap CI = [-0.128776, +0.177475]
Full lower on 19/46 origins
SAS lower on 27/46 origins
```

95% CI 跨过 0。因此当前论文不把 0.21% 点估计差异解释成稳定性能差距。

安全表述：

> 在 168 h publisher-compatible sum-of-DMA MAE 上，SAS-Norm-only 与完整 STaR-GNN 的点估计近似持平；完整 STaR-GNN 同时取得更低的 MAPE、RMSE、更高的 NSE、更低的 aggregate-demand MAE，以及更小的 Day-1-to-Day-7 相对误差增长。

不使用选择性四舍五入、截断坐标或手工修改 CSV 来改变排序。

## 4. 最终四模型 publisher-compatible 消融

### 24 h

| Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---:|---:|---:|---:|
| DCRNN | 11.917304 | 2.212928 | 6.848257 | 0.970419 |
| DCRNN + SAS-Norm | 10.467994 | 2.010448 | 6.133886 | 0.976269 |
| DCRNN + FA-DPR | 11.238099 | 1.944550 | 6.079036 | 0.976691 |
| **STaR-GNN** | **9.424199** | **1.804574** | **5.534656** | **0.980679** |

### 168 h

| Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---:|---:|---:|---:|
| DCRNN | 16.800744 | 3.248413 | 9.817428 | 0.939504 |
| **DCRNN + SAS-Norm** | **12.207835** | 2.102380 | 6.468312 | 0.973739 |
| DCRNN + FA-DPR | 14.085994 | 3.277716 | 9.332415 | 0.945334 |
| STaR-GNN | 12.233590 | **2.013774** | **6.160881** | **0.976176** |

manuscript-facing factorial cell audit = **30/32**。

两个透明例外：

1. FA-DPR 168 h MAPE `3.277716%` 略高于 DCRNN `3.248413%`；
2. STaR-GNN 168 h publisher MAE `12.233590` 略高于 SAS-Norm-only `12.207835`。

旧 **31/32** 只属于 aggregate-demand internal hierarchy，不再作为当前正文验收数字。

## 5. 总体模型比较核对

STaR-GNN manuscript-facing total：

| Horizon | MAE | MAPE (%) | RMSE | NSE |
|---|---:|---:|---:|---:|
| 24 h | 9.424199 | 1.804574 | 5.534656 | 0.980679 |
| 168 h | 12.233590 | 2.013774 | 6.160881 | 0.976176 |

MAE reduction：

- 24 h vs GRU/LSTM/MSNet/MSCMNet variants：34.9%--46.7%；
- 24 h vs DCRNN/STGCN：20.9%--23.7%；
- 168 h vs GRU/LSTM/MSNet/MSCMNet variants：18.2%--34.5%；
- 168 h vs DCRNN/STGCN：16.0%--27.2%。

来源边界：GRU/LSTM/MSNet/MSCMNet variants 为 Que et al. (2024) reported results；DCRNN/STGCN/STaR-GNN 为当前 common-46 复评。不能写成 9 个模型全部在同一训练代码条件下重训。

## 6. Figure 2 最终设计核对

Figure 2 是纯 factorial ablation，**不含 STGCN**。

### Panel (a)

SAS-Norm / FA-DPR / STaR-GNN 相对 DCRNN 的 day-wise publisher-compatible MAE reduction：

\[
\Delta MAE_d=\frac{MAE_{DCRNN,d}-MAE_{model,d}}{MAE_{DCRNN,d}}\times100\%.
\]

### Panel (b)

四个 factorial variants 相对各自 Day 1 的 MAE change：

\[
G_d=\frac{MAE_d-MAE_{Day1}}{MAE_{Day1}}\times100\%.
\]

Day 7：

```text
DCRNN                 +38.245%
DCRNN + FA-DPR        +11.934%
DCRNN + SAS-Norm       +2.643%
STaR-GNN               +1.698%
```

因此 Figure 2 的主信息是 module contribution 与 lead-time robustness，而不是放大 SAS 与 Full 的 0.21% overall MAE 点估计差异。

## 7. Figure 3--5 核对

### Figure 3

只比较 DCRNN、STGCN、STaR-GNN：

```text
24 h  vs DCRNN 45/46
24 h  vs STGCN 45/46
168 h vs DCRNN 46/46
168 h vs STGCN 40/46
```

这是 baseline robustness，不是消融。

### Figure 4

10 DMA × 2 horizons × 2 baselines = 40 comparisons，全部 MAE reduction > 0，范围约 `1.2611%--61.1981%`。

### Figure 5

代表样本固定使用 median-error proximity rule：

```text
common index = 70
STGCN = 14.653121
DCRNN = 15.516927
STaR-GNN = 12.182450
```

不是人工挑选最好样本。

## 8. 数据、图和训练协议核对

### 数据

- 2021-01-01 至 2023-03-05；19,056 h；
- train 至 2022-12-15 23:00；17,136 h；
- Test 自 2022-12-16；1,920 h；
- 10 DMA；
- history = 672 h；
- horizons = 24 / 168 h；
- stride = 24 h；
- common-46 = 46 origins。

### 图

- training-only Pearson；
- `A_ij=max(r_ij,0)` for `i != j`；
- diagonal = 0；
- no threshold / Top-K；
- random-walk normalization；
- static graph；
- 24 h / 168 h 共用；
- diffusion `K=2`。

### 训练

- hidden = 32；layers = 1；batch = 16；
- lr = 0.0003；weight decay = 0；
- inverse-sigmoid scheduled sampling；`cl_decay_steps=500`；
- state loss weight = 0.03；
- max epochs = 100；patience = 15；
- seed = 0；
- Test teacher forcing = 0；
- validation-first / test-once；
- `test_targets_used_for_training_or_selection=false`。

## 9. 已完成的仓库修正

本分支已完成：

- 四模型 Table 2 生成逻辑；
- STGCN 从 ablation table / ablation figures / Figure 2 中移除；
- Markdown manuscript tables 统一 3 位小数；
- CSV/JSON 保留完整精度；
- Figure 2 改为 module improvement + long-horizon degradation；
- Full-vs-SAS 7-origin moving-block bootstrap guardrail；
- README / README_EN / METHOD / RESULTS / FULL_PIPELINE / PLOTTING / paper README / captions / metric conventions 同步；
- 旧 `MANUSCRIPT_FIGURES_CN.md` 明确降级为历史设计稿；
- `paper/reports/TEST_RESULTS_CN.md` 明确标记为 legacy aggregate-demand internal report。

冻结 checkpoint、预测数组和真实测试结果没有为形成更漂亮排序而修改。

## 10. 当前权威入口

- 结果：`docs/RESULTS_AND_ARTIFACTS_CN.md`
- 方法：`docs/METHOD_CN.md`
- 图表：`docs/MANUSCRIPT_FIGURES_FINAL_CN.md`
- 作图：`docs/PLOTTING_CN.md`
- 指标：`paper/tables/literature/METRIC_CONVENTIONS.md`
- 本审计：`docs/RESULT_CONSISTENCY_AUDIT_CN.md`

最终原则：**真实结果透明保留；实验分类正确；图表回答科学问题；不通过视觉或精度技巧制造不存在的优势。**
