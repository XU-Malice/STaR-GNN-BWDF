# STaR-GNN-BWDF 文档索引

本目录同时包含“完整复现流程”“方法说明”“最终论文结果”“历史图表设计记录”等不同用途的文档。为避免旧内部命名或旧指标口径与当前 Journal of Hydrology 稿件混淆，建议按下面顺序阅读。

## 1. 新读者推荐顺序

1. [`../README.md`](../README.md) — 仓库入口、最终指标、快速验证和 Figure 1--5 概览；
2. [`METHOD_CN.md`](METHOD_CN.md) — SAS-Norm、FA-DPR、Pearson 功能图与源码对应；
3. [`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md) — 最终 publisher-compatible 结果、Table 1--3、Figure 1--5；
4. [`PLOTTING_CN.md`](PLOTTING_CN.md) — 从冻结预测重新生成最终 5 张正文图；
5. [`FULL_PIPELINE_CN.md`](FULL_PIPELINE_CN.md) — 从环境、数据、图、训练到 Test/clean-room 的完整流程；
6. [`RELEASE_CN.md`](RELEASE_CN.md) — GitHub Release、冻结资产和独立验收。

## 2. 当前 manuscript-facing 权威文档

### 指标定义

[`../paper/tables/literature/METRIC_CONVENTIONS.md`](../paper/tables/literature/METRIC_CONVENTIONS.md)

正文 total MAE 采用 DMA A--J MAE 求和；MAPE/RMSE/NSE 在小时总需求序列上计算。STaR-GNN 最终 MAE：24 h `9.424199`、168 h `12.233590`。

### 精确正文表格

```text
paper/tables/literature/table_literature_comparison_common46.*
paper/tables/literature/table_ablation_common46.*
paper/tables/literature/table_star_gnn_dma_common46.*
```

### Figure 1--5 最终设计

[`MANUSCRIPT_FIGURES_FINAL_CN.md`](MANUSCRIPT_FIGURES_FINAL_CN.md)

### Figure 1--5 作图流程

[`PLOTTING_CN.md`](PLOTTING_CN.md)

### Figure captions

[`../paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md`](../paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md)

## 3. 文档状态说明

| 文件 | 当前用途 | 是否 manuscript-facing |
|---|---|:---:|
| `METHOD_CN.md` | 最终方法名与源码对应 | ✓ |
| `RESULTS_AND_ARTIFACTS_CN.md` | 最终结果/工件入口 | ✓ |
| `MANUSCRIPT_FIGURES_FINAL_CN.md` | 最终 Figure 1--5 设计 | ✓ |
| `PLOTTING_CN.md` | 最终作图教程 | ✓ |
| `FULL_PIPELINE_CN.md` | 完整复现与代码流转 | 部分 |
| `RELEASE_CN.md` | 发布与 clean-room | 否，工程发布文档 |
| `MANUSCRIPT_FIGURES_CN.md` | 历史初始图表设计记录 | **否，已替代** |

`FULL_PIPELINE_CN.md` 中若出现 `State`、`Base` 等词，通常指源码/冻结工件的内部兼容标签；论文正文名称以 `METHOD_CN.md` 为准：

- `State` / `dssn_sasr` → **SAS-Norm**
- `FA-DPR` / `fa_dpr` → **FA-DPR**
- `Full` / `full` → **STaR-GNN**
- `Base` / `backbone` → **DCRNN**

## 4. 两套 MAE 不要混用

### 正文 publisher-compatible MAE

```text
MAE_publisher = sum(DMA A--J MAE)
```

STaR-GNN：

- 24 h = 9.424199
- 168 h = 12.233590

### 内部 aggregate-demand MAE

```text
MAE_agg = MAE(sum prediction, sum target)
```

STaR-GNN：

- 24 h = 4.360841
- 168 h = 4.919812

后者只用于运行诊断、legacy `test_*` 图和部分 aggregate-demand trajectory 分析，不作为正文跨模型 total MAE。

## 5. 当前消融解释边界

publisher-compatible 消融为 **30/32**。必须透明保留：

- FA-DPR 168 h MAPE 略差于 DCRNN；
- SAS-Norm-only 168 h MAE `12.207835` 略低于 Full `12.233590`。

因此不要使用旧 aggregate-demand 报告中的“Full 在两个 horizon 所有指标均严格优于单模块”作为正文结论。

## 6. 最终 Results 证据链

```text
Table 1 + Figure 1  → 总体预测精度
Table 2 + Figure 2  → 消融与长时域稳定性
Figure 3            → 46 个测试起点上的稳健性
Table 3 + Figure 4  → 跨 DMA 空间一致性
Figure 5            → 代表性一周预测行为
```

Figure 1--5 的底层审计数据位于 `paper/tables/manuscript/`。

## 7. Legacy / 内部诊断入口

以下内容仍可用于复现与补充分析，但不是当前正文权威结果：

- `paper/reports/TEST_RESULTS_CN.md`
- `paper/tables/test_*`
- `paper/figures/test_*`
- 旧 aggregate-demand Day 1--Day 7 图

查看这些文件时，应先确认它们使用的是 publisher-compatible 还是 aggregate-demand 口径。