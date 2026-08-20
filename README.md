# STaR-GNN：多 DMA 小时需水预测

[English](README_EN.md)｜[文档索引](docs/README.md)｜[完整复现教程](docs/FULL_PIPELINE_CN.md)｜[方法说明](docs/METHOD_CN.md)｜[结果与工件](docs/RESULTS_AND_ARTIFACTS_CN.md)｜[最终作图教程](docs/PLOTTING_CN.md)

本仓库是 STaR-GNN 的独立可复现实现，用于 10 个分区计量区域（DMA）的 24 h day-ahead 与 168 h week-ahead 联合需水预测。仓库包含 split-aware 数据预处理、仅训练期 Pearson 功能图、DCRNN/STGCN 图时空基线、SAS-Norm 与 FA-DPR 消融、冻结 checkpoint、common-46 Test 复评，以及 Journal of Hydrology 稿件所用表格、图件和审计工件。

> **当前 manuscript-facing 结果统一采用 publisher-compatible 口径。** 不要把内部 aggregate-demand MAE（例如 STaR-GNN 的 4.360841/4.919812）与论文正文的 total MAE 混用。详细定义见 [`paper/tables/literature/METRIC_CONVENTIONS.md`](paper/tables/literature/METRIC_CONVENTIONS.md)。

## 1. 最终论文指标口径

与 Que et al. (2024) Supplementary Tables 的 total 口径保持一致：

- **total MAE** = DMA A--J 十个 MAE 的求和；
- **total MAPE / RMSE / NSE** = 在 A--J 小时需求求和后的总需求序列上计算；
- 主 Test = `common_46`，即两个预测任务共同的 46 个测试起点；
- Test 阶段 `teacher forcing=0`，未来真实需求不进入模型输入或参数选择。

STaR-GNN 的最终 manuscript-facing 结果为：

| Horizon | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---:|---:|---:|---:|
| 24 h | **9.424199** | **1.804574** | **5.534656** | **0.980679** |
| 168 h | **12.233590** | **2.013774** | **6.160881** | **0.976176** |

完整九模型比较见 [`table_literature_comparison_common46.md`](paper/tables/literature/table_literature_comparison_common46.md)。其中 GRU、LSTM、MSNet 和 MSCMNet variants 为 Que et al. (2024) 的 reported results；DCRNN、STGCN 与 STaR-GNN 为当前 common-46 复评结果，不能表述为所有模型均在完全相同代码条件下重训。

## 2. 最终消融结果

论文中的公开名称与内部兼容键对应如下：

| 论文名称 | 内部键 | SAS-Norm | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` / `Base` | × | × |
| DCRNN + SAS-Norm | `dssn_sasr` / `State` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` / `FA-DPR` | × | ✓ |
| STaR-GNN | `full` / `Full` | ✓ | ✓ |

publisher-compatible 消融为 **30/32**，两个真实例外必须保留：

1. FA-DPR 的 168 h MAPE（3.277716%）略高于 DCRNN（3.248413%）；
2. 168 h 下 SAS-Norm-only 的 sum-of-DMA MAE 为 **12.207835**，略低于完整 STaR-GNN 的 **12.233590**（约 0.21%）。

因此，正确结论是：完整模型在 24 h 的四项指标均最优；在 168 h 的 MAPE、RMSE 和 NSE 最优，而 SAS-Norm-only 在 publisher-compatible MAE 上略低。完整表见 [`table_ablation_common46.md`](paper/tables/literature/table_ablation_common46.md)。

## 3. 最终正文图表

正文结果证据链固定为：

1. **Overall predictive accuracy**：Table 1 + Figure 1；
2. **Ablation and component contributions**：Table 2 + Figure 2；
3. **Robustness across forecast origins**：Figure 3；
4. **Spatial consistency across DMAs**：Table 3 + Figure 4；
5. **Representative weekly forecasting behavior**：Figure 5。

正文三张表：

- `paper/tables/literature/table_literature_comparison_common46.*`
- `paper/tables/literature/table_ablation_common46.*`
- `paper/tables/literature/table_star_gnn_dma_common46.*`

正文五张图：

- `paper/figures/manuscript_fig1_relative_improvement.*`
- `paper/figures/manuscript_fig2_day1_day7_publisher_mae.*`
- `paper/figures/manuscript_fig3_origin_ecdf.*`
- `paper/figures/manuscript_fig4_dma_mae_improvement.*`
- `paper/figures/manuscript_fig5_representative_168h_trajectory.*`

最终图表设计说明见 [`docs/MANUSCRIPT_FIGURES_FINAL_CN.md`](docs/MANUSCRIPT_FIGURES_FINAL_CN.md)，Figure captions 见 [`paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md`](paper/captions/MANUSCRIPT_RESULT_FIGURE_CAPTIONS.md)。

## 4. 关键实证事实

- 168 h 的 Day 7 相对 Day 1 publisher-compatible MAE：DCRNN `+38.25%`、FA-DPR `+11.93%`、SAS-Norm `+2.64%`、STaR-GNN `+1.70%`；
- STaR-GNN 对 DCRNN 的逐起点胜率：24 h 为 `45/46`，168 h 为 `46/46`；
- STaR-GNN 对 STGCN 的逐起点胜率：24 h 为 `45/46`，168 h 为 `40/46`；
- DMA-level MAE 相比 DCRNN/STGCN，在 `10 DMA × 2 horizon × 2 baseline = 40` 个比较中 **40/40 均为正改善**，但改善幅度具有空间异质性；
- 168 h 的长期 MAE 稳定性主要由 SAS-Norm 贡献，因此不应声称 Full 在 168 h MAE 上严格全面优于 SAS-Norm-only。

这些结果均可在 `paper/tables/manuscript/` 的 CSV/JSON 审计文件中追溯。

## 5. 快速安装与冻结结果验证

```bash
conda env create -f environment.yml
conda activate star-gnn-bwdf
python -m pip install -e .

bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

GitHub Release 保存冻结 checkpoint、预测、指标和校验清单；原始 BWDF/处理数据不随仓库重新分发。首次运行若缺少论文数据与训练期 Pearson 图，验证入口会按固定配置自动构建，但不会重新训练模型。

从原始数据完整训练论文实验：

```bash
bash scripts/reproduce/train_from_scratch.sh \
  --device auto \
  --evaluation-device cpu \
  --seeds 0
```

完整的数据、构图、训练、Test 与 clean-room 流程见 [`docs/FULL_PIPELINE_CN.md`](docs/FULL_PIPELINE_CN.md)。

## 6. 重新生成最终论文图

第一阶段从冻结预测生成 manuscript audit 数据和基础 Figure 1--5：

```bash
python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820
```

第二阶段根据冻结审计结果生成**最终版 Figure 2 和 Figure 3**：

```bash
python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures
```

第二阶段会覆盖第一阶段的 Figure 2/3。完整输入、输出、审计字段、成功标志和故障排查见 [`docs/PLOTTING_CN.md`](docs/PLOTTING_CN.md)。

## 7. 数据与图协议

- 数据期：2021-01-01 至 2023-03-05，小时分辨率；
- 训练截止：2022-12-15 23:00；Test 从 2022-12-16 开始；
- 历史窗口：672 h；预测范围：24 h 与 168 h；
- 图：仅训练期 DMA Pearson 正相关，零对角，无阈值/Top-K，随机游走归一化；
- 24 h 与 168 h 共用同一固定功能图；
- 主 Test：common-46；
- 参数仅根据 Validation 确定。

## 8. 仓库结构

```text
configs/                 数据、图、模型和论文冻结配置
src/dma_wdf/             数据、图、模型、训练、评估核心源码
scripts/data/            数据预处理与质量检查
scripts/graph/           Pearson 图构建与验证
scripts/train/           基线训练入口
scripts/innovation/      STaR-GNN 训练与评估入口
scripts/reproduce/       从零复现、冻结验证、论文表图与审计
paper/tables/literature/ 正文 Table 1--3 与指标口径
paper/tables/manuscript/ Figure 1--5 的审计 CSV/JSON
paper/figures/           正文图及补充图 PNG/PDF
paper/captions/          最终 Figure captions
docs/                    方法、结果、复现、作图和发布文档
```

旧的 `test_overall_*`、`test_ablation_*`、`test_star_gnn_dma_metrics.*` 和 aggregate-demand Day 1--Day 7 图保留为 Supplementary/内部诊断，不再承担正文主要科学结论。

## 9. 测试、数据与引用

```bash
bash scripts/reproduce/smoke_test.sh
python -m pytest tests -q
```

原始数据获取见 [`data/README.md`](data/README.md)。发布与 clean-room 说明见 [`docs/RELEASE_CN.md`](docs/RELEASE_CN.md)。论文正式发表后请使用最终 DOI 更新 [`CITATION.cff`](CITATION.cff)。