# 实验结果、来源与论文工件

本文档集中说明实验口径、冻结 Test 结果、结果来源以及论文表图位置。环境安装、
数据预处理、训练、Test 和 checkpoint 复评命令统一见
[`FULL_PIPELINE_CN.md`](FULL_PIPELINE_CN.md)，不在这里重复。

## 1. 论文模型命名

论文中的 DCRNN 就是消融中的 Base，代码对应 `variant=backbone`。主表使用以下名称：

| 论文名称 | 内部键 | State | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` | × | × |
| DCRNN + State | `dssn_sasr` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` | × | ✓ |
| STaR-GNN | `full` | ✓ | ✓ |
| STGCN | 独立基线 | × | × |

冻结工件只保留 `star_gnn/Base` 这一套 DCRNN，不再发布历史
`baselines/dcrnn` 重复 checkpoint。

## 2. 选择协议与公平性

最终设置只根据 Validation 确定：

```yaml
learning_rate: 0.0003
weight_decay: 0.0
cl_decay_steps: 500
state_loss_weight: 0.03
max_epochs: 100
seed: 0
```

候选参数在读取 Test 前预先声明。最终配置在 Validation 的 32 项方向关系中通过
28 项；参数固定后才执行 common-46 Test。Test 不参与参数选择、early stopping
或组件取舍。

四个消融单元共享数据、Pearson 图、DCRNN hidden size、decoder、优化器和训练协议。
关系按四项指标分别判断：MAE、MAPE、RMSE 越低越好，NSE 越高越好。

- State 优于 DCRNN：验证状态变换的独立贡献；
- FA-DPR 优于 DCRNN：验证预测对齐检索的独立贡献；
- STaR-GNN 优于 State：验证 FA-DPR 的边际贡献；
- STaR-GNN 优于 FA-DPR：验证 State 的边际贡献。

## 3. common-46 冻结 Test 结果

| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |
|---|---|---:|---:|---:|---:|
| 24 h | DCRNN | 5.356315 | 2.212928 | 6.848257 | 0.970419 |
| 24 h | DCRNN + State | 4.895320 | 2.010448 | 6.133886 | 0.976269 |
| 24 h | DCRNN + FA-DPR | 4.739336 | 1.944550 | 6.079036 | 0.976691 |
| 24 h | **STaR-GNN** | **4.360841** | **1.804574** | **5.534656** | **0.980679** |
| 168 h | DCRNN | 7.734838 | 3.248413 | 9.817428 | 0.939504 |
| 168 h | DCRNN + State | 5.122511 | 2.102380 | 6.468312 | 0.973739 |
| 168 h | DCRNN + FA-DPR | 7.578056 | 3.277716 | 9.332415 | 0.945334 |
| 168 h | **STaR-GNN** | **4.919812** | **2.013774** | **6.160881** | **0.976176** |

主要结论如下：

- STaR-GNN 在 24 h 和 168 h 的全部四项指标上均优于两个单组件模型；
- State 在两个任务的八项指标上均优于 DCRNN；
- FA-DPR 在 24 h 的四项指标上均优于 DCRNN；
- FA-DPR 在 168 h 改善 MAE、RMSE 和 NSE，但 MAPE 为 3.277716%，略高于
  DCRNN 的 3.248413%；因此 Test 方向关系为 31/32，而不是 32/32。

这一例外必须透明报告。论文不得把真实的区域或预测距离异质性改写成所有子表均满足
固定排序。

## 4. 指标从哪里产生

最终指标不是从训练日志或 Validation 表复制，而由冻结最佳 checkpoint 重新推理生成：

```text
scripts/reproduce/reproduce.py
  -> scripts/innovation/evaluate_star_dcrnn.py
  -> run_star_checkpoint_evaluation()
  -> prepare_dcrnn_test_data()
  -> protocol_indices["common_46"]
  -> metrics_aggregate_total_common_46.csv
```

每份 `test_summary.json` 都记录：

- checkpoint SHA-256；
- 图工件与训练需求 SHA-256；
- operational、strict-within-test 和 common-46 样本数；
- `teacher_forcing_ratio=0.0`；
- `test_targets_used_for_training_or_selection=false`。

冻结文件由 `results/paper/frozen_v1/MANIFEST.json` 和 `CHECKSUMS.sha256` 保护。
重新推理命令会对10个物理 checkpoint 的40项指标进行差异审计；物理工件与
论文五个模型名称一一对应。

## 5. 指标口径

本仓库的主表使用十个 DMA 聚合需求的 MAE、MAPE、RMSE 和 NSE。与 MSCMNet
原文比较时，还会生成其补充材料使用的 publisher `total` 口径，其中 MAE 是
A--J 十个 DMA MAE 之和。两种 MAE 定义必须分别标注，不得直接混在同一列比较。

## 6. 论文表格

运行 `verify_pretrained.sh` 或 `build_detailed_test_artifacts.py` 后生成：

| 文件 | 内容 |
|---|---|
| `paper/tables/test_all_models_common46.csv` | 五个公开模型、两个任务的10行 Test 总表 |
| `paper/tables/test_ablation_common46.csv` | DCRNN、State、FA-DPR、STaR-GNN 四单元消融 |
| `paper/tables/test_dma_metrics_long.csv` | task × model × DMA 的完整四指标长表 |
| `paper/tables/test_dma_*_wide_24h.csv` | 24 h 的 A--J DMA 宽表 |
| `paper/tables/test_dma_*_wide_168h.csv` | 168 h 的 A--J DMA 宽表 |
| `paper/tables/test_day1_day7_metrics.csv` | 168 h 切成七个连续 24 h 区间 |
| `paper/tables/test_day1_day7_*_wide.csv` | 每项指标的 Day 1--Day 7 宽表 |
| `paper/tables/pearson_correlation.csv` | 训练期 Pearson 相关矩阵 |
| `paper/tables/pearson_adjacency.csv` | 正相关邻接矩阵 |
| `paper/tables/pearson_random_walk.csv` | 模型实际使用的传播矩阵 |
| `paper/tables/test_hierarchy.json` | 论文关系自动审计 |

DMA 表用于分析区域异质性；Day 1--Day 7 表用于分析预测距离效应。这些结果由
46 个共同预测起点上的真实预测重新计算，不是从总体指标按比例拆分。

## 7. 论文图件

所有图同时输出 PNG 和 PDF：

| 图件 | 内容 |
|---|---|
| `test_overall_24h/168h.*` | STGCN、DCRNN 与 STaR-GNN 总体比较 |
| `test_ablation_24h/168h.*` | 四单元消融 |
| `test_dma_mae_24h/168h.*` | A--J DMA 异质性 |
| `test_day1_day7_models.*` | 168 h 跨模型逐日误差 |
| `test_day1_day7_ablation.*` | 168 h 消融逐日误差 |
| `pearson_correlation_heatmap.*` | 训练期功能关联热力图 |

更适合放在正文的通常是总体比较、核心消融、Day 1--Day 7 和 Pearson 热力图；
完整 DMA 四指标表可放补充材料。自动生成的中文解读位于
`paper/reports/TEST_RESULTS_CN.md`。

## 8. 一键生成与审计

冻结 checkpoint 和预测已经安装时执行：

```bash
bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

只重新生成论文表图：

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

python scripts/reproduce/build_detailed_test_artifacts.py \
  --release results/paper/frozen_v1 \
  --graph artifacts/graphs/bwdf_pearson_static_graph.npz \
  --output paper
```

正文和补充材料应直接使用这些脚本生成的 CSV、Markdown、PNG 和 PDF。不得手工修改
CSV 来强制形成预期排序；若重新训练结果不同，应先审计环境、数据、图、checkpoint
和协议，而不是调整 Test 指标。
