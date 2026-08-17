# common-46 Test 结果与论文图表说明

> 本目录由冻结 checkpoint 自动生成。参数在 Validation 阶段确定，Test 只执行最终报告；所有预测均关闭 teacher forcing。

## 1. 总体 Test 对比

| task | model | MAE | MAPE | RMSE | NSE |
|---|---|---:|---:|---:|---:|
| 24h | STGCN | 5.850690 | 2.424526 | 7.904592 | 0.960590 |
| 24h | DCRNN | 5.356315 | 2.212928 | 6.848257 | 0.970419 |
| 24h | State | 4.895320 | 2.010448 | 6.133886 | 0.976269 |
| 24h | FA-DPR | 4.739336 | 1.944550 | 6.079036 | 0.976691 |
| 24h | Full | 4.360841 | 1.804574 | 5.534656 | 0.980679 |
| 168h | STGCN | 8.574033 | 3.575848 | 10.305691 | 0.933337 |
| 168h | DCRNN | 7.734838 | 3.248413 | 9.817428 | 0.939504 |
| 168h | State | 5.122511 | 2.102380 | 6.468312 | 0.973739 |
| 168h | FA-DPR | 7.578056 | 3.277716 | 9.332415 | 0.945334 |
| 168h | Full | 4.919812 | 2.013774 | 6.160881 | 0.976176 |

## 2. 组件消融

| task | model | MAE | MAPE | RMSE | NSE |
|---|---|---:|---:|---:|---:|
| 24h | DCRNN | 5.356315 | 2.212928 | 6.848257 | 0.970419 |
| 24h | State | 4.895320 | 2.010448 | 6.133886 | 0.976269 |
| 24h | FA-DPR | 4.739336 | 1.944550 | 6.079036 | 0.976691 |
| 24h | Full | 4.360841 | 1.804574 | 5.534656 | 0.980679 |
| 168h | DCRNN | 7.734838 | 3.248413 | 9.817428 | 0.939504 |
| 168h | State | 5.122511 | 2.102380 | 6.468312 | 0.973739 |
| 168h | FA-DPR | 7.578056 | 3.277716 | 9.332415 | 0.945334 |
| 168h | Full | 4.919812 | 2.013774 | 6.160881 | 0.976176 |

Full 在两个预测范围、四项指标上均优于 State 和 FA-DPR；State 在八项比较中均优于 DCRNN。FA-DPR 相对 DCRNN 的 168 h MAPE 是唯一例外，完整记录为 31/32，而不是删除该结果。

## 3. 168 h 逐日分析

| day | STGCN | DCRNN | State | FA-DPR | Full |
|---:|---:|---:|---:|---:|---:|
| 1.000000 | 8.873282 | 6.864114 | 5.161433 | 5.263806 | 5.178844 |
| 2.000000 | 8.717121 | 7.451384 | 4.872711 | 6.517680 | 4.871463 |
| 3.000000 | 8.690786 | 7.826405 | 4.948184 | 7.697594 | 4.881252 |
| 4.000000 | 8.433139 | 7.641306 | 5.063941 | 8.190442 | 4.751058 |
| 5.000000 | 8.401340 | 7.481204 | 5.174514 | 8.379114 | 4.886332 |
| 6.000000 | 8.424156 | 7.677867 | 5.236640 | 8.495382 | 4.894216 |
| 7.000000 | 8.478408 | 9.201582 | 5.400153 | 8.502374 | 4.975517 |

逐日表按 common-46 的 168 h 预测切分为七个连续 24 h 区间，指标在各日的十 DMA 总需求序列上重算。逐日结果用于解释误差随预测距离的变化，不参与参数选择。

## 4. DMA 异质性

完整 A--J 四指标见 `tables/test_dma_metrics_long.csv`，宽表见 `tables/test_dma_*_wide.csv`，图见 `figures/test_dma_mae_24h.*` 和 `figures/test_dma_mae_168h.*`。

## 5. Pearson 功能图

图仅由训练期 10 个 DMA 构建；正相关无向边 45/45。相关矩阵、邻接矩阵、随机游走矩阵和节点统计均保存在 `tables/`。

## 6. 自动层级验收

```json
{
  "protocol": "common_46",
  "test_origins": 46,
  "selection_policy": "validation_first_test_once",
  "aggregate_relations": {
    "24h_State_vs_DCRNN": "4/4",
    "24h_FA-DPR_vs_DCRNN": "4/4",
    "24h_Full_vs_State": "4/4",
    "24h_Full_vs_FA-DPR": "4/4",
    "24h_Full_vs_STGCN": "4/4",
    "168h_State_vs_DCRNN": "4/4",
    "168h_FA-DPR_vs_DCRNN": "3/4",
    "168h_Full_vs_State": "4/4",
    "168h_Full_vs_FA-DPR": "4/4",
    "168h_Full_vs_STGCN": "4/4"
  },
  "daily_relations": {
    "Day1-7_State_vs_DCRNN": "28/28",
    "Day1-7_FA-DPR_vs_DCRNN": "15/28",
    "Day1-7_Full_vs_State": "26/28",
    "Day1-7_Full_vs_FA-DPR": "28/28",
    "Day1-7_Full_vs_STGCN": "28/28"
  },
  "transparent_exception": "FA-DPR 168h MAPE is slightly worse than DCRNN; all other registered factorial Test relations pass (31/32)."
}
```

## 7. 图件索引

- `figures/test_overall_24h.*`、`test_overall_168h.*`：STGCN/DCRNN/Full 总体对比。
- `figures/test_ablation_24h.*`、`test_ablation_168h.*`：四单元消融。
- `figures/test_day1_day7_models.*`：STGCN/DCRNN/Full 逐日对比。
- `figures/test_day1_day7_ablation.*`：DCRNN/State/FA-DPR/Full 逐日消融。
- `figures/test_dma_mae_*.{png,pdf}`：十 DMA 的区域异质性。
- `figures/pearson_correlation_heatmap.*`：训练期 Pearson 功能关联热力图。
