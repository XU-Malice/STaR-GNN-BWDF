# 论文表格与图件

执行以下命令后，本目录会由冻结 common-46 Test 预测自动填充：

```bash
bash scripts/reproduce/verify_pretrained.sh
```

- `tables/test_all_models_common46.*`：STGCN、DCRNN、State、FA-DPR 和 Full；
- `tables/test_ablation_common46.*`：DCRNN/State/FA-DPR/Full；
- `tables/test_dma_*`：A--J 十个 DMA 的四项指标；
- `tables/test_day1_day7_*`：168 h 切分为七个连续 24 h 区间；
- `tables/pearson_*`：训练期相关、邻接、随机游走和节点统计；
- `tables/literature/`：MSCMNet 文献值与 STGCN/DCRNN/Full 对比；
- `figures/*.png`：README/预览使用；
- `figures/*.pdf`：论文矢量图；
- `reports/TEST_RESULTS_CN.md`：中文结果解读和层级审计。

这些文件不用于选参，只用于从冻结 Test 输出生成可审计的论文材料。

指标定义、冻结结果、文件用途和论文放置建议见
[`../docs/RESULTS_AND_ARTIFACTS_CN.md`](../docs/RESULTS_AND_ARTIFACTS_CN.md)。
