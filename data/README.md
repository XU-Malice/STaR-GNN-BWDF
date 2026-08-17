# BWDF 数据获取与本地目录

本仓库不重新分发原始供水数据。环境文件把
[`wf4bwdf`](https://github.com/WaterFutures/wf4bwdf) 固定到 commit
`5da2d47752190dd69bc3ee612dff043a52e25a2b`，数据管道通过其公开加载接口读取
BWDF 完整数据。

从零构建：

```bash
bash scripts/data/run_pipeline.sh
```

如果已经克隆 wf4bwdf：

```bash
bash scripts/data/run_pipeline.sh --wf4bwdf-repo repos/wf4bwdf
```

输出在 `data/processed/data_build/`，该目录不会提交到 Git。数据报告记录输入
哈希、时间范围、行数、缺失值、插值、IQR 阈值和 scaler 拟合范围。

checkpoint 复评需要相同处理数据；作者可用
`scripts/reproduce/import_local_artifacts.py` 从旧工程复制，外部用户也可从原始
数据重新执行管道。

预处理的完整调用链、每个输出文件及防泄漏检查见
[`../docs/FULL_PIPELINE_CN.md`](../docs/FULL_PIPELINE_CN.md#4-数据预处理从命令到函数再到输出)。
