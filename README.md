# STaR-GNN：多 DMA 小时需水图预测

[English](README_EN.md)｜[完整中文教程](docs/FULL_PIPELINE_CN.md)｜[方法说明](docs/METHOD_CN.md)｜[实验结果与论文工件](docs/RESULTS_AND_ARTIFACTS_CN.md)｜[GitHub 发布](docs/RELEASE_CN.md)

本仓库是论文的独立、可复现实现，面向 10 个分区计量区域（DMA）的
24 h day-ahead 与 168 h week-ahead 联合需水预测。仓库保留数据预处理、
训练期 Pearson 功能图、DCRNN/STGCN 基线、四单元消融、冻结 checkpoint、
common-46 Test 复评，以及论文表格和图件生成流程。

公开方法名为 **STaR-GNN**。源码中的 `star_dcrnn` 仅为兼容已冻结
checkpoint 的内部键，不表示论文方法只是对一个已有模型的简单改名。

## 从哪里开始

如果是第一次使用本仓库，建议按以下顺序进行：

1. 按 [`docs/FULL_PIPELINE_CN.md`](docs/FULL_PIPELINE_CN.md) 的快速路线完成环境和源码检查；
2. 使用 `verify_pretrained.sh` 验证冻结 checkpoint 和 common-46 Test；
3. 需要重新生成数据、Pearson 图、训练结果或论文图表时，继续按同一教程逐阶段执行；
4. 论文指标、表格和图件说明见
   [`docs/RESULTS_AND_ARTIFACTS_CN.md`](docs/RESULTS_AND_ARTIFACTS_CN.md)；
5. 上传 GitHub 前，按 [`docs/RELEASE_CN.md`](docs/RELEASE_CN.md) 完成两轮 clean-room 验收。

完整教程不是命令列表，而是逐步说明“入口脚本 → Python 函数 → 下游调用 → 输出文件
→ 成功标准”。其中还包含每个目录和主要脚本的用途、单模型训练/Test 命令、后台运行、
进度监控和常见故障处理。

### 论文命名约定

论文中的 **DCRNN 就是消融中的 Base**。它在代码中对应 `variant=backbone`；
State、FA-DPR 和 Full 均在同一个 DCRNN 主干上逐项启用组件：

| 论文名称 | 内部 variant | State | FA-DPR |
|---|---|---:|---:|
| DCRNN | `backbone` | × | × |
| DCRNN + State | `dssn_sasr` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` | × | ✓ |
| STaR-GNN | `full` | ✓ | ✓ |

冻结目录保留 `Base` 键名作为唯一 DCRNN checkpoint 身份；不再保存第二套
`baselines/dcrnn` 工件，论文正文和主表统一写作 `DCRNN (Base)`。

## 核心结论

论文参数只在 Validation 上确定：

```yaml
learning_rate: 0.0003
weight_decay: 0.0
cl_decay_steps: 500
state_loss_weight: 0.03
max_epochs: 100
seed: 0
```

common-46 Test 的四单元消融如下。MAE、MAPE、RMSE 越低越好，NSE 越高越好。

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

关键层级为：

- STaR-GNN 在 24 h 和 168 h 的四项指标上都优于两个单组件模型；
- DCRNN + State 在两个任务的八项比较中都优于 DCRNN；
- DCRNN + FA-DPR 在 24 h 全面优于 DCRNN，在 168 h 改善 MAE、RMSE、NSE；
- FA-DPR 的 168 h MAPE 比 DCRNN 高 0.029303 个百分点，是唯一透明保留的
  例外，因此总体消融关系为 31/32，而不是伪造为 32/32。

STGCN、DCRNN 与 STaR-GNN 的对比，十 DMA 指标，168 h 的 Day 1--Day 7
表格及 Pearson 图表均由冻结预测自动生成，见
[`paper/reports/TEST_RESULTS_CN.md`](paper/reports/TEST_RESULTS_CN.md)。

## 两种复现方式

### A. 验证论文 checkpoint（推荐先做）

下载 GitHub Release 中的冻结工件并解压到
`results/paper/frozen_v1/`，然后执行：

```bash
conda env create -f environment.yml
conda activate star-gnn-bwdf
python -m pip install -e .

bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

该入口依次检查文件 SHA-256、10 个 checkpoint 元数据、common-46 样本数、
Test 隔离字段、冻结指标、消融层级，并重新推理全部模型。GPU 复评只对指标
跨 CUDA/cuDNN 环境的末位浮点误差使用 `5e-4` 绝对与相对容差（0.05%），
并输出40项指标差异审计表；checkpoint 哈希、协议字段和样本索引仍须完全一致。
最终详细审计还会核对DMA/逐日/Pearson表格行数、图件非空性和旧HPO/SGDR
执行代码隔离；该扫描由Python完成，不依赖服务器是否安装`rg`。

作者在同一台机器迁移旧项目工件时，可一次完成导入和验收：

```bash
bash scripts/reproduce/setup_from_existing.sh \
  /path/to/DMA-WDF \
  --re-evaluate \
  --device cuda:0
```

上传GitHub前的最高强度验证是clean-room从零复现。它创建全新Conda环境和纯净
源码副本，从原始数据重新预处理、构图、训练10组模型、执行common-46并逐项
对照冻结结果：

```bash
bash scripts/reproduce/validate_clean_room.sh \
  --workspace /path/to/new-clean-room \
  --frozen-release results/paper/frozen_v1 \
  --device cuda:0 \
  --evaluation-device cuda:0
```

clean-room 的原理、完整阶段和监控方式见
[`docs/FULL_PIPELINE_CN.md`](docs/FULL_PIPELINE_CN.md)；上传前和从 GitHub 实际下载后的
两轮独立验收步骤见 [`docs/RELEASE_CN.md`](docs/RELEASE_CN.md)。

发布前建议使用带阶段状态和最终清单的全面验收入口。该入口会原子修复上次
中断留下的半成品目录，并把旧目录保留为 `*.incomplete.<timestamp>`：

```bash
bash scripts/reproduce/validate_everything.sh \
  /path/to/DMA-WDF \
  --device cuda:0
```

全面验收覆盖 10/10 checkpoint、10/10 冻结预测、源码测试、SHA-256、
common-46 协议、10 组 GPU 复推理以及总体/消融/DMA/Day1--Day7/Pearson
论文表图。它不重新训练模型；完整从零训练使用下一节的独立入口。

如果冻结工件已经在当前仓库，可直接执行最终收口入口。它不会重新训练，而是将
DCRNN/Base 合并为唯一 checkpoint、清理已合并的旧文档、复推理10组模型、重建
论文表图、审计 GitHub 发布边界并生成 checkpoint Release 资产：

```bash
bash scripts/reproduce/finalize_public_release.sh --device cuda:0
```

成功标志是 `results/public_release_validation/latest_run_dir.txt` 指向目录中的
`STATUS=SUCCESS` 和 `CURRENT=DONE`。

### B. 从原始 BWDF 数据重新训练

```bash
bash scripts/reproduce/train_from_scratch.sh \
  --device auto \
  --evaluation-device cpu \
  --seeds 0
```

流程会先完成数据与图构建，再完成全部10个模型/任务；只有全部训练完成并写入
`TRAINING_COMPLETE` 后才读取 Test target。输出目录非空时拒绝静默覆盖。

单阶段命令、内部函数调用链、每个文件的用途和完整输出清单见
[`docs/FULL_PIPELINE_CN.md`](docs/FULL_PIPELINE_CN.md)。当前版本只训练和发布10组，
不存在第二套 DCRNN checkpoint。

## 代码流转

```text
wf4bwdf 原始数据
  -> split-aware 清洗与特征
  -> 仅训练期 Pearson 功能图
  -> DCRNN(Base) / STGCN / State / FA-DPR / Full
  -> Validation 选择 checkpoint
  -> TRAINING_COMPLETE
  -> common-46 Test（teacher forcing=0）
  -> 总体表 / DMA表 / Day1-Day7表 / Pearson图
```

每个阶段的入口、函数调用、输出和防泄漏条件均见
[`docs/FULL_PIPELINE_CN.md`](docs/FULL_PIPELINE_CN.md)。

## 目录结构

```text
configs/                 数据、图、模型和唯一论文参数
src/dma_wdf/             数据、图、模型、训练、评估核心源码
scripts/data/            数据构建与质量检查
scripts/graph/           Pearson 图构建、复算和稳定性验证
scripts/train/           DCRNN/STGCN 训练入口
scripts/innovation/      STaR-GNN 训练与 Test 入口
scripts/reproduce/       从零复现、checkpoint 验证和论文制图
tests/                   防泄漏、shape、梯度、协议回归测试
paper/tables/            自动生成的 Test、DMA、逐日和 Pearson 表
paper/figures/           PNG 预览与 PDF 矢量图
paper/reports/           中文结果说明和层级审计
results/paper/frozen_v1/ 本地冻结 checkpoint（GitHub Release 工件）
```

开发阶段的 HPO、旧 SGDR、候选搜索脚本和临时日志不属于公开仓库。
`configs/paper/protocol.yaml` 是论文口径和最终参数的唯一注册表；其他配置是
数据/模型底层依赖，不代表保留了多组候选参数。

## 环境与测试

记录环境：Python 3.11.15、PyTorch 2.9.1+cu128、CUDA 12.8、
NumPy 2.4.6、pandas 3.0.5。

```bash
bash scripts/reproduce/smoke_test.sh
python -m pytest tests -q
```

## 数据与图协议

- 数据期：2021-01-01 至 2023-03-05，小时分辨率；
- 训练截止：2022-12-15 23:00，Test 从 2022-12-16 开始；
- 历史窗口：672 h；预测范围：24 h 与 168 h；
- 图：训练期 DMA Pearson 正相关，零对角，无阈值/Top-K，随机游走归一化；
- 主 Test：common-46，46 个共同预测起点；
- Test：teacher forcing=0，未来需求不进入模型或选参流程。

原始数据不随仓库重新分发，获取方式见 [`data/README.md`](data/README.md)。

## 引用与许可

请参见 [`CITATION.cff`](CITATION.cff) 与 [`LICENSE`](LICENSE)。论文正式发表后，
请用最终 DOI 更新 citation metadata。
