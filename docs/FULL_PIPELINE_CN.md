# STaR-GNN-BWDF 完整复现与代码流转教程

本文档面向第一次拿到本仓库的读者，说明如何从空白环境开始完成以下工作：

- 创建并核验 Conda 环境；
- 从原始 BWDF 数据重新预处理；
- 仅使用训练期数据构建 Pearson 功能图；
- 单独训练或一次训练论文全部模型；
- 使用最佳 Validation checkpoint 执行 common-46 Test；
- 验证 GitHub Release 中的冻结 checkpoint；
- 生成总体、消融、DMA、Day 1--Day 7 和 Pearson 图表；
- 在全新目录和全新环境中执行 clean-room 验收；
- 根据日志、状态文件和输出目录判断每一步是否成功。

文中的命令均在仓库根目录执行。示例根目录为：

```bash
cd /home/dengxu/projects/STaR-GNN-BWDF
```

如果你的仓库位于其他位置，只需替换这一行，不要修改代码中的相对路径。

---

## 1. 先理解论文名与代码名

论文中只有一个 DCRNN 主干。`Base` 不是另一个模型，而是关闭两个创新组件后的
DCRNN。冻结目录保留内部键名 `Base`，但它就是唯一的 DCRNN checkpoint 身份。

| 论文中的名称 | 代码 `variant` | State | FA-DPR | 含义 |
|---|---|---:|---:|---|
| DCRNN | `backbone` | × | × | 基础 DCRNN，也是消融中的 Base |
| DCRNN + State | `dssn_sasr` | ✓ | × | 只加入状态变换 |
| DCRNN + FA-DPR | `fa_dpr` | × | ✓ | 只加入预测对齐日模式检索 |
| STaR-GNN | `full` | ✓ | ✓ | 同时启用两个组件 |
| STGCN | 不适用 | × | × | 外部图时空基线 |

因此，论文表格应写成 `DCRNN`、`DCRNN + State`、`DCRNN + FA-DPR`、
`STaR-GNN` 和 `STGCN`。源码里的 `Base` 仅是 DCRNN 的内部变体标签。
冻结发布与从零复现均为5个模型×2个任务，共10组，不再额外训练或保存第二套 DCRNN。

---

## 2. 四种运行路线

根据目的选择入口，不必每次都从头训练。

| 目的 | 推荐命令 | 是否训练 | 典型耗时 |
|---|---|---:|---:|
| 检查源码、环境和单元测试 | `bash scripts/reproduce/smoke_test.sh` | 否 | 数十秒 |
| 验证冻结 checkpoint 和论文结果 | `bash scripts/reproduce/verify_pretrained.sh --re-evaluate --device cuda:0` | 否 | 数分钟 |
| 从原始数据重新训练论文实验 | `bash scripts/reproduce/train_from_scratch.sh ...` | 是 | 单卡约 10--15 h |
| 模拟外部读者从零验收 | `bash scripts/reproduce/validate_clean_room.sh ...` | 是 | 环境安装加完整训练 |

建议顺序是：先验证源码，再验证冻结 checkpoint，最后才运行耗时较长的从零训练。

---

## 3. 环境安装与基础检查

### 3.1 创建环境

```bash
conda env create -f environment.yml
conda activate star-gnn-bwdf
python -m pip install -e .
```

`environment.yml` 会完成三件事：

1. 创建 Python 3.11 环境；
2. 根据 `requirements-lock.txt` 安装论文记录的精确版本；
3. 以 editable 模式安装当前仓库，使 `src/dma_wdf/` 可以被脚本导入。

其中 `wf4bwdf` 固定到 commit
`5da2d47752190dd69bc3ee612dff043a52e25a2b`，避免上游数据接口变化。

### 3.2 检查环境

```bash
python scripts/reproduce/check_environment.py
```

该脚本检查 Python、PyTorch、NumPy、pandas、PyYAML、PyArrow、SciPy、CUDA、
cuDNN 和可见 GPU，并输出实际版本。它不读取数据，也不训练模型。

### 3.3 检查公开源码未损坏

```bash
bash scripts/reproduce/verify_source.sh
```

该脚本依据 `SOURCE_CHECKSUMS.sha256` 逐个校验公开源码。成功标志为：

```text
公开源码文件校验：PASS
```

### 3.4 运行测试

在尚未生成处理数据时，只运行不依赖数据工件的测试：

```bash
bash scripts/reproduce/smoke_test.sh --source-only
```

数据预处理完成后运行完整测试：

```bash
bash scripts/reproduce/smoke_test.sh
```

第二条命令会额外检查 24 h/168 h 样本索引、common-46 计数、数据划分和防泄漏约束。

---

## 4. 数据预处理：从命令到函数再到输出

### 4.1 原始数据来源

仓库不重新分发原始供水数据。数据由固定版本的 `wf4bwdf` 提供。默认运行：

```bash
bash scripts/data/run_pipeline.sh
```

如果已经在本地克隆 `wf4bwdf`，可以显式指定：

```bash
bash scripts/data/run_pipeline.sh \
  --wf4bwdf-repo /path/to/wf4bwdf
```

自定义输出目录：

```bash
bash scripts/data/run_pipeline.sh \
  --output-dir /path/to/data_build
```

完整参数可执行：

```bash
bash scripts/data/run_pipeline.sh --help
```

### 4.2 实际调用链

| 层级 | 文件/函数 | 作用 |
|---|---|---|
| Bash 入口 | `scripts/data/run_pipeline.sh` | 检查配置和依赖，按顺序执行构建与三类验证 |
| Python 模块入口 | `python -m dma_wdf.data.pipeline` | 解析命令行参数并调用 `main()` |
| 总编排函数 | `dma_wdf.data.pipeline.run_data_build()` | 从原始数据到模型输入表和样本索引的完整编排 |
| 原始数据 | `loader.load_raw_dataset()` | 调用 `wf4bwdf.load_complete_dataset()` 读取需求、气象、日历和 DMA 属性 |
| 时间裁剪 | `loader.select_period()` | 保留 2021-01-01 至 2023-03-05 的论文时间段 |
| 分区插值 | `interpolation.interpolate_by_splits()` | 训练集与测试集分别插值，禁止跨边界填补 |
| 异常阈值 | `outlier_detection.fit_iqr_thresholds()` | 仅用训练期拟合 Tukey IQR 阈值 |
| 异常处理 | `outlier_detection.apply_iqr_thresholds()` | 冻结阈值后分别应用到训练集和测试集 |
| 气象字段 | `weather_features.rename_weather()` | 统一 Rain、Temperature、Humidity、Windspeed 名称 |
| 日历特征 | `temporal_features.build_temporal_features()` | 构建 hour、weekday、holiday 等时间特征 |
| 样本索引 | `sliding_window.build_sample_index()` | 生成 24 h 与 168 h 滑动窗口索引 |
| 指标自检 | `metrics.compute_metrics()` | 检查完美预测和训练均值预测的指标实现 |
| 产物验证 | `scripts/data/validate_preprocessing.py::validate()` | 检查行数、边界、缺失值、IQR 拟合范围和索引 |
| 质量检查 | `dma_wdf.quality.inspect_processed` | 生成处理后数据质量报告 |

`scripts/data/run_pipeline.sh` 的四步顺序为：

1. `run_data_build()` 构建处理数据；
2. `validate_preprocessing.py` 验证防泄漏和物理产物；
3. `inspect_processed` 输出质量检查；
4. `compare_paper.sh` 对照论文协议。

### 4.3 处理逻辑

核心时间边界来自 `configs/data/paper_split.yaml`：

| 分区 | 开始 | 结束 | 小时数 |
|---|---|---|---:|
| 全部论文期 | 2021-01-01 00:00 | 2023-03-05 23:00 | 19,056 |
| 训练期 | 2021-01-01 00:00 | 2022-12-15 23:00 | 17,136 |
| 测试期 | 2022-12-16 00:00 | 2023-03-05 23:00 | 1,920 |

防泄漏规则来自 `configs/data/preprocessing.yaml`：

- 训练期和测试期分别插值；
- IQR 阈值只在 17,136 条训练期记录上拟合；
- 同一组冻结阈值应用于两个分区；
- scaler 只在训练样本上拟合；
- 预测历史长度固定为 672 h，步长为 24 h。

### 4.4 输出文件

默认输出目录为 `data/processed/data_build/`。

| 文件 | 内容 | 下游使用者 |
|---|---|---|
| `demand_hourly.parquet` | 清洗后的 10 个 DMA 小时需水 | 构图、训练、测试 |
| `demand_interpolated_before_outliers.parquet` | 异常值处理前的插值需求 | 数据审计 |
| `demand_outlier_mask.parquet` | 被 IQR 判定为异常的位置 | 数据审计 |
| `weather_hourly.parquet` | 统一字段后的小时气象 | 模型输入构建器 |
| `temporal_hourly.parquet` | 小时、星期、节假日等日历特征 | 历史输入和 future calendar |
| `combined_hourly_features.parquet` | 需求、气象和时间特征合并表 | 人工检查和扩展实验 |
| `sample_index_single_step_24h.csv` | 24 h 任务的样本起点和划分 | DCRNN/STGCN 数据集构建器 |
| `sample_index_multi_step_168h.csv` | 168 h 任务的样本起点和划分 | DCRNN/STGCN 数据集构建器 |
| `demand_iqr_thresholds.csv` | 每个 DMA 的训练期 Q1/Q3/IQR 阈值 | 防泄漏审计 |
| `interpolation_split_profile.csv` | 分区插值统计 | 数据质量报告 |
| `split_summary.json` | 训练/测试边界和行数 | 测试与 clean-room 审计 |
| `quality_checks.json` | 所有质量检查及总状态 | 自动化验收 |
| `data_build_report.md` | 人可读的数据构建报告 | 论文补充材料准备 |
| `status.json` | `all_passed` 和样本摘要 | 流程成功判断 |

质量报告另存于 `results/data_quality/`。

### 4.5 如何判断数据步骤成功

```bash
python - <<'PY'
from pathlib import Path
import json

path = Path("data/processed/data_build/status.json")
payload = json.loads(path.read_text(encoding="utf-8"))
print("all_passed =", payload["all_passed"])
print(payload["split_summary"])
PY
```

必须看到 `all_passed = True`，并且训练/测试行数分别为 17,136 和 1,920。

---

## 5. Pearson 功能图：构建、复算与图件

### 5.1 运行命令

```bash
bash scripts/graph/run_graph_pipeline.sh
```

使用自定义图配置时，配置路径是第一个位置参数：

```bash
bash scripts/graph/run_graph_pipeline.sh \
  configs/graph/pearson_static.yaml
```

### 5.2 实际调用链

| 层级 | 文件/函数 | 作用 |
|---|---|---|
| Bash 入口 | `scripts/graph/run_graph_pipeline.sh` | 先构图，再独立复算验证 |
| 构图入口 | `build_static_graph.py::main()` | 读取配置并调用 `build()` |
| 构图编排 | `build_static_graph.py::build()` | 截取训练期需求、构图、保存矩阵和图件 |
| 数学实现 | `data.graph.build_pearson_graph()` | 计算相关矩阵、正相关邻接和随机游走矩阵 |
| 数学检查 | `data.graph.validate_graph()` | 检查形状、对称性、对角、非负性和行和 |
| 保存图 | `data.graph.save_graph()` | 将矩阵和图身份写入压缩 NPZ |
| 保存表 | `data.graph.save_matrix_csvs()` | 输出 Pearson、邻接、随机游走 CSV |
| 构图可视化 | `data.graph.plot_graph_diagnostics()` | 输出热力图、加权网络和节点度图 |
| 独立验证 | `validate_static_graph.py::validate()` | 从训练需求重新构图并逐元素比较 |
| 稳定性 | `compute_segment_stability()` | 比较四个训练期分段的相关结构 |
| Bootstrap | `moving_block_bootstrap()` | 168 h block bootstrap 计算边置信区间 |

图的固定定义为：

\[
A_{ij}=\begin{cases}
\max(r_{ij},0), & i\ne j,\\
0, & i=j,
\end{cases}
\qquad P=D^{-1}A.
\]

不设置相关性阈值，不做 Top-K，不取绝对值，不加入邻接自环。全部矩阵只由训练期
17,136 行需求计算。

### 5.3 输出文件

模型读取：

```text
artifacts/graphs/bwdf_pearson_static_graph.npz
```

该 NPZ 包含 `static_corr`、`static_adj`、`random_walk`、DMA 顺序、训练起止时间、
训练行数、图方法和训练需求 SHA-256。

人类可读和诊断文件位于 `results/graph/pearson_static/`：

| 文件 | 含义 |
|---|---|
| `static_corr.csv` | 10×10 Pearson 相关矩阵 |
| `static_adj.csv` | 截断负相关并清零对角后的邻接矩阵 |
| `random_walk.csv` | DCRNN 使用的行归一化传播矩阵 |
| `graph_metadata.json` | 图身份和训练期信息 |
| `static_corr_heatmap.png` | Pearson 热力图 |
| `static_adj_heatmap.png` | 正相关邻接热力图 |
| `weighted_network.png` | DMA 加权功能关联图 |
| `weighted_degree.png` | 节点加权度柱状图 |
| `segment_stability.json/csv` | 分段稳定性 |
| `edge_bootstrap_intervals.csv` | 边权 block-bootstrap 区间 |
| `validation_report.json` | 独立复算验证结论 |

### 5.4 如何判断图步骤成功

```bash
python - <<'PY'
from pathlib import Path
import json

path = Path("results/graph/pearson_static/validation_report.json")
report = json.loads(path.read_text(encoding="utf-8"))
print("all_passed =", report["all_passed"])
print("fit_rows =", report["fit_rows"])
print("fit_start =", report["fit_start"])
print("fit_end =", report["fit_end"])
PY
```

必须满足 `all_passed=True`、`fit_rows=17136`，并且复算的三个矩阵误差均不超过
`1e-10`。

---

## 6. 单个模型训练

一次性复现全部实验见第 8 节。本节用于调试或只训练一个模型。

### 6.1 训练 STGCN

```bash
python scripts/train/train_model.py \
  --model stgcn \
  --task 24h \
  --config configs/paper/stgcn_24h.yaml \
  --seed 0 \
  --device cuda:0 \
  --output-dir results/manual/stgcn/24h/seed_0
```

将 `24h` 和配置替换为 `168h` 即可训练周尺度任务。

调用链为：

```text
train_model.py::main
  -> load_config_with_inheritance
  -> prepare_forecast_training_data
  -> set_reproducible_seed
  -> build_stgcn_model
  -> train_stgcn
```

### 6.2 训练 DCRNN（Base）

论文中的 DCRNN 是 `backbone` 变体：

```bash
python scripts/innovation/train_star_dcrnn.py \
  --variant backbone \
  --task 24h \
  --config configs/paper/star_gnn_24h.yaml \
  --seed 0 \
  --device cuda:0 \
  --output-dir results/manual/dcrnn/24h/seed_0
```

### 6.3 训练 DCRNN + State

```bash
python scripts/innovation/train_star_dcrnn.py \
  --variant dssn_sasr \
  --task 24h \
  --config configs/paper/star_gnn_24h.yaml \
  --seed 0 \
  --device cuda:0 \
  --output-dir results/manual/state/24h/seed_0
```

### 6.4 训练 DCRNN + FA-DPR

```bash
python scripts/innovation/train_star_dcrnn.py \
  --variant fa_dpr \
  --task 24h \
  --config configs/paper/star_gnn_24h.yaml \
  --seed 0 \
  --device cuda:0 \
  --output-dir results/manual/fa_dpr/24h/seed_0
```

### 6.5 训练完整 STaR-GNN

```bash
python scripts/innovation/train_star_dcrnn.py \
  --variant full \
  --task 24h \
  --config configs/paper/star_gnn_24h.yaml \
  --seed 0 \
  --device cuda:0 \
  --output-dir results/manual/full/24h/seed_0
```

四个 DCRNN 变体的内部调用链相同：

```text
train_star_dcrnn.py::main
  -> load_config_with_inheritance
  -> prepare_dcrnn_training_data
  -> set_reproducible_seed
  -> build_star_dcrnn_model(variant=...)
  -> train_star_dcrnn
```

其中 `prepare_dcrnn_training_data()` 会：

- 读取需求、气象、时间特征和任务样本索引；
- 将 development 样本分为 fit 与 Validation；
- 只用 fit 数据拟合 demand/weather scaler；
- 生成 `x_past`、`future_exog` 和 `y_scaled`；
- 保留未来已知日历，但不向模型提供未来需求。

`build_star_dcrnn_model()` 根据 `variant` 启用 `DSSNSASR` 和/或
`ForecastAlignedDailyPatternRetrieval`。`train_star_dcrnn()` 使用 Adam、MAE、
inverse-sigmoid scheduled sampling、梯度裁剪和 Validation early stopping。

### 6.6 训练输出

每个训练目录包含：

| 文件/目录 | 作用 |
|---|---|
| `checkpoint_best.pt` | Validation normalized MAE 最低时的 checkpoint |
| `checkpoint_last.pt` | 最后一个 epoch，用于 `--resume`；正式完成后可按配置删除 |
| `history.csv` | 每个 epoch 的训练、Validation 四指标、teacher forcing 和耗时 |
| `training_summary.json` | 完成状态、最佳 epoch、运行时和配置摘要 |
| `tensorboard/` | TensorBoard event 文件 |

查看训练曲线：

```bash
tensorboard --logdir results/manual --port 6006
```

继续中断的同一训练目录：

```bash
python scripts/innovation/train_star_dcrnn.py \
  --variant full \
  --task 24h \
  --config configs/paper/star_gnn_24h.yaml \
  --seed 0 \
  --device cuda:0 \
  --output-dir results/manual/full/24h/seed_0 \
  --resume
```

脚本不会静默覆盖非空目录。`--overwrite` 会先把旧目录移动到带时间戳的备份目录。

---

## 7. 单个 checkpoint 测试

### 7.1 测试 STGCN

```bash
python scripts/evaluate/test_model.py \
  --model stgcn \
  --task 24h \
  --checkpoint results/manual/stgcn/24h/seed_0/checkpoint_best.pt \
  --output-dir results/manual/stgcn/24h/seed_0/evaluation \
  --device cuda:0
```

### 7.2 测试 DCRNN、State、FA-DPR 或 Full

以完整 STaR-GNN 为例：

```bash
python scripts/innovation/evaluate_star_dcrnn.py \
  --variant full \
  --task 24h \
  --checkpoint results/manual/full/24h/seed_0/checkpoint_best.pt \
  --output-dir results/manual/full/24h/seed_0/evaluation \
  --device cuda:0
```

将 `--variant` 改为 `backbone`、`dssn_sasr` 或 `fa_dpr` 可测试其余变体。

调用链为：

```text
evaluate_star_dcrnn.py::main
  -> run_star_checkpoint_evaluation
  -> torch.load(checkpoint)
  -> prepare_dcrnn_test_data
  -> build_star_dcrnn_model
  -> validate checkpoint graph identity
  -> load_state_dict(strict=True)
  -> model(..., teacher_forcing_ratio=0.0)
  -> evaluate_predictions / evaluate_aggregate_total_predictions
  -> 保存指标、预测、机制诊断和 test_summary
```

测试目录包含：

| 文件 | 内容 |
|---|---|
| `metrics_operational.csv` | 可运行测试起点的 DMA 级指标 |
| `metrics_strict_within_test.csv` | 历史和目标严格位于 Test 内的指标 |
| `metrics_common_46.csv` | 46 个共同起点的 DMA A--J 指标 |
| `metrics_aggregate_total_common_46.csv` | 论文总体 MAE/MAPE/RMSE/NSE |
| `predictions.npz` | 预测和真实值，用于逐日与 DMA 图表 |
| `mechanism_diagnostics.npz` | State/FA-DPR 内部诊断，若该变体启用 |
| `paper_comparison.csv` | 与水务领域报告结果的统一口径比较 |
| `test_summary.json` | checkpoint SHA、图身份、协议计数和防泄漏字段 |

检查 Test 是否合规：

```bash
python - <<'PY'
from pathlib import Path
import json

path = Path("results/manual/full/24h/seed_0/evaluation/test_summary.json")
summary = json.loads(path.read_text(encoding="utf-8"))
print("status =", summary["status"])
print("common_46 =", summary["protocol_counts"]["common_46"])
print("teacher_forcing_ratio =", summary["teacher_forcing_ratio"])
print("test_used_for_selection =", summary["test_targets_used_for_training_or_selection"])
PY
```

正确结果必须是：`completed`、`46`、`0.0` 和 `False`。

---

## 8. 一条命令从原始数据复现全部实验

```bash
bash scripts/reproduce/train_from_scratch.sh \
  --device auto \
  --evaluation-device cpu \
  --seeds 0
```

如果物理 GPU 6 通过 `CUDA_VISIBLE_DEVICES` 暴露给程序，它在 PyTorch 中会成为
逻辑设备 `cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=6 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash scripts/reproduce/train_from_scratch.sh \
  --device cuda:0 \
  --evaluation-device cuda:0 \
  --seeds 0
```

### 8.1 编排调用链

```text
train_from_scratch.sh
  -> check_environment.py
  -> reproduce.py::main
       -> scripts/data/run_pipeline.sh
       -> scripts/graph/run_graph_pipeline.sh
       -> _train(...) 逐项训练
       -> 写 TRAINING_COMPLETE
       -> _evaluate(...) 逐项 common-46 Test
       -> build_paper_tables.py
       -> build_detailed_test_artifacts.py
```

`reproduce.py::_train()` 只复用同时具有 `checkpoint_best.pt` 和
`training_summary.json(status=completed)` 的完整目录；非空的半成品目录会直接报错，
避免把不同运行混在一起。

`reproduce.py::_evaluate()` 只在所有训练完成并写入 `TRAINING_COMPLETE` 后执行。
这是 Test target 与训练/选参之间的硬边界。

### 8.2 常用参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--device` | `auto` | 训练设备 |
| `--evaluation-device` | `cpu` | Test 推理设备 |
| `--seeds` | `0` | 逗号分隔随机种子，如 `0,1,2,3,4` |
| `--output` | `results/paper/reproduction` | 完整复现输出根目录 |
| `--skip-data` | 关闭 | 已确认数据和图时跳过重建 |
| `--skip-baselines` | 关闭 | 调试时跳过基线，不用于正式全流程 |
| `--control-dir` | 无 | 持续写入 `CURRENT`，便于后台监控 |

### 8.3 后台运行和监控

```bash
mkdir -p results/logs
STAMP="$(date +%Y%m%d-%H%M%S)"
CONTROL="results/training_control/${STAMP}"
LOG="results/logs/from_scratch_${STAMP}.log"

nohup env \
  CUDA_VISIBLE_DEVICES=6 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  bash scripts/reproduce/train_from_scratch.sh \
    --device cuda:0 \
    --evaluation-device cuda:0 \
    --seeds 0 \
    --control-dir "${CONTROL}" \
  > "${LOG}" 2>&1 &

PID=$!
printf '%s\n' "${PID}" > results/logs/from_scratch.pid
printf '%s\n' "${LOG}" > results/logs/from_scratch_latest_log.txt
```

查看状态：

```bash
PID="$(cat results/logs/from_scratch.pid)"
LOG="$(cat results/logs/from_scratch_latest_log.txt)"

kill -0 "${PID}" 2>/dev/null && ps -fp "${PID}" || echo "主进程已结束"
cat "${CONTROL}/CURRENT"
tail -n 100 "${LOG}"
```

不要只根据 GPU 利用率判断任务是否结束；Validation、文件保存和两个任务切换时 GPU
可能短暂为 0。应同时检查 PID、`CURRENT`、日志更新时间和输出文件。

---

## 9. 冻结 checkpoint 验证

GitHub Release 中的冻结工件应解压到：

```text
results/paper/frozen_v1/
```

### 9.1 只验证，不重新推理

```bash
bash scripts/reproduce/verify_pretrained.sh
```

内部调用：

```text
verify_pretrained.sh
  -> check_environment.py
  -> verify_paper_release.py
       -> _verify_checksums
       -> _audit_checkpoint
       -> _collect_release_metrics
       -> _verify_expected
       -> _verify_paper_hierarchy
  -> build_paper_tables.py
  -> build_detailed_test_artifacts.py
```

该模式检查 SHA-256、checkpoint 元数据、46 个 common 起点、teacher forcing、
`test_targets_used_for_training_or_selection=false`、注册四指标和 31/32 消融关系，
然后直接从冻结预测生成表图。

### 9.2 重新执行 checkpoint 推理

```bash
bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

重新推理会生成 `results/paper/verification/<timestamp>/`，其中：

| 文件/目录 | 作用 |
|---|---|
| `reevaluation_metric_differences.csv` | 每个模型、任务、指标的冻结值与复算值 |
| `reevaluation_summary.json` | 40 项比较通过数和最大误差 |
| `star/` | 四个 DCRNN 变体的复算产物 |
| `baseline/` | 兼容性基线的复算产物 |

GPU/cuDNN 的末位浮点差异使用 `5e-4` 绝对容差和 `5e-4` 相对容差；checkpoint
SHA、任务、seed、图身份、common-46 数量和防泄漏字段仍要求完全一致。

### 9.3 论文层级要求

四个指标逐项判断，误差指标越低越好，NSE 越高越好：

- `DCRNN + State` 优于 `DCRNN`；
- `STaR-GNN` 优于 `DCRNN + State`；
- `STaR-GNN` 优于 `DCRNN + FA-DPR`；
- 168 h 的 FA-DPR MAPE 例外被透明保留，因此总关系为 31/32。

---

## 10. 生成论文表格和图

如果冻结预测已存在，可单独重新制表作图，不需要训练：

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

`build_detailed_test_artifacts.py::main()` 的处理顺序是：

1. `_collect_aggregate()` 收集所有模型的 common-46 总体四指标；
2. `_collect_dma()` 收集 A--J 的 MAE、MAPE、RMSE、NSE；
3. `_load_common_predictions()` 读取共同 46 个预测起点；
4. `_collect_daily_168h()` 把 168 h 切成七个 24 h 区间；
5. `_hierarchy_report()` 计算完整模型、组件和基线之间的方向关系；
6. `_plot_metric_panels()` 生成总体和消融四指标图；
7. `_plot_daily()` 生成 Day 1--Day 7 曲线；
8. `_plot_dma_mae()` 生成 DMA 异质性图；
9. `_build_graph_artifacts()` 生成 Pearson 表和热力图；
10. `_write_report()` 生成中文 Test 结果说明。

主要表格：

| 文件 | 内容 |
|---|---|
| `paper/tables/test_all_models_common46.csv` | 全部模型、两个任务、四指标 |
| `paper/tables/test_ablation_common46.csv` | DCRNN 及两个组件的消融表 |
| `paper/tables/test_dma_metrics_long.csv` | DMA A--J 的长表 |
| `paper/tables/test_dma_*_wide_24h.csv` | 24 h 每项指标的 DMA 宽表 |
| `paper/tables/test_dma_*_wide_168h.csv` | 168 h 每项指标的 DMA 宽表 |
| `paper/tables/test_day1_day7_metrics.csv` | 168 h 七日逐日长表 |
| `paper/tables/test_day1_day7_*_wide.csv` | 每项指标的 Day 1--Day 7 宽表 |
| `paper/tables/pearson_correlation.csv` | Pearson 矩阵 |
| `paper/tables/pearson_adjacency.csv` | 正相关邻接矩阵 |
| `paper/tables/pearson_random_walk.csv` | 随机游走矩阵 |
| `paper/tables/test_hierarchy.json` | 论文层级自动审计 |

主要图件同时输出 PNG 和 PDF：

```text
paper/figures/test_overall_24h.*
paper/figures/test_overall_168h.*
paper/figures/test_ablation_24h.*
paper/figures/test_ablation_168h.*
paper/figures/test_dma_mae_24h.*
paper/figures/test_dma_mae_168h.*
paper/figures/test_day1_day7_models.*
paper/figures/test_day1_day7_ablation.*
paper/figures/pearson_correlation_heatmap.*
```

中文解释位于 `paper/reports/TEST_RESULTS_CN.md`。

对生成物执行只读审计：

```bash
python scripts/reproduce/audit_release_inventory.py \
  --require-paper-artifacts
```

---

## 11. 全面 checkpoint 发布验收

作者从旧工程导入本地冻结工件时使用：

```bash
bash scripts/reproduce/validate_everything.sh \
  /path/to/DMA-WDF \
  --device cuda:0
```

该流程不重新训练，依次完成：源码 SHA、环境、原子导入、物理清单、测试套件、
checkpoint 元数据、common-46、重新推理、论文图表和旧 HPO/SGDR 隔离。

状态目录：

```bash
DIR="$(cat results/release_validation/latest_run_dir.txt)"
cat "${DIR}/STATUS"
cat "${DIR}/CURRENT"
cat "${DIR}/FINAL_REPORT.txt"
```

只有 `STATUS=SUCCESS`、`CURRENT=DONE` 且 `FINAL_REPORT.txt` 全部为 PASS 才表示通过。
历史失败目录不会使后来的成功目录失效；应以 `latest_run_dir.txt` 指向的运行目录为准。

如果10组冻结工件已经在当前仓库，不需要再从旧工程导入，也不需要重新训练。直接运行：

```bash
bash scripts/reproduce/finalize_public_release.sh --device cuda:0
```

该脚本明确不调用训练入口，依次执行 DCRNN/Base 唯一化、源码与测试、10组 checkpoint
复推理、40项指标审计、论文表图重建、公开仓库边界审计和 Release 资产打包。最终状态：

```bash
DIR="$(cat results/public_release_validation/latest_run_dir.txt)"
cat "${DIR}/STATUS"
cat "${DIR}/CURRENT"
cat "${DIR}/FINAL_REPORT.txt"
```

---

## 12. Clean-room 从零验收

这是上传 GitHub 前最严格的检查。它不会复用当前环境、处理数据、Pearson 图或训练
结果，只会装入冻结 Release asset 用于独立验证和结果对照。

```bash
cd /home/dengxu/projects/STaR-GNN-BWDF
source /home/dengxu/miniconda3/etc/profile.d/conda.sh

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="cleanroom_validation_${STAMP}.log"
WORKSPACE="/home/dengxu/projects/STaR-GNN-BWDF-cleanroom-${STAMP}"

nohup env \
  CUDA_VISIBLE_DEVICES=6 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  bash scripts/reproduce/validate_clean_room.sh \
    --workspace "${WORKSPACE}" \
    --frozen-release results/paper/frozen_v1 \
    --device cuda:0 \
    --evaluation-device cuda:0 \
  > "${LOG}" 2>&1 &

PID=$!
printf '%s\n' "${PID}" > cleanroom_validation.pid
printf '%s\n' "${LOG}" > cleanroom_validation_latest_log.txt
```

11 个阶段依次为：

1. 复制 `SOURCE_CHECKSUMS.sha256` 登记的公开源码并装入冻结 Release；
2. 从 `environment.yml` 创建全新 Conda prefix；
3. 源码 SHA、环境和非数据测试；
4. 冻结 checkpoint 离线验证；
5. 原始数据预处理和训练期 Pearson 图；
6. 数据生成后的完整测试；
7. 从零训练并执行 common-46；
8. 从零结果、图身份、防泄漏和论文层级审计；
9. 用新数据和新图复推理冻结 checkpoint；
10. 从冻结预测生成并审计论文表图；
11. 再次验证公开源码未被运行过程修改。

监控：

```bash
DIR="$(cat results/cleanroom_validation/latest_run_dir.txt)"
LOG="$(cat cleanroom_validation_latest_log.txt)"
PID="$(cat cleanroom_validation.pid)"

kill -0 "${PID}" 2>/dev/null && ps -fp "${PID}" || echo "主进程已结束"
echo -n "STATUS:  "; cat "${DIR}/STATUS"
echo -n "CURRENT: "; cat "${DIR}/CURRENT"
tail -n 100 "${LOG}"
```

最终必须看到 `STATUS=SUCCESS`、`CURRENT=DONE` 和：

```text
STaR-GNN-BWDF clean-room从零复现：PASS
```

若失败，`STATUS` 会记录 `FAILED exit_code=...`，`CURRENT` 保留失败阶段。不要删除
失败目录；先根据日志最后一个 Traceback 修复，再使用新的时间戳创建全新 clean-room。

---

## 13. 目录说明

| 目录 | 主要内容 | 是否提交 Git |
|---|---|---:|
| `.github/workflows/` | GitHub Actions 源码和单元测试 | 是 |
| `configs/data/` | 时间划分、插值、IQR 和特征配置 | 是 |
| `configs/graph/` | Pearson 图固定协议 | 是 |
| `configs/model/` | DCRNN、STGCN、STaR-GNN 结构参数 | 是 |
| `configs/train/` | 模型训练底层配置 | 是 |
| `configs/paper/` | 唯一论文参数与任务入口 | 是 |
| `configs/evaluation/` | MSCMNet 文献结果和指标口径 | 是 |
| `data/` | 数据获取说明；处理数据在 `data/processed/` | 说明提交，处理数据不提交 |
| `artifacts/graphs/` | 训练期 Pearson 图 NPZ | 小型固定工件可随 Release |
| `src/dma_wdf/data/` | 数据加载、插值、异常值、窗口和图 | 是 |
| `src/dma_wdf/models/` | DCRNN、STGCN、State、FA-DPR、STaR-GNN | 是 |
| `src/dma_wdf/training/` | 训练循环、早停、checkpoint、TensorBoard | 是 |
| `src/dma_wdf/evaluation/` | checkpoint 加载、推理和四指标 | 是 |
| `src/dma_wdf/quality/` | 数据和协议质量报告 | 是 |
| `scripts/data/` | 数据构建的命令行入口 | 是 |
| `scripts/graph/` | Pearson 图构建与复算入口 | 是 |
| `scripts/train/` | STGCN和底层DCRNN通用实现；论文DCRNN由`backbone`入口调用 | 是 |
| `scripts/innovation/` | 四个 DCRNN 变体的训练和评估入口 | 是 |
| `scripts/evaluate/` | 基线 Test 入口 | 是 |
| `scripts/reproduce/` | 从零复现、checkpoint 验证、clean-room 和制图 | 是 |
| `tests/` | shape、梯度、协议、防泄漏和工件回归测试 | 是 |
| `paper/tables/` | 论文 CSV/Markdown 表格 | 是 |
| `paper/figures/` | 论文 PNG/PDF 图件 | 是 |
| `paper/reports/` | 中文结果说明和层级审计 | 是 |
| `results/paper/frozen_v1/` | checkpoint、预测和 Test 汇总 | 建议作为 GitHub Release asset |
| `results/paper/reproduction/` | 从零训练结果 | 否 |
| `results/release_validation/` | 本地发布验收日志 | 否 |
| `results/cleanroom_validation/` | clean-room 控制状态 | 否 |

---

## 14. 主要脚本索引

| 脚本 | 用途 | 是否训练 |
|---|---|---:|
| `scripts/data/run_pipeline.sh` | 原始 BWDF → 处理数据和索引 | 否 |
| `scripts/data/validate_preprocessing.py` | 验证数据划分和防泄漏 | 否 |
| `scripts/graph/run_graph_pipeline.sh` | 构建并复算 Pearson 图 | 否 |
| `scripts/train/train_model.py` | 训练 STGCN 或兼容基线 | 是 |
| `scripts/innovation/train_star_dcrnn.py` | 训练 DCRNN/State/FA-DPR/Full | 是 |
| `scripts/evaluate/test_model.py` | 测试 STGCN 或兼容基线 | 否 |
| `scripts/innovation/evaluate_star_dcrnn.py` | 测试四个 DCRNN 变体 | 否 |
| `scripts/reproduce/train_from_scratch.sh` | 从原始数据执行全部论文流程 | 是 |
| `scripts/reproduce/verify_pretrained.sh` | 验证冻结 checkpoint 并制图 | 否 |
| `scripts/reproduce/build_paper_tables.py` | 生成总体/消融文献表 | 否 |
| `scripts/reproduce/build_detailed_test_artifacts.py` | 生成 DMA、逐日和 Pearson 表图 | 否 |
| `scripts/reproduce/audit_release_inventory.py` | 只读检查发布工件物理完整性 | 否 |
| `scripts/reproduce/audit_public_repository.py` | 审计拟上传文件、文档、密钥、大文件和唯一模型身份 | 否 |
| `scripts/reproduce/consolidate_dcrnn_base_release.py` | 删除重复DCRNN并迁移索引、重建SHA | 否 |
| `scripts/reproduce/finalize_public_release.sh` | 无重训完成唯一化、复推理、制图和打包 | 否 |
| `scripts/reproduce/package_frozen_release.py` | 生成确定性的GitHub Release checkpoint资产 | 否 |
| `scripts/reproduce/validate_everything.sh` | 作者本地发布前全面验收 | 否 |
| `scripts/reproduce/validate_clean_room.sh` | 全新环境和目录的从零验收 | 是 |
| `scripts/reproduce/verify_source.sh` | 校验公开源码 SHA | 否 |
| `scripts/reproduce/smoke_test.sh` | 编译、配置、模型和防泄漏测试 | 否 |

---

## 15. 常见问题

### 15.1 为什么 `nvidia-smi` 显示 GPU 6，但程序日志写 `cuda:0`？

设置 `CUDA_VISIBLE_DEVICES=6` 后，物理 GPU 6 是进程唯一可见设备，PyTorch 会将它
重新编号为逻辑 `cuda:0`。这是正常行为。

### 15.2 为什么运行过程中 GPU 利用率会短暂为 0？

epoch 结束、Validation、checkpoint 保存、数据加载和任务切换都可能短暂不使用 GPU。
应根据 PID、日志更新时间和 `CURRENT` 综合判断。

### 15.3 为什么训练提前结束，没有达到 100 epoch？

`max_epochs=100` 是上限。Validation 连续 15 个 epoch 不改善时 early stopping 会结束
训练，并保留最佳 Validation checkpoint。这是正式协议的一部分。

### 15.4 为什么 checkpoint 复推理和冻结值最后几位不同？

不同 CUDA/cuDNN 内核可能引入极小归约差异。复推理只对四指标使用预注册容差；
checkpoint SHA、common-46 索引、图身份和防泄漏字段仍严格一致。

### 15.5 为什么 source-only 测试通过，完整测试却提示样本索引不存在？

完整测试依赖 `data/processed/data_build/sample_index_*.csv`。先执行数据预处理，再运行
不带 `--source-only` 的 `smoke_test.sh`。

### 15.6 为什么输出目录非空时训练拒绝启动？

这是防止将两个实验混在一起。完整运行可使用 `--resume` 继续，或显式使用
`--overwrite` 先归档旧目录。不要直接删除仍需审计的失败结果。

### 15.7 什么结果可以写入论文？

论文结果必须来自：固定 `configs/paper/protocol.yaml`、seed 0、最佳 Validation
checkpoint、teacher forcing 为 0 的 common-46 Test。调试用 `--max-epochs`、
`--max-train-batches` 或不同输出目录产生的结果不能替代正式结果。

---

## 16. 上传 GitHub 前的最小通过条件

- `verify_source.sh` 通过；
- 完整 `smoke_test.sh` 通过；
- 冻结 checkpoint 的 SHA、元数据和 common-46 通过；
- checkpoint 重新推理的四指标审计通过；
- 数据预处理和 Pearson 独立复算通过；
- 总体、消融、DMA、Day 1--Day 7、Pearson 表图均存在且非空；
- clean-room 最终 `STATUS=SUCCESS`；
- README、本文档和实际命令保持一致；
- 不上传旧 HPO、旧 SGDR、候选搜索、服务器日志、PID、Conda 环境或临时压缩包；
- 大型 checkpoint 和预测文件作为 GitHub Release asset 发布，并提供 SHA-256。

一键收口通过后，Release 资产位于：

```text
dist/STaR-GNN-BWDF-frozen-v1.tar.gz
dist/STaR-GNN-BWDF-frozen-v1.tar.gz.sha256
```

上传后还应在另一个新目录 `git clone` 仓库，下载 Release asset，再执行一次第 12 节，
以确认公开仓库本身能够独立复现。

---

## 17. 相关文档

- 方法结构和公式对应：[`METHOD_CN.md`](METHOD_CN.md)；
- 冻结结果、指标来源和论文表图：
  [`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md)；
- GitHub 上传、Release 资产和发布后复验：[`RELEASE_CN.md`](RELEASE_CN.md)。
