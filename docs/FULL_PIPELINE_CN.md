# STaR-GNN-BWDF 完整复现与代码流转教程

本文档给出当前仓库的权威复现流程：环境 → 数据 → Pearson 图 → 模型训练/冻结 checkpoint → common-46 Test → manuscript-facing 表格/图件 → clean-room 验收。

当前 Journal of Hydrology 稿件的结果与图表口径以 [`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md) 和 [`MANUSCRIPT_FIGURES_FINAL_CN.md`](MANUSCRIPT_FIGURES_FINAL_CN.md) 为准。

## 1. 论文模型与代码名

正式 factorial ablation 只有四个模型：

| 论文名称 | Internal variant | SAS-Norm | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` / `Base` | × | × |
| DCRNN + SAS-Norm | `dssn_sasr` / `State` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` / `FA-DPR` | × | ✓ |
| STaR-GNN | `full` / `Full` | ✓ | ✓ |

STGCN 是独立图时空 baseline，不属于消融。

冻结发布仍包含 5 个模型身份 × 2 个任务 = 10 组评估：DCRNN/Base、SAS-Norm-only、FA-DPR-only、Full、STGCN。

## 2. 论文冻结设置

```yaml
learning_rate: 0.0003
weight_decay: 0.0
cl_decay_steps: 500
state_loss_weight: 0.03
max_epochs: 100
seed: 0
```

共同设置：

- hidden size = 32；
- recurrent layers = 1；
- batch size = 16；
- early-stopping patience = 15；
- diffusion step `K=2`；
- Test teacher forcing = 0。

冻结参数只根据 Validation 确定。Test 不参与超参数选择、early stopping 或模块取舍。

## 3. 环境安装

```bash
conda env create -f environment.yml
conda activate star-gnn-bwdf
python -m pip install -e .
```

检查：

```bash
python scripts/reproduce/check_environment.py
bash scripts/reproduce/verify_source.sh
bash scripts/reproduce/smoke_test.sh --source-only
```

## 4. 数据预处理

默认：

```bash
bash scripts/data/run_pipeline.sh
```

数据协议：

| 项目 | 设置 |
|---|---|
| 全期 | 2021-01-01 00:00 至 2023-03-05 23:00 |
| 总小时数 | 19,056 |
| 训练期 | 至 2022-12-15 23:00 |
| 训练小时数 | 17,136 |
| Test | 自 2022-12-16 00:00 |
| Test 小时数 | 1,920 |
| DMA | 10 个（A--J） |
| history | 672 h |
| horizon | 24 h / 168 h |
| stride | 24 h |

数据防泄漏规则：

- train/test 分开插值；
- IQR 阈值只在训练期拟合；
- scaler 只在训练样本拟合；
- Test target 不用于训练或参数选择。

数据清洗的投稿示例由 `render_supplementary_figures.py` 按固定规则生成：DMA F 代表最高缺失率，DMA C 代表最高异常值数量。对应的 Supplementary Fig. S1 只保存图件和选择元数据，不在仓库中重复发布原始需水片段。

主要输出：

```text
data/processed/data_build/demand_hourly.parquet
data/processed/data_build/weather_hourly.parquet
data/processed/data_build/temporal_hourly.parquet
data/processed/data_build/sample_index_single_step_24h.csv
data/processed/data_build/sample_index_multi_step_168h.csv
data/processed/data_build/status.json
```

检查：

```bash
python - <<'PY'
from pathlib import Path
import json
p = Path('data/processed/data_build/status.json')
x = json.loads(p.read_text(encoding='utf-8'))
print('all_passed =', x['all_passed'])
print(x['split_summary'])
PY
```

必须看到 `all_passed=True`。

## 5. Pearson 功能图

```bash
bash scripts/graph/run_graph_pipeline.sh
```

固定定义：

\[
A_{ij}=\begin{cases}
\max(r_{ij},0), & i\neq j,\\
0, & i=j,
\end{cases}
\qquad
P=D^{-1}A.
\]

协议：

- 只使用训练期需求；
- Pearson correlation；
- negative correlations clip to zero；
- zero diagonal；
- no threshold / Top-K；
- no self-loop in adjacency；
- static graph；
- random-walk normalization；
- 24 h / 168 h 共用同一图。

输出：

```text
artifacts/graphs/bwdf_pearson_static_graph.npz
results/graph/pearson_static/
```

DCRNN model config 使用 `matrix_key=random_walk`、`max_diffusion_step=2`。

## 6. 验证冻结 checkpoint（推荐先做）

```bash
bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

验证内容至少包括：

- checkpoint SHA-256；
- 图身份；
- common-46 index；
- Test teacher forcing = 0；
- `test_targets_used_for_training_or_selection=false`；
- 重新推理指标与冻结指标一致；
- manuscript-facing metric convention；
- 表格和图件生成器。

## 7. 从原始数据重新训练

```bash
bash scripts/reproduce/train_from_scratch.sh \
  --device auto \
  --evaluation-device cpu \
  --seeds 0
```

正式论文冻结 release 使用 seed 0。开发阶段若进行多 seed 稳健性分析，应与冻结 manuscript result 明确区分。

### 单模型配置

DCRNN：

```text
configs/train/dcrnn_24h.yaml
configs/train/dcrnn_168h.yaml
```

STaR-GNN：

```text
configs/paper/star_gnn_24h.yaml
configs/paper/star_gnn_168h.yaml
```

## 8. common-46 Test

两个任务最终正文比较统一使用 46 个共同 forecast origins。

168 h 的 46 sequences 来自：

```text
80 test days - 28 within-test history days - 7 forecast days + 1 = 46
```

Test 推理不使用 target teacher forcing。

每个评估目录生成：

```text
metrics_common_46.csv
metrics_aggregate_total_common_46.csv
predictions.npz
test_summary.json
```

## 9. 两套 MAE

### 9.1 Manuscript-facing publisher-compatible MAE

\[
MAE_{publisher}=\sum_{i=A}^{J} MAE_i.
\]

用于：

- Table 1 overall comparison；
- Table 2 factorial ablation；
- Figure 1；
- Figure 2；
- Figure 3。

STaR-GNN：

```text
24 h  9.424199
168 h 12.233590
```

### 9.2 Internal aggregate-demand MAE

\[
MAE_{agg}=MAE\left(\sum_i\hat y_i,\sum_i y_i\right).
\]

用于 aggregate-demand trajectory / operational diagnostics。

STaR-GNN：

```text
24 h  4.360841
168 h 4.919812
```

不要混用。

## 10. 最终 Table 1--3

### Table 1

```text
paper/tables/literature/table_literature_comparison_common46.csv
```

9 models：GRU、LSTM、MSNet、MSCMNet_WM、MSCMNet_M、MSCMNet_W、DCRNN、STGCN、STaR-GNN。

GRU/LSTM/MSNet/MSCMNet 为 Que et al. (2024) reported results；DCRNN/STGCN/STaR-GNN 为本仓库 common-46 复评。

### Table 2

```text
paper/tables/literature/table_ablation_common46.csv
```

严格四模型：

```text
DCRNN
DCRNN + SAS-Norm
DCRNN + FA-DPR
STaR-GNN
```

STGCN 不进入 Table 2。

### Table 3

```text
paper/tables/literature/table_star_gnn_dma_common46.csv
```

报告 STaR-GNN 的 DMA A--J MAE/MAPE/RMSE/NSE。

## 11. 30/32 与旧 31/32

当前 **manuscript-facing publisher-compatible factorial cell audit = 30/32**。

两个透明例外：

1. FA-DPR 168 h MAPE `3.277716%` 略高于 DCRNN `3.248413%`；
2. STaR-GNN 168 h publisher-compatible MAE `12.233590` 略高于 SAS-Norm-only `12.207835`。

旧 `31/32` 属于 legacy/internal aggregate-demand hierarchy：在 aggregate-demand MAE 下 Full `4.919812` 优于 SAS-Norm `5.122511`，因此当时只有 FA-DPR 168 h MAPE 一个例外。

不要把 31/32 与当前正文 30/32 当作同一统计口径。

## 12. 为什么 168 h MAE 不作为 Full 明显劣势

publisher-compatible：

```text
SAS-Norm-only 12.207835
STaR-GNN      12.233590
```

差：

```text
0.025755 ≈ 0.21%
```

相邻 168 h forecast origins 每隔 24 h 启动，预测窗口高度重叠，因此不能简单假设 46 origins 独立。最终 Figure 2 audit 使用 7-origin moving-block bootstrap。Full−SAS 的均值差 95% CI 跨过 0，所以正文应写“点估计近似持平”，而不是“Full 明显退化”。

## 13. 重建 manuscript-facing 表格

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout
```

必须看到：

```text
Metric convention audit: PASS
Factorial ablation model-set audit: PASS (4 models, no STGCN)
Publisher-compatible factorial cell audit: 30/32 PASS
```

## 14. 重建 absolute reference figures

```bash
python scripts/reproduce/build_literature_figures.py \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --ablation-table paper/tables/literature/table_ablation_common46.csv \
  --dma-table paper/tables/literature/table_star_gnn_dma_common46.csv \
  --output paper/figures
```

必须看到：

```text
Overall figure audit: PASS
Factorial ablation figure audit: PASS (4 models, no STGCN)
```

## 15. 重建正文 Figure 1--5

Stage 1：

```bash
python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820
```

Stage 2：

```bash
python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures \
  --block-bootstrap-iterations 50000 \
  --block-bootstrap-length 7 \
  --block-bootstrap-seed 20260820
```

最终 Figure 2 是纯四模型消融，不包含 STGCN。

Figure 3 是 baseline robustness，比较 DCRNN/STGCN/STaR-GNN。

## 16. 预期 Figure 2/3 审计

Day 7 relative to Day 1：

```text
DCRNN                 +38.25%
DCRNN + FA-DPR        +11.93%
DCRNN + SAS-Norm       +2.64%
STaR-GNN               +1.70%
```

Figure 3 paired wins：

```text
24 h  vs DCRNN 45/46
24 h  vs STGCN 45/46
168 h vs DCRNN 46/46
168 h vs STGCN 40/46
```

## 17. 语法与测试

```bash
python -m py_compile \
  scripts/reproduce/build_paper_tables.py \
  scripts/reproduce/build_literature_figures.py \
  scripts/reproduce/build_manuscript_results_figures.py \
  scripts/reproduce/refine_manuscript_results_figures.py

python -m pytest tests/test_paper_artifacts.py -q
```

然后：

```bash
git status --short
```

## 18. clean-room 验收

```bash
bash scripts/reproduce/validate_clean_room.sh \
  --workspace /path/to/new-clean-room \
  --frozen-release results/paper/frozen_v1 \
  --device cuda:0 \
  --evaluation-device cuda:0
```

完整 clean-room 需重新执行数据、图、训练、Test 与结果生成，且不得从冻结 Test 结果反向选择参数。

## 19. 论文结果使用原则

1. 不手工修改 CSV 形成预期排序；
2. 不用选择性小数位掩盖 168 h 0.21% MAE 差异；
3. 表格正文统一 3 位小数，audit 保留完整精度；
4. baseline comparison 与 factorial ablation 分开；
5. reported literature results 与 re-evaluated baselines 明确区分来源；
6. 不通过截断 y 轴夸大微小差异；
7. Figure 2 强调 module contribution 与 lead-time stability；
8. Figure 3 强调 forecast-origin robustness；
9. Figure 4 强调 DMA consistency；
10. Figure 5 使用 pre-specified median-error sample selection。

这样可以保证仓库、结果表、图件与 Journal of Hydrology 稿件始终使用同一实验定义。
