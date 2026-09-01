# Que et al. 六种时序基线：实现、审计与运行

本文档对应 Water Research X 2024 论文 *Water demand forecasting in
multiple district metered areas based on a multi-scale correction module
neural network architecture*（DOI: `10.1016/j.wroa.2024.100269`）。

## 审计结论

| 位置 | GRU/LSTM | MSNet | MSCMNet_M | MSCMNet_WM/MW | MSCMNet_W | 结论 |
|---|---|---|---|---|---|---|
| 本仓库审计前 | 无训练实现 | 无 | 无 | 无 | 无 | 只保存论文 reported metrics |
| `DMA-WDF` | 无 | 无 | 无 | 无 | 无 | 只有论文指标 YAML |
| `BWDF` 审计包 | 有研究脚本，但 20 个正式模型未跑完 | 有候选 | 有候选 | 有候选 | 有候选 | 尚不能称为完成复现 |
| 本分支 | 已实现 | 已实现 | 已实现 | 已实现，`MW` 为别名 | 已实现 | 共用无泄漏数据、训练和 common-46 评估 |

`BWDF` 候选实现不能直接作为正式结果，原因包括：

- MSNet 与部分 MSCMNet 正式配置默认使用 `full_iqr15_interpolate`，IQR
  阈值由完整论文期（包括 Test）拟合；
- 部分 GRU/LSTM 诊断按不同指标选择不同 epoch，不能代表单一冻结
  checkpoint；
- GRU/LSTM 的完整 10 DMA × 2 模型正式队列未完成；
- MSNet 的早期正式候选使用二维卷积，而论文明确说明 CAM 使用 1D
  convolution；
- FC1/FC2 候选采用残差、零初始化和额外辅助损失等未由论文明确给出的
  设计；
- FC2 的 168 h 递归函数每个日步重复调用模型两次，第一次结果被丢弃；
- 审计包中的状态文档在最后一句中途结束，因此不能把该文档视为完整的
  最终验收记录。

## 本仓库实现

核心代码：

- `src/dma_wdf/models/mscmnet.py`：六种模型及 `MSCMNet_MW` 别名；
- `src/dma_wdf/data/mscmnet_dataset.py`：common-46、FC1/FC2 数据与泄漏
  审计；
- `configs/model/mscmnet_baselines.yaml`：论文 Table 3 与 Supplement S3
  的逐 DMA 参数；
- `scripts/train/train_temporal_baselines.py`：统一训练、24 h 直接预测、
  168 h 日递归预测和 S1 指标；
- `scripts/train/run_temporal_baselines_gpu6.sh`：GPU6 显存预检、smoke 与
  正式六模型队列。

模型语义：

| 名称 | 输入与校正 |
|---|---|
| GRU | 每个 DMA 独立模型，仅历史需求 |
| LSTM | 每个 DMA 独立模型，仅历史需求 |
| MSNet | 10 个 1D CNN-Attention-LSTM 分支，联合全连接输出 |
| MSCMNet_M | MSNet + 预报日 4 个气象与 5 个时间特征的 FC1 |
| MSCMNet_WM/MW | MSCMNet_M + 历史 DMA 日需求份额及日最高/最低温度的 FC2 |
| MSCMNet_W | 去除气象；保留时间特征 FC1 与仅需求份额 FC2 |

## 2026-08-31 正文与补充材料复核

新上传的 `(9).docx` 与先前 `(8).docx` 的 SHA256 均为
`8163f939c1b7dd14376027110b6f61c9c7b12b186637c292a4c87f4aa2ee88ee`，
内容完全相同。补充材料提供 S1 指标、S2 搜索空间、S3 最优参数以及 S4
校正前后曲线；网络数据流仍需以正文 Fig. 7、Table 3 和 Eq. (2)–(7) 为准。

| 项目 | 论文证据 | 旧实现 | 本次修正 |
|---|---|---|---|
| CAM 顺序 | Fig. 7：Conv1D 与 Attention 交替 3 次 | 3 个 Conv 后接 3 个 Attention | 按图交替 |
| Attention | Eq. (4)–(7)：单头 Q/K/V 加权 | MultiheadAttention + residual + LayerNorm | 单头 scaled dot-product，无残差/LayerNorm |
| CAM 输出 | Table 3：进入 LSTM 前为 1 channel | 每层均等于 DMA LSTM hidden size | 最后一层固定为 1 channel |
| CAM 中间宽度 | 未公开 | 绑定 LSTM hidden size | 显式参数，默认诊断值 `16,16` |
| FC2 前端 | Fig. 7：CAM-LSTM 权重预测模块 | 直接 LSTM | 增加同序 CAM |
| FC1/FC2 | Table 3：nodes/dropout/LSTM hidden | 已匹配 | 保持不变 |
| 归一化 | 仅说明输入归一化，算法未公开 | train-only Z-score | 保留 Z-score，并加入独立 MinMax 诊断 |
| batch/loss/seed | 未公开 | 8/MSE/20240604 | 继续明确标为复现假设 |

此前固定 epoch 的正式运行已经证明程序链完整，但没有数值复现 S1：GRU/LSTM
只达到部分一致，MSNet 及三种 MSCMNet 的 total MAE 相对论文高约
34%–75%，且模型排序与论文相反。新结构首先只重跑 MSNet，用它隔离 CAM 与
归一化影响；如果 MSNet 的四项 total 指标和 DMA 误差分布明显靠近 S1，再把
同一设置扩展到 M、WM/MW 和 W。这样可以避免在主干仍错误时盲目重复四个大模型。

正式协议强制：

- 插值在 train/Test 分区内独立完成；
- IQR 阈值仅用 train 拟合并冻结；
- 所有 scaler 仅用训练张量拟合；
- 固定论文 best epoch 后保存一个 checkpoint；
- 同一 checkpoint 同时用于 24 h 与 168 h，并且所有指标共用；
- 168 h 只把前一日预测追加到需求历史；未来真实需求从不进入模型；
- Test 仅采用论文 46 个共同起点；
- total MAE 是十个 DMA MAE 之和，其他 total 指标在小时总需求上计算。

## 论文未公开的实现细节

再次逐页核对论文 Fig. 7、Table 3 与补充材料后，以下内容不再属于假设：

- CAM 按 `Conv1D → Attention` 交替三次，而不是先连续卷积再连续注意力；
- Table 3 给出 forecast branch 的 CAM 输入/输出张量形状，并明确 CAM 输出
  为 1 channel 后进入 LSTM；两层中间卷积宽度没有公开；
- Eq. (4)–(7) 描述单头 Q/K/V attention；正文没有明确 QK 分数缩放分母，
  也没有给出 residual attention、multi-head 或 LayerNorm；
- FC2 的日需求份额/温度序列同样经过 CAM-LSTM，而不是直接进入 LSTM；
- Table 3 给出的 FC1/FC2 节点数、dropout、LSTM 层数和 hidden size 已逐项
  写入配置。

论文没有公开作者训练代码，也没有完整说明 padding、batch size、loss、
归一化算法与随机种子。因此，本仓库仍将以下选择显式写入配置和
checkpoint，不能宣称与作者私有代码逐行相同：

- same-length padding、ReLU，以及未公开的两层中间卷积宽度；默认
  `16 → 16 → 1` 来自此前 BWDF 诊断中较接近论文的候选，并非论文参数；
- Adam、MSE、batch size 8、seed `20240604`；
- 默认 train-only Z-score；另提供 train-only MinMax 诊断开关，因为论文只说
  输入归一化而未披露具体算法，二者必须写入不同输出目录比较；
- FC1/FC2 按论文文字采用直接全连接校正；
- 默认不加入论文未报告的辅助 total/share loss。

这是一套可审计的论文结构重建；是否数值复现 S1 必须由服务器正式运行
结果判定，不能用“局部指标优于论文”替代协议一致性。

## 优化器与 CAM 时间轴诊断

正式 99 epoch 优化器矩阵表明，当前重建使用 PyTorch `Adam` 的 coupled
weight decay 时，即使 decay 只有 `0.0001`，CAM 卷积和注意力权重也会被压到
接近零，预测退化为不随输入起点变化的固定 24 小时模板。`AdamW(0.1)` 能保留
论文的 `Conv1D → Attention` 三次交替结构及非零输入敏感性，因此后续结构兼容
诊断固定使用 `replace + AdamW(0.1)`。`skip_final` 虽然总指标更接近补充材料，
但它删除 Figure 7 的末级 Attention，只能作为机理诊断，不能标记为论文实现。

论文的 CAM 输入写为 `d_i × 24 × 10`，但没有交代这两个时间轴在 CNN、Attention
和 LSTM 之间如何展平，也没有公开 QK 分数是否除以特征维平方根或训练 batch
size。以下脚本穷举仍与图 7 相容的 12 组选择：

- `full_history_flat`：将全部历史小时展平后执行 CAM 和 LSTM（仅诊断）；
- `per_day_flat`：每天独立执行 24 小时 CAM，再展平给逐小时 LSTM；
- `per_day_vectors`：每天独立执行 CAM，并将一天 24 个输出作为一个 LSTM
  时间步；
- attention scaling 为 `sqrt_dim` 或 `none`，batch size 为 8 或 16。

2026-09-01 的 12 组正式诊断固定了 `replace + AdamW`。其中
`full_history_flat + none + 8` 最接近论文的八个汇总数，但预测日变化仅为
真值的约 44%，存在明显过度平滑。`per_day_vectors + sqrt_dim + 8` 的逐 DMA
指标平均相关性约为 0.96，并且更符合补充材料给出的 CAM 张量形状，因此被选为
后续四个联合模型的证据优先设置。该选择仍属于重建推断，不表示作者未公开的
实现细节已被唯一确定。

```bash
nohup bash scripts/train/run_que_cam_layout_diagnostics_gpu6.sh \
  > logs/que_cam_layout_diagnostics_launcher.log 2>&1 &
tail -f logs/que_cam_layout_diagnostics_launcher.log
```

脚本支持断点续跑，会验证 46 个共同起点、24 h/168 h 张量、正式 epoch、优化器
和 resolved config；最终摘要还记录跨预测起点标准差、相邻起点变化和 168 h
逐日变化，避免把固定日模板误判成有效时间动态。

固定上述选择后，可一次后台运行 MSNet、MSCMNet_M、MSCMNet_WM 和 MSCMNet_W：

```bash
nohup bash scripts/train/run_que_selected_joint_baselines_gpu6.sh \
  > logs/que_selected_joint_baselines_launcher.log 2>&1 &
tail -f logs/que_selected_joint_baselines_launcher.log
```

## GPU6 运行

```bash
cd ~/projects/STaR-GNN-BWDF
conda activate bwdf311
python -m pip install -e ".[dev,model]"

# 前台运行：先检查物理 GPU6，再 smoke，最后跑六模型。
bash scripts/train/run_temporal_baselines_gpu6.sh
```

后台运行：

```bash
mkdir -p logs
nohup bash scripts/train/run_temporal_baselines_gpu6.sh \
  > logs/temporal_baselines_gpu6.log 2>&1 &
echo $! > logs/temporal_baselines_gpu6.pid
```

监看：

```bash
tail -f logs/temporal_baselines_gpu6.log
nvidia-smi -i 6
```

如只运行一个模型，可设置环境变量：

```bash
MODEL=mscmnet_w bash scripts/train/run_temporal_baselines_gpu6.sh
```

建议先用修正后的主干对 MSNet 做最小归一化诊断，而不是立即重跑六模型：

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/train/train_temporal_baselines.py \
  --model msnet --device cuda:0 --normalization zscore \
  --output-root results/temporal_baselines_cam_table3_zscore

CUDA_VISIBLE_DEVICES=6 python scripts/train/train_temporal_baselines.py \
  --model msnet --device cuda:0 --normalization minmax \
  --output-root results/temporal_baselines_cam_table3_minmax
```

两次运行共用论文固定 99 epoch、common-46 和同一评估口径；只比较论文未公开的
归一化选择。选定更接近 S1 且 DMA 级误差结构合理的设置后，再运行三种校正模型。

正式输出位于 `results/temporal_baselines/<model>/seed_20240604/`，包含：

- `checkpoint_*.pt`；
- `predictions_common46.npz`；
- `metrics.csv`；
- `loss_curve.csv`；
- `resolved_config.yaml`；
- `status.json`。

`metrics.csv` 中的 `paper_value` 只用于并排核查，绝不参与训练、epoch
选择或模型选择。

## 2026-09-01 正式矩阵验收与下一轮诊断

`que_reproduction_matrix_20260831` 的 18/18 个任务均完成协议校验。物理 GPU 6
峰值显存 1613 MiB、最低剩余显存 22479 MiB、最高温度 60°C，不存在 OOM 或
热失速证据。Z-score 在六个模型上均优于 MinMax；`1,1,1`、`8,8,1`、
`16,16,1`、`32,32,1` 四种 MSNet CAM 宽度的预测几乎完全相同，因此归一化
和中间宽度都不是当前 MSNet 失败的主要原因。

Z-score 的 total 指标显示：GRU/LSTM 训练链有效但只部分接近 S1；MSNet 的
24 h MAE/RMSE/NSE 为 `42.124/38.348/0.072`，相对论文
`15.537/9.526/0.929` 明显失败。MSNet 每个 DMA 的预测在时间轴上的平均标准差
仅约 `0.298`，说明模型主要学到了静态 DMA 轮廓。最大 24 h MAE 偏差来自
DMA E（本地 `12.169`，论文 `1.867`）。三种校正模型能部分绕过退化主干，
但 total MAE 仍比论文高约 44%–49%，局部最大偏差转移至 DMA I。

最可疑的机制是末级 `1 channel` scaled-dot-product attention：它用所有时间
点的值加权替换每个局部时间表示，而当前结构没有论文未说明的 residual path。
为避免把猜测伪装成论文实现，`replace` 保持正式默认；CLI 新增三个带 provenance
的诊断值：

- `final_residual`：仅末级注意力采用残差更新；
- `skip_final`：跳过末级一通道注意力；
- `residual`：三个注意力阶段都采用残差更新。

三组均只运行 Z-score MSNet 和论文固定 99 epoch，不再重复已被淘汰的 MinMax
与 CAM 宽度扫描。服务器一键后台命令：

```bash
cd ~/projects/STaR-GNN-BWDF || exit 1
conda activate bwdf311
python -m pip install -e ".[dev,model]"
mkdir -p logs
nohup bash scripts/train/run_que_attention_diagnostics_gpu6.sh \
  > logs/que_attention_diagnostics_launcher.log 2>&1 &
echo $! > logs/que_attention_diagnostics_launcher.pid
```

预计显存仍低于 2 GiB；按正式矩阵实测速度，三组通常约 40–50 分钟。脚本会自动
续跑、逐组校验、统计时间变化幅度，并生成
`~/projects/que_attention_diagnostics_20260901.tar.gz`。只有候选同时恢复时间变化、
显著改善 RMSE/NSE 且 DMA 误差结构靠近 S1，才继续扩展到 M/WM/W；不能仅按
total MAE 选择候选。

### 注意力诊断结论与优化器语义审计

`que_attention_diagnostics_20260901` 的三组任务均通过，但
`final_residual`、`skip_final` 和 `residual` 的预测最大差异仅为
`3.8e-6`，24 h total MAE 均约 `42.124`，DMA 时间标准差均约
`0.298`。配置、checkpoint 哈希和实际参数存在差异，因此不是 CLI 未生效或
导入错误包。

checkpoint 参数审计表明，论文 S3-3 报告的 MSNet `weight_decay=0.1` 在当前
PyTorch coupled-L2 `Adam` 中使所有参与前向的参数发生塌缩：卷积参数总范数约
`6.2e-4`、注意力约 `5.7e-4`、LSTM 约 `7.7e-2`、联合输出权重约
`3.2e-3`；相比之下联合输出 bias 范数为 `0.876`。`skip_final` 中被跳过且
没有梯度的末级 attention 保留初始化权重，而其余参与计算的权重仍被衰减，进一步
证明当前静态预测来自“高 coupled weight decay + 近似 bias-only 输出”，不能再
归因于单独的末级 attention。

补充材料只给出 `Weight_decay=0.1`，没有公开 Adam 的框架实现、coupled/decoupled
语义、参数分组及 bias 是否正则化。下一轮因此保持论文数值可追踪，同时一次性比较：

- literal `replace` + Adam：weight decay `0`、`1e-4`、`1e-2`；
- literal `replace` + AdamW：论文数值 `0.1`；
- `residual` 和 `skip_final` + Adam：weight decay `0`。

运行：

```bash
cd ~/projects/STaR-GNN-BWDF || exit 1
git pull --ff-only
conda activate bwdf311
python -m pip install -e ".[dev,model]"
mkdir -p logs
nohup bash scripts/train/run_que_optimizer_diagnostics_gpu6.sh \
  > logs/que_optimizer_diagnostics_launcher.log 2>&1 &
echo $! > logs/que_optimizer_diagnostics_launcher.pid
```

该矩阵共六组正式 99 epoch MSNet，预计约 80–90 分钟。结果摘要同时记录四项
指标、时间标准差和各参数组范数；只有先恢复非零网络权重和时间动态后，才重新判断
attention 结构。所有 override 都写入 resolved config、status 和 checkpoint，
不会覆盖论文 S3-3 原值或被误标为已复现结果。
