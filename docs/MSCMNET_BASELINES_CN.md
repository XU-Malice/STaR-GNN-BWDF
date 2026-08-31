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

论文没有公开作者训练代码，也没有完整说明 CNN channel 数、padding、
Attention 的框架细节、batch size、loss 与随机种子。因此，本仓库将以下
选择显式写入配置和 checkpoint，不能宣称与作者私有代码逐行相同：

- same-length 1D convolution、ReLU、residual self-attention 与 LayerNorm；
- Adam、MSE、batch size 8、seed `20240604`；
- FC1/FC2 按论文文字采用直接全连接校正；
- 默认不加入论文未报告的辅助 total/share loss。

这是一套可审计的论文结构重建；是否数值复现 S1 必须由服务器正式运行
结果判定，不能用“局部指标优于论文”替代协议一致性。

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

正式输出位于 `results/temporal_baselines/<model>/seed_20240604/`，包含：

- `checkpoint_*.pt`；
- `predictions_common46.npz`；
- `metrics.csv`；
- `loss_curve.csv`；
- `resolved_config.yaml`；
- `status.json`。

`metrics.csv` 中的 `paper_value` 只用于并排核查，绝不参与训练、epoch
选择或模型选择。
