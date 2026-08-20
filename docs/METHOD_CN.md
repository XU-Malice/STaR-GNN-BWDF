# STaR-GNN 方法与实现对应

本文档说明论文方法名称与仓库内部兼容键之间的关系。**论文与正文统一使用 SAS-Norm 和 FA-DPR；`State`、`dssn_sasr`、`Full`、`Base` 仅是冻结工件或源码中的兼容标签。**

## 1. 多 DMA 图预测问题

系统包含 10 个 DMA 节点。模型使用过去 672 h 预测未来 24 h 或 168 h：

```text
X: [batch, 672, 10, features]
Y: [batch, horizon, 10]
```

图不是管网物理拓扑，而是仅由训练期 DMA 需求序列构建的 Pearson 功能关联图。邻接定义为：

\[
A_{ij}=\begin{cases}
\max(r_{ij},0), & i\neq j,\\
0, & i=j,
\end{cases}
\qquad P=D^{-1}A.
\]

即：保留正相关权重、对角线为 0、不使用阈值或 Top-K，并进行随机游走归一化。24 h 和 168 h 共用同一个固定图；模型中的二阶扩散递归单元使用该图传播 DMA 间功能性依赖。

## 2. SAS-Norm：Seasonally Anchored State Normalization

SAS-Norm 的目标是把跨日需求状态与日内形状分离，避免日均水平/波动尺度的漂移与小时级形状变化混在同一表示中。

每个 DMA 的 672 h 历史被切为 28 个 24 h 日片段。对历史日 `d`：

```text
mu[d,n]    = mean_h x[d,h,n]
sigma[d,n] = max(std_h x[d,h,n], epsilon)
z[d,h,n]   = (x[d,h,n] - mu[d,n]) / sigma[d,n]
```

其中：

- `mu[d,n]` 表示 DMA `n` 在第 `d` 个历史日的需求水平；
- `sigma[d,n]` 表示该日的波动尺度；
- `z[d,h,n]` 表示去除日状态后的日内形状。

编码器主要建模归一化后的日内动态。预测端再通过季节锚定的状态恢复把预测形状映射回需求空间。未来真实状态只用于辅助状态损失监督，不进入预测输入，因此 Test 阶段不存在未来需求泄漏。

源码对应：

- `src/dma_wdf/models/star_components.py`
- 内部类/组件：日切片归一化与季节状态恢复
- 变体键：`dssn_sasr`
- 冻结工件兼容名：`State`

论文正文应写 **DCRNN + SAS-Norm**，而不是 `DCRNN + State`。

## 3. FA-DPR：Forecast-Aligned Daily Pattern Retrieval

FA-DPR 解决另一个问题：在 672 h 长历史中，不同未来预测位置对应的有效历史日并不固定。

编码器的 672 个隐藏状态按自然日聚合成 28 个历史日 token。FA-DPR 在每个 decoder step 根据当前未来位置重新检索：

```text
query_t = f(previous_decoder_state, known_future_calendar_t)
key_d, value_d = g(encoder_day_token_d)
attention[t,d] = softmax(query_t · key_d / sqrt(attention_dim))
context_t = sum_d attention[t,d] value_d
decoder_state_t = decoder_state_t + gate_t * context_t
```

因此预测第几小时、星期位置、已知日历条件和 decoder 当前状态都可以改变所读取的历史日。检索对象只有 28 个 daily tokens，而不是对全部 672 h 做全量 self-attention。

源码对应：

- `ForecastAlignedDailyPatternRetrieval`
- 变体键：`fa_dpr`
- 冻结工件兼容名：`FA-DPR`

## 4. Full STaR-GNN

完整 STaR-GNN 同时启用 SAS-Norm 与 FA-DPR：

- **SAS-Norm** 在观测/输出空间分离跨日状态和日内形状；
- **FA-DPR** 在隐藏空间根据未来位置动态选择历史日模式；
- Pearson 功能图提供跨 DMA 的固定传播通道。

二者作用位置不同，因此不是对同一周期规律的重复参数化。

| 论文名称 | 内部 variant | SAS-Norm | FA-DPR |
|---|---|:---:|:---:|
| DCRNN | `backbone` | × | × |
| DCRNN + SAS-Norm | `dssn_sasr` | ✓ | × |
| DCRNN + FA-DPR | `fa_dpr` | × | ✓ |
| STaR-GNN | `full` | ✓ | ✓ |

四个变体共享相同的图、主干、hidden size、decoder、优化器和训练协议。前向、scheduled sampling 和 Test 推理由 `src/dma_wdf/models/star_dcrnn.py` 与 `src/dma_wdf/training/star_engine.py` 统一实现。

冻结目录中的 `Base` 是 DCRNN 的内部兼容标签，论文正文只写 DCRNN；不存在第二套独立 `baselines/dcrnn` checkpoint。

## 5. 最终训练与评估边界

论文冻结参数：

```yaml
learning_rate: 0.0003
weight_decay: 0.0
cl_decay_steps: 500
state_loss_weight: 0.03
max_epochs: 100
seed: 0
```

参数只根据 Validation 确定；参数冻结后才进行 common-46 Test。Test 阶段关闭 teacher forcing，未来需求不进入模型输入、early stopping 或组件选择。

## 6. 如何阅读最终结果

论文主结果不使用内部 aggregate-demand MAE 作为跨模型 MAE 对比。最终 manuscript-facing 口径、9 模型比较、publisher-compatible 消融、DMA-level 结果见：

- [`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md)
- [`../paper/tables/literature/METRIC_CONVENTIONS.md`](../paper/tables/literature/METRIC_CONVENTIONS.md)

最终 Figure 1--5 的科学问题和解释边界见：

- [`MANUSCRIPT_FIGURES_FINAL_CN.md`](MANUSCRIPT_FIGURES_FINAL_CN.md)
- [`PLOTTING_CN.md`](PLOTTING_CN.md)

特别注意：168 h 下 SAS-Norm-only 的 publisher-compatible MAE（12.207835）略低于 Full（12.233590），因此不能宣称 Full 在所有 168 h 指标上严格优于单模块；Full 的 168 h 优势体现在 MAPE、RMSE、NSE 以及更完整的预测行为证据上。