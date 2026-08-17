# STaR-GNN 方法与实现对应

## 1. 多 DMA 图预测问题

系统包含 10 个 DMA 节点。模型使用过去 672 h 预测未来 24 h 或 168 h：

```text
X: [batch, 672, 10, features]
Y: [batch, horizon, 10]
```

图不是管网物理拓扑，而是训练期需求序列的 Pearson 功能关联。正相关边经
随机游走归一化后进入二阶扩散递归编码器，从而联合描述 DMA 间同步变化和
小时级动态。

## 2. State：日状态—形状建模

每个 DMA 的 672 h 历史被切成 28 个 24 h 片段。对历史日 `d`：

```text
mu[d,n]    = mean_h x[d,h,n]
sigma[d,n] = max(std_h x[d,h,n], epsilon)
z[d,h,n]   = (x[d,h,n] - mu[d,n]) / sigma[d,n]
```

编码器主要预测局部日内形状 `z`。未来状态由最近日状态与两个对应星期状态的
可学习凸组合恢复，十个 DMA 仅增加均值/尺度各一个权重。真实未来状态只作为
辅助损失监督，不进入预测输入。

源码对应：`src/dma_wdf/models/star_components.py` 中的日切片与恢复组件，
变体键为 `dssn_sasr`。

## 3. FA-DPR：预测对齐的日模式检索

编码器的 672 个隐藏状态按日池化为 28 个历史 token。与旧版一次性注意力不同，
FA-DPR 在每个 decoder step 重新计算检索：

```text
query_t = f(previous_decoder_state, known_future_calendar_t)
key_d, value_d = g(encoder_day_token_d)
attention[t,d] = softmax(query_t · key_d / sqrt(attention_dim))
context_t = sum_d attention[t,d] value_d
decoder_state_t = decoder_state_t + gate_t * context_t
```

因此未来第几小时、星期几、是否节假日以及 decoder 当前状态都可以改变所读取
的历史日。计算对象只有 28 个日 token，而不是对 672 h 做全量 self-attention。

源码对应 `ForecastAlignedDailyPatternRetrieval`，变体键为 `fa_dpr`。

## 4. Full STaR-GNN

Full 同时启用 State 与 FA-DPR：State 在观测/输出空间分离跨日状态和日内形状，
FA-DPR 在隐藏空间缓解长历史压缩。两者分别位于编码前后不同位置，因此不是对
同一周期规律的重复参数化。

四单元共享图、hidden size、decoder、优化器和训练协议：

| 变体 | State | FA-DPR |
|---|:---:|:---:|
| DCRNN / Base (`backbone`) | × | × |
| DCRNN + State (`dssn_sasr`) | ✓ | × |
| DCRNN + FA-DPR (`fa_dpr`) | × | ✓ |
| STaR-GNN (`full`) | ✓ | ✓ |

模型前向、scheduled sampling 和 Test 推理均由
`src/dma_wdf/models/star_dcrnn.py` 与 `src/dma_wdf/training/star_engine.py`
统一实现，避免四个变体使用不同数据或 decoder。

冻结目录中的 `Base` 是内部兼容标签，论文正文统一写作 DCRNN。历史
冻结发布只保留 `Base/backbone` 这一套 DCRNN checkpoint，不再保存
`baselines/dcrnn` 重复工件。
实验口径和完整结果见
[`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md)。
