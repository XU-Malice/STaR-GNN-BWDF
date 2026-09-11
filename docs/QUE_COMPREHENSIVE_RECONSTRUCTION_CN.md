# Que 六模型数值重建计划

目标是用固定 seed 20240604，找到同时接近论文正文和补充表格的 GRU、LSTM、MSNet、MSCMNet_M、MSCMNet_WM、MSCMNet_W 配置。误差指标显著低于论文也会计入偏离；不以超过论文作为选取依据。DMA A–J 是供水分区。

用户允许调整论文给出的训练参数。本轮因此以已有结果作为线索，扩大训练参数与校正方式搜索，不把论文中的最佳轮数当作不可调整的约束，也不把循环网络输入结构当作主要原因。

## 历史结果给出的方向

下表是截至 2026-09-06 已核验的、固定 seed 20240604 的历史候选。这里按总体 8 项指标的最坏偏离选取整组配置，不能理解为这些配置也最接近各 DMA 的 80 项指标。误差百分比为 MAE、MAPE、RMSE 六项中的最大绝对相对偏差；NSE 为两个时域中的最大绝对差。

| 模型 | 总体误差指标最大偏离 | NSE 最大绝对差 | 本轮重点 |
|---|---:|---:|---|
| GRU | 18.31% | 0.03063 | MinMax/Z-score、批大小、学习率与较短训练，随后按 DMA 选定独立网络配置 |
| LSTM | 19.10% | 0.00485 | 保留已修正的 S3 权重衰减起点，重点寻找误差指标与 NSE 同时接近的训练设置 |
| MSNet | 8.80% | 0.02340 | 围绕已有配置调整损失、正则化与训练程度，兼顾总体和各 DMA |
| MSCMNet_M | 5.52% | 0.00403 | 比较直接校正与零初始化残差校正，细化批大小和训练轮数 |
| MSCMNet_WM | 2.93% | 0.00122 | 保留小权重份额辅助监督候选，检查各 DMA 偏差，细化辅助权重 |
| MSCMNet_W | 32.79% | 0.02300 | 较高优先级：校正方式、批大小、轮数、学习率、份额监督联合检查 |

历史较近候选含缩短轮数、Huber、残差校正或辅助监督，属于数值重建候选。这些调整会完整记录，不伪装成作者公布的配置。

## 一次启动的搜索流程

采用有限的分阶段搜索，而非把所有参数做笛卡尔积。每阶段结束自动生成下一阶段任务；选取依据及候选清单写入文件并冻结，重启时继续原清单。

| 阶段 | 设计 | 训练次数上限 |
|---|---|---:|
| A：基础覆盖 | 六模型 × Z-score/MinMax × Adam/AdamW × batch 4/8/16，共 72 项；M/W 增加残差校正 12 项；WM 增加小权重份额监督 4 项；另保留下述 7 个历史配置 | 95 |
| B：训练程度搜索 | 每模型选取最多两个较近候选，交叉测试论文轮数的 0.35/0.5/1/2/4 倍和学习率的 0.3/1/3 倍；去掉原设置 | 168 |
| C：局部细化 | 根据前两阶段结果细化 MAE/Huber 损失、batch 1/2/32、训练期归一化统计、正则化；M/W 增加 50/100 轮；W/WM 细化份额监督；GRU/LSTM 补充 0.1 倍轮数及权重衰减 0/0.001；少量全连接和 CAM 宽度候选作为补充 | 152 |
| D：GRU/LSTM 分区配置选择 | 从本轮有效候选中，为各 DMA 选择一个完整的独立网络；重新计算组合后的两个时域和总体结果，保存可复用的十网络配置与权重 | 无新增训练 |

总上限 415 次，重复配置自动去除，实际任务数通常较少。不会追加随机种子。默认仍使用历史的逐小时 GRU/LSTM 输入；新增逐日输入支持仅供显式试验，未加入默认搜索。联合模型的小规模结构候选也只承担补充作用。

七个额外历史配置为：GRU 的 Z-score/AdamW/batch16/0.35 倍轮数；GRU 与 LSTM 的 Z-score/AdamW/batch8/固定100轮；LSTM 的 Z-score/Adam/batch8/零权重衰减；MSNet 的 Z-score/AdamW/batch8/Huber；M 的 Z-score/AdamW/batch4/零初始化残差/Huber；W 的 Z-score/AdamW/batch1/零初始化残差/50轮。将这些配置提前纳入，避免丢掉已知较接近的结果，也允许其继续参与学习率和轮数细化。

阶段 B/C 会利用论文测试表的距离选取候选。这是用户要求的、明确以已发表数值为目标的重建搜索，不是独立测试集上的无偏性能估计。

## 如何判断接近

每模型有两套同时存在的比较：总体 8 项，以及 A–J 共 80 项。每个候选必须由同一套冻结权重计算两个时域；不得为不同指标或不同预测长度挑选不同权重。GRU/LSTM 本来就是十个独立网络，阶段 D 允许十个 DMA 使用不同的完整训练配置，但每个 DMA 的同一网络必须同时给出 24 h 和 168 h 预测。联合模型禁止拆列拼接。

操作判据为误差指标相对偏差不超过 5%，NSE 绝对差不超过 0.01。这是此次搜索约定的接近阈值，非论文定义。选取分数等权考虑总体与 DMA 两组的平均标准化偏差，并报告最差值、95 分位及逐项通过数；不能靠十个 DMA 的相互抵消掩盖分区偏差。

`pooled` 为当前主要口径，`origin_mean` 为另行列出的汇总假设。每种口径都计算完整表格并各自排名，不允许逐模型、逐时域或逐指标混用口径拼一张更接近的表。论文表中的 total MAE 按 DMA MAE 之和比较，物理总需求的 MAE 另列，两者不能混淆。

历史审计已经发现：固定当前真实值后，部分论文 RMSE/NSE 组合在 pooled 口径下无法同时满足上述阈值。因此，本轮可以系统寻找有限搜索范围内最接近的配置，但不能预先保证全部 528 项均达标。若仍存在差距，汇总会给出确切数值和对应配置，不把技术运行通过写成论文复现成功。

## 执行、恢复与证据

- 单 GPU 顺序运行，默认物理 GPU 6；训练前检查可用显存、磁盘、项目导入路径和 CPU 回归测试。
- A/B/C 每阶段的全部实际命令先进入原训练器的真实参数解析和配置检查，在设备解析前退出，不读取训练数据、不创建权重。全部通过才开始该阶段；保存 `stage_a_command_preflight.json` 等检查记录。若训练进程仍以参数错误常见退出码 2 返回，立即停止，不连续尝试其余候选。
- 对源码、数据、请求参数、归一化统计、权重和预测建立校验记录。已完成且证据一致的候选跳过；部分完成的候选在恢复时单独处理。
- 测试真实值、预测起点和 A–J 顺序固定，测试期不参与归一化参数拟合。保存实际归一化参数，既支持训练窗口统计，也支持训练期唯一行统计。
- 一项训练或校验失败会单独记录，不把它计为论文指标不接近；其他候选可继续。中断时只处理当前启动器持有的子进程。
- 启动脚本使用 GPU 锁，避免重复启动；不停止其他用户的任务，也不覆盖历史结果包。
- 完成后自动汇总六模型逐项差距、完整候选排名、有效配置和运行证据，打包紧凑结果，训练权重留在服务器。

历史单任务耗时差异较大：GRU/LSTM/MSNet 通常约 10–13 分钟，M/W 论文轮数较短，WM 约 7 分钟。轮数倍增与小批次会延长时间。本轮宜预留数天；准确剩余时间以启动后的实际耗时为依据。可取消墙钟时间限制而保留 415 候选上限，使有限队列自动跑完；也可设置预算，在完整任务之间暂停并随后恢复。

## 命令

更新到对应提交并激活 `bwdf311` 后，先预览实际清单：

```bash
bash scripts/train/run_que_comprehensive_reconstruction_gpu6.sh --dry-run
```

后台运行整个有限队列：

```bash
mkdir -p logs
nohup bash scripts/train/run_que_comprehensive_reconstruction_gpu6.sh --budget-hours 0 \
  > logs/que_comprehensive_reconstruction_launcher.log 2>&1 &
echo "$!" > logs/que_comprehensive_reconstruction_launcher.pid
```

`--budget-hours 0` 取消墙钟时间限制，仍保留有限候选上限。中断后重新执行同一命令，会验证已完成结果并继续。修复后的默认 run-tag 为 `que_comprehensive_reconstruction_20260907`。

查看进度，无需粘贴 Python 代码，也不会启动训练：

```bash
bash scripts/train/run_que_comprehensive_reconstruction_gpu6.sh --status
```

最终结果目录为 `results/que_comprehensive_reconstruction_20260907`。`closest_complete_configurations.json` 分别给出兼顾全部88项、侧重总体8项、最坏偏差最小的完整候选；`recurrent_assembled` 保存 GRU/LSTM 分区配置及结果。紧凑压缩包自动写入项目的上级目录，文件名为 `que_comprehensive_reconstruction_20260907_compact.tar.gz`。

运行期间不要更新源码或改动输入数据；本队列会检查其一致性。已有日志中的 `PASS` 只表示技术检查通过，是否接近须看最终数值比较。

## 2026-09-07 启动参数修复与历史结果复用

提交 `a42c2848b0886688ee1a13eeb989ecc4fc4c7df4` 的启动器把 CAM 宽度构造成一个字符串 `16,16,1`，而训练器的 `--cam-channel-sizes` 要求三个独立数字 `16 16 1`。上传结果证实：A 阶段 GRU/LSTM 各 14 项成功，另 67 项全部在解析阶段退出，尚未训练；B/C/D 未执行。原来的函数级训练测试和模拟队列测试没有覆盖这个实际 CLI 契约，新增测试直接运行训练器的真实解析与配置校验。

已核验的 compact 包 SHA256 为 `20e9023455ccb1f579775df5037c5679089f4a0b7d54a4033923f3e6366ca906`。28 项成功结果中，168 个非权重证据文件哈希全部吻合；46 个起点与数据审计一致；2,464 项指标独立重算最大差 `3.55e-15`。这些任务累计训练约 6 小时 49 分钟。compact 未包含 280 个 `.pt`，恢复时必须校验服务器原件。

更新修复提交后，以新目录恢复整轮计划：

```bash
mkdir -p logs
nohup bash scripts/train/run_que_comprehensive_reconstruction_gpu6.sh \
  --reuse-from "$PWD/results/que_comprehensive_reconstruction_20260906" \
  --budget-hours 0 \
  >> logs/que_comprehensive_reconstruction_20260907_launcher.log 2>&1 &
echo "$!" > logs/que_comprehensive_reconstruction_20260907_launcher.pid
```

复用前核验原 manifest、源码快照、实际数据指纹、参数、预测、指标、权重和完成凭据。训练器、模型、数据处理源码及配置必须与原运行一致。本修复不改动它们；仅修改调度、命令检查、恢复及相关测试与文档。核验失败时停止并保留原件，不把证据缺失当作可复用。

28 项通过核验的结果复制到新队列，标为 `PASS(reused)`。原训练 `status.json` 中的提交、时间和参数保持原样，另外保存 `reused_source_provenance.json` 说明来源；新完成凭据同时校验此来源记录。旧目录不改写。其后补跑 67 项未训练候选，并自动继续 B/C/D，固定一个 seed，仍受原有限计划上限约束。

同一新目录再次启动时可以省略 `--reuse-from`，程序会从新 manifest 恢复来源。查询旧失败队列需显式使用 `--run-tag que_comprehensive_reconstruction_20260906 --status`；不带 run-tag 的 `--status` 查询修复后的新队列。

## 2026-09-08 显式共享 GPU 7

用户服务器的8张GPU均已有计算进程。GPU7快照显示剩余8838 MiB，vLLM与另一Python进程仍占用显存。原启动器要求GPU没有计算进程，所以会在资源预检停止；这与模型训练是否正常、能否接近论文无关。

新增共享入口只在显式启用时允许其他计算进程存在。默认独占检查保留。共享入口仍执行相同A/B/C/D单seed搜索，训练器、模型、精度、batch、学习率、轮数和评价代码保持不变。

共享入口使用以下运行策略：

- 物理GPU7映射为训练进程的逻辑`cuda:0`，一次运行一个候选。
- 启动前至少8192 MiB空闲；PyTorch缓存分配器限额6 GiB，另留2 GiB启动余量。CUDA上下文创建后，再要求至少7 GiB空闲；训练器自身的空闲显存检查为6 GiB。
- 限额通过PyTorch的 `torch.cuda.set_per_process_memory_fraction` 设置，按设备总可见显存计算比例。它约束缓存分配器，**不是整个进程所有CUDA分配的硬配额**。共享进程的显存变化和算力竞争仍可能影响运行与速度。见 [PyTorch 2.9 官方说明](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.memory.set_per_process_memory_fraction.html)。
- 显存不足时状态为`waiting_gpu`，每30秒查询一次，达到门槛后自动开始。每次等待都检查源代码与数据仍一致，遵守墙钟预算与中断信号。
- 初始化后显存下降或训练OOM会记录资源报告并最多重试2次，保持原训练设置。重试后仍不足的候选记为资源失败，继续其他候选；不能因此把该配置判断为论文指标不接近。无法取得全部模型有效候选时，原自适应阶段保护仍生效。
- 每次训练的资源报告保存在该队列日志目录的`*_resource_attempt_*.json`，包括分配器限额、实际峰值和退出原因。既有vLLM/Python进程只读取状态，队列不会向它们发送信号。

源码调度策略已改变，因此共享入口使用新的`que_comprehensive_reconstruction_shared_20260908`目录。从原始20260906目录核验并复制28个GRU/LSTM结果，不重训，也不修改20260906或20260907目录。

```bash
mkdir -p logs
nohup bash scripts/train/run_que_comprehensive_reconstruction_shared_gpu7.sh \
  --reuse-from "$PWD/results/que_comprehensive_reconstruction_20260906" \
  --budget-hours 0 \
  >> logs/que_shared_gpu7_launcher.log 2>&1 &
echo "$!" > logs/que_shared_gpu7_launcher.pid
```

查看共享队列（无需激活训练环境，不会访问GPU或启动训练）：

```bash
bash scripts/train/run_que_comprehensive_reconstruction_shared_gpu7.sh --status
```

完成后结果包为项目上级目录的`que_comprehensive_reconstruction_shared_20260908_compact.tar.gz`。同一命令再次启动会验证并跳过完成项。`--budget-hours 0`允许长期等待资源，但候选搜索仍是有限计划。更改共享限额会改变运行清单签名，应使用新run-tag；不要在运行期间编辑源码或数据。
