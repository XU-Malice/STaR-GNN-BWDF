# 六模型总体定向搜索

本工具对应2026-09-10用户确认的目标：固定训练seed20240604，只比较六模型各自24h和168h的总体MAE、MAPE、RMSE、NSE，共48项。各DMA的指标接近程度不参与候选选择，模型之间可以采用不同训练设置。误差指标相对差不超过5%，NSE绝对差不超过0.01；改善和恶化都计入与论文的距离。

## 评价限制

在当前固定common46真实值与pooled口径下，RMSE²=(1−NSE)Var(y)。24h总体真实值方差为1585.4421529337021，GRU论文RMSE10.194与NSE0.916、LSTM论文RMSE9.711与NSE0.920，分别无法同时落入上述容差。即便为论文三位小数增加±0.0005的舍入区间，仍不相交。因此程序保留原容差，报告最接近结果和不可相容目标，不承诺48项全部通过，也不通过改标签、调整预测值或混合指标定义来宣称复现。

`pooled`为原始主评价口径；`origin_mean`将每个预测起点的完整指标算完后取平均，是单列的诊断假设。两种口径各自生成完整的六模型表，均不为单个模型或指标自动切换。所有选择都受到已公开测试表反馈，应表述为数值重建，不能据此宣称恢复作者原始实现或完成独立测试验证。

## 一次启动后的流程

1. **外部安装。** 从指定Git提交导出源码到`~/projects/que_total_focus_tools/<commit>`。只执行git fetch及git archive，不合并到正在训练的工作区。比较关键模型、训练器、数据处理和配置源码，要求与现有训练一致。
2. **立即执行CPU核验。** 核验现有A/B/C成功任务的冻结计划、原始数据、请求、权重、归一化、原始预测和完成凭据。所有任务直接从原目录读取，故无需现有仅支持A的跨目录复制机制。训练未完成或证据异常的任务单独列为排除项。
3. **总体优先组合搜索。** 对GRU/LSTM分别计算全部完整候选作为基准，然后进行最多12个起点、每个最多8轮逐DMA替换，以及总计最多300个双DMA替换提案。目标按总体8项的最大容差倍数、平均容差倍数依次排序；DMA指标不参与主次排序。一个DMA对应一个完整独立网络及其归一化参数，24h/168h使用同源预测。保留最优完整候选作为基准，不进行模型平均、预测校准或训练。
4. **随旧队列更新。** 每60秒查看状态，每增加16个成功任务重新核验/汇总；未变化的GRU/LSTM候选池使用已核验缓存。两种口径的组合单独保存。
5. **自动补充训练。** 旧队列进入终态且释放项目GPU7锁后才开始；现有vLLM等进程不受操作。按总体最接近的两个完整候选生成GRU/LSTM各最多24组，共最多48组。组合调整学习率、训练轮数，另比较batch2/4/16/32和MSE/MAE/Huber。已跑设置和失败设置去重。所有实际命令先由原训练器解析和加载配置进行CPU预检。每完成4项再进行组合搜索和汇总。
6. **结束并打包。** 有限搜索结束即退出，不以不可实现的48/48作为无限运行条件。完整权重、配置、预测与逐DMA来源保留在服务器；紧凑结果包不包含`.pt`，上传供分析即可。

新增训练同样使用物理GPU7，PyTorch分配器限额6GiB，启动余量2GiB。分配器限额不等于整个CUDA进程的硬配额。显存不够时等待；可核验的资源失败最多重试2次，旧尝试完整归档。训练命令退出码2立即停止，数值发散任务记录失败。终止新队列时只清理它持有的训练子进程；不停止原队列或其他GPU作业。

## 安装及查看

从项目目录执行，将`COMMIT`替换为此次发布的完整提交号：

```bash
(
set -e -o pipefail
cd ~/projects/STaR-GNN-BWDF
timeout 180s git fetch origin feat/mscmnet-baselines
tools_commit=COMMIT
git show "${tools_commit}:scripts/reproduce/install_que_total_focus.sh" | bash -s -- "$tools_commit"
)
```

安装器调用已有`bwdf311`环境，不重新pip安装。先运行新工具CPU测试，再后台启动。安装失败会输出预检日志；重复安装不会启动第二个实例或覆盖运行源码。

安装器从指定Git提交核验必需目录及项目文件，只有该提交实际包含`uv.lock`时才导出它。不能根据本地工作区是否存在锁文件判断提交内容。2026-09-10首版安装失败发生在Git导出阶段，尚未启动新搜索或补充训练；重新运行修复后的安装器即可，原训练队列和已有结果不受该次失败影响。

## 联合模型收尾后专注 GRU/LSTM

2026-09-10用户接受 MSNet、MSCMNet_M、MSCMNet_WM、MSCMNet_W 当前最接近论文的候选，要求保存模型及尝试记录，并将后续资源集中到 GRU/LSTM。原始 C 阶段按 GRU、LSTM、MSNet、M、WM、W 的顺序执行；进入 MSNet 后，本次调用不会再训练 GRU/LSTM。原 D 仅组合已有独立 DMA 网络并重算指标，仍有可能改善循环模型总体值。

安装时将 `--close-joint-first` 作为提交号之后的第二个参数：

```bash
git show "${tools_commit}:scripts/reproduce/install_que_total_focus.sh" | bash -s -- "$tools_commit" --close-joint-first
```

该模式使用独立结果目录 `results/que_recurrent_focus_20260910` 和 `logs/que_recurrent_focus_launcher.log`。按以下顺序自动执行：

1. 从所有已核验完整候选的原始预测重新计算 pooled TOTAL 指标，为四个联合模型各选一个完整配置，同时保存24h和168h八项指标。选择优先最小化最大标准化差距，再比较平均差距；不会跨候选拼指标。
2. 复制完整权重、预测、实际配置、scaler、训练曲线及审计文件，保存 A/B/C 全部计划和尝试记录。成功、复用、失败、运行中、待执行保持区分；仅尝试过的设置进入已训练去重清单。原源码、配置、数据哈希及工具源码一并记录。
3. 生成 `joint_closeout/joint_models_complete.tar.gz` 和 SHA256；逐文件以及逐压缩包成员核验后才写入 `completion_manifest.json` 的 READY 状态。归档接受当前差距，不声称四个模型全部达到原容差。
4. 确认完整 C 计划中 GRU/LSTM 没有未结束任务，再核对原共享 GPU7 启动器的用户、目录、命令和进程身份，通过绑定的 pidfd 发送一个 SIGTERM。原队列自己的清理路径关闭训练子进程；新流程等待旧队列终止状态和原 GPU 锁释放，不改写原队列状态或计划。
5. 固定归档中的四个联合模型，仅为 GRU/LSTM 执行总体指标组合搜索和最多各24组补充训练。补充计划排除本轮已经尝试的参数，包括失败或未通过结果核验的设置；只使用seed20240604。共享GPU7维持6GiB分配器限额和2GiB启动余量。

缺失 Python/libc pidfd 包装器时，停止器仅在已核对的 Linux x86_64/aarch64 LP64 ABI 上调用相同的内核 pidfd 接口。系统拒绝或不支持该接口时保留归档并停止交接，不对数值PID或其他GPU进程发送信号。归档或身份核验失败也不会中断旧队列。

四个联合模型的完整压缩包始终保留在服务器上；最终 `que_recurrent_focus_20260910_compact.tar.gz` 提供指标、配置、历史和核验记录，不嵌入该权重压缩包。已有模型组合是针对已公开测试目标的数值搜索，当前固定真值与 pooled 口径下的部分 RMSE/NSE 目标不相容，因此按原容差报告最接近结果，不承诺48项全部匹配。

服务器执行 editable install 后，`src/star_gnn_bwdf.egg-info/` 下会出现 `PKG-INFO`、`SOURCES.txt` 等安装元数据；纯 Git 导出的外部工具通常不包含它们。跨安装的数值源码比较仅将该目录中列明的包装信息单独记录，仍对其余 `src`、全部配置、训练入口及共享GPU运行脚本做完整文件集合和逐字节比较。未知文件、Python代码和配置不享受该处理。原服务器目录的全量冻结指纹（包括安装元数据）及外部工具自身的全量指纹继续核验。`training_source_compatibility.json` 保存检查清单和两侧安装元数据哈希，不能通过删除原目录的 egg-info 来修复此问题。

查看进度：

```bash
cat ~/projects/STaR-GNN-BWDF/results/que_total_focus_20260910/current_table.md
tail -n 30 ~/projects/STaR-GNN-BWDF/logs/que_total_focus_launcher.log
```

外部`run_que_total_focus.py --status`显示旧队列状态、CPU搜索模型/口径、总体接近项以及补充训练完成数。`--status`不创建结果也不访问GPU。

主要结果位于`results/que_total_focus_20260910`：

- `total_comparison.tsv`和`current_table.md`：主口径下每个模型最接近的完整候选/网络组合。
- `total_comparison_diagnostic_origin_mean.tsv`：独立的诊断口径表。
- `selection.json`：两个口径分别选取的完整配置与来源。
- `audits/`：全部候选指标、两种口径的比较和RMSE/NSE可行区间。
- `cohorts/`：完整网络、源配置、归一化审计、两时域预测和搜索轨迹；缓存复用前再次核验。
- `followup_plan.json`：冻结的补充训练设置及父候选。`followup_records/`与`logs/`保存每项实际命令、预检、资源报告与完成状态。
- `campaign_status.json`：原子更新的流程状态。`completed_search`只代表有限流程结束，不代表48项复现成功。

同一工具提交再次启动会核验已有完成项并跳过训练。已失败或未完整结束的尝试默认保留且不重复训练；需要再次尝试应单独规划，不能覆盖旧证据。工具版本或计划参数改变时必须使用新输出目录；不能修改既有冻结清单。
