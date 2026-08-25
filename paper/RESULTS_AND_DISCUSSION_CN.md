# 3. Results and discussion

## 3.1 Overall performance across forecasting horizons and model architectures

为检验 STaR-GNN 在不同预测时域和模型架构下的总体有效性，[Table 1. Overall forecasting performance of the comparison models for the 24 h and 168 h prediction horizons](https://github.com/XU-Malice/STaR-GNN-BWDF/blob/main/paper/tables/submission/table1_overall_performance.md) 汇总了各模型在 24 h 和 168 h 任务上的四项评价指标，[Fig. 1. Overall forecasting performance across prediction horizons and model families](https://github.com/XU-Malice/STaR-GNN-BWDF/blob/main/paper/figures/submission/main_fig1_overall_performance.pdf) 进一步给出了 STaR-GNN 相对于各对比模型的改善幅度。其中，Fig. 1a 表示 MAE、MAPE 和 RMSE 的相对降幅，Fig. 1b 表示 NSE 的绝对提升量，即 $\Delta\mathrm{NSE}=\mathrm{NSE}_{\mathrm{STaR-GNN}}-\mathrm{NSE}_{\mathrm{baseline}}$。因此，两个面板中的正值均表示 STaR-GNN 性能更优。

STaR-GNN 在两个预测时域的四项指标上均取得最优结果。对于 24 h 预测，其 MAE、MAPE、RMSE 和 NSE 分别为 9.424、1.805%、5.535 和 0.981。与 GRU、LSTM、MSNet 及三种 MSCMNet 变体相比，STaR-GNN 的 MAE 降低 34.9%–46.7%，MAPE 降低 30.6%–43.6%，RMSE 降低 27.0%–45.7%，NSE 提高 0.022–0.065。以这些模型中综合表现最强的 MSCMNet-W 为例，STaR-GNN 将 MAE、MAPE 和 RMSE 分别降低 34.9%、30.6% 和 27.0%，同时将 NSE 从 0.959 提高至 0.981。上述结果说明，STaR-GNN 的优势并非局限于某一误差指标，而是同时体现在平均误差、相对误差、较大偏差控制和整体拟合能力上。

相对于图时空模型，STaR-GNN 同样保持明显优势。与 DCRNN 相比，24 h MAE、MAPE 和 RMSE 分别降低 20.9%、18.5% 和 19.2%，NSE 提高 0.010；与 STGCN 相比，三项误差分别降低 23.7%、25.6% 和 30.0%，NSE 提高 0.020。这表明 STaR-GNN 的性能增益并不只是来自图结构建模本身，针对跨日状态变化和预测位置所设计的时序机制进一步提升了图预测框架的精度。

当预测时域由 24 h 延长至 168 h 时，各对比模型普遍出现误差增大或 NSE 下降，但 STaR-GNN 仍保持最优表现，其 MAE、MAPE、RMSE 和 NSE 分别为 12.234、2.014%、6.161 和 0.976。与六种时序或多尺度模型相比，STaR-GNN 的 MAE、MAPE 和 RMSE 分别降低 18.2%–34.5%、22.5%–37.1% 和 20.6%–45.7%，NSE 提高 0.016–0.058。与 DCRNN 相比，三项误差分别降低 27.2%、38.0% 和 37.2%，NSE 提高 0.037；与 STGCN 相比，三项误差分别降低 16.0%、43.7% 和 40.2%，NSE 提高 0.043。相较于 24 h 任务，STaR-GNN 在 168 h 任务上对 MAPE、RMSE 和 NSE 的改善更为突出，说明其优势在误差更易累积的周尺度预测中得到进一步放大。

总体而言，Table 1 和 Fig. 1 共同形成了“绝对指标—相对改善”的第一层证据：Table 1 用于判断不同模型的实际误差水平，Fig. 1 用于比较改善幅度及其跨指标、跨时域的一致性。需要指出的是，这些结果支持“STaR-GNN 优于所有对比模型”，但并不意味着任意图模型均优于时序模型；例如，在 168 h 任务上，DCRNN 和 STGCN 并未在所有指标上超过 MSCMNet-W。

## 3.2 Component contributions and lead-time behavior

为进一步确定总体性能增益的来源，[Table 2. Factorial ablation of SAS-Norm and FA-DPR for the 24 h and 168 h prediction horizons](https://github.com/XU-Malice/STaR-GNN-BWDF/blob/main/paper/tables/submission/table2_factorial_ablation.md) 比较了 DCRNN、DCRNN + SAS-Norm、DCRNN + FA-DPR 和完整 STaR-GNN；[Fig. 2. Component contributions and lead-time stability during 168 h forecasting](https://github.com/XU-Malice/STaR-GNN-BWDF/blob/main/paper/figures/submission/main_fig2_ablation_leadtime.pdf) 则从 MAE、MAPE、RMSE 和 NSE 四个方面展示三个模型变体相对于 DCRNN 的逐日改善及 95% 置信区间。由于 SAS-Norm 与完整模型的部分结果非常接近，图中采用轻微水平错位，并结合不同的线型和标记，使两条曲线在不改变横坐标含义的情况下保持可辨识性。

在 24 h 任务上，单独引入 SAS-Norm 后，MAE、MAPE 和 RMSE 分别降低 12.2%、9.1% 和 10.4%；单独引入 FA-DPR 后，三项误差分别降低 5.7%、12.1% 和 11.2%。完整 STaR-GNN 的对应降幅达到 20.9%、18.5% 和 19.2%，并取得最高 NSE。两个单模块均能改善日尺度预测，但完整模型的改善幅度高于任一单模块，表明两者作用并非完全重合。

在 168 h 任务上，SAS-Norm 是降低总体误差的主要贡献模块。相对于 DCRNN，DCRNN + SAS-Norm 将 MAE、MAPE 和 RMSE 分别降低 27.3%、35.3% 和 34.1%，并将 NSE 从 0.940 提高至 0.974。FA-DPR 单独使用时将 MAE 降低 16.2%、RMSE 降低 4.9%，NSE 提高 0.006，但 MAPE 由 3.248% 增至 3.278%，说明其独立作用具有明显的指标依赖性。完整 STaR-GNN 在 MAPE、RMSE 和 NSE 上分别取得 2.014%、6.161 和 0.976的最优结果。

对于 168 h MAE，DCRNN + SAS-Norm 的结果为 12.208，完整 STaR-GNN 为 12.234，二者仅相差 0.026，约占 SAS-Norm 结果的 0.21%。进一步的配对移动块分析表明，完整模型与 SAS-Norm 的平均 MAE 差值的 95% 置信区间为 −0.129–0.177，包含 0。因此，不能据此声称任一模型在该指标上具有稳定优势；更准确的结论是，两者在 168 h MAE 上基本持平，而完整模型在 MAPE、RMSE 和 NSE 上表现更为均衡。

逐预测日结果揭示了不同模块随预测提前期变化的作用。完整 STaR-GNN 相对于 DCRNN 的 MAE 降幅从第 1 天的 13.4% 增至第 7 天的 34.0%，对应的 95% 置信区间分别为 10.6%–16.1% 和 30.3%–38.0%；其在第 1 天改善 43/46 个测试起点，在第 7 天改善全部 46 个测试起点。SAS-Norm 的 MAE 降幅由 14.2% 增至 33.9%，呈现出与完整模型相近的周尺度误差控制能力。FA-DPR 的 MAE 降幅则由第 1 天的 2.7% 增至第 7 天的 18.5%，但其 MAPE 和 RMSE 在部分中后期预测日出现负改善，且置信区间较宽。

这些结果说明，SAS-Norm 主要负责缓解需求水平和波动尺度随预测日变化所引起的误差累积，是周尺度绝对精度提升的主要来源；FA-DPR 能够补充与预测位置相关的历史日模式，但其单独作用并未在所有指标上稳定成立。二者联合后，模型在平均误差、相对误差、大偏差控制和整体拟合之间取得了更平衡的结果。因此，消融实验支持两个模块具有差异化作用，但对 FA-DPR 的机制解释应限定为“补充并改善部分长提前期预测”，而不应扩大为“单独改善所有周尺度指标”。

## 3.3 Temporal and spatial robustness

平均指标可能受到少数有利测试日期或少数 DMA 的影响。为判断 STaR-GNN 的总体优势是否具有时间和空间一致性，[Fig. 3. Temporal and spatial robustness of STaR-GNN across all four metrics](https://github.com/XU-Malice/STaR-GNN-BWDF/blob/main/paper/figures/submission/main_fig3_temporal_spatial_robustness.pdf) 分别汇总了 46 个统一测试起点和 10 个 DMA 上的配对比较结果。Fig. 3a 表示测试起点层面的时间一致性，Fig. 3b 表示 DMA 层面的空间一致性；颜色表示 STaR-GNN 取得改善的比较比例，单元格第一行给出平均误差降幅或平均 $\Delta\mathrm{NSE}$，第二行给出改善次数与总比较次数。

在 24 h 任务上，STaR-GNN 相对于 DCRNN 和 STGCN 的逐起点平均 MAE 降幅分别为 19.3% 和 22.6%，两组比较均在 45/46 个测试起点上取得改善。相对于 DCRNN 的 MAE 降幅的 95% 置信区间为 16.7%–21.6%，相对于 STGCN 的区间为 20.6%–25.0%。对于 MAPE、RMSE 和 NSE，STaR-GNN 在多数而非全部测试起点上占优，说明 24 h 总体优势具有较强一致性，但局部预测日期仍可能出现性能波动。

在 168 h 任务上，STaR-GNN 相对于 DCRNN 的时间一致性进一步增强。其在全部 46 个测试起点上降低 MAE，平均降幅为 26.7%，95% 置信区间为 24.1%–29.2%；MAPE、RMSE 和 NSE 均在 45/46 个测试起点上改善。相对于 STGCN，STaR-GNN 在 40/46 个测试起点上降低 MAE，平均降幅为 15.1%，95% 置信区间为 9.8%–19.8%。MAPE、RMSE 和 NSE 的平均改善仍为正，但改善次数分别为 36/46、35/46 和 35/46，表明相对于 STGCN 的周尺度优势并非在每一个测试时段上都成立。

空间层面的结果更加一致。在 10 个 DMA、2 个预测时域、2 个图模型基线和4项指标组成的 160 个比较中，STaR-GNN 改善了 158 项；其中，所有 40 个 DMA–预测时域–基线组合的 MAE 均得到降低。仅有的两个例外均出现在 168 h 任务的 DMA G：相对于 STGCN，STaR-GNN 的 RMSE 为 1.469，而 STGCN 为 1.451；相应 NSE 分别为 0.878 和 0.881。虽然这两个差异的数值较小，但仍表明模型优势不应表述为对所有 DMA 和指标的绝对支配。

不同 DMA 的改善幅度存在明显差异。例如，部分 DMA 的 MAE 降幅超过 40%，而另一些 DMA 的改善不足 5%。这种空间异质性说明，系统级平均结果不能替代分区层面的评价。各 DMA 的绝对指标见 [Table S1. DMA-level forecasting performance](https://github.com/XU-Malice/STaR-GNN-BWDF/blob/main/paper/tables/submission/tableS1_dma_metrics.md)，四项指标的逐 DMA 改善见 [Fig. S1. DMA-level improvement of STaR-GNN across four metrics](https://github.com/XU-Malice/STaR-GNN-BWDF/blob/main/paper/figures/supplementary/supp_figS1_dma_improvement.pdf)。此外，[Fig. S2. Distribution of per-origin total MAE across the common test period](https://github.com/XU-Malice/STaR-GNN-BWDF/blob/main/paper/figures/supplementary/supp_figS2_origin_ecdf.pdf) 给出了逐测试起点 MAE 的经验累积分布，用于验证平均值改善并非由少数极端样本主导。

因此，Fig. 3 及补充材料形成了“平均性能—时间一致性—空间一致性—分布特征”的第三层证据。结果表明，STaR-GNN 的总体改善具有较强的时间和空间普遍性，同时保留了对局部失效情况和空间异质性的必要边界。

## 3.4 Week-ahead demand dynamics

前述指标和配对分析说明了模型在统计意义上的总体表现，但不能直接展示误差在一周需求过程中的具体分布。为此，[Fig. 4. Week-ahead demand dynamics from population-level error structure to a representative forecast](https://github.com/XU-Malice/STaR-GNN-BWDF/blob/main/paper/figures/submission/main_fig4_week_ahead_dynamics.pdf) 按照“总体误差结构—代表性预测轨迹—局部误差定位”的顺序展示 168 h 预测行为。Fig. 4a 基于全部测试起点和七个预测日，给出 24 个日内小时位置上的平均系统总需求绝对误差及移动块 95% 置信区间；Fig. 4b 展示一个按预设规则选取的代表性一周预测；Fig. 4c 给出相同预测起点上的逐小时绝对误差。

在所有日内小时位置上取平均后，STaR-GNN 的系统总需求绝对误差为 4.920 L s$^{-1}$，DCRNN 和 STGCN 分别为 7.735 和 8.574 L s$^{-1}$，对应降幅为 36.4% 和 42.6%。STaR-GNN 在 24 个日内小时位置中的 22 个位置上低于 DCRNN，并在 23 个位置上低于 STGCN。这表明模型优势覆盖了大部分日内周期，而不是仅集中在低需求时段或少数小时；与此同时，仍有 1–2 个小时位置未取得最低平均误差，因此不能将该结果解释为所有时刻均严格最优。

代表性预测并非依据曲线视觉效果选取，而是从 46 个测试起点中选择 STaR-GNN 总 MAE 最接近其中位数的样本。该规则选中统一测试索引 70，其 STaR-GNN 总 MAE 为 12.182，接近全部测试起点的中位数 12.193；同一预测起点上，DCRNN 和 STGCN 的总 MAE 分别为 15.517 和 14.653。Fig. 4b 表明，三个模型均能够再现一周需求的基本日周期，但 STaR-GNN 在多数日需求水平转换和局部峰谷附近与观测曲线更为接近。Fig. 4c 进一步表明，STaR-GNN 在大部分时段保持较低误差，但在少数快速变化时刻仍出现较大偏差。

该图的作用不是通过单个案例再次进行模型排序，而是将基于全部测试样本得到的统计结果映射到可解释的一周需求过程。总体日内误差分析提供群体层面的证据，代表性轨迹用于说明这些改善在实际预测曲线中的表现形式，逐小时误差则揭示模型仍需改进的局部时段。

## 3.5 Discussion

四组实验形成了由总体到局部、由数值到预测行为的递进证据链。首先，STaR-GNN 在 24 h 和 168 h 任务的四项总体指标上均优于全部对比模型，证明了完整方法的总体有效性。其次，因子消融表明 SAS-Norm 是周尺度误差降低的主要来源，而 FA-DPR 提供具有预测提前期和指标依赖性的补充作用；完整模型在 MAPE、RMSE 和 NSE 上形成更加均衡的结果。再次，逐测试起点和逐 DMA 分析说明平均改善并非主要来自少数有利日期或分区。最后，日内误差分布及代表性一周轨迹将统计改进落实到需求峰谷、日尺度转换和局部误差位置上。

从模型行为看，24 h 预测主要涉及相邻日之间的短期延续，而 168 h 预测同时受到需求状态变化、工作日结构和误差递推的影响。STaR-GNN 在 168 h MAPE、RMSE 和 NSE 上相对于图模型取得更大改善，与其将日尺度需求状态和日内形状分离、并根据未来预测位置检索历史日模式的设计目标相一致。不过，消融结果也表明 FA-DPR 的独立贡献并非对所有指标都为正，因此当前证据更支持两个模块的互补作用，而不是将全部提升归因于单一组件。

从供水运行角度看，24 h 预测可服务于次日供水计划和泵组运行安排，168 h 预测则为周尺度蓄水调度、设备维护和运行资源配置提供更长的决策提前量。STaR-GNN 在周尺度任务中的误差控制优势，以及其在多数 DMA 和日内小时位置上的一致改善，表明该模型具有支持多分区协同预测的潜力。同时，不同 DMA 的改善幅度和 DMA G 的局部例外说明，实际应用仍应保留分区级监测与误差诊断，而不能仅依据系统总需求指标判断模型可靠性。

本研究仍存在若干边界。首先，实验基于一个真实多 DMA 供水系统，模型在不同城市、气候和用水结构下的可迁移性仍需进一步验证。其次，当前图结构刻画的是由历史需求形成的功能关联，而非完整的水力拓扑，其静态形式也不能反映区域依赖随季节或运行工况变化的过程。再次，GRU、LSTM、MSNet 和 MSCMNet 系列的结果采用原研究报告值，而 DCRNN、STGCN 和 STaR-GNN 的结果来自本研究的统一测试，因此跨来源比较主要用于建立与既有研究的量级联系；本文关于组件作用、时间稳健性和空间稳健性的结论则建立在同一测试协议下的直接比较上。最后，Fig. 4 的代表性轨迹仅用于解释总体统计结果，不应代替全测试期评价。

综合来看，STaR-GNN 的优势不仅表现为总体误差降低，还体现为跨预测时域、测试日期、DMA 和日内周期的较强一致性。与单纯增加模型复杂度相比，针对多 DMA 需求中日尺度状态变化和预测位置依赖进行显式建模，是其在周尺度预测中保持较高精度的关键。

