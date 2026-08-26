# DMA-level performance analysis：图形选择与论证设计

## 1. 科学问题

推荐小节标题为 **Cross-DMA consistency and horizon-dependent heterogeneity**。该小节不再重复系统总需求层面的“谁最好、提高多少”，而回答两个更具体的问题：

1. STaR-GNN 相对各类基线的改善是否广泛分布于不同 DMA，而非由少数高需水分区主导？
2. 当预测时域从 24 h 延长至 168 h 时，STaR-GNN 相对每个 DMA 的最强局部竞争者仍保留多大优势，例外集中在哪里？

## 2. 文献中的组织规律

多站点水文论文通常先给出跨站点分布，再定位站点差异，而不是只报告平均名次。例如，Mosaffa et al. 先用箱线图和 CDF 概括 530 个站点的指标分布，再用带正负号的站点差值图定位改善和退化，并进一步分析网络属性；Aerts et al. 也先展示 299 个流域的 CDF，再报告流域级性能差异。WRR 中，Ouyang et al. 使用带 1:1 线的流域级散点和空间图，同时明确报告提出方法在部分流域退化；Song et al. 则将总体 CDF、NSE 差值地图和按流域属性分层的箱线图依次组织。

参考来源：

- [Mosaffa et al. (2026), HESS](https://hess.copernicus.org/articles/30/2079/2026/)
- [Aerts et al. (2024), HESS](https://hess.copernicus.org/articles/28/5011/2024/)
- [Ouyang et al. (2025), Water Resources Research](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR036593)
- [Song et al. (2025), Water Resources Research](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024WR038928)
- [Que et al. (2024), Water Research X](https://doi.org/10.1016/j.wroa.2024.100260)

这些论文共同提供了两个可迁移原则：**分布图负责回答总体覆盖，带符号的站点差值负责回答空间位置和例外；负差异应保留并讨论，而不是从图中剔除。**

## 3. 候选图比较

| 候选图 | 能回答的问题 | 优点 | 主要局限 | 处理决定 |
|---|---|---|---|---|
| within-DMA ranking boxplot | 模型在多少 DMA–指标组合中名次靠前？ | 单位无关、紧凑 | 名次把微小差异和巨大差异视为相同；四项相关指标被当作 40 个等价样本；不能识别 DMA | 从正文删除 |
| first-place count heatmap | 每个 DMA 有几项指标第一？ | 保留 DMA 身份 | 丢失具体指标、差距和第二名信息；与排名箱线图重复胜场结论 | 从正文删除，不移入补充材料 |
| raw-metric boxplot | 各模型在 DMA 间的绝对指标分布如何？ | 水文论文常见 | MAE/RMSE 受 DMA 需求尺度支配，不能公平汇总跨 DMA 差异 | 不采用 |
| 1:1 paired scatter | 提出方法是否优于一个指定基线？ | JoH/WRR 常见，例外清楚 | 八个基线和四个指标会形成大量 panel；误差与 NSE 的优劣方向不同 | 不作为主图 |
| signed pairwise distribution | 相对每个基线的改善是否跨 DMA 广泛存在？ | 保留差距、方向和模型族；不混合 DMA 绝对尺度 | 不直接标出异常 DMA | 采用为 Fig. 2 |
| strongest-competitor dumbbell | 每个 DMA 相对最强局部替代方案的优势及其时域变化如何？ | 竞争标准保守；保留 DMA、幅度、正负和 24→168 h 变化 | 只聚焦 STaR-GNN 与局部最强竞争者 | 采用为 Fig. 3 |
| DMA 属性相关散点 | 哪些需求或图属性解释局部增益？ | 可产生机制解释 | 仅 10 个 DMA，现有图结构指标与增益无稳定关系；易产生过度解释 | 不采用 |

## 4. 最终两图逻辑

### Main Fig. 2 — cross-DMA pairwise breadth

四个 panel 分别为 MAE、MAPE、RMSE 和 NSE。纵轴为八种基线，横轴为 STaR-GNN 相对该基线的有符号改善；误差指标使用相对降幅，NSE 使用绝对提升。每个小点代表一个 DMA，大点为中位数，线段为四分位距；圆和方形分别表示 24 h 与 168 h。六种时序模型与两种图模型之间用水平线分隔。

该图回答“改善是否跨模型族和 DMA 广泛存在”，并替代原排名分布。零线左侧的数据完整保留。

### Main Fig. 3 — local margin and horizon transition

四个 panel 仍对应四项指标。纵轴为 DMA A–J；横轴为 STaR-GNN 相对该“预测时域–DMA–指标”组合中最强非 STaR-GNN 方法的有符号改善。每个 DMA 的 24 h 圆点与 168 h 方点用线连接，直接展示竞争幅度随预测时域的变化；红色点表示 STaR-GNN 不是该局部组合的最优方法。

该图在 Fig. 2 的分布证据之上增加三个信息：具体 DMA 身份、保守的最强竞争者边界，以及 24 h 到 168 h 的方向变化。因此两图是“跨 DMA 分布 → DMA 定位与时域演变”的递进关系，而不是重复统计胜场。

## 5. 数据完整性和解释边界

输入表包含 180 行，即 2 个预测时域 × 10 个 DMA × 9 个模型；主键无重复、四项指标无缺失，每个预测时域–DMA 组合均含九种模型。Figure 2 在每个预测时域内使用 \(10\text{ DMAs}\times8\text{ baselines}\times4\text{ metrics}=320\) 个有符号描述性对比，两个预测时域合计 640 个。这里的“320”是覆盖度的计数单位，不是 320 个相互独立的统计样本，也不用于显著性推断。Figure 3 在每个预测时域内使用 \(10\times4=40\) 个逐单元最强竞争者比较，两个预测时域合计 80 个。

## 6. 模拟 Journal of Hydrology 审稿审查

| 审稿风险 | 终稿处理 | 结论 |
|---|---|---|
| 将 320 个对比误读为独立重复 | 正文和 Fig. 2 图注均给出 \(10\times8\times4\) 的构成，并明确仅作描述性覆盖评价 | 通过 |
| 图离开正文后无法理解正负方向和符号 | 图内增加正负方向及点、区间、连线和红色标记说明；caption 定义误差降幅和 \(\Delta\mathrm{NSE}\) | 通过 |
| Fig. 2 与 Fig. 3 重复“多数获胜” | Fig. 2 回答相对各基线的跨 DMA 覆盖；Fig. 3 回答相对局部最强对手的保守边界及 24→168 h 变化 | 通过 |
| 选择性隐藏不占优的 DMA | 所有负值保留；正文明确报告 A、E、G 及 I 的例外 | 通过 |
| 从十个 DMA 过度推断成因 | 不作需求或运行机制的因果归因，仅提出延长时域后需保留 DMA 级监测 | 通过 |
| 写成逐格结果清单 | 论证按“覆盖广度—局部最强边界—时域异质性—解释边界”组织，仅报告支撑主结论的统计量 | 通过 |

24 h 时，STaR-GNN 在 40 个 DMA–指标组合中的 36 个超过全部八种竞争方法；四个例外均位于 DMA A。168 h 时，该数量为 27/40；DMA A、E 和 G 的四项指标均由其他方法取得更优值，DMA I 的 NSE 也不是最优。正文据此表述为“跨多数 DMA 保持优势”，不表述为“每个 DMA 全面获胜”。造成这些局部差异的需求机制不能由现有十节点结果直接识别，因此不作因果归因。

完整绝对数值保留在 Supplementary Table S1；旧排名图和首位计数热图因信息损失和相互重复，不再作为正文或补充图件。
