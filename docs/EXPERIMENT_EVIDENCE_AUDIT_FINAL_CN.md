# STaR-GNN 实验证据审计与 Journal of Hydrology 最终计划

本文件记录参考论文拆解、仓库证据审计、差距分析、审稿迭代及最终执行边界。它是实验设计审计文件，不是论文正文。资料判定优先级为：上传全文与补充材料、冻结仓库结果、期刊网页、其他来源。无法由全文或补充材料确认的项目均标记为“现有材料无法确认”。

## A. 五篇参考论文的 Results and Discussion 结构

### A1. 逐篇结构与图表分工

| 论文 | Results / Results and Discussion 结构 | Baselines 与指标 | Table 的作用 | Figure 的作用 | 从总体性能进入深入分析的路径 |
|---|---|---|---|---|---|
| Que et al. (2024) | Model Performance Analysis；高不确定 DMA、总需水、极端需水 DMA；Training Convergence；Model Interpretability；Discussion | GRU、LSTM、MSNet、3 个 MSCMNet 变体；MAE、MAPE、RMSE、NSE；24/168 h；DMA A–J 与 total | Supplementary S1 给出 6 个模型在各 DMA/total、两时域、四指标的绝对值；S2–S3 为搜索空间与参数 | Fig. 1 雷达图概括 DMA/total 多指标；Fig. 2 改善率；Fig. 3 46 个序列的 total-demand 误差箱线图；Fig. 4 极端 DMA 周轨迹与逐日 MAE；Fig. 5 收敛；Fig. 6 ApEn/SHAP；Fig. S4 校正前后轨迹 | 总体/DMA 覆盖 → total-demand 分布 → 极端 DMA 案例 → 收敛与解释。Supplement 仅给 A 类模型 DMA 平均指标，不能恢复其逐起点或逐小时输出。 |
| Lin et al. (2025) | 3.1 参数；3.2 多时间尺度比较；3.3 消融 | 1D-CNN、LSTM、Transformer、DLinear、PatchTST；MAE、RMSE、MAPE、R²；1/6/12/24/48 h | Table 3 完整多时域指标；Tables 4–5 归一化和组件消融 | Figs. 5–6 雷达图视觉化 Table 3；Fig. 7 五个时域的末四日轨迹 | 多时域总体指标 → 轨迹误差位置 → 组件消融。雷达图与表存在部分重复，轨迹提供新的过程信息。 |
| Pesantez et al. (2025) | 4.1 Input-output；4.1.1 Error analysis；4.1.2 Feature importance；4.2 Output-only；4.3 Missing data | IONET/LSTM/LGSSM 及 output-only 变体；NSI、MAPE、RMSE；24 h | Tables 4–5 报告 119 天的 min/mean/max；缺测表位于 Supplement（细节现有材料无法确认） | Figs. 5/9 按星期箱线图；Figs. 6/10 代表性日曲线；Fig. 7 小时/日误差分布；Fig. 8 特征重要性 | 平均性能 → 按星期与测试期分布 → 日内曲线 → 特征作用 → 缺测敏感性。 |
| Huang et al. (2025) | Scenario 1/2/3 嵌入 3.4 Results；3.5 Discussion | HOA-HSS-BiLSTM 及 LSTM/GRU/TCN/ANN 变体；MAPE、RMSE、R²、NSE、运行时间、误差分位数 | Tables 4–6 分别回答总体模型、调度策略和样本构造 | Taylor 图概括复合性能；预测/绝对误差曲线定位时段；CDF/误差分布/箱线图刻画点误差 | 每个 scenario 只验证一个设计问题：总体竞争力 → 学习策略 → 样本构造。 |
| Xue and Zhu (2025) | 4.1 Key findings；4.2 Existing models；4.3 Implications and interpretation；4.4 Strengths/limitations | LSTM、Informer、STGCN 与 SiSGNN；多种图配置；MAE、RMSE、NSE；T+1/3/6/9 | Tables 1–2 给多提前期绝对指标 | Fig. 5 典型洪峰过程；Figs. 6–7 图配置/周期组件；Fig. 8 预定义与学习图热图 | 总体比较 → 图配置与周期机制 → learned adjacency → 案例 → 含义和局限。 |

### A2. 大型参考论文实验对照表

符号：✅ 明确报告；◐ 相近分析；❌ 未报告；— 不适用。

| 实验分析类型 | Que 2024 | Lin 2025 | Pesantez 2025 | Huang 2025 | Xue & Zhu 2025 |
|---|---|---|---|---|---|
| Overall performance | ✅ S1 绝对指标；Fig. 1 | ✅ Table 3 | ✅ Tables 4–5 | ✅ Tables 4–6 | ✅ Tables 1–2 |
| Multiple horizons | ✅ 24/168 h | ✅ 1–48 h | ❌ 固定 24 h | ❌ 固定时域 | ✅ T+1/3/6/9 |
| DMA/site/basin-level | ✅ 10 DMA + total | ❌ 单系统 | ❌ 单系统 | ✅ 两个 DMA | ◐ 图配置而非站点分布 |
| Relative improvement | ✅ Fig. 2 | ◐ 文本百分比 | ◐ 相对讨论 | ◐ scenario 间差异 | ◐ 表内比较 |
| Sample-level distribution | ✅ Fig. 3 箱线图 | ❌ | ✅ 星期箱线图与误差分布 | ✅ CDF/分布/箱线图 | ❌ |
| Paired comparison / CI | ❌ | ❌ | ❌；LGSSM 有预测区间 | ❌ | ❌ |
| Representative trajectory | ✅ Fig. 4 | ✅ Fig. 7 | ✅ Figs. 6/10 | ✅ Figs. 5–6 | ✅ Fig. 5 |
| Difficult/extreme conditions | ✅ 高/低平均需水 DMA 的自然极端周 | ❌ | ◐ weekday/weekend | ◐ 误差尾部 | ✅ sudden-change/peak case |
| Ablation | ✅ correction comparison | ✅ Tables 4–5 | ◐ I/O 与 output-only | ✅ Scenarios 2–3 | ✅ 图与周期组件配置 |
| Lead-time analysis | ◐ 24 vs 168 h | ✅ 多时域 | ❌ | ❌ | ✅ 多提前期 |
| Training convergence | ✅ Fig. 5 | ❌ | ❌ | ❌ | ❌ |
| Parameter sensitivity | ✅ S2–S3 | ◐ 参数设定 | ❌ | ◐ HOA 参数 | ❌ |
| Feature/model interpretation | ✅ ApEn/SHAP | ❌ | ✅ feature importance | ❌ | ✅ 图配置与邻接热图 |
| Practical implications | ✅ Discussion | ◐ | ✅ | ✅ 3.5 | ✅ 4.3 |
| Explicit limitations | ✅ | ◐ | ◐ | ✅ | ✅ 4.4 |

结论不是“每篇论文都使用同一套图”，而是 Table 先建立总体量级，Figure 再回答分布、时序过程、困难条件或机制等不同科学问题。Taylor/radar/convergence/SHAP 并非 JoH 方法论文的固定清单。

## B. STaR-GNN 实验可行性审计

状态：① 已有真实结果；② 可由已有 prediction outputs 重算；③ 运行已有分析/推理代码即可且无需重训；④ 需重训；⑤ 当前无法获得。

| 实验类型 | 参考论文中是否常见 | A 类模型 | B 类模型 | C 类模型 | 真实数据来源 | 当前状态与工作量 | 建议 |
|---|---|---:|---:|---:|---|---|---|
| 24/168 h 总体四指标 | 常见 | ① | ① | ① | Que S1；frozen common-46 | 已有，低 | 正文 Table 1/2 |
| 10 DMA 四指标 | 多站点研究常见 | ① | ① | ② | Que S1；metrics/predictions | 已有/可重算，低 | Table S1；正文概括 |
| 各 DMA relative improvement | 常见 | ② | ② | ② | DMA 平均指标 | 已有，低 | Fig. 2 |
| Cross-DMA distribution | 常见 | ② | ② | ② | DMA 平均指标 | 已有，低 | Fig. 2；描述性 |
| DMA absolute best-baseline comparison | 价值高 | ② | ② | — | 全九模型 DMA 指标 | 已有，低 | Fig. 3/Table S1–S2 |
| 46-origin 四指标分布 | 常见且重要 | ⑤ | ② | ② | frozen predictions | 新增统计，低 | Fig. 5；仅 B/C |
| Boxplot / ECDF | 常见 | ⑤ | ③ | ③ | frozen predictions | MAE ECDF 已有，低 | Supplement；主图不用重复 |
| Paired win rate | 高水平 ML 常见 | ⑤ | ② | ② | 同一 common-46 | 新增，低 | Fig. 5/Table S3 |
| Moving-block CI | 统计价值高 | ⑤ | ③ | ③ | 有序 common-46 | 消融已有；总体新增，低 | Figs. 4–5 |
| Day 1–Day 7 | 长时域研究常见 | ⑤ | ② | ① | 168 h predictions | 已有，低 | Fig. 4 |
| Component × lead time | 创新验证必要 | — | — | ① | factorial outputs | 已有，低 | Fig. 4 |
| Representative 168 h trajectory | 常见 | ⑤ | ① | ① | frozen predictions | 已有，低 | Fig. 6 |
| Hourly absolute error/profile | 常见 | ⑤ | ① | ① | frozen predictions | 已有，低 | Fig. 6 |
| Difficult-demand windows | 常见/价值高 | ⑤ | ② | ② | observed targets + predictions | 新增，低 | Fig. 5d/Table S3 |
| Extreme cases | 常见但易选择偏差 | ⑤ | ② | ② | predictions | 可做，低 | 不另设主图；中位规则替代 |
| Training convergence | 偶见 | 论文中有，仓库无原始值 | ① | ① | histories | 已有，中 | 不做；不回答核心 claim |
| Parameter count | 偶见 | ⑤ | ③ | ①/③ | summaries/checkpoints | 部分已有，中 | 暂不做；协议不完整 |
| Unified inference time | 偶见 | ⑤ | ③ | ③ | checkpoints | 需统一硬件推理，中 | Supplement 可选，非阻断 |
| Feature importance | 部分论文使用 | 论文中有 SHAP | ⑤ | ⑤ | 无统一输入归因输出 | 当前无法公平比较 | 不做 |
| SAS-Norm mechanism diagnostic | 创新相关 | — | — | ③ | mechanism outputs/history | 可做，中 | Supplement 可选；消融为主证据 |
| FA-DPR attention diagnostic | 创新相关 | — | — | ③ | attention/gates | 可做，中 | 不以注意力作因果证明 |
| Static graph description | 图论文常见 | — | ① | ① | adjacency/node metrics | 已有 | 方法/Supplement |
| Learned graph interpretation | 图论文常见 | — | ⑤ | ⑤ | 模型未学习动态图 | 不适用 | 不做，不虚构 learned graph |
| Missing-data robustness | 部分论文使用 | ⑤ | ④ | ④ | 需构造缺测并重训/评估 | 高 | 不做；本文无缺测稳健性 claim |

严格边界：A 类模型只有论文及 Supplementary S1 的总体/DMA 平均指标。不能从这些值构造 46-origin 分布、CDF、bootstrap、Day 1–Day 7 或预测曲线。

## C. 实验差距分析

### 已有优势

- 九模型、两时域、四指标和 10 DMA 的证据覆盖度高于仅报告单系统平均值的研究。
- 2×2 factorial ablation 比“逐个删除组件”更清楚地区分 SAS-Norm、FA-DPR 与完整模型。
- 对重叠的 168 h 测试窗口采用 ordered moving-block bootstrap，比把 46 个起点当独立样本更严谨。
- DMA 级分析保留全部局部失败，不以“每个 DMA 全面最优”包装结果。
- 代表性周轨迹由预注册式中位误差规则确定，不按视觉效果挑选。

### 初始关键缺口

1. **Major：平均性能之外缺少四指标的逐起点稳健性。** 原 MAE ECDF 不能代表 MAPE、RMSE、NSE，也不能处理重叠窗口。
2. **Minor：困难条件只有代表性轨迹中的定性表述。** 需要用仅由观测定义、与模型误差无关的难度指标做透明分层。
3. **Minor：3.5 独立综合讨论重复前文，削弱 Results and Discussion 的递进。**
4. **Minor：**跨来源 baseline 与同协议 baseline 的推断边界需要贯穿正文，而不能只藏在表注。

### 不应机械补充的项目

- Radar/Taylor 图会重新编码已有绝对指标，信息增量不足。
- 训练损失不能直接支持测试期泛化，不作为核心证据。
- SHAP 不能解释本文图结构或两个创新组件；无可靠统一归因输出时不做。
- Learned adjacency 与动态图不适用于当前固定功能图。
- Missing-data retraining、更多预测时域和大规模参数敏感性不直接支撑论文当前核心 claim，成本高且会扩散故事。

## D–H. PASS 版 Results and Discussion 结构与图表计划

| Section | Scientific question | Models / metrics | Table / Figure 与 panel | 数据与状态 | Main claim | Limit of inference | Supplement / priority |
|---|---|---|---|---|---|---|---|
| **3.1 Predictive performance across forecasting horizons and DMAs** | 总体优势是否跨时域、指标和 DMA 广泛成立；局部边界在哪里？ | 九模型；MAE/MAPE/RMSE/NSE；24/168 h | Table 1 绝对系统值；Fig. 1 相对总体改善；Fig. 2 各基线跨 DMA 效应分布；Fig. 3 与局部最优基线的绝对指标比较 | Table/Figs. 1–3 均为① | STaR-GNN 在系统级全部指标最优，优势由多数 DMA 支撑，但长时域存在 A/E/G 与 I-NSE 等局部例外 | 跨来源模型只用于性能定位；DMA 比较是描述性且只有 10 个区域，不能作空间因果归因 | Tables S1–S2；最高优先级 |
| **3.2 Component contributions and lead-time dependence** | 两组件各贡献什么，贡献如何随 Day 1–7 变化？ | DCRNN、SAS-only、FA-only、Full；四指标 | Table 2 factorial；Fig. 4 四指标逐日 paired improvement + moving-block CI | ① | SAS-Norm 是长时域稳定性的主要来源；FA-DPR 贡献较小且依赖指标；Full 的综合指标更均衡 | Ablation 支持功能贡献，不证明物理因果；Full 与 SAS-only 的 168 h MAE 不可声称有稳定差异 | 精确日级 CSV；最高优先级 |
| **3.3 Robustness across forecast origins and demand conditions** | 平均优势是否跨测试起点成立，在高波动窗口是否保持？ | 同协议 DCRNN/STGCN/STaR-GNN；四指标 | Fig. 5a–b 46-origin 三误差配对分布与块 CI；c NSE；d 观测定义的高波动四分位 win rate | ②→①，无重训 | 168 h 改善在绝大多数起点和高波动窗口保持；24 h MAPE/RMSE 的均值 CI 对 DCRNN 跨零，需诚实限定 | 仅适用于 B 类同协议模型；高波动 n=12 为描述性分层，不作多重显著性检验 | Table S3；最高优先级 |
| **3.4 Week-ahead demand dynamics and practical implications** | 统计优势在实际周轨迹中表现为何，仍有哪些误差和应用边界？ | DCRNN/STGCN/STaR-GNN；aggregate absolute error | Fig. 6a 全测试窗口日内误差；b 中位规则周轨迹；c 同窗逐小时误差 | ① | 优势覆盖大多数日内小时，典型周中更好地跟踪水平转换；快速变化仍产生尖峰 | 单系统、不评价控制成本、不提供概率区间、固定功能图；不把典型案例当总体证明 | 选择规则 JSON/轨迹 CSV；高优先级 |

递进关系为：系统与空间范围（在哪里成立）→ 组件和提前期（由什么贡献、何时出现）→ 测试样本和困难条件（是否稳定）→ 真实周行为、应用意义与边界（如何表现、能推断到哪里）。没有单列 3.5–3.7，discussion 在每节局部展开，并在 3.4 收束。

## I–K. Journal of Hydrology / 时空预测审稿审计

### Round 1 — FAIL

**Major**

- 仅有系统/DMA 平均值和 MAE 过程图，无法证明四指标优势在 46 个重叠预测起点上的稳定性。

**Minor**

- 缺少与预测误差无关的困难条件定义。
- 独立 3.5 重复结论；章节故事松散。
- 跨来源 baseline 的公平性边界需更显式。

### 修改记录

1. 新增 Fig. 5 与 Table S3，只对同协议模型做逐起点配对分析。
2. 采用 ordered seven-origin moving-block bootstrap，保留 168 h 窗口重叠依赖。
3. 以观测需求的 normalized mean absolute ramp 定义高波动窗口；阈值为各时域 46 个起点的第 75 百分位，不参考模型误差。
4. 将正文压缩为 3.1–3.4；总体与 DMA 合并，实际周行为与 discussion/limitations 合并。
5. 明确 A 类模型不进入 origin-level 统计；不将注意力权重解释为因果机制。

### Round 2 — PASS

审稿检查的 22 项要求均已满足或有合理边界：核心创新由 factorial ablation 和 lead-time evidence 支撑；24/168 h、10 DMA 和四指标完整；平均精度、逐起点稳健性、困难条件和真实周轨迹形成互补证据；baseline 来源严格区分；局部失败和统计不确定性未隐藏；重复图型、无科学问题的实验和过度机制归因均已删除或拒绝。效率和缺测实验不是当前 claim 的必要条件，因此不构成“实验验证不足”的 major concern。

## L. 执行清单

1. **P1 — 统计与审计**：校验 release checksum；生成 46-origin 四指标、moving-block CI、win rate、NMAR 高波动分层和 Table S3。
2. **P2 — 投稿图件**：生成 Main Fig. 5；将现有周轨迹顺延为 Main Fig. 6；核对 PDF/SVG/300-dpi PNG、轴、单位、图例和碰撞。
3. **P3 — 正文与复现**：将中英文 Results and Discussion 改为 3.1–3.4；更新 captions、README、实验设计、artifact audit、tests 与 checksum；执行全仓审计后合并。
