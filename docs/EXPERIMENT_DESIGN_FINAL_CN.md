# Journal of Hydrology 投稿版实验设计与证据链

本文件定义 STaR-GNN 论文实验部分的最终组织。目标不是堆叠更多 benchmark，而是同时满足：

1. **Journal of Hydrology 优先**：围绕真实多 DMA 供水需求、预测时域、空间异质性和周尺度运行意义展开；
2. **高水平时空预测论文的实验严谨性**：统一协议、强 baseline、严格 factorial ablation、paired robustness、长时域行为和可复现审计；
3. **Nature-style claim-driven evidence architecture**：每张主图回答一个 Results-level scientific question，panel 承担不同推理角色；
4. **figures4papers-style publication graphics**：固定视觉层级、最终出版尺寸、克制配色、直接标注关键量、PDF/SVG 可编辑矢量输出。

---

## 1. 正文只保留两张主表

### Table 1 — Overall forecasting performance

回答：**STaR-GNN 是否在 24 h 和 168 h 两个预测时域上取得总体优势？**

包含：

- GRU、LSTM、MSNet、MSCMNet-WM、MSCMNet-M、MSCMNet-W；
- DCRNN、STGCN、STaR-GNN。

前六个时序/多尺度模型使用 Que et al. (2024) 的结果，并通过表注说明来源，不再使用符号标记；DCRNN、STGCN、STaR-GNN 使用本文评价流程。正文显示统一 3 位小数，审计 CSV 保留可用精度。

### Table 2 — Factorial ablation

回答：**SAS-Norm 和 FA-DPR 分别贡献什么？**

只包含：

1. DCRNN；
2. DCRNN + SAS-Norm；
3. DCRNN + FA-DPR；
4. STaR-GNN。

STGCN 是独立 graph baseline，不属于消融。

168 h total MAE 中 SAS-Norm-only 为 12.208，STaR-GNN 为 12.234；完整模型在 MAPE、RMSE 和 NSE 上最好。该 0.21% MAE 点估计差异不用于声称任一模型稳定占优，长时域稳定性由 Main Fig. 5 和 moving-block bootstrap 进一步限定。

### Supplementary Table S1 — DMA-level metrics

并列报告全部九种模型在 DMA A--J 的四指标绝对值。正文用 Main Fig. 2 概括相对各基线的跨 DMA 改善分布，用 Main Figs. 3--4 分别呈现 24 h 与 168 h 下 STaR-GNN 和局部最优基线的绝对性能。

---

## 2. 正文七张核心结果图

## Main Figure 1 — Overall four-metric performance

**Results-level question：**STaR-GNN 是否在 24 h 与 168 h、四个互补指标上都保持总体优势？

Panel a 用热图展示相对六个 published sequence/multiscale models 与 DCRNN、STGCN 的 MAE/MAPE/RMSE 降幅；Panel b 展示 NSE 绝对增益。正值始终代表 STaR-GNN 更优。该图与 Table 1 构成首个实验章节：表给绝对数值，图给改善幅度与跨预测时域的一致性。

严谨表述是“STaR-GNN 在全部比较模型上取得更优总体指标”，不能扩大为“所有图模型都全面优于时序模型”；168 h 下 DCRNN/STGCN 并不在所有指标上优于 MSCMNet-W。

---

## Main Figure 2 — DMA-level performance breadth

**Results-level question：**相对不同模型族的改善是否广泛分布于 DMA，而非由少数分区驱动？

四个 panel 分别展示 MAE、MAPE、RMSE 和 NSE。纵轴为八种基线，横轴为 STaR-GNN 相对每个基线的有符号逐 DMA 改善；小点保留十个 DMA，大点与线段给出中位数和四分位距，圆和方形区分 24 h 与 168 h。误差指标采用相对降幅，NSE 采用绝对增益，正值始终代表 STaR-GNN 更优。

Main inference：STaR-GNN 的改善在时序模型和图模型两类基线中均具有广泛的跨 DMA 覆盖，但分布的负向尾部说明其并非在所有局部任务上占优。

---

## Main Figure 3 — 24 h DMA-level absolute performance

**Results-level question：**日前预测的系统级优势落到各 DMA 后，在哪里保持、接近或反转？

四个 panel 分别给出 MAE、MAPE、RMSE 和 NSE 的绝对值。每个 DMA–指标组合独立选择最强非 STaR-GNN 方法，并与 STaR-GNN 采用成对柱比较；局部最优基线占优时，灰色柱改为橙色。

Main inference：24 h 的四个例外集中于 DMA A。STaR-GNN 在其余九个 DMA 上均保持四指标领先，但不同 DMA 的柱高差异显示领先幅度并不均衡。

---

## Main Figure 4 — 168 h DMA-level absolute performance

**Results-level question：**预测时域延长后，哪些 DMA 的局部竞争关系发生改变？

沿用 Main Fig. 3 的四指标分面、成对柱和颜色编码，但各指标根据 168 h 数据使用独立纵轴，避免跨时域共用尺度压缩局部差异。

Main inference：DMA A、E 和 G 的四项指标及 DMA I 的 NSE 由其他方法取得更优结果。与 Main Fig. 3 对照可见，预测时域延长并未造成所有 DMA 同步退化，而是改变了分区从模型结构中获得的相对收益。

---

## Main Figure 5 — Four-metric ablation and lead-time stability

**Results-level question：**SAS-Norm 与 FA-DPR 如何影响一周预测中的准确性和稳定性？

四个 panel 分别展示 MAE、MAPE、RMSE、NSE。SAS-Norm-only、FA-DPR-only 与 Full 均按相同测试窗口和预测日相对 DCRNN 计算方向统一的 paired improvement；95% CI 采用 ordered seven-window moving-block bootstrap。

同一 forecast day 内的三个模型点做轻微水平错位，并同时使用不同 marker/linestyle，解决 SAS-Norm 与 STaR-GNN 曲线几乎重合的问题。Main inference：SAS-Norm 是周尺度改善的主要来源；FA-DPR-only 的作用具有指标依赖性；完整模型在 168 h 的 MAPE、RMSE、NSE 上最好。

---

## Main Figure 6 — Forecast-origin and difficult-window robustness

**Results-level question：**系统级平均优势是否跨测试起点成立，并能否在需求快速变化的窗口中保持？

Panels a–b 展示 24 h 与 168 h 下相对 DCRNN/STGCN 的逐起点 MAE、MAPE、RMSE 降幅及 seven-origin moving-block bootstrap 95% CI；Panel c 展示 NSE 绝对提升；Panel d 展示由观测 normalized mean absolute ramp 定义的高波动四分位窗口中的胜出数。仅使用同协议模型，不从已发表 DMA 平均值反推样本分布。

Main inference：168 h 的四指标改善在绝大多数预测起点中保持；24 h 相对 DCRNN 的 MAPE/RMSE 区间跨零。高波动窗口总体仍以正向结果为主，但相对 STGCN 的覆盖度收窄。

---

## Main Figure 7 — Week-ahead demand dynamics

**Results-level question：**统计上的改进在真实一周需求轨迹中具体表现为什么？

采用 scale-to-instance chain。

### Panel a — Population-level diurnal aggregate-error profile

使用全部共同测试窗口 × 7 forecast days，将 168 h 折叠为 24 h 日内周期。对 DCRNN、STGCN、STaR-GNN 比较 aggregate-demand absolute error，并使用 moving-block bootstrap 95% CI。

推理角色：**population-level diagnostic**。

该 panel 检验优势是否贯穿日内周期，而不是只发生在低负荷或少数时段。

### Panel b — Representative 168 h aggregate-demand trajectory

代表性测试窗口使用预先固定的 median-error proximity rule，不依据图像外观挑选。Observed 使用黑色；STaR-GNN 为深蓝 hero line；DCRNN/STGCN 降低视觉权重。

推理角色：**representative instance**。

### Panel c — Corresponding absolute error

展示同一 representative window 的 aggregate-demand absolute error，并标出连续七个 forecast days。

推理角色：**instance-level error localization**。

需求单位统一为 L s⁻¹，与原始 BWDF net inflow 定义一致。

---

## 3. 统一 publication visual system

不再坚持旧绿色。最终 hero system：

- STaR-GNN：deep blue `#0F4D92`；
- DCRNN：dark gray `#5C5C5C`；
- STGCN：light gray `#A6A6A6`；
- DCRNN + SAS-Norm：soft blue `#8FB6D5`；
- DCRNN + FA-DPR：muted violet `#9A86B8`；
- Observed：near-black `#1F1F1F`。

原则：

- proposed method 是唯一 hero hue；
- baseline 不与 proposed model 争夺视觉注意力；
- variants 属于 proposed-family，使用低饱和近邻色；
- 同一模型在所有 panel 中永不换色；
- 同时使用 linestyle/marker，保证灰度打印仍可识别；
- 主图按约 190 mm 双栏宽度直接设计，不先画超大图再缩小；
- PDF + editable SVG 为主矢量输出，PNG 300 dpi 只用于预览。

统一样式由：

```text
scripts/reproduce/manuscript_plot_style.py
```

管理。

---

## 4. Results 最终章节结构

### 3.1 Predictive performance across forecasting horizons and DMAs

证据：Table 1 + Main Fig. 1 + Main Figs. 2--4 + Supplementary Table S1。

写法：claim → quantitative evidence → sequence/multiscale comparison → graph baseline comparison → bounded inference。

### 3.2 Component contributions and lead-time dependence

证据：Table 2 + Main Fig. 5。

开头直接给 scientific finding，不以“Table 2 shows...”起句。

建议核心句：

> The two proposed modules contributed to week-ahead forecasting in distinct but complementary ways: SAS-Norm accounted for most of the reduction in absolute MAE, whereas FA-DPR primarily reduced the accumulation of error with increasing lead time.

### 3.3 Robustness across forecast origins and demand conditions

证据：Main Fig. 6 + Supplementary Table S3。

只对同协议 DCRNN、STGCN、STaR-GNN 做配对统计，并明确 overlapping 168 h windows 的移动块处理。困难条件只由观测需求定义。

### 3.4 Week-ahead demand dynamics and practical implications

证据：Main Fig. 7。

代表案例定义为 population analysis 的 instance-level validation，而不是额外 leaderboard；实际意义与限制由该过程证据自然导出，不另设独立 summary/discussion 小节。

---

## 5. Results 写作规则

每个 subsection 均按：

> **Claim → quantitative evidence → comparison → local mechanism interpretation → boundary → next question**

避免连续使用：

> Table X shows... Figure Y shows... As can be seen...

Results 可以包含与当前证据直接绑定的局部解释；跨文献综合、广义机制和工程意义放在 Discussion。

不要隐藏边界：

- FA-DPR 168 h MAPE 略差于 DCRNN；
- SAS-Norm-only 的 168 h total MAE 点估计略低于 Full；
- Full-vs-SAS 的差异由 ordered moving-block analysis 限定，不将 0.21% 放大为稳定优劣关系。

---

## 6. 当前冻结结果之外的可选补充实验

这些实验尚未纳入当前主结果，在没有真实运行结果前不得写入论文结论。

1. **Efficiency analysis**：parameter count、inference time、GPU memory，至少比较 DCRNN / STGCN / STaR-GNN。
2. **FA-DPR alignment diagnostic**：在 evaluation 中错位/打乱 future calendar alignment，检验 168 h 性能是否按预期下降。该 targeted misalignment 比普通 remove-module ablation 更能支持 FA-DPR 的机制解释。
3. **Missing-data robustness**：对输入窗口施加可控比例的随机缺测与连续缺测，报告四指标相对无缺测条件的退化，并明确插补策略。

---

## 8. 权威生成入口

最终投稿图只由：

```text
scripts/reproduce/render_submission_figures.py
```

生成。

最终投稿显示表只由：

```text
scripts/reproduce/render_submission_tables.py
```

生成。

旧 `manuscript_fig1...5` 和 `test_*` 图保留用于历史复现/诊断，但不再是投稿版权威图件。
