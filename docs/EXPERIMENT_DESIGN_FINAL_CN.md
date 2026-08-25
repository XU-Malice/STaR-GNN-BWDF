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

前六个时序/多尺度模型使用 Que et al. (2024) 已发表结果，并通过表内分组和表注说明来源，不再使用符号标记；DCRNN、STGCN、STaR-GNN 为 common-46 协议下重新评价。正文显示统一 3 位小数，审计 CSV 保留完整精度。

### Table 2 — Factorial ablation

回答：**SAS-Norm 和 FA-DPR 分别贡献什么？**

只包含：

1. DCRNN；
2. DCRNN + SAS-Norm；
3. DCRNN + FA-DPR；
4. STaR-GNN。

STGCN 是独立 graph baseline，不属于消融。

168 h total MAE 中 SAS-Norm-only 为 12.208，STaR-GNN 为 12.234；完整模型在 MAPE、RMSE 和 NSE 上最好。该 0.21% MAE 点估计差异不用于声称任一模型稳定占优，长时域稳定性由 Main Fig. 2 和 moving-block bootstrap 进一步限定。

### Supplementary Table S1 — DMA-level metrics

并列报告 DCRNN、STGCN、STaR-GNN 在 DMA A--J 的四指标绝对值。正文用 Main Fig. 3b 概括空间一致性，Supplementary Fig. S1 展示逐 DMA 的改善幅度。

---

## 2. 正文四张核心结果图

## Main Figure 1 — Overall four-metric performance

**Results-level question：**STaR-GNN 是否在 24 h 与 168 h、四个互补指标上都保持总体优势？

Panel a 用热图展示相对六个 published sequence/multiscale models 与 DCRNN、STGCN 的 MAE/MAPE/RMSE 降幅；Panel b 展示 NSE 绝对增益。正值始终代表 STaR-GNN 更优。该图与 Table 1 构成首个实验章节：表给绝对数值，图给改善幅度与跨预测时域的一致性。

严谨表述是“STaR-GNN 在全部比较模型上取得更优总体指标”，不能扩大为“所有图模型都全面优于时序模型”；168 h 下 DCRNN/STGCN 并不在所有指标上优于 MSCMNet-W。

---

## Main Figure 2 — Four-metric ablation and lead-time stability

**Results-level question：**SAS-Norm 与 FA-DPR 如何影响一周预测中的准确性和稳定性？

四个 panel 分别展示 MAE、MAPE、RMSE、NSE。SAS-Norm-only、FA-DPR-only 与 Full 均按相同 origin、相同 forecast day 相对 DCRNN 计算方向统一的 paired improvement；95% CI 采用 ordered seven-origin moving-block bootstrap。

同一 forecast day 内的三个模型点做轻微水平错位，并同时使用不同 marker/linestyle，解决 SAS-Norm 与 STaR-GNN 曲线几乎重合的问题。正值始终代表优于 DCRNN；误差指标使用相对降幅，NSE 使用绝对增益。

Main inference：SAS-Norm 是周尺度性能改善的主要来源；FA-DPR-only 的逐日结果显示其作用并非在所有指标上独立成立，完整模型在 168 h 的 MAPE、RMSE、NSE 上最好。SAS-only 与 Full 的 168 h MAE 只差约 0.21%，不得写成稳定显著差异。

---

## Main Figure 3 — Four-metric temporal and spatial robustness

**Results-level question：**总体优势是否只来自少数有利日期、少数 DMA 或单一指标？

Panel a 汇总 46 个 common forecast origins，Panel b 汇总 10 个 DMA；列为 24 h / 168 h × DCRNN / STGCN，行为 MAE/MAPE/RMSE/NSE。颜色编码“改善比较所占比例”，单元格文本同时给出 mean improvement 与 wins/comparisons。

该编码让四个量纲不同的指标使用同一可比颜色语义，同时保留实际改善幅度。逐 DMA 共有 160 个 horizon–baseline–metric comparisons，其中 158 个改善；两个例外均为 168 h、DMA G、相对 STGCN 的 RMSE 与 NSE，必须在正文和 caption 中如实报告。

---

## Main Figure 4 — Week-ahead demand dynamics

**Results-level question：**统计上的改进在真实一周需求轨迹中具体表现为什么？

采用 scale-to-instance chain。

### Panel a — Population-level diurnal aggregate-error profile

使用全部 46 common origins × 7 forecast days，将 168 h 折叠为 24 h 日内周期。对 DCRNN、STGCN、STaR-GNN 比较 aggregate-demand absolute error，并使用 moving-block bootstrap 95% CI。

推理角色：**population-level diagnostic**。

该 panel 检验优势是否贯穿日内周期，而不是只发生在低负荷或少数时段。

### Panel b — Representative 168 h aggregate-demand trajectory

代表性 origin 使用预先固定的 median-error proximity rule，不依据图像外观挑选。Observed 使用黑色；STaR-GNN 为深蓝 hero line；DCRNN/STGCN 降低视觉权重。

推理角色：**representative instance**。

### Panel c — Corresponding absolute error

展示同一 representative origin 的 aggregate-demand absolute error，并标出连续七个 forecast days。

推理角色：**instance-level error localization**。

需求单位统一为 L s⁻¹，与原始 BWDF net inflow 定义一致。

---

## 3. Supplementary figures

### Figure S1 — Detailed four-metric DMA improvements

以 2×2 panel 展示每个 DMA 的 MAE/MAPE/RMSE 相对降幅和 NSE 绝对增益。采用以 0 为中心的发散配色，因为 160 个单元格中确实有两个负值；蓝色为改善，红色为退化。

### Figure S2 — Per-origin ECDF

保留 DCRNN / STGCN / STaR-GNN 的 24 h 和 168 h total-MAE ECDF，作为 Main Fig. 3a 的 distributional reassurance。

---

## 4. 统一 publication visual system

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

## 5. Results 最终章节结构

### 4.1 STaR-GNN consistently improves day-ahead and week-ahead multi-DMA forecasting

证据：Table 1 + Main Fig. 1。

写法：claim → quantitative evidence → sequence/multiscale comparison → graph baseline comparison → bounded inference。

### 4.2 State normalization drives week-ahead accuracy while future-aware retrieval improves lead-time stability

证据：Table 2 + Main Fig. 2。

开头直接给 scientific finding，不以“Table 2 shows...”起句。

建议核心句：

> The two proposed modules contributed to week-ahead forecasting in distinct but complementary ways: SAS-Norm accounted for most of the reduction in absolute MAE, whereas FA-DPR primarily reduced the accumulation of error with increasing lead time.

### 4.3 Forecasting gains remain consistent across test origins and DMAs

证据：Main Fig. 3。

建议核心句：

> The average accuracy improvement was not driven by a small subset of favorable forecasting periods or DMAs.

### 4.4 STaR-GNN preserves week-ahead demand dynamics across the diurnal cycle

证据：Main Fig. 4。

代表案例定义为 population analysis 的 instance-level validation，而不是额外 leaderboard。

---

## 6. Results 写作规则

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

## 7. 当前冻结结果之外的可选补充实验

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
