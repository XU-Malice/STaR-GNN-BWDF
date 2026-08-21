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

- GRU†、LSTM†、MSNet†、MSCMNet-WM†、MSCMNet-M†、MSCMNet-W†；
- DCRNN、STGCN、STaR-GNN。

其中 † 表示 Que et al. (2024) 已发表结果；DCRNN、STGCN、STaR-GNN 为 common-46 协议下重新评价。正文显示统一 3 位小数，审计 CSV 保留完整精度。

### Table 2 — Factorial ablation

回答：**SAS-Norm 和 FA-DPR 分别贡献什么？**

只包含：

1. DCRNN；
2. DCRNN + SAS-Norm；
3. DCRNN + FA-DPR；
4. STaR-GNN。

STGCN 是独立 graph baseline，不属于消融。

168 h publisher-compatible MAE 中 SAS-Norm-only 为 12.208，STaR-GNN 为 12.234；完整模型在 MAPE、RMSE 和 NSE 上最好。该 0.21% MAE 点估计差异不用于声称任一模型稳定占优，长时域稳定性由 Main Fig. 1 和 moving-block bootstrap 进一步限定。

### Supplementary Table S1 — DMA-level metrics

原正文 DMA A--J 全指标表移至 Supplementary。正文用 Main Fig. 2b 回答空间一致性，避免重复。

---

## 2. 正文只保留三张核心结果图

## Main Figure 1 — Ablation mechanism and lead-time stability

**Results-level question：**为什么预测时域从 1 d 延伸到 7 d 后，STaR-GNN 仍能保持稳定？

### Panel a — Absolute day-wise publisher-compatible MAE

四个 factorial variants 的 Day 1--Day 7 绝对 MAE，并给出 ordered 7-origin moving-block bootstrap 95% CI。

推理角色：**primary quantitative evidence**。

### Panel b — Lead-time degradation

对每个模型计算：

\[
100\times\frac{MAE_d-MAE_{Day1}}{MAE_{Day1}}.
\]

只直接标注 Day 7 端点：

- DCRNN：约 +38.25%；
- DCRNN + FA-DPR：约 +11.93%；
- DCRNN + SAS-Norm：约 +2.64%；
- STaR-GNN：约 +1.70%。

推理角色：**mechanism-discriminating lead-time stress test**。

### Main inference

SAS-Norm 是周尺度绝对 MAE 改善的主要来源；FA-DPR 更明显地抑制随 lead time 增加产生的误差累积。完整模型将两者结合后获得最稳定的 Day-1-to-Day-7 误差演化，并在 168 h 的 MAPE/RMSE/NSE 上取得最好结果。

---

## Main Figure 2 — Temporal and spatial robustness

**Results-level question：**总体优势是否只来自少数有利日期或少数 DMA？

### Panel a — Paired origin-level MAE improvement

对相同 common test origin 直接计算：

\[
\Delta MAE_s = MAE_{baseline,s}-MAE_{STaR,s}.
\]

分别展示：

- 24 h vs DCRNN；
- 24 h vs STGCN；
- 168 h vs DCRNN；
- 168 h vs STGCN。

每组显示 46 个 paired differences、moving-block bootstrap mean 95% CI，并直接标注 win count：45/46、45/46、46/46、40/46。

推理角色：**temporal robustness / paired validation**。

相比 ECDF，paired difference 更直接利用同一 forecast origin 的配对结构；ECDF 降级为 Supplementary Fig. S2。

### Panel b — DMA-level MAE reduction heatmap

10 DMA × 2 horizons × 2 graph baselines，共 40 个比较。所有单元格均为正改善，范围约 1.26%--61.20%。

采用单向 sequential blue heatmap，而不是正负 diverging colormap；背景越深表示 MAE reduction 越大。深色单元格使用白字，浅色单元格使用黑字。

推理角色：**spatial stratification**。

### Main inference

STaR-GNN 的平均提升不是由少数容易预测的日期或高需求 DMA 驱动；改善方向在时间与空间上均保持一致，但不同 DMA 的收益幅度具有明显异质性。

---

## Main Figure 3 — Week-ahead demand dynamics

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

### Figure S1 — Relative improvement over all baselines

原 manuscript relative-improvement heatmap 降级为 Supplementary：Table 1 已经给出绝对性能，S1 用于概括相对改善范围。

视觉上使用 sequential blue，不使用当前全部为正值却以 0 为中心的 RdBu diverging colormap。Published reference models 与 re-evaluated graph baselines 用分隔线明确区分。

### Figure S2 — Per-origin ECDF

保留 DCRNN / STGCN / STaR-GNN 的 24 h 和 168 h ECDF，作为 Main Fig. 2a paired-difference analysis 的 distributional reassurance。

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

证据：Table 1。

写法：claim → quantitative evidence → sequence/multiscale comparison → graph baseline comparison → bounded inference。

### 4.2 State normalization drives week-ahead accuracy while future-aware retrieval improves lead-time stability

证据：Table 2 + Main Fig. 1。

开头直接给 scientific finding，不以“Table 2 shows...”起句。

建议核心句：

> The two proposed modules contributed to week-ahead forecasting in distinct but complementary ways: SAS-Norm accounted for most of the reduction in absolute MAE, whereas FA-DPR primarily reduced the accumulation of error with increasing lead time.

### 4.3 Forecasting gains remain consistent across test origins and DMAs

证据：Main Fig. 2。

建议核心句：

> The average accuracy improvement was not driven by a small subset of favorable forecasting periods or DMAs.

### 4.4 STaR-GNN preserves week-ahead demand dynamics across the diurnal cycle

证据：Main Fig. 3。

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
- SAS-Norm-only 的 168 h publisher-compatible MAE 点估计略低于 Full；
- Full-vs-SAS 的差异由 ordered moving-block analysis 限定，不将 0.21% 放大为稳定优劣关系。

---

## 7. 当前冻结结果之外建议补充的高标准实验

这些实验尚未纳入当前主结果，在没有真实运行结果前不得写入论文结论。

1. **Multi-seed replication**：DCRNN、STGCN、SAS-Norm、FA-DPR、STaR-GNN 至少 3 seeds，最好 0--4；Supplementary 报 mean ± SD。
2. **Efficiency analysis**：parameter count、inference time、GPU memory，至少比较 DCRNN / STGCN / STaR-GNN。
3. **FA-DPR alignment diagnostic**：在 evaluation 中错位/打乱 future calendar alignment，检验 168 h 性能是否按预期下降。该 targeted misalignment 比普通 remove-module ablation 更能支持 FA-DPR 的机制解释。

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
