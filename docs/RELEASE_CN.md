# GitHub 发布与独立复现指南

本文件面向仓库维护者。普通使用者的环境、训练和冻结 checkpoint 验证见 [`FULL_PIPELINE_CN.md`](FULL_PIPELINE_CN.md)；最终实验设计见 [`EXPERIMENT_DESIGN_FINAL_CN.md`](EXPERIMENT_DESIGN_FINAL_CN.md)；结果口径见 [`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md)。

## 1. 发布边界

Git 仓库包含：

- `configs/`、`src/`、`scripts/`、`tests/`；
- README、`docs/`、环境/许可/引用信息；
- `paper/` 中可审计的小型表格、caption、submission PNG/PDF/SVG 和 legacy diagnostics；
- `SOURCE_CHECKSUMS.sha256`。

大型冻结 checkpoint 与预测作为 GitHub Release asset 发布，不写入普通 Git 历史。原始/处理 BWDF 数据遵循上游许可和可重建原则。

## 2. 当前正式 manuscript contract

- **Main Table 1**：9-model overall comparison；
- **Main Table 2**：4-model factorial ablation = DCRNN / DCRNN + SAS-Norm / DCRNN + FA-DPR / STaR-GNN；
- **Supplementary Table S1**：DMA A--J detailed metrics；
- **Supplementary Table S2**：DMA-local strongest competitors and signed margins；
- **Supplementary Table S3**：forecast-origin robustness and high-variability stratum；
- **Main Fig. 1**：overall four-metric improvement；
- **Main Figs. 2–4**：DMA breadth and horizon-specific local absolute performance；
- **Main Fig. 5**：factorial ablation + lead-time stability；
- **Main Fig. 6**：forecast-origin + difficult-window robustness；
- **Main Fig. 7**：population-to-instance week-ahead dynamics；
- **Supplementary Figs. S1–S3**：data cleaning examples / weekly demand patterns / common-46 origin MAE distributions。

STGCN 是独立 graph baseline，不进入 factorial ablation。Manuscript MAE = DMA A--J MAE 之和；internal aggregate-demand MAE 单独保留。Manuscript factorial cell audit = 30/32；legacy aggregate-demand hierarchy = 31/32，仅内部诊断。

投稿版权威目录：

```text
paper/tables/submission/
paper/figures/submission/
paper/tables/manuscript/submission/
```

旧 `paper/figures/manuscript_fig1...5` 和 `test_*` 继续保留，但不是投稿版权威图件。

## 3. 源码 SHA 的正确使用

修改任何已登记源码、配置、测试或文档后，旧 `SOURCE_CHECKSUMS.sha256` 必然失效。确认当前分支就是准备发布的版本后重新封存：

```bash
python scripts/reproduce/regenerate_source_checksums.py
bash scripts/reproduce/verify_source.sh
```

发布后的普通用户只执行 `verify_source.sh`，不应先重生成清单。

## 4. 推荐的一键最终收口

冻结工件已在当前仓库时：

```bash
bash scripts/reproduce/finalize_public_release.sh \
  --device cuda:0
```

该入口不重新训练，依次完成：

1. 清理弃用入口；
2. DCRNN/Base 唯一化；
3. 重生成并验证 source SHA；
4. 校验 10 组冻结 checkpoint、协议和 internal aggregate diagnostics；
5. 重新执行 10 组 common-46 推理；
6. 重建 full-precision source tables、submission display tables、Main Fig. 1--7、Supplementary Figs. S1--S3 和独立审计 CSV/JSON；
7. 审计公开仓库结构、四模型消融、submission artifacts 与大文件边界；
8. 生成 Release asset（除非使用 `--skip-package`）。

成功报告应包含：

```text
Main Table 1 overall: PASS
Main Table 2 factorial ablation: 4 models / no STGCN / 30/32 PASS
Supplementary Table S1 DMA metrics: PASS
Supplementary Tables S2--S3: PASS
Main Fig. 1--7: PASS
Supplementary Figs. S1--S3: PASS
7-origin moving-block bootstrap: PASS
```

## 5. 只做冻结验证与重建投稿表图

```bash
bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

该入口使用同一个 canonical submission renderer，不再执行旧 Stage-1/Stage-2 同名覆盖流程。

只想重建表图时：

```bash
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

python scripts/reproduce/render_submission_tables.py

python scripts/reproduce/render_submission_figures.py \
  --release results/paper/frozen_v1 \
  --block-length 7 \
  --bootstrap-iterations 50000 \
  --bootstrap-seed 20260821

PYTHONPATH=scripts/reproduce python \
  scripts/reproduce/render_supplementary_figures.py
```

若不使用环境中固定版本的 `wf4bwdf`，可通过 `--wf4bwdf-repo repos/wf4bwdf` 显式指定与 `data/README.md` 一致的本地 checkout。

## 6. 修改后最低限度本地检查

```bash
python -m py_compile \
  scripts/reproduce/build_paper_tables.py \
  scripts/reproduce/manuscript_plot_style.py \
  scripts/reproduce/render_submission_tables.py \
  scripts/reproduce/render_submission_figures.py \
  scripts/reproduce/render_supplementary_figures.py \
  scripts/reproduce/audit_public_repository.py \
  scripts/reproduce/regenerate_source_checksums.py

python -m pytest \
  tests/test_paper_release.py \
  tests/test_paper_artifacts.py \
  tests/test_submission_artifacts.py \
  -q
```

随后：

```bash
python scripts/reproduce/regenerate_source_checksums.py
bash scripts/reproduce/verify_source.sh

python scripts/reproduce/audit_public_repository.py \
  --require-frozen \
  --require-paper-artifacts
```

## 7. 上传 GitHub 前

先查看：

```bash
git status --short
git diff --stat
git diff -- SOURCE_CHECKSUMS.sha256
```

不要使用 `git add .` 或 `git add -A`。只暂存本次确认的源文件、submission 表图、审计工件和重新生成的 SOURCE_CHECKSUMS。

最终图件必须满足：

- Main Figs. 1–4：总体四指标、跨 DMA 覆盖和两个预测时域的局部绝对性能；
- Main Fig. 5：四模型 factorial ablation 与逐日提前期稳定性，无 STGCN；
- Main Fig. 6：46 个共同起点的 paired effects 与高波动窗口；
- Main Fig. 7：population diurnal profile + deterministic representative trajectory + local error；
- Supplementary Figs. S1–S3：分别承担数据清洗、周期需求结构和起点 MAE 分布，不重复主图；
- STaR-GNN 全文固定 deep-blue hero visual role；
- PDF/SVG 文字可编辑，PNG 300 dpi 仅作预览。

## 8. GitHub Release

若只需要重新打包冻结工件：

```bash
python scripts/reproduce/package_frozen_release.py
```

输出：

```text
dist/STaR-GNN-BWDF-frozen-v1.tar.gz
dist/STaR-GNN-BWDF-frozen-v1.tar.gz.sha256
```

Release 页面至少记录源码 commit/tag、资产 SHA-256、解压位置、验证命令、环境和 common-46 协议。

## 9. 发布后复验

从 GitHub 新克隆目录重新执行：

```bash
bash scripts/reproduce/verify_source.sh
bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

需要最高强度复现时，再在独立路径执行 clean-room，从原始数据重新预处理、构图、训练和 Test。不要通过修改 Test 样本、指标定义或 Test 后重选参数来消除差异。

最终发布原则：**冻结预测不为排序而修改；两套 MAE 明确分离；四模型消融与外部 baseline 分开；每张主图回答一个独立 Results-level question；表格负责精确值，主图负责不可替代的推理证据。**
