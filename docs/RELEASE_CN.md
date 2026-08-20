# GitHub 发布与独立复现指南

本文件面向仓库维护者。普通使用者的环境、训练和冻结 checkpoint 验证见 [`FULL_PIPELINE_CN.md`](FULL_PIPELINE_CN.md)；结果口径见 [`RESULTS_AND_ARTIFACTS_CN.md`](RESULTS_AND_ARTIFACTS_CN.md)。

## 1. 发布边界

Git 仓库包含：

- `configs/`、`src/`、`scripts/`、`tests/`；
- README、`docs/`、环境/许可/引用信息；
- `paper/` 中可审计的小型表格、caption、PNG/PDF；
- `SOURCE_CHECKSUMS.sha256`。

大型冻结 checkpoint 与预测作为 GitHub Release asset 发布，不写入普通 Git 历史。原始/处理 BWDF 数据遵循上游许可和可重建原则。

## 2. 当前正式论文口径

- Table 1：9-model overall comparison；
- Table 2：**4-model factorial ablation** = DCRNN / DCRNN + SAS-Norm / DCRNN + FA-DPR / STaR-GNN；
- STGCN 是独立 graph baseline，不进入 Table 2 / Figure 2；
- manuscript-facing MAE = sum of DMA A--J MAEs；
- internal aggregate-demand MAE 单独保留；
- manuscript factorial cell audit = 30/32；
- legacy aggregate-demand hierarchy = 31/32，仅内部诊断。

## 3. 源码 SHA 的正确使用

修改任何已登记源码、配置、测试或文档后，旧 `SOURCE_CHECKSUMS.sha256` 必然失效。发布维护者应在**确认当前分支内容就是准备发布的版本后**重新封存：

```bash
python scripts/reproduce/regenerate_source_checksums.py
bash scripts/reproduce/verify_source.sh
```

`regenerate_source_checksums.py` 只用于发布准备；发布后的用户正常只执行 `verify_source.sh`，不应先重新生成清单。

## 4. 推荐的一键最终收口

冻结工件已在当前仓库时：

```bash
bash scripts/reproduce/finalize_public_release.sh \
  --device cuda:0
```

该入口不重新训练，依次完成：

1. 清理弃用入口；
2. DCRNN/Base 唯一化；
3. 重新封存并验证 source SHA；
4. 校验冻结 checkpoint 与内部 aggregate 诊断；
5. 重新执行 10 组 common-46 推理；
6. 重建 Table 1--3、Figure 1--5 和 manuscript audits；
7. 审计公开仓库结构、四模型消融、图表与大文件边界；
8. 生成 Release asset（除非使用 `--skip-package`）。

成功报告应明确包含：

```text
Table 2 factorial ablation: 4 models / no STGCN / 30/32 PASS
Figure 1--5: PASS
Full-vs-SAS 168 h moving-block bootstrap audit: PASS
```

## 5. 只做冻结验证与重建表图

```bash
bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

该入口现在同样会重建最终 Figure 1--5，而不是只生成旧 absolute reference figures。

## 6. 修改后最低限度本地检查

```bash
python -m py_compile \
  scripts/reproduce/build_paper_tables.py \
  scripts/reproduce/build_literature_figures.py \
  scripts/reproduce/build_manuscript_results_figures.py \
  scripts/reproduce/refine_manuscript_results_figures.py \
  scripts/reproduce/audit_public_repository.py \
  scripts/reproduce/regenerate_source_checksums.py

python -m pytest \
  tests/test_paper_release.py \
  tests/test_paper_artifacts.py \
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

不要使用 `git add .` 或 `git add -A`。只暂存本次确认的源文件、最终表图和重新生成的 SOURCE_CHECKSUMS。

最终 Figure 2 应满足：

- 不含 STGCN；
- Panel (a) 为 SAS-Norm / FA-DPR / STaR-GNN 相对 DCRNN 的 day-wise MAE reduction；
- Panel (b) 为四个 factorial variants 相对 Day 1 的 MAE change；
- 168 h Full−SAS block-bootstrap 95% CI 跨 0。

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

最终发布原则：**冻结预测不为排序而修改；两套 MAE 明确分离；四模型消融与外部 baseline 分开；表格统一精度；图件回答科学问题而不是放大微小差异。**
