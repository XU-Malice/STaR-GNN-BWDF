# GitHub 发布与独立复现指南

本文档只面向仓库维护者，说明发布前迁移、clean-room 验收、GitHub 上传和发布后
复验。普通使用者的安装、训练和 checkpoint 验证见
[`FULL_PIPELINE_CN.md`](FULL_PIPELINE_CN.md)。

本仓库的发布边界参考三类官方要求：NeurIPS 的代码与数据指南要求实验贡献所依赖的
代码尽量自包含、可执行，并包含训练、评估和依赖说明；ACM Artifact Review 强调
工件可获取、可运行和结果可验证；GitHub 普通 Git 历史会阻止超过100 MiB的文件，
因此源码/配置/小型图表进入 Git，checkpoint 与冻结预测进入同仓库的 Release asset。

- NeurIPS Code and Data Submission Guidelines：
  <https://neurips.cc/public/guides/CodeSubmissionPolicy>
- ACM Artifact Review and Badging：
  <https://www.acm.org/publications/policies/artifact-review-and-badging-current>
- GitHub large files：
  <https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github>
- GitHub Releases：
  <https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases>

## 1. 发布内容

Git 仓库提交以下小型、可审计内容：

- `configs/`、`src/`、`scripts/`、`tests/`；
- `README.md`、`README_EN.md`、`docs/`、`data/README.md`；
- `paper/` 中的 CSV、Markdown、PNG 和 PDF；
- 环境文件、许可、引用信息和源码 SHA 清单。

大文件不直接写入 Git 历史，而作为 GitHub Release asset 发布：

- `STaR-GNN-BWDF-frozen-v1.tar.gz`；
- 对应的 `.sha256` 文件；
- 与源码一致的版本标签。

Release 包应包含冻结 checkpoint、预测、Test 摘要、指标和 manifest。原始 BWDF
数据遵循其上游许可，不随本仓库重新分发。

不得上传：Conda 环境、缓存、服务器日志、PID 文件、失败临时目录、旧 HPO、旧
SGDR、候选搜索结果、个人路径或本地补丁压缩包。

## 2. 作者本机导入与全面验收

如果冻结工件仍在旧项目 `/path/to/DMA-WDF`，在新仓库根目录执行：

```bash
bash scripts/reproduce/validate_everything.sh \
  /path/to/DMA-WDF \
  --device cuda:0
```

该命令会原子导入工件，验证源码、测试套件、10 个物理 checkpoint、10 份预测、
common-46、40 项复推理指标以及总体/消融/DMA/逐日/Pearson 表图。中断留下的
半成品不会被静默复用。

检查最近一次报告：

```bash
DIR="$(cat results/release_validation/latest_run_dir.txt)"
cat "${DIR}/STATUS"
cat "${DIR}/CURRENT"
cat "${DIR}/FINAL_REPORT.txt"
cat "${DIR}/checkpoint_reevaluation/reevaluation_summary.json"
```

只有 `STATUS=SUCCESS`、`CURRENT=DONE` 且报告逐项为 PASS，才进入 clean-room。

若冻结工件已经在当前仓库，推荐直接执行不重训的最终收口：

```bash
bash scripts/reproduce/finalize_public_release.sh --device cuda:0
```

它会删除重复的 `baselines/dcrnn`，只保留 `star_gnn/Base` 作为论文 DCRNN，完成
10组 checkpoint 复推理、40项指标审计、表图重建、仓库边界审计，并自动生成第5节
所需的两个 Release 文件。

## 3. 上传前 clean-room

clean-room 会复制 `SOURCE_CHECKSUMS.sha256` 登记的纯净源码，创建全新 Conda
prefix，从原始数据重新预处理、构图、训练、Test 和制图。冻结 checkpoint 只用于
身份验证和结果对照，不作为新训练的初始化权重。

```bash
cd /home/dengxu/projects/STaR-GNN-BWDF

STAMP="$(date +%Y%m%d-%H%M%S)"
WORKSPACE="/home/dengxu/projects/STaR-GNN-BWDF-cleanroom-${STAMP}"
LOG="cleanroom_validation_${STAMP}.log"

nohup env \
  CUDA_VISIBLE_DEVICES=6 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  bash scripts/reproduce/validate_clean_room.sh \
    --workspace "${WORKSPACE}" \
    --frozen-release results/paper/frozen_v1 \
    --device cuda:0 \
    --evaluation-device cuda:0 \
  > "${LOG}" 2>&1 &

printf '%s\n' "$!" > cleanroom_validation.pid
printf '%s\n' "${LOG}" > cleanroom_validation_latest_log.txt
```

单张 RTX 4090 串行完整训练通常需要约 10--15 小时，并建议预留至少 25 GB
磁盘。监控方式：

```bash
DIR="$(cat results/cleanroom_validation/latest_run_dir.txt)"
LOG="$(cat cleanroom_validation_latest_log.txt)"

cat "${DIR}/STATUS"
cat "${DIR}/CURRENT"
tail -n 100 "${LOG}"
```

成功报告必须包括：全新环境、数据预处理、训练期 Pearson 图、完整测试套件、从零
训练、common-46 Test、40 项指标对照、论文层级、冻结 checkpoint 新数据复推理和
源码未被修改。

## 4. Git 初始化与上传前检查

```bash
bash scripts/reproduce/verify_source.sh
bash scripts/reproduce/smoke_test.sh --source-only
python scripts/reproduce/audit_public_repository.py \
  --require-frozen --require-paper-artifacts
git status --short
```

确认 `.gitignore` 排除了数据、环境、缓存、日志和大文件。然后人工检查：

```bash
find paper/tables -type f | sort
find paper/figures -type f | sort
sed -n '1,260p' paper/reports/TEST_RESULTS_CN.md
```

此仓库的脚本不会自动执行 `git push`。提交说明应指出源码版本、论文参数、Release
资产名和 SHA-256。

## 5. GitHub Release

若尚未打包，执行：

```bash
python scripts/reproduce/package_frozen_release.py
```

该脚本先完整复核冻结 `CHECKSUMS.sha256` 和唯一10组模型，再生成确定性压缩包：

```text
dist/STaR-GNN-BWDF-frozen-v1.tar.gz
dist/STaR-GNN-BWDF-frozen-v1.tar.gz.sha256
```

Release 页面至少写明：

- 与源码 commit 对应的 tag；
- 资产文件名、大小和 SHA-256；
- 解压位置 `results/paper/frozen_v1/`；
- checkpoint 验证命令；
- 记录环境和 common-46 协议；
- 原始数据获取链接。

用户下载后应能执行：

```bash
sha256sum -c STaR-GNN-BWDF-frozen-v1.tar.gz.sha256
tar --no-same-owner -xzf STaR-GNN-BWDF-frozen-v1.tar.gz

bash scripts/reproduce/verify_pretrained.sh \
  --re-evaluate \
  --device cuda:0
```

## 6. GitHub 实际下载后的第二轮验收

上传后不要在作者工作目录自证。必须新建目录，从 GitHub 实际克隆源码并下载
Release 资产：

```bash
cd /home/dengxu/projects
git clone <公开仓库URL> STaR-GNN-BWDF-github-check
cd STaR-GNN-BWDF-github-check

sha256sum -c STaR-GNN-BWDF-frozen-v1.tar.gz.sha256
```

随后先运行冻结验证，再在另一个全新路径执行第 3 节的 clean-room。第二轮不得引用
第一次 clean-room 的目录，也不得复用作者旧项目的处理数据、图或训练结果。

## 7. 失败处理

脚本会保留失败目录。优先查看：

- `results/release_validation/<时间戳>/CURRENT`；
- `results/cleanroom_validation/<时间戳>/CURRENT`；
- `FINAL_REPORT.txt`（成功后生成）；
- `from_scratch_audit/from_scratch_metric_differences.csv`；
- `from_scratch_audit/from_scratch_summary.json`；
- 后台日志和失败 clean-room 工作目录。

不要通过修改 Test 样本、指标定义或 Test 后重选参数来消除失败。应先判断差异来自
依赖版本、数据身份、图身份、checkpoint 身份、随机性还是 GPU 浮点归约，再修复并
启动一轮全新的 clean-room。

## 8. 最终发布判定

同时满足以下条件才建议公开仓库地址：

- 作者本机 `validate_everything.sh` 全部通过；
- 上传前 clean-room 全部通过；
- GitHub 克隆和 Release 下载后的冻结验证通过；
- GitHub 下载后的第二次 clean-room 通过；
- README、完整教程和实际脚本参数一致；
- Git 仓库与 Release 中没有未授权数据或开发临时文件。
