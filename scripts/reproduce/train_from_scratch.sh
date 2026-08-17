#!/usr/bin/env bash
# ============================================================
# 原始 BWDF 数据 -> 论文表格：从零复现入口
# ============================================================
#
# 默认训练 seed=0 的 10 个任务：
#   STGCN × {24h,168h}
#   DCRNN(Base)/State/FA-DPR/Full × {24h,168h}
#
# 数据预处理和训练期 Pearson 图会先执行；所有训练完成后才创建
# TRAINING_COMPLETE 标记并读取 Test target，从流程上阻止 Test 参与选参。
# 输出目录非空时脚本不会静默覆盖。
#
# 用法：
#   bash scripts/reproduce/train_from_scratch.sh --device auto
#
# 多随机种子：
#   bash scripts/reproduce/train_from_scratch.sh \
#       --device auto --seeds 0,1,2,3,4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

python scripts/reproduce/check_environment.py
python scripts/reproduce/reproduce.py "$@"

echo "============================================"
echo "从零复现：PASS"
echo "默认输出：results/paper/reproduction"
echo "============================================"
