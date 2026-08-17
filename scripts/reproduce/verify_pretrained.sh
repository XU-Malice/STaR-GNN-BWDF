#!/usr/bin/env bash
# ============================================================
# 冻结 checkpoint 验证入口（不训练）
# ============================================================
#
# 默认执行：
#   1. 校验冻结发布内每个文件的 SHA-256；
#   2. 校验 10 个 checkpoint 的任务、seed、模型和冻结超参数；
#   3. 校验每次评估都使用 common-46 且 Test 未参与训练/选参；
#   4. 校验注册的四指标与 31/32 消融关系；
#   5. 从冻结 predictions.npz 生成总体、DMA、逐日和 Pearson 图表。
#
# 如需重新执行全部 checkpoint 推理：
#   bash scripts/reproduce/verify_pretrained.sh \
#       --re-evaluate --device cuda:0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

python scripts/reproduce/check_environment.py
python scripts/reproduce/verify_paper_release.py "$@"
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout
python scripts/reproduce/build_detailed_test_artifacts.py

echo "============================================"
echo "冻结 checkpoint、common-46 Test 与论文图表：PASS"
echo "结果说明：paper/reports/TEST_RESULTS_CN.md"
echo "============================================"
