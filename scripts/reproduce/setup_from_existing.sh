#!/usr/bin/env bash
# ============================================================
# 作者本机一键发布前验收
# ============================================================
#
# 将旧 DMA-WDF 项目中已经冻结的大文件迁移到当前独立仓库，随后执行：
# 环境检查 -> 完整测试套件 -> checkpoint哈希验证 -> 论文表图生成。
# 可选 --re-evaluate 会再次推理唯一10个checkpoint。
#
# 用法：
#   bash scripts/reproduce/setup_from_existing.sh \
#       /path/to/DMA-WDF \
#       --re-evaluate --device cuda:0

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "用法：$0 OLD_DMA_WDF_ROOT [verify_pretrained参数...]" >&2
    exit 2
fi

SOURCE_ROOT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[1/6] 验证公开源码文件"
bash scripts/reproduce/verify_source.sh

echo "[2/6] 安装当前仓库（不下载或升级依赖）"
python -m pip install -e . --no-deps

echo "[3/6] 导入冻结checkpoint、处理数据和Pearson图"
python scripts/reproduce/import_local_artifacts.py "${SOURCE_ROOT}"

echo "[4/6] 运行源码、协议、防泄漏和模型单元测试"
bash scripts/reproduce/smoke_test.sh

echo "[5/6] 验证checkpoint并生成论文表图"
bash scripts/reproduce/verify_pretrained.sh "$@"

echo "[6/6] 审计公开仓库结构、唯一模型身份和旧HPO/SGDR隔离"
python scripts/reproduce/audit_public_repository.py \
  --require-frozen \
  --require-paper-artifacts

echo "============================================"
echo "独立仓库发布前验收：PASS"
echo "项目目录：${PROJECT_ROOT}"
echo "论文结果：${PROJECT_ROOT}/paper"
echo "============================================"
