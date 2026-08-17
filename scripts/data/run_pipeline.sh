#!/usr/bin/env bash
# ============================================================
# DMA-WDF：完整数据预处理与质量验证
# ============================================================
#
# 执行顺序：
#   1. 构建论文协议数据集
#   2. 验证预处理产物及防泄漏约束
#   3. 运行通用数据质量检查
#   4. 运行 MSCMNet 论文协议对比
#
# 用法：
#   bash scripts/data/run_pipeline.sh
#
#   bash scripts/data/run_pipeline.sh \
#       --output-dir /tmp/dma_wdf_data_build
#
#   bash scripts/data/run_pipeline.sh \
#       --wf4bwdf-repo repos/wf4bwdf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="${PROJECT_ROOT}/configs/data/paper_split.yaml"
PREPROCESSING_CONFIG="${PROJECT_ROOT}/configs/data/preprocessing.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed/data_build"
RESULT_DIR="${PROJECT_ROOT}/results"
WF4BWDF_REPO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --preprocessing-config)
            PREPROCESSING_CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --wf4bwdf-repo)
            WF4BWDF_REPO="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法："
            echo "  bash scripts/data/run_pipeline.sh [选项]"
            echo ""
            echo "选项："
            echo "  --config PATH               论文划分配置"
            echo "  --preprocessing-config PATH 预处理配置"
            echo "  --output-dir PATH           数据输出目录"
            echo "  --wf4bwdf-repo PATH         wf4bwdf本地仓库"
            exit 0
            ;;
        *)
            echo "错误：未知参数 $1" >&2
            exit 2
            ;;
    esac
done

echo "============================================"
echo "DMA-WDF 数据预处理管道"
echo "============================================"
echo "项目目录:   ${PROJECT_ROOT}"
echo "论文配置:   ${CONFIG}"
echo "预处理配置: ${PREPROCESSING_CONFIG}"
echo "输出目录:   ${OUTPUT_DIR}"
echo "Conda环境:  ${CONDA_DEFAULT_ENV:-未检测到}"
echo "Python路径: $(command -v python)"
echo ""

if [[ ! -f "${CONFIG}" ]]; then
    echo "错误：论文配置不存在：${CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${PREPROCESSING_CONFIG}" ]]; then
    echo "错误：预处理配置不存在：${PREPROCESSING_CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/scripts/data/validate_preprocessing.py" ]]; then
    echo "错误：缺少预处理验证脚本：" >&2
    echo "  ${PROJECT_ROOT}/scripts/data/validate_preprocessing.py" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/scripts/data/compare_paper.sh" ]]; then
    echo "错误：缺少论文协议验证脚本：" >&2
    echo "  ${PROJECT_ROOT}/scripts/data/compare_paper.sh" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

python - <<'PY'
import sys
import dma_wdf
import wf4bwdf

print("Python版本：", sys.version.split()[0])
print("dma_wdf：已加载")
print("wf4bwdf：已加载")
PY

PIPELINE_ARGS=(
    --root "${PROJECT_ROOT}"
    --config "${CONFIG}"
    --preprocessing-config "${PREPROCESSING_CONFIG}"
    --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${WF4BWDF_REPO}" ]]; then
    PIPELINE_ARGS+=(--wf4bwdf-repo "${WF4BWDF_REPO}")
fi

echo ""
echo "[第1步/4] 构建论文协议数据集..."
python -m dma_wdf.data.pipeline "${PIPELINE_ARGS[@]}"

echo ""
echo "[第2步/4] 验证预处理产物与防泄漏约束..."
python "${PROJECT_ROOT}/scripts/data/validate_preprocessing.py" \
    --root "${PROJECT_ROOT}" \
    --data-dir "${OUTPUT_DIR}"

echo ""
echo "[第3步/4] 运行数据质量检查..."
python -m dma_wdf.quality.inspect_processed \
    --data-dir "${OUTPUT_DIR}" \
    --output-dir "${RESULT_DIR}/data_quality"

echo ""
echo "[第4步/4] 运行论文协议对比..."
bash "${PROJECT_ROOT}/scripts/data/compare_paper.sh" \
    "${OUTPUT_DIR}" \
    "${PROJECT_ROOT}/configs/mscmnet/supplementary_tables"

echo ""
echo "============================================"
echo "数据预处理管道运行完毕"
echo "============================================"
echo "处理数据:   ${OUTPUT_DIR}"
echo "质量报告:   ${RESULT_DIR}/data_quality/"
echo "论文报告:   ${RESULT_DIR}/paper_comparison/compare_paper_report.md"
echo "============================================"
