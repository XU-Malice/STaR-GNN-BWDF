#!/usr/bin/env bash
# ============================================================
# MSCMNet 论文协议独立对比验证
# ============================================================
#
# 作用：
#   读取已经生成的预处理数据，不重新处理数据，
#   检查时间范围、特征、样本数、缺失值和指标实现
#   是否符合 MSCMNet / BWDF 论文协议。
#
# 用法：
#   bash scripts/data/compare_paper.sh
#
#   bash scripts/data/compare_paper.sh \
#       data/processed/data_build
#
#   bash scripts/data/compare_paper.sh \
#       data/processed/data_build \
#       configs/mscmnet/supplementary_tables

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_DIR="${1:-${PROJECT_ROOT}/data/processed/data_build}"
S1_DIR="${2:-${PROJECT_ROOT}/configs/mscmnet/supplementary_tables}"
REPORT_DIR="${PROJECT_ROOT}/results/paper_comparison"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "MSCMNet 论文协议对比验证"
echo "============================================"
echo "项目目录:   ${PROJECT_ROOT}"
echo "数据目录:   ${DATA_DIR}"
echo "S1表格目录: ${S1_DIR}"
echo "报告目录:   ${REPORT_DIR}"
echo "Python路径: $(command -v python)"
echo ""

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "错误：预处理数据目录不存在：" >&2
    echo "  ${DATA_DIR}" >&2
    echo "请先执行：" >&2
    echo "  bash scripts/data/run_pipeline.sh" >&2
    exit 1
fi

REQUIRED_FILES=(
    "demand_hourly.parquet"
    "weather_hourly.parquet"
    "temporal_hourly.parquet"
    "sample_index_single_step_24h.csv"
    "sample_index_multi_step_168h.csv"
)

for filename in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${DATA_DIR}/${filename}" ]]; then
        echo "错误：缺少必要文件：" >&2
        echo "  ${DATA_DIR}/${filename}" >&2
        exit 1
    fi
done

python - <<'PY'
import dma_wdf
import pandas
import pyarrow

print("论文协议验证依赖：PASS")
PY

mkdir -p "${REPORT_DIR}"

COMPARE_ARGS=(
    --data-dir "${DATA_DIR}"
    --output-dir "${REPORT_DIR}"
)

S1_FILE=""

if [[ -d "${S1_DIR}" ]]; then
    S1_FILE="$(
        find "${S1_DIR}" \
            -maxdepth 1 \
            -type f \
            -name '*.csv' \
            -print \
            -quit
    )"
fi

if [[ -n "${S1_FILE}" ]]; then
    echo "检测到S1补充表格，将运行协议验证和S1数值对比。"
    COMPARE_ARGS+=(--s1-dir "${S1_DIR}")
else
    echo "未检测到S1补充表格，仅运行论文协议和指标健全性验证。"
fi

echo ""
python -m dma_wdf.quality.compare_paper "${COMPARE_ARGS[@]}"

REPORT_FILE="${REPORT_DIR}/compare_paper_report.md"
JSON_FILE="${REPORT_DIR}/paper_comparison.json"
CHECKS_FILE="${REPORT_DIR}/paper_comparison_checks.csv"
CHARACTERISTICS_FILE="${REPORT_DIR}/data_characteristics.csv"

for path in \
    "${REPORT_FILE}" \
    "${JSON_FILE}" \
    "${CHECKS_FILE}" \
    "${CHARACTERISTICS_FILE}"
do
    if [[ ! -f "${path}" ]]; then
        echo "错误：论文协议验证没有生成预期文件：" >&2
        echo "  ${path}" >&2
        exit 1
    fi
done

echo ""
echo "============================================"
echo "论文协议对比验证完成"
echo "============================================"
echo "Markdown报告: ${REPORT_FILE}"
echo "JSON结果:     ${JSON_FILE}"
echo "检查明细:     ${CHECKS_FILE}"
echo "数据特征:     ${CHARACTERISTICS_FILE}"
echo ""

grep "Overall pass rate" "${REPORT_FILE}" || true