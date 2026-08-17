#!/bin/bash
# ============================================================
# 数据质量检查
# ============================================================
# 检查原始 wf4bwdf 数据或预处理后的数据。
#
# 用法:
#   ./scripts/data/inspect_data.sh raw                  # 检查原始数据
#   ./scripts/data/inspect_data.sh processed [DATA_DIR]  # 检查处理后数据

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-raw}"
DATA_DIR="${2:-${PROJECT_ROOT}/data/processed/data_build}"

cd "${PROJECT_ROOT}"

case "${MODE}" in
    raw)
        echo "检查 wf4bwdf 原始数据..."
        python -m dma_wdf.quality.inspect_raw \
            --output-dir "${PROJECT_ROOT}/results/data_quality"
        ;;
    processed)
        echo "检查处理后数据: ${DATA_DIR}"
        python -m dma_wdf.quality.inspect_processed \
            --data-dir "${DATA_DIR}" \
            --output-dir "${PROJECT_ROOT}/results/data_quality"
        ;;
    *)
        echo "用法: $0 {raw|processed} [data_dir]"
        exit 1
        ;;
esac
