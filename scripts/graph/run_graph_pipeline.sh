#!/usr/bin/env bash
# Build and independently validate the shared BWDF Pearson graph.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${1:-${PROJECT_ROOT}/configs/graph/pearson_static.yaml}"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "BWDF Pearson静态图构建"
echo "============================================"
echo "项目目录: ${PROJECT_ROOT}"
echo "图配置:   ${CONFIG}"
echo "Conda环境:${CONDA_DEFAULT_ENV:-未检测到}"
echo "Python:    $(command -v python)"

python - <<'PY'
import dma_wdf
import matplotlib
import numpy
import pandas
print("图构建依赖检查：PASS")
PY

echo ""
echo "[第1步/2] 构建共享Pearson静态图..."
python "${PROJECT_ROOT}/scripts/graph/build_static_graph.py" \
    --root "${PROJECT_ROOT}" \
    --config "${CONFIG}"

echo ""
echo "[第2步/2] 独立验证图文件..."
python "${PROJECT_ROOT}/scripts/graph/validate_static_graph.py" \
    --root "${PROJECT_ROOT}" \
    --config "${CONFIG}"

echo ""
echo "============================================"
echo "Pearson静态图构建与验证完成"
echo "图文件: ${PROJECT_ROOT}/artifacts/graphs/bwdf_pearson_static_graph.npz"
echo "报告:   ${PROJECT_ROOT}/results/graph/pearson_static/"
echo "============================================"
