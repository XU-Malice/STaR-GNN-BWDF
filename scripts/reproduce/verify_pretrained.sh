#!/usr/bin/env bash
# ============================================================
# 冻结 checkpoint 验证入口（不训练）
# ============================================================
#
# 默认：
#   1. 检查环境、处理数据与训练期 Pearson 图；
#   2. 验证冻结文件 SHA、10 个 checkpoint、common-46 与 Test 隔离；
#   3. 可选重新执行全部 checkpoint 推理；
#   4. 重建 manuscript-facing Table 1--3；
#   5. 重建 legacy aggregate-demand 诊断工件；
#   6. 重建最终 Figure 1--5 与 moving-block bootstrap audit。
#
# 内部 aggregate-demand hierarchy 保留 31/32 作为冻结诊断；
# 当前 manuscript-facing factorial ablation 为 4 models / no STGCN / 30/32。
#
# 用法：
#   bash scripts/reproduce/verify_pretrained.sh \
#       --re-evaluate --device cuda:0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

FROZEN_ROOT="${PROJECT_ROOT}/results/paper/frozen_v1"
FROZEN_MANIFEST="${FROZEN_ROOT}/CHECKSUMS.sha256"
GRAPH_PATH="${PROJECT_ROOT}/artifacts/graphs/bwdf_pearson_static_graph.npz"

REQUIRED_DATA=(
  "${PROJECT_ROOT}/data/processed/data_build/demand_hourly.parquet"
  "${PROJECT_ROOT}/data/processed/data_build/weather_hourly.parquet"
  "${PROJECT_ROOT}/data/processed/data_build/temporal_hourly.parquet"
  "${PROJECT_ROOT}/data/processed/data_build/combined_hourly_features.parquet"
  "${PROJECT_ROOT}/data/processed/data_build/sample_index_multi_step_24h.csv"
  "${PROJECT_ROOT}/data/processed/data_build/sample_index_multi_step_168h.csv"
)

if [[ ! -f "${FROZEN_MANIFEST}" ]]; then
  echo "ERROR：未找到冻结发布清单：${FROZEN_MANIFEST}" >&2
  echo "请先从 GitHub Release 下载并在仓库根目录解压冻结工件。" >&2
  exit 1
fi

python scripts/reproduce/check_environment.py

MISSING_DATA=()
for path in "${REQUIRED_DATA[@]}"; do
  if [[ ! -f "${path}" ]]; then
    MISSING_DATA+=("${path}")
  fi
done

DATA_REBUILT=false
if [[ ${#MISSING_DATA[@]} -gt 0 ]]; then
  echo "============================================"
  echo "缺少论文处理数据，将按注册协议自动执行预处理"
  printf 'MISSING: %s\n' "${MISSING_DATA[@]}"
  echo "============================================"
  bash scripts/data/run_pipeline.sh
  DATA_REBUILT=true
else
  echo "论文处理数据已存在：REUSE"
fi

if [[ "${DATA_REBUILT}" == "true" || ! -f "${GRAPH_PATH}" ]]; then
  echo "============================================"
  echo "构建并验证训练期 Pearson 功能图"
  echo "============================================"
  bash scripts/graph/run_graph_pipeline.sh
else
  echo "Pearson 功能图已存在：REUSE"
fi

for path in "${REQUIRED_DATA[@]}" "${GRAPH_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR：自动准备后仍缺少必要工件：${path}" >&2
    exit 1
  fi
done

python scripts/reproduce/verify_paper_release.py "$@"

python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

python scripts/reproduce/build_detailed_test_artifacts.py

python scripts/reproduce/build_literature_figures.py \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --ablation-table paper/tables/literature/table_ablation_common46.csv \
  --dma-table paper/tables/literature/table_star_gnn_dma_common46.csv \
  --output paper/figures

python scripts/reproduce/build_manuscript_results_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --figure-output paper/figures \
  --table-output paper/tables/manuscript \
  --bootstrap-iterations 5000 \
  --bootstrap-seed 20260820

python scripts/reproduce/refine_manuscript_results_figures.py \
  --table-dir paper/tables/manuscript \
  --figure-dir paper/figures \
  --block-bootstrap-iterations 50000 \
  --block-bootstrap-length 7 \
  --block-bootstrap-seed 20260820

echo "============================================"
echo "冻结 checkpoint、common-46 Test 与 manuscript artifacts：PASS"
echo "Table 1：publisher-compatible 9-model overall comparison"
echo "Table 2：4-model factorial ablation / no STGCN / 30/32"
echo "Table 3：STaR-GNN DMA-level metrics"
echo "Figure 1--5：final manuscript design"
echo "168 h Full-vs-SAS：7-origin moving-block bootstrap guardrail"
echo "legacy aggregate-demand hierarchy：31/32 internal diagnostic only"
echo "结果说明：docs/RESULTS_AND_ARTIFACTS_CN.md"
echo "============================================"
