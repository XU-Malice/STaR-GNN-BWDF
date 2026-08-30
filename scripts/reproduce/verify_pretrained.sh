#!/usr/bin/env bash
# ============================================================
# 冻结 checkpoint 验证入口（不训练）
# ============================================================
#
# 默认：
#   1. 检查环境、处理数据与训练期 Pearson 图；
#   2. 验证冻结文件 SHA、10 个 checkpoint、common-46 与 Test 隔离；
#   3. 可选重新执行全部 checkpoint 推理；
#   4. 重建全精度论文源表和 legacy aggregate-demand 诊断；
#   5. 生成投稿显示 Table 1--2 / Tables S1--S3；
#   6. 通过唯一 canonical renderer 生成 Main Fig. 1--7。
#
# 内部 aggregate-demand hierarchy 保留 31/32 作为冻结诊断；
# manuscript-facing factorial ablation 为 4 models / no STGCN / 30/32。
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

# Full-precision source tables.
python scripts/reproduce/build_paper_tables.py \
  --input results/paper/frozen_v1 \
  --output paper/tables/literature \
  --frozen-layout

# Legacy aggregate-demand diagnostics retained for backward-compatible audit.
python scripts/reproduce/build_detailed_test_artifacts.py

# Submission-display tables: 2 main + 3 supplementary.
python scripts/reproduce/render_submission_tables.py \
  --source-dir paper/tables/literature \
  --output-dir paper/tables/submission \
  --release results/paper/frozen_v1

# Canonical submission figures: no Stage-1/Stage-2 overwrite chain.
python scripts/reproduce/render_submission_figures.py \
  --release results/paper/frozen_v1 \
  --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
  --dma-table paper/tables/literature/table_all_models_dma.csv \
  --main-output paper/figures/submission \
  --audit-output paper/tables/manuscript/submission \
  --block-length 7 \
  --bootstrap-iterations 50000 \
  --bootstrap-seed 20260821

echo "============================================"
echo "冻结 checkpoint、common-46 Test 与 submission artifacts：PASS"
echo "Main Table 1：publisher-compatible overall comparison"
echo "Main Table 2：4-model factorial ablation / no STGCN / 30/32"
echo "Supplementary Tables S1--S3：DMA details + origin robustness"
echo "Main Fig. 1--7：overall / DMA / ablation / robustness / week dynamics"
echo "legacy aggregate-demand hierarchy：31/32 internal diagnostic only"
echo "实验设计：docs/EXPERIMENT_DESIGN_FINAL_CN.md"
echo "============================================"
