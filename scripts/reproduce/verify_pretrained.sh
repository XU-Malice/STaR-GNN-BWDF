#!/usr/bin/env bash
# ============================================================
# 冻结 checkpoint 验证入口（不训练）
# ============================================================
#
# 默认执行：
#   1. 校验 Python 环境和冻结发布目录；
#   2. 缺少处理数据时，按论文协议自动运行数据预处理；
#   3. 缺少 Pearson 图或数据刚重建时，自动重建训练期功能图；
#   4. 校验冻结发布内每个文件的 SHA-256；
#   5. 校验 10 个 checkpoint 的任务、seed、模型和冻结超参数；
#   6. 校验 common-46、Test 隔离和冻结指标；
#   7. 生成 publisher-compatible 主比较/消融表及 STaR-GNN DMA 表；
#   8. 生成内部逐日/Pearson 等诊断工件；
#   9. 最后覆盖生成论文主图：9 模型总体、publisher 消融、STaR-GNN DMA 四指标。
#
# 内部 aggregate-demand 消融关系仍作为冻结工件审计；论文正文消融采用
# publisher-compatible 口径，并由 build_paper_tables.py 单独审计为 30/32。
#
# 如需重新执行全部 checkpoint 推理：
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
  echo "检测到全新服务器缺少论文处理数据，将自动执行预处理"
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

echo "============================================"
echo "冻结 checkpoint、common-46 Test 与论文图表：PASS"
echo "论文主比较：publisher-compatible 9 模型口径"
echo "论文消融：publisher-compatible 图模型/模块口径（30/32）"
echo "论文 DMA：STaR-GNN DMA-level MAE/MAPE/RMSE/NSE"
echo "内部逐日分析：aggregate-demand 口径"
echo "结果说明：docs/RESULTS_AND_ARTIFACTS_CN.md"
echo "============================================"
