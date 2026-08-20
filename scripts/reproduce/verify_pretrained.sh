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
#   6. 校验每次评估都使用 common-46 且 Test 未参与训练/选参；
#   7. 校验注册的四指标与 31/32 消融关系；
#   8. 生成统一 aggregate-demand 表图、DMA/逐日/Pearson 图；
#   9. 生成与 MSCMNet 补充材料同口径的 9 模型总体比较图。
#
# 如需重新执行全部 checkpoint 推理：
#   bash scripts/reproduce/verify_pretrained.sh \
#       --re-evaluate --device cuda:0
#
# 说明：
#   GitHub Release 不重新分发上游 BWDF 原始/处理数据。因此，全新服务器
#   首次运行时需要能够访问 environment.yml 中固定版本的 wf4bwdf 上游源。
#   已存在并通过路径检查的数据和图会直接复用，不会重新训练任何模型。

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
  --table paper/tables/literature/table_literature_comparison_common46.csv \
  --output paper/figures

echo "============================================"
echo "冻结 checkpoint、common-46 Test 与论文图表：PASS"
echo "总体比较图：publisher-compatible 9 模型口径"
echo "消融/逐日图：aggregate-demand 口径"
echo "DMA 图：DMA-level 口径"
echo "结果说明：paper/reports/TEST_RESULTS_CN.md"
echo "============================================"
