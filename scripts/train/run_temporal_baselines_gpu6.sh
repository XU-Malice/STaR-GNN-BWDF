#!/usr/bin/env bash
# Run the six Que et al. temporal baselines on physical GPU 6.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_ID="${GPU_ID:-6}"
MODEL="${MODEL:-all}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-8192}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/temporal_baselines}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "错误：未找到 nvidia-smi。" >&2
    exit 1
fi

GPU_ROW="$(
    nvidia-smi \
        --id="${GPU_ID}" \
        --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits
)"
GPU_FREE_MIB="$(printf '%s\n' "${GPU_ROW}" | awk -F',' '{gsub(/ /,"",$5); print $5}')"

echo "GPU 预检: ${GPU_ROW}"
if [[ ! "${GPU_FREE_MIB}" =~ ^[0-9]+$ ]]; then
    echo "错误：无法解析 GPU ${GPU_ID} 的可用显存。" >&2
    exit 1
fi
if (( GPU_FREE_MIB < MINIMUM_FREE_MIB )); then
    echo "错误：GPU ${GPU_ID} 只有 ${GPU_FREE_MIB} MiB 可用，至少需要 ${MINIMUM_FREE_MIB} MiB。" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

AUDIT_FILES=(
    "data/processed/data_build/quality_checks.json"
    "data/processed/data_build/demand_iqr_thresholds.csv"
    "data/processed/data_build/interpolation_split_profile.csv"
)
for path in "${AUDIT_FILES[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "未检测到完整的无泄漏数据审计工件，先重建数据。"
        bash scripts/data/run_pipeline.sh
        break
    fi
done

echo "运行 1 epoch / 1 batch GPU smoke test..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train/train_temporal_baselines.py \
    --model mscmnet_w \
    --device cuda:0 \
    --max-epochs 1 \
    --max-train-batches 1 \
    --output-root results/temporal_baselines_smoke \
    --overwrite

echo "Smoke test 通过，开始正式 ${MODEL} 队列。"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train/train_temporal_baselines.py \
    --model "${MODEL}" \
    --device cuda:0 \
    --output-root "${OUTPUT_ROOT}"
