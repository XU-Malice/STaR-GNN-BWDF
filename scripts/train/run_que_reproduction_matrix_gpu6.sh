#!/usr/bin/env bash
# Queue, validate, and package the Que et al. temporal-baseline reproduction.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_ID="${GPU_ID:-6}"
RUN_TAG="${RUN_TAG:-que_reproduction_matrix_20260831}"
SEED="${SEED:-20240604}"
CAM_WIDTH_SWEEP="${CAM_WIDTH_SWEEP:-1}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-8192}"
RESULT_ROOT="${PROJECT_ROOT}/results/${RUN_TAG}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_TAG}"
BUNDLE_PATH="${BUNDLE_PATH:-${PROJECT_ROOT}/../${RUN_TAG}.tar.gz}"
MATRIX_STATUS="${LOG_ROOT}/matrix_status.tsv"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
exec 9>"${LOG_ROOT}/run.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "错误：同一结果目录已有一个复现队列在运行。" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "错误：未找到 nvidia-smi。" >&2
    exit 1
fi

GPU_ROW="$(
    nvidia-smi --id="${GPU_ID}" \
        --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits
)"
GPU_FREE_MIB="$(printf '%s\n' "${GPU_ROW}" | awk -F',' '{gsub(/ /,"",$5); print $5}')"
echo "GPU 预检: ${GPU_ROW}"
if [[ ! "${GPU_FREE_MIB}" =~ ^[0-9]+$ ]] || (( GPU_FREE_MIB < MINIMUM_FREE_MIB )); then
    echo "错误：GPU ${GPU_ID} 可用显存不足 ${MINIMUM_FREE_MIB} MiB。" >&2
    exit 1
fi

python - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the active Python environment")
PY

{
    date --iso-8601=seconds
    git rev-parse HEAD
    git status --short
    python --version
    python -m pip freeze
    nvidia-smi --id="${GPU_ID}"
} >"${LOG_ROOT}/environment_and_git.txt" 2>&1

printf 'case\tmodel\tnormalization\tcam_channels\texit_code\tvalidation\tlog\n' \
    >"${MATRIX_STATUS}"

nvidia-smi --id="${GPU_ID}" \
    --query-gpu=timestamp,index,temperature.gpu,power.draw,memory.used,memory.free,utilization.gpu \
    --format=csv -l 60 >"${LOG_ROOT}/gpu_monitor.csv" 2>&1 &
MONITOR_PID=$!
cleanup() {
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

is_complete() {
    local status_path="$1"
    local metrics_path="$2"
    python - "${status_path}" "${metrics_path}" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd

status_path, metrics_path = map(Path, sys.argv[1:])
if not status_path.is_file() or not metrics_path.is_file():
    raise SystemExit(1)
status = json.loads(status_path.read_text(encoding="utf-8"))
metrics = pd.read_csv(metrics_path)
ok = (
    status.get("status") == "completed"
    and status.get("formal_protocol") is True
    and len(metrics) == 88
    and metrics["value"].map(math.isfinite).all()
)
raise SystemExit(0 if ok else 1)
PY
}

run_case() {
    local case_name="$1"
    local model="$2"
    local normalization="$3"
    local channels="$4"
    local output_root="$5"
    local run_dir="${output_root}/${model}/seed_${SEED}"
    local status_path="${run_dir}/status.json"
    local metrics_path="${run_dir}/metrics.csv"
    local log_path="${LOG_ROOT}/${case_name}.log"

    mkdir -p "$(dirname "${log_path}")" "${output_root}"
    if is_complete "${status_path}" "${metrics_path}"; then
        echo "跳过已完成任务: ${case_name}"
        printf '%s\t%s\t%s\t%s\t0\tPASS(existing)\t%s\n' \
            "${case_name}" "${model}" "${normalization}" "${channels}" \
            "${log_path}" >>"${MATRIX_STATUS}"
        return 0
    fi

    local overwrite_args=()
    if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
        overwrite_args+=(--overwrite)
    fi
    local channel_args=()
    if [[ -n "${channels}" ]]; then
        IFS=',' read -r c1 c2 c3 <<<"${channels}"
        channel_args+=(--cam-channel-sizes "${c1}" "${c2}" "${c3}")
    fi

    echo "开始任务: ${case_name} model=${model} normalization=${normalization} channels=${channels:-default}"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train/train_temporal_baselines.py \
        --model "${model}" \
        --device cuda:0 \
        --seed "${SEED}" \
        --normalization "${normalization}" \
        --output-root "${output_root}" \
        "${channel_args[@]}" \
        "${overwrite_args[@]}" \
        2>&1 | tee "${log_path}"
    local train_rc=${PIPESTATUS[0]}
    set -e

    local validation="FAIL"
    if (( train_rc == 0 )) && is_complete "${status_path}" "${metrics_path}"; then
        validation="PASS"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${case_name}" "${model}" "${normalization}" "${channels}" \
        "${train_rc}" "${validation}" "${log_path}" >>"${MATRIX_STATUS}"
    echo "结束任务: ${case_name} exit=${train_rc} validation=${validation}"
}

# Core matrix: all six paper baselines under both plausible unpublished scalers.
# MSNet is intentionally first so its two diagnostic results are available early.
CORE_MODELS=(msnet mscmnet_wm mscmnet_m mscmnet_w gru lstm)
NORMALIZATIONS=(zscore minmax)
for model in "${CORE_MODELS[@]}"; do
    for normalization in "${NORMALIZATIONS[@]}"; do
        case_name="core/${model}_${normalization}"
        run_case \
            "${case_name}" "${model}" "${normalization}" "16,16,1" \
            "${RESULT_ROOT}/core/${normalization}"
    done
done

# The paper fixes the final CAM output channel at one but does not publish the
# two intermediate widths. This bounded MSNet-only sweep isolates that ambiguity.
if [[ "${CAM_WIDTH_SWEEP}" == "1" ]]; then
    for width in 1 8 32; do
        for normalization in "${NORMALIZATIONS[@]}"; do
            channels="${width},${width},1"
            case_name="cam_width_sweep/c${width}-${width}-1/msnet_${normalization}"
            run_case \
                "${case_name}" "msnet" "${normalization}" "${channels}" \
                "${RESULT_ROOT}/cam_width_sweep/c${width}-${width}-1/${normalization}"
        done
    done
fi

set +e
python scripts/reproduce/summarize_que_reproduction_matrix.py \
    "${RESULT_ROOT}" --summary-dir "${RESULT_ROOT}/_summary" \
    >"${LOG_ROOT}/summary.log" 2>&1
SUMMARY_RC=$?
set -e

cleanup
trap - EXIT INT TERM

tar -czf "${BUNDLE_PATH}" \
    -C "${PROJECT_ROOT}" \
    "results/${RUN_TAG}" \
    "logs/${RUN_TAG}"
sha256sum "${BUNDLE_PATH}" | tee "${BUNDLE_PATH}.sha256"

FAILED_CASES="$(awk -F '\t' 'NR > 1 && $6 !~ /^PASS/ {count++} END {print count+0}' "${MATRIX_STATUS}")"
echo "队列结束：失败任务=${FAILED_CASES}，汇总校验退出码=${SUMMARY_RC}"
echo "结果包：${BUNDLE_PATH}"
if (( FAILED_CASES > 0 || SUMMARY_RC != 0 )); then
    exit 1
fi
