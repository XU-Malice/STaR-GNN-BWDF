#!/usr/bin/env bash
# Isolate the temporal-collapse mechanism in the paper-style MSNet CAM.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_ID="${GPU_ID:-6}"
RUN_TAG="${RUN_TAG:-que_attention_diagnostics_20260901}"
SEED="${SEED:-20240604}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-8192}"
RESULT_ROOT="${PROJECT_ROOT}/results/${RUN_TAG}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_TAG}"
BUNDLE_PATH="${BUNDLE_PATH:-${PROJECT_ROOT}/../${RUN_TAG}.tar.gz}"
STATUS_TSV="${LOG_ROOT}/diagnostic_status.tsv"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

exec 9>"${LOG_ROOT}/run.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "错误：同一诊断目录已有任务在运行。" >&2
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

{
    date --iso-8601=seconds
    git rev-parse HEAD
    git status --short
    python --version
    python - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY
    nvidia-smi --id="${GPU_ID}"
} >"${LOG_ROOT}/environment_and_git.txt" 2>&1

nvidia-smi --id="${GPU_ID}" \
    --query-gpu=timestamp,index,temperature.gpu,power.draw,memory.used,memory.free,utilization.gpu \
    --format=csv -l 60 >"${LOG_ROOT}/gpu_monitor.csv" 2>&1 &
MONITOR_PID=$!
cleanup() {
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

printf 'attention_update\texit_code\tvalidation\tlog\n' >"${STATUS_TSV}"

is_complete() {
    local run_dir="$1"
    local expected_mode="$2"
    python - "${run_dir}" "${expected_mode}" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

run_dir = Path(sys.argv[1])
expected_mode = sys.argv[2]
status_path = run_dir / "status.json"
metrics_path = run_dir / "metrics.csv"
config_path = run_dir / "resolved_config.yaml"
if not all(path.is_file() for path in (status_path, metrics_path, config_path)):
    raise SystemExit(1)
status = json.loads(status_path.read_text(encoding="utf-8"))
metrics = pd.read_csv(metrics_path)
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
ok = (
    status.get("status") == "completed"
    and status.get("formal_protocol") is True
    and len(metrics) == 88
    and metrics["value"].map(math.isfinite).all()
    and config["cam"].get("attention_update") == expected_mode
)
raise SystemExit(0 if ok else 1)
PY
}

for mode in final_residual skip_final residual; do
    output_root="${RESULT_ROOT}/${mode}"
    run_dir="${output_root}/msnet/seed_${SEED}"
    log_path="${LOG_ROOT}/msnet_${mode}_zscore.log"
    if is_complete "${run_dir}" "${mode}"; then
        echo "跳过已完成任务: ${mode}"
        printf '%s\t0\tPASS(existing)\t%s\n' "${mode}" "${log_path}" >>"${STATUS_TSV}"
        continue
    fi
    overwrite_args=()
    if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
        overwrite_args+=(--overwrite)
    fi
    echo "开始诊断: ${mode}"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train/train_temporal_baselines.py \
        --model msnet \
        --device cuda:0 \
        --seed "${SEED}" \
        --normalization zscore \
        --cam-attention-update "${mode}" \
        --output-root "${output_root}" \
        "${overwrite_args[@]}" \
        2>&1 | tee "${log_path}"
    train_rc=${PIPESTATUS[0]}
    set -e
    validation="FAIL"
    if (( train_rc == 0 )) && is_complete "${run_dir}" "${mode}"; then
        validation="PASS"
    fi
    printf '%s\t%s\t%s\t%s\n' \
        "${mode}" "${train_rc}" "${validation}" "${log_path}" >>"${STATUS_TSV}"
done

python - "${RESULT_ROOT}" "${SEED}" >"${RESULT_ROOT}/diagnostic_summary.tsv" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
seed = sys.argv[2]
print("attention_update\ttask\tMAE\tMAPE\tRMSE\tNSE\tmean_dma_temporal_std")
for mode in ("final_residual", "skip_final", "residual"):
    run = root / mode / "msnet" / f"seed_{seed}"
    metrics = pd.read_csv(run / "metrics.csv")
    predictions = np.load(run / "predictions_common46.npz")
    for task, key in (("24h", "y_pred_24h"), ("168h", "y_pred_168h")):
        total = metrics[(metrics["task"] == task) & (metrics["series"] == "total")]
        values = dict(zip(total["metric"], total["value"]))
        temporal_std = predictions[key].reshape(-1, 10).std(axis=0).mean()
        print(
            f"{mode}\t{task}\t{values['MAE']:.6f}\t{values['MAPE']:.6f}\t"
            f"{values['RMSE']:.6f}\t{values['NSE']:.6f}\t{temporal_std:.6f}"
        )
PY

cleanup
trap - EXIT INT TERM

tar -czf "${BUNDLE_PATH}" \
    -C "${PROJECT_ROOT}" \
    "results/${RUN_TAG}" \
    "logs/${RUN_TAG}"
sha256sum "${BUNDLE_PATH}" | tee "${BUNDLE_PATH}.sha256"

failed="$(awk -F '\t' 'NR > 1 && $3 !~ /^PASS/ {count++} END {print count+0}' "${STATUS_TSV}")"
echo "诊断结束：失败任务=${failed}"
echo "摘要：${RESULT_ROOT}/diagnostic_summary.tsv"
echo "结果包：${BUNDLE_PATH}"
if (( failed > 0 )); then
    exit 1
fi
