#!/usr/bin/env bash
# Diagnose MSCMNet correction calibration, batch updates and FC2 supervision.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_ID="${GPU_ID:-6}"
RUN_TAG="${RUN_TAG:-que_correction_calibration_diagnostics_20260902}"
SEED="${SEED:-20240604}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-8192}"
RESULT_ROOT="${PROJECT_ROOT}/results/${RUN_TAG}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_TAG}"
BUNDLE_PATH="${BUNDLE_PATH:-${PROJECT_ROOT}/../${RUN_TAG}.tar.gz}"
STATUS_TSV="${LOG_ROOT}/diagnostic_status.tsv"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec 9>"${LOG_ROOT}/run.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "错误：同一诊断目录已有任务在运行。" >&2
    exit 1
fi

GPU_ROW="$(nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits)"
GPU_FREE_MIB="$(printf '%s\n' "${GPU_ROW}" | awk -F',' '{gsub(/ /,"",$5); print $5}')"
echo "GPU 预检: ${GPU_ROW}"
if [[ ! "${GPU_FREE_MIB}" =~ ^[0-9]+$ ]] || (( GPU_FREE_MIB < MINIMUM_FREE_MIB )); then
    echo "错误：GPU ${GPU_ID} 可用显存不足 ${MINIMUM_FREE_MIB} MiB。" >&2
    exit 1
fi

if ! {
    date --iso-8601=seconds
    git rev-parse HEAD
    git status --short
    python --version
    PROJECT_ROOT_PY="${PROJECT_ROOT}" python - <<'PY'
import inspect
import os
from pathlib import Path

import torch
import dma_wdf
from dma_wdf.models.mscmnet import MSCMNetM

expected = (Path(os.environ["PROJECT_ROOT_PY"]) / "src").resolve()
for source in (Path(dma_wdf.__file__).resolve(), Path(inspect.getsourcefile(MSCMNetM)).resolve()):
    if not source.is_relative_to(expected):
        raise SystemExit(f"Wrong dma_wdf source: {source}; expected below {expected}")
print({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "dma_wdf_file": dma_wdf.__file__,
    "mscmnet_source": inspect.getsourcefile(MSCMNetM),
})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY
    nvidia-smi --id="${GPU_ID}"
} >"${LOG_ROOT}/environment_and_git.txt" 2>&1; then
    echo "错误：环境与源码导入预检失败。" >&2
    sed -n '1,240p' "${LOG_ROOT}/environment_and_git.txt" >&2
    exit 1
fi

nvidia-smi --id="${GPU_ID}" \
    --query-gpu=timestamp,index,temperature.gpu,power.draw,memory.used,memory.free,utilization.gpu \
    --format=csv -l 60 >"${LOG_ROOT}/gpu_monitor.csv" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "${MONITOR_PID}" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

printf 'case\tmodel\tbatch_size\tcorrection_mode\tzero_init\tshare_weight\texit_code\tvalidation\tlog\n' \
    >"${STATUS_TSV}"

is_complete() {
    local run_dir="$1"
    local model="$2"
    local batch="$3"
    local mode="$4"
    local zero="$5"
    local share="$6"
    python - "${run_dir}" "${model}" "${batch}" "${mode}" "${zero}" "${share}" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

run = Path(sys.argv[1])
model, batch, mode = sys.argv[2], int(sys.argv[3]), sys.argv[4]
zero, share = bool(int(sys.argv[5])), float(sys.argv[6])
required = [run / name for name in ("status.json", "metrics.csv", "loss_curve.csv", "resolved_config.yaml", "predictions_common46.npz")]
if not all(path.is_file() for path in required):
    raise SystemExit(1)
status = json.loads(required[0].read_text(encoding="utf-8"))
metrics = pd.read_csv(required[1])
config = yaml.safe_load(required[3].read_text(encoding="utf-8"))
model_config = config["model"]
expected_share = share if model in {"mscmnet_w", "mscmnet_wm"} else 0.0
ok = (
    status.get("status") == "completed"
    and status.get("formal_protocol") is True
    and status.get("single_frozen_checkpoint_for_24h_and_168h") is True
    and status.get("prediction_24h_shape") == [46, 24, 10]
    and status.get("prediction_168h_shape") == [46, 168, 10]
    and status.get("correction_mode") == mode
    and bool(status.get("zero_init_correction")) is zero
    and math.isclose(float(status.get("fc2_share_supervision_weight", -1)), expected_share)
    and len(metrics) == 88
    and metrics["value"].map(math.isfinite).all()
    and model_config.get("correction_mode") == mode
    and bool(model_config.get("zero_init_correction")) is zero
    and int(config["training"].get("batch_size")) == batch
)
raise SystemExit(0 if ok else 1)
PY
}

CASES=(
    m_direct_b1
    m_direct_b4
    w_direct_b1
    w_direct_b4
    m_residual_zero_b8
    w_residual_zero_share01_b8
    wm_residual_zero_share01_b8
    w_direct_share01_b8
    wm_direct_share01_b8
)
MODELS=(mscmnet_m mscmnet_m mscmnet_w mscmnet_w mscmnet_m mscmnet_w mscmnet_wm mscmnet_w mscmnet_wm)
BATCHES=(1 4 1 4 8 8 8 8 8)
MODES=(direct direct direct direct residual residual residual direct direct)
ZERO_INIT=(0 0 0 0 1 1 1 0 0)
SHARE_WEIGHTS=(0 0 0 0 0 0.1 0.1 0.1 0.1)

for index in "${!CASES[@]}"; do
    case_name="${CASES[$index]}"
    model="${MODELS[$index]}"
    batch="${BATCHES[$index]}"
    mode="${MODES[$index]}"
    zero="${ZERO_INIT[$index]}"
    share="${SHARE_WEIGHTS[$index]}"
    output_root="${RESULT_ROOT}/${case_name}"
    run_dir="${output_root}/${model}/seed_${SEED}"
    log_path="${LOG_ROOT}/${case_name}.log"
    if is_complete "${run_dir}" "${model}" "${batch}" "${mode}" "${zero}" "${share}"; then
        echo "跳过已完成任务: ${case_name}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t0\tPASS(existing)\t%s\n' \
            "${case_name}" "${model}" "${batch}" "${mode}" "${zero}" "${share}" "${log_path}" \
            >>"${STATUS_TSV}"
        continue
    fi
    overwrite_args=()
    if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
        overwrite_args+=(--overwrite)
    fi
    zero_args=()
    if [[ "${zero}" == "1" ]]; then zero_args+=(--zero-init-correction); fi
    share_args=()
    if [[ "${model}" != "mscmnet_m" ]]; then
        share_args+=(--fc2-share-supervision-weight "${share}")
    fi
    echo "开始诊断: ${case_name}"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train/train_temporal_baselines.py \
        --model "${model}" \
        --device cuda:0 \
        --seed "${SEED}" \
        --normalization zscore \
        --optimizer adamw \
        --cam-attention-update replace \
        --cam-attention-scaling sqrt_dim \
        --cam-temporal-layout per_day_vectors \
        --batch-size "${batch}" \
        --correction-mode "${mode}" \
        --output-root "${output_root}" \
        "${zero_args[@]}" \
        "${share_args[@]}" \
        "${overwrite_args[@]}" \
        2>&1 | tee "${log_path}"
    train_rc=${PIPESTATUS[0]}
    set -e
    validation="FAIL"
    if (( train_rc == 0 )) && is_complete "${run_dir}" "${model}" "${batch}" "${mode}" "${zero}" "${share}"; then
        validation="PASS"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${case_name}" "${model}" "${batch}" "${mode}" "${zero}" "${share}" \
        "${train_rc}" "${validation}" "${log_path}" >>"${STATUS_TSV}"
done

python - "${RESULT_ROOT}" "${SEED}" >"${RESULT_ROOT}/diagnostic_summary.tsv" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

root, seed = Path(sys.argv[1]), sys.argv[2]
print("case\tmodel\tbatch_size\tcorrection_mode\tzero_init\tshare_weight\ttask\tMAE\tMAPE\tRMSE\tNSE\tmean_paper_relative_error\ttotal_bias\tbias_mse_share\toracle_debiased_rmse\torigin_sensitivity\tadjacent_change_ratio\tdaily_change_ratio")
for case in sorted(path for path in root.iterdir() if path.is_dir()):
    model_dirs = [path for path in case.iterdir() if path.is_dir()]
    if len(model_dirs) != 1:
        continue
    model = model_dirs[0].name
    run = model_dirs[0] / f"seed_{seed}"
    required = [run / name for name in ("metrics.csv", "predictions_common46.npz", "resolved_config.yaml")]
    if not all(path.is_file() for path in required):
        continue
    metrics = pd.read_csv(required[0])
    arrays = np.load(required[1])
    config = yaml.safe_load(required[2].read_text(encoding="utf-8"))
    mc = config["model"]
    share = float(mc.get("fc2", {}).get("share_supervision_weight", 0.0))
    for task in ("24h", "168h"):
        total = metrics[(metrics["task"] == task) & (metrics["series"] == "total")]
        values = dict(zip(total["metric"], total["value"]))
        paper = dict(zip(total["metric"], total["paper_value"]))
        relative = np.mean([abs(values[key] - paper[key]) / abs(paper[key]) for key in values])
        truth, pred = arrays[f"y_true_{task}"], arrays[f"y_pred_{task}"]
        truth_total, pred_total = truth.sum(axis=2), pred.sum(axis=2)
        bias = float((pred_total - truth_total).mean())
        bias_share = bias ** 2 / float(np.mean((pred_total - truth_total) ** 2))
        debiased = float(np.sqrt(np.mean((pred_total - bias - truth_total) ** 2)))
        origin = float(pred.std(axis=0).mean() / truth.std(axis=0).mean())
        adjacent = float(np.abs(np.diff(pred, axis=0)).mean() / np.abs(np.diff(truth, axis=0)).mean())
        daily = float("nan") if pred.shape[1] < 48 else float(np.abs(pred[:, 24:] - pred[:, :-24]).mean() / np.abs(truth[:, 24:] - truth[:, :-24]).mean())
        print(f"{case.name}\t{model}\t{config['training']['batch_size']}\t{mc.get('correction_mode','direct')}\t{int(bool(mc.get('zero_init_correction',False)))}\t{share}\t{task}\t{values['MAE']:.6f}\t{values['MAPE']:.6f}\t{values['RMSE']:.6f}\t{values['NSE']:.6f}\t{relative:.6f}\t{bias:.6f}\t{bias_share:.6f}\t{debiased:.6f}\t{origin:.6f}\t{adjacent:.6f}\t{daily:.6f}")
PY

cleanup
trap - EXIT INT TERM
tar -czf "${BUNDLE_PATH}" -C "${PROJECT_ROOT}" "results/${RUN_TAG}" "logs/${RUN_TAG}"
sha256sum "${BUNDLE_PATH}" | tee "${BUNDLE_PATH}.sha256"

failed="$(awk -F '\t' 'NR > 1 && $8 !~ /^PASS/ {count++} END {print count+0}' "${STATUS_TSV}")"
echo "校正标定诊断结束：失败任务=${failed}"
echo "摘要：${RESULT_ROOT}/diagnostic_summary.tsv"
echo "结果包：${BUNDLE_PATH}"
if (( failed > 0 )); then exit 1; fi
