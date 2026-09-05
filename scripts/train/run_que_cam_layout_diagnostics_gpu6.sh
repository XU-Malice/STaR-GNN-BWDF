#!/usr/bin/env bash
# Diagnose paper-compatible CAM time-axis, attention-scaling and batch choices.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_ID="${GPU_ID:-6}"
RUN_TAG="${RUN_TAG:-que_cam_layout_diagnostics_20260901}"
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
from dma_wdf.models.mscmnet import ConvAttentionBlock

expected_root = (Path(os.environ["PROJECT_ROOT_PY"]) / "src").resolve()
sources = (
    Path(dma_wdf.__file__).resolve(),
    Path(inspect.getsourcefile(ConvAttentionBlock)).resolve(),
)
for source in sources:
    if not source.is_relative_to(expected_root):
        raise SystemExit(
            f"Wrong dma_wdf source: {source}; expected a file below {expected_root}"
        )
print({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "dma_wdf_file": dma_wdf.__file__,
    "mscmnet_source": inspect.getsourcefile(ConvAttentionBlock),
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
cleanup() {
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

printf 'case\ttemporal_layout\tattention_scaling\tbatch_size\texit_code\tvalidation\tlog\n' \
    >"${STATUS_TSV}"

is_complete() {
    local run_dir="$1"
    local expected_layout="$2"
    local expected_scaling="$3"
    local expected_batch="$4"
    python - "${run_dir}" "${expected_layout}" "${expected_scaling}" "${expected_batch}" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

run_dir = Path(sys.argv[1])
layout, scaling, batch = sys.argv[2], sys.argv[3], int(sys.argv[4])
paths = [run_dir / name for name in ("status.json", "metrics.csv", "resolved_config.yaml")]
if not all(path.is_file() for path in paths):
    raise SystemExit(1)
status = json.loads(paths[0].read_text(encoding="utf-8"))
metrics = pd.read_csv(paths[1])
config = yaml.safe_load(paths[2].read_text(encoding="utf-8"))
ok = (
    status.get("status") == "completed"
    and status.get("formal_protocol") is True
    and status.get("optimizer") == "adamw"
    and math.isclose(status.get("effective_weight_decay", -1.0), 0.1)
    and status.get("prediction_24h_shape") == [46, 24, 10]
    and status.get("prediction_168h_shape") == [46, 168, 10]
    and len(metrics) == 88
    and metrics["value"].map(math.isfinite).all()
    and config["cam"].get("attention_update") == "replace"
    and config["cam"].get("temporal_layout") == layout
    and config["cam"].get("attention_scaling") == scaling
    and int(config["training"].get("batch_size")) == batch
)
raise SystemExit(0 if ok else 1)
PY
}

LAYOUTS=(full_history_flat per_day_flat per_day_vectors)
SCALINGS=(sqrt_dim none)
BATCHES=(8 16)

for layout in "${LAYOUTS[@]}"; do
    for scaling in "${SCALINGS[@]}"; do
        for batch in "${BATCHES[@]}"; do
            case_name="${layout}__${scaling}__b${batch}"
            output_root="${RESULT_ROOT}/${case_name}"
            run_dir="${output_root}/msnet/seed_${SEED}"
            log_path="${LOG_ROOT}/${case_name}.log"
            if is_complete "${run_dir}" "${layout}" "${scaling}" "${batch}"; then
                echo "跳过已完成任务: ${case_name}"
                printf '%s\t%s\t%s\t%s\t0\tPASS(existing)\t%s\n' \
                    "${case_name}" "${layout}" "${scaling}" "${batch}" "${log_path}" \
                    >>"${STATUS_TSV}"
                continue
            fi
            overwrite_args=()
            if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
                overwrite_args+=(--overwrite)
            fi
            echo "开始诊断: ${case_name}"
            set +e
            CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train/train_temporal_baselines.py \
                --model msnet \
                --device cuda:0 \
                --seed "${SEED}" \
                --normalization zscore \
                --optimizer adamw \
                --joint-weight-decay 0.1 \
                --cam-attention-update replace \
                --cam-attention-scaling "${scaling}" \
                --cam-temporal-layout "${layout}" \
                --batch-size "${batch}" \
                --output-root "${output_root}" \
                "${overwrite_args[@]}" \
                2>&1 | tee "${log_path}"
            train_rc=${PIPESTATUS[0]}
            set -e
            validation="FAIL"
            if (( train_rc == 0 )) && is_complete "${run_dir}" "${layout}" "${scaling}" "${batch}"; then
                validation="PASS"
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${case_name}" "${layout}" "${scaling}" "${batch}" \
                "${train_rc}" "${validation}" "${log_path}" >>"${STATUS_TSV}"
        done
    done
done

python - "${RESULT_ROOT}" "${SEED}" >"${RESULT_ROOT}/diagnostic_summary.tsv" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

root = Path(sys.argv[1])
seed = sys.argv[2]
paper = {
    "24h": {"MAE": 15.537, "MAPE": 0.032, "RMSE": 9.526, "NSE": 0.929},
    "168h": {"MAE": 15.908, "MAPE": 0.032, "RMSE": 9.698, "NSE": 0.930},
}
header = (
    "case\ttemporal_layout\tattention_scaling\tbatch_size\ttask\tfinal_loss\t"
    "MAE\tMAPE\tRMSE\tNSE\tpaper_relative_error\twithin_lead_std\t"
    "origin_std\ttruth_origin_std\torigin_sensitivity_ratio\t"
    "adjacent_origin_change\ttruth_adjacent_origin_change\tdaily_change\t"
    "truth_daily_change\tconv_norm\tattention_norm\tlstm_norm\t"
    "joint_weight_norm\tjoint_bias_norm"
)
print(header)
for case_root in sorted(path for path in root.iterdir() if path.is_dir()):
    run = case_root / "msnet" / f"seed_{seed}"
    required = [run / name for name in ("resolved_config.yaml", "metrics.csv", "loss_curve.csv", "predictions_common46.npz", "checkpoint_msnet.pt")]
    if not all(path.is_file() for path in required):
        continue
    config = yaml.safe_load(required[0].read_text(encoding="utf-8"))
    metrics = pd.read_csv(required[1])
    final_loss = pd.read_csv(required[2])["train_loss"].iloc[-1]
    arrays = np.load(required[3])
    checkpoint = torch.load(required[4], map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]

    def group_norm(pattern):
        return sum(
            float(tensor.float().square().sum())
            for name, tensor in state.items()
            if pattern in name
        ) ** 0.5

    norms = {
        "conv": group_norm(".cam.convolutions."),
        "attention": group_norm(".cam.attention."),
        "lstm": group_norm(".lstm."),
        "joint_weight": float(state["joint_fully_connected.weight"].float().norm()),
        "joint_bias": float(state["joint_fully_connected.bias"].float().norm()),
    }
    for task, pred_key, true_key in (
        ("24h", "y_pred_24h", "y_true_24h"),
        ("168h", "y_pred_168h", "y_true_168h"),
    ):
        total = metrics[(metrics["task"] == task) & (metrics["series"] == "total")]
        values = dict(zip(total["metric"], total["value"]))
        relative = np.mean([
            abs(values[name] - paper[task][name]) / abs(paper[task][name])
            for name in ("MAE", "MAPE", "RMSE", "NSE")
        ])
        pred = arrays[pred_key]
        truth = arrays[true_key]
        within = pred.std(axis=1).mean()
        origin_std = pred.std(axis=0).mean()
        truth_origin_std = truth.std(axis=0).mean()
        sensitivity = origin_std / truth_origin_std
        adjacent = np.abs(np.diff(pred, axis=0)).mean()
        truth_adjacent = np.abs(np.diff(truth, axis=0)).mean()
        if pred.shape[1] >= 48:
            daily = np.abs(pred[:, 24:] - pred[:, :-24]).mean()
            truth_daily = np.abs(truth[:, 24:] - truth[:, :-24]).mean()
        else:
            daily = float("nan")
            truth_daily = float("nan")
        print(
            f"{case_root.name}\t{config['cam']['temporal_layout']}\t"
            f"{config['cam']['attention_scaling']}\t{config['training']['batch_size']}\t"
            f"{task}\t{final_loss:.8f}\t{values['MAE']:.6f}\t{values['MAPE']:.6f}\t"
            f"{values['RMSE']:.6f}\t{values['NSE']:.6f}\t{relative:.6f}\t"
            f"{within:.6f}\t{origin_std:.6f}\t{truth_origin_std:.6f}\t"
            f"{sensitivity:.6f}\t{adjacent:.6f}\t{truth_adjacent:.6f}\t"
            f"{daily:.6f}\t{truth_daily:.6f}\t{norms['conv']:.6f}\t"
            f"{norms['attention']:.6f}\t{norms['lstm']:.6f}\t"
            f"{norms['joint_weight']:.6f}\t{norms['joint_bias']:.6f}"
        )
PY

cleanup
trap - EXIT INT TERM

tar -czf "${BUNDLE_PATH}" \
    -C "${PROJECT_ROOT}" \
    "results/${RUN_TAG}" \
    "logs/${RUN_TAG}"
sha256sum "${BUNDLE_PATH}" | tee "${BUNDLE_PATH}.sha256"

failed="$(awk -F '\t' 'NR > 1 && $6 !~ /^PASS/ {count++} END {print count+0}' "${STATUS_TSV}")"
echo "CAM 布局诊断结束：失败任务=${failed}"
echo "摘要：${RESULT_ROOT}/diagnostic_summary.tsv"
echo "结果包：${BUNDLE_PATH}"
if (( failed > 0 )); then
    exit 1
fi
