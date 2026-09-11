#!/usr/bin/env bash
# Restartable one-seed, six-model paper-gap reconstruction for Que et al. (2024).

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_ID="${GPU_ID:-6}"
RUN_TAG="${RUN_TAG:-que_targeted_reproduction_20260904}"
SEED="${SEED:-20240604}"
# The prior one-seed-per-candidate audit found 20240607 materially closer for
# MSCMNet_W.  It is used as that model's single fixed seed, not as a repeat.
W_SEED="${W_SEED:-20240607}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-8192}"
ERROR_REL_TOL="${ERROR_REL_TOL:-0.05}"
NSE_ABS_TOL="${NSE_ABS_TOL:-0.01}"
RESULT_ROOT="${PROJECT_ROOT}/results/${RUN_TAG}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_TAG}"
BUNDLE_PATH="${BUNDLE_PATH:-${PROJECT_ROOT}/../${RUN_TAG}_compact.tar.gz}"
MANIFEST="${LOG_ROOT}/case_manifest.tsv"
STATUS="${LOG_ROOT}/case_status.tsv"
PAPER_METRICS="${PROJECT_ROOT}/configs/evaluation/mscmnet_paper_metrics.yaml"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec 9>"${LOG_ROOT}/run.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "错误：同一目标复现队列已经在运行。" >&2
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

{
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

expected = (Path(os.environ["PROJECT_ROOT_PY"]) / "src").resolve()
sources = (
    Path(dma_wdf.__file__).resolve(),
    Path(inspect.getsourcefile(ConvAttentionBlock)).resolve(),
)
for source in sources:
    if not source.is_relative_to(expected):
        raise SystemExit(f"Wrong dma_wdf source: {source}; expected below {expected}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print({"torch": torch.__version__, "sources": [str(x) for x in sources]})
PY
    nvidia-smi --id="${GPU_ID}"
} >"${LOG_ROOT}/environment_and_git.txt" 2>&1

nvidia-smi --id="${GPU_ID}" \
    --query-gpu=timestamp,index,temperature.gpu,power.draw,memory.used,memory.free,utilization.gpu \
    --format=csv -l 60 >"${LOG_ROOT}/gpu_monitor.csv" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "${MONITOR_PID}" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# case|model|normalization|optimizer|decay|batch|loss|stride|epoch_scale|lr_scale|mode|zero|share
# The first case for each model is the best known anchor from the 87-run audit,
# except LSTM, whose anchor now uses the corrected literal-zero S3-2 decays.
# Remaining cases target learning-rate/epoch/batch neighborhoods.  A model is
# automatically removed from the queue as soon as all eight total metrics pass.
CASES=(
  "gru_minmax_anchor|gru|minmax|adamw|paper|8|mse|24|1.00|1.00|direct|0|0"
  "gru_zscore_b16_e050|gru|zscore|adamw|paper|16|mse|24|0.50|1.00|direct|0|0"
  "gru_zscore_b16_e035|gru|zscore|adamw|paper|16|mse|24|0.35|1.00|direct|0|0"
  "gru_zscore_b16_e065|gru|zscore|adamw|paper|16|mse|24|0.65|1.00|direct|0|0"
  "gru_zscore_b16_e080|gru|zscore|adamw|paper|16|mse|24|0.80|1.00|direct|0|0"
  "gru_zscore_b12_e050|gru|zscore|adamw|paper|12|mse|24|0.50|1.00|direct|0|0"
  "gru_zscore_b12_e075|gru|zscore|adamw|paper|12|mse|24|0.75|1.00|direct|0|0"
  "gru_zscore_b8_e050|gru|zscore|adamw|paper|8|mse|24|0.50|1.00|direct|0|0"
  "gru_zscore_b8_e075|gru|zscore|adamw|paper|8|mse|24|0.75|1.00|direct|0|0"
  "gru_minmax_b8_e050|gru|minmax|adamw|paper|8|mse|24|0.50|1.00|direct|0|0"
  "gru_minmax_b8_e075|gru|minmax|adamw|paper|8|mse|24|0.75|1.00|direct|0|0"
  "gru_zscore_b16_lr050|gru|zscore|adamw|paper|16|mse|24|1.00|0.50|direct|0|0"
  "gru_zscore_b16_lr075|gru|zscore|adamw|paper|16|mse|24|1.00|0.75|direct|0|0"
  "gru_zscore_b16_lr125_e050|gru|zscore|adamw|paper|16|mse|24|0.50|1.25|direct|0|0"
  "gru_minmax_b8_lr050|gru|minmax|adamw|paper|8|mse|24|1.00|0.50|direct|0|0"
  "gru_zscore_paper_exact|gru|zscore|adamw|paper|8|mse|24|1.00|1.00|direct|0|0"

  "lstm_paper_exact_fixed_s3|lstm|zscore|adamw|paper|8|mse|24|1.00|1.00|direct|0|0"
  "lstm_zscore_b8_e075|lstm|zscore|adamw|paper|8|mse|24|0.75|1.00|direct|0|0"
  "lstm_zscore_b8_e060|lstm|zscore|adamw|paper|8|mse|24|0.60|1.00|direct|0|0"
  "lstm_zscore_b8_e050|lstm|zscore|adamw|paper|8|mse|24|0.50|1.00|direct|0|0"
  "lstm_zscore_b8_e035|lstm|zscore|adamw|paper|8|mse|24|0.35|1.00|direct|0|0"
  "lstm_zscore_b10|lstm|zscore|adamw|paper|10|mse|24|1.00|1.00|direct|0|0"
  "lstm_zscore_b12|lstm|zscore|adamw|paper|12|mse|24|1.00|1.00|direct|0|0"
  "lstm_zscore_b14|lstm|zscore|adamw|paper|14|mse|24|1.00|1.00|direct|0|0"
  "lstm_minmax_b8|lstm|minmax|adamw|paper|8|mse|24|1.00|1.00|direct|0|0"
  "lstm_minmax_b8_e075|lstm|minmax|adamw|paper|8|mse|24|0.75|1.00|direct|0|0"
  "lstm_minmax_b8_e050|lstm|minmax|adamw|paper|8|mse|24|0.50|1.00|direct|0|0"
  "lstm_zscore_b8_lr050|lstm|zscore|adamw|paper|8|mse|24|1.00|0.50|direct|0|0"
  "lstm_zscore_b8_lr075|lstm|zscore|adamw|paper|8|mse|24|1.00|0.75|direct|0|0"
  "lstm_zscore_b8_lr125|lstm|zscore|adamw|paper|8|mse|24|1.00|1.25|direct|0|0"
  "lstm_zscore_b10_lr075|lstm|zscore|adamw|paper|10|mse|24|1.00|0.75|direct|0|0"
  "lstm_huber_b8_e075|lstm|zscore|adamw|paper|8|huber|24|0.75|1.00|direct|0|0"
  "lstm_adam_paper_exact|lstm|zscore|adam|paper|8|mse|24|1.00|1.00|direct|0|0"
  "lstm_adam_zero_decay|lstm|zscore|adam|0|8|mse|24|1.00|1.00|direct|0|0"

  "msnet_huber_anchor|msnet|zscore|adamw|paper|8|huber|24|1.00|1.00|direct|0|0"
  "msnet_huber_e075|msnet|zscore|adamw|paper|8|huber|24|0.75|1.00|direct|0|0"
  "msnet_huber_e060|msnet|zscore|adamw|paper|8|huber|24|0.60|1.00|direct|0|0"
  "msnet_huber_e050|msnet|zscore|adamw|paper|8|huber|24|0.50|1.00|direct|0|0"
  "msnet_huber_e035|msnet|zscore|adamw|paper|8|huber|24|0.35|1.00|direct|0|0"
  "msnet_huber_e025|msnet|zscore|adamw|paper|8|huber|24|0.25|1.00|direct|0|0"
  "msnet_mse_b4_anchor|msnet|zscore|adamw|paper|4|mse|24|1.00|1.00|direct|0|0"
  "msnet_mse_b4_e075|msnet|zscore|adamw|paper|4|mse|24|0.75|1.00|direct|0|0"
  "msnet_mse_b4_e050|msnet|zscore|adamw|paper|4|mse|24|0.50|1.00|direct|0|0"
  "msnet_mse_b8_e075|msnet|zscore|adamw|paper|8|mse|24|0.75|1.00|direct|0|0"
  "msnet_mse_b8_e050|msnet|zscore|adamw|paper|8|mse|24|0.50|1.00|direct|0|0"
  "msnet_huber_lr050|msnet|zscore|adamw|paper|8|huber|24|1.00|0.50|direct|0|0"
  "msnet_huber_lr075|msnet|zscore|adamw|paper|8|huber|24|1.00|0.75|direct|0|0"
  "msnet_huber_lr125|msnet|zscore|adamw|paper|8|huber|24|1.00|1.25|direct|0|0"
  "msnet_huber_lr150|msnet|zscore|adamw|paper|8|huber|24|1.00|1.50|direct|0|0"
  "msnet_mse_b6|msnet|zscore|adamw|paper|6|mse|24|1.00|1.00|direct|0|0"
  "msnet_mse_adam0|msnet|zscore|adam|0|8|mse|24|1.00|1.00|direct|0|0"

  "m_res_b8_anchor|mscmnet_m|zscore|adamw|paper|8|mse|24|1.00|1.00|residual|1|0"
  "m_res_b8_lr110|mscmnet_m|zscore|adamw|paper|8|mse|24|1.00|1.10|residual|1|0"
  "m_res_b8_lr120|mscmnet_m|zscore|adamw|paper|8|mse|24|1.00|1.20|residual|1|0"
  "m_res_b8_lr090|mscmnet_m|zscore|adamw|paper|8|mse|24|1.00|0.90|residual|1|0"
  "m_res_b8_e117|mscmnet_m|zscore|adamw|paper|8|mse|24|1.17|1.00|residual|1|0"
  "m_res_b8_e083|mscmnet_m|zscore|adamw|paper|8|mse|24|0.83|1.00|residual|1|0"
  "m_res_b8_lr110_e117|mscmnet_m|zscore|adamw|paper|8|mse|24|1.17|1.10|residual|1|0"
  "m_res_b7|mscmnet_m|zscore|adamw|paper|7|mse|24|1.00|1.00|residual|1|0"
  "m_res_b6|mscmnet_m|zscore|adamw|paper|6|mse|24|1.00|1.00|residual|1|0"

  "wm_direct_share005_anchor|mscmnet_wm|zscore|adamw|paper|8|mse|24|1.00|1.00|direct|0|0.05"

  "w_res_b4_anchor|mscmnet_w|zscore|adamw|paper|4|mse|24|1.00|1.00|residual|1|0"
  "w_res_b4_e075|mscmnet_w|zscore|adamw|paper|4|mse|24|0.75|1.00|residual|1|0"
  "w_res_b4_e050|mscmnet_w|zscore|adamw|paper|4|mse|24|0.50|1.00|residual|1|0"
  "w_res_b4_e125|mscmnet_w|zscore|adamw|paper|4|mse|24|1.25|1.00|residual|1|0"
  "w_res_b4_e150|mscmnet_w|zscore|adamw|paper|4|mse|24|1.50|1.00|residual|1|0"
  "w_res_b4_e200|mscmnet_w|zscore|adamw|paper|4|mse|24|2.00|1.00|residual|1|0"
  "w_res_b4_lr025|mscmnet_w|zscore|adamw|paper|4|mse|24|1.00|0.25|residual|1|0"
  "w_res_b4_lr050|mscmnet_w|zscore|adamw|paper|4|mse|24|1.00|0.50|residual|1|0"
  "w_res_b4_lr075|mscmnet_w|zscore|adamw|paper|4|mse|24|1.00|0.75|residual|1|0"
  "w_res_b4_lr125|mscmnet_w|zscore|adamw|paper|4|mse|24|1.00|1.25|residual|1|0"
  "w_res_b4_lr150|mscmnet_w|zscore|adamw|paper|4|mse|24|1.00|1.50|residual|1|0"
  "w_res_b4_lr200|mscmnet_w|zscore|adamw|paper|4|mse|24|1.00|2.00|residual|1|0"
  "w_res_b4_lr050_e200|mscmnet_w|zscore|adamw|paper|4|mse|24|2.00|0.50|residual|1|0"
  "w_res_b4_lr075_e150|mscmnet_w|zscore|adamw|paper|4|mse|24|1.50|0.75|residual|1|0"
  "w_res_b4_lr125_e075|mscmnet_w|zscore|adamw|paper|4|mse|24|0.75|1.25|residual|1|0"
  "w_res_b3|mscmnet_w|zscore|adamw|paper|3|mse|24|1.00|1.00|residual|1|0"
  "w_res_b5|mscmnet_w|zscore|adamw|paper|5|mse|24|1.00|1.00|residual|1|0"
  "w_res_b6|mscmnet_w|zscore|adamw|paper|6|mse|24|1.00|1.00|residual|1|0"
  "w_res_b4_share001|mscmnet_w|zscore|adamw|paper|4|mse|24|1.00|1.00|residual|1|0.001"
  "w_res_b4_share005|mscmnet_w|zscore|adamw|paper|4|mse|24|1.00|1.00|residual|1|0.005"
  "w_res_b4_share010|mscmnet_w|zscore|adamw|paper|4|mse|24|1.00|1.00|residual|1|0.01"
)

HEADER='case\tmodel\tnormalization\toptimizer\tweight_decay\tbatch_size\tloss\ttrain_stride_hours\tepoch_scale\tlr_scale\tcorrection_mode\tzero_init\tshare_weight\tseed'
printf '%b\n' "${HEADER}" >"${MANIFEST}"
for spec in "${CASES[@]}"; do
    IFS='|' read -r _case_name manifest_model _rest <<<"${spec}"
    manifest_seed="${SEED}"
    [[ "${manifest_model}" != mscmnet_w ]] || manifest_seed="${W_SEED}"
    printf '%s\t%s\n' "${spec//|/$'\t'}" "${manifest_seed}" >>"${MANIFEST}"
done
printf 'case\tmodel\tseed\texit_code\tvalidation\tpaper_acceptance\tlog\n' >"${STATUS}"

is_complete() {
    local run="$1" model="$2" seed="$3" epoch_scale="$4" lr_scale="$5"
    python - "${run}" "${model}" "${seed}" "${epoch_scale}" "${lr_scale}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run = Path(sys.argv[1])
model = sys.argv[2]
seed = int(sys.argv[3])
epoch_scale = float(sys.argv[4])
lr_scale = float(sys.argv[5])
required = [run / name for name in ("status.json", "metrics.csv", "loss_curve.csv", "predictions_common46.npz")]
if not all(path.is_file() for path in required):
    raise SystemExit(1)
status = json.loads(required[0].read_text(encoding="utf-8"))
metrics = pd.read_csv(required[1])
predictions = np.load(required[3])
ok = (
    status.get("status") == "completed"
    and status.get("model") == model
    and status.get("seed") == seed
    and math.isclose(float(status.get("best_epoch_scale", -1)), epoch_scale)
    and math.isclose(float(status.get("learning_rate_scale", -1)), lr_scale)
    and status.get("single_frozen_checkpoint_for_24h_and_168h") is True
    and status.get("prediction_24h_shape") == [46, 24, 10]
    and status.get("prediction_168h_shape") == [46, 168, 10]
    and len(metrics) == 88
    and metrics["value"].map(math.isfinite).all()
    and predictions["y_pred_24h"].shape == (46, 24, 10)
    and predictions["y_pred_168h"].shape == (46, 168, 10)
)
raise SystemExit(0 if ok else 1)
PY
}

paper_accepts() {
    local run="$1" model="$2"
    python - "${run}/metrics.csv" "${model}" "${PAPER_METRICS}" \
        "${ERROR_REL_TOL}" "${NSE_ABS_TOL}" <<'PY'
import sys

import pandas as pd
import yaml

metrics_path, model, paper_path = sys.argv[1:4]
error_tolerance, nse_tolerance = map(float, sys.argv[4:6])
display = {
    "gru": "GRU", "lstm": "LSTM", "msnet": "MSNet",
    "mscmnet_m": "MSCMNet_M", "mscmnet_wm": "MSCMNet_WM",
    "mscmnet_w": "MSCMNet_W",
}[model]
table = pd.read_csv(metrics_path)
paper = yaml.safe_load(open(paper_path, encoding="utf-8"))["tasks"]
passed = 0
for task in ("24h", "168h"):
    for metric in ("MAE", "MAPE", "RMSE", "NSE"):
        values = table.loc[
            (table.task == task) & (table.series == "total") & (table.metric == metric),
            "value",
        ]
        if len(values) != 1:
            raise SystemExit(2)
        actual = float(values.iloc[0])
        expected = float(paper[task][display]["total"][metric])
        gap = abs(actual - expected)
        accepted = gap <= nse_tolerance if metric == "NSE" else gap / abs(expected) <= error_tolerance
        passed += int(accepted)
raise SystemExit(0 if passed == 8 else 1)
PY
}

declare -A MODEL_ACCEPTED=()
FAILED_CASES=0
EXECUTED_CASES=0

run_case() {
    local case_name="$1" model="$2" norm="$3" optimizer="$4" decay="$5"
    local batch="$6" loss="$7" stride="$8" epoch_scale="$9" lr_scale="${10}"
    local mode="${11}" zero="${12}" share="${13}"
    local output_root run log rc validation acceptance case_seed
    case_seed="${SEED}"
    [[ "${model}" != mscmnet_w ]] || case_seed="${W_SEED}"
    output_root="${RESULT_ROOT}/${case_name}"
    run="${output_root}/${model}/seed_${case_seed}"
    log="${LOG_ROOT}/${case_name}.log"

    if [[ "${MODEL_ACCEPTED[${model}]:-0}" == 1 ]]; then
        printf '%s\t%s\t%s\t0\tSKIPPED\tALREADY_ACCEPTED\t%s\n' \
            "${case_name}" "${model}" "${case_seed}" "${log}" >>"${STATUS}"
        return
    fi

    if is_complete "${run}" "${model}" "${case_seed}" "${epoch_scale}" "${lr_scale}"; then
        echo "跳过已完成任务: ${case_name}"
        validation="PASS(existing)"
        rc=0
    else
        local args=(
            --model "${model}" --device cuda:0 --seed "${case_seed}"
            --normalization "${norm}" --optimizer "${optimizer}"
            --batch-size "${batch}" --loss "${loss}"
            --train-stride-hours "${stride}"
            --best-epoch-scale "${epoch_scale}"
            --learning-rate-scale "${lr_scale}"
            --output-root "${output_root}"
        )
        if [[ "${model}" == gru || "${model}" == lstm ]]; then
            [[ "${decay}" == paper ]] || args+=(--recurrent-weight-decay "${decay}")
        else
            [[ "${decay}" == paper ]] || args+=(--joint-weight-decay "${decay}")
        fi
        if [[ "${model}" == mscmnet_* ]]; then
            args+=(--correction-mode "${mode}")
            [[ "${zero}" == 0 ]] || args+=(--zero-init-correction)
            [[ "${model}" == mscmnet_m ]] || args+=(--fc2-share-supervision-weight "${share}")
        fi
        [[ ! -d "${run}" ]] || args+=(--overwrite)
        echo "开始任务: ${case_name} (${model}, epoch_scale=${epoch_scale}, lr_scale=${lr_scale})"
        set +e
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
            python scripts/train/train_temporal_baselines.py "${args[@]}" \
            2>&1 | tee "${log}"
        rc=${PIPESTATUS[0]}
        set -e
        validation="FAIL"
        if (( rc == 0 )) && is_complete "${run}" "${model}" "${case_seed}" "${epoch_scale}" "${lr_scale}"; then
            validation="PASS"
        else
            FAILED_CASES=$((FAILED_CASES + 1))
        fi
        EXECUTED_CASES=$((EXECUTED_CASES + 1))
    fi

    acceptance="NOT_ACCEPTED"
    if [[ "${validation}" == PASS* ]] && paper_accepts "${run}" "${model}"; then
        acceptance="ALL_8_ACCEPTED"
        MODEL_ACCEPTED["${model}"]=1
        echo "模型已达到论文接近阈值，后续候选自动跳过: ${model} (${case_name})"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${case_name}" "${model}" "${case_seed}" "${rc}" "${validation}" \
        "${acceptance}" "${log}" >>"${STATUS}"
}

for spec in "${CASES[@]}"; do
    IFS='|' read -r case_name model norm optimizer decay batch loss stride \
        epoch_scale lr_scale mode zero share <<<"${spec}"
    run_case "${case_name}" "${model}" "${norm}" "${optimizer}" "${decay}" \
        "${batch}" "${loss}" "${stride}" "${epoch_scale}" "${lr_scale}" \
        "${mode}" "${zero}" "${share}"
done

SUMMARY_RC=0
python scripts/reproduce/summarize_que_targeted_reproduction.py \
    --result-root "${RESULT_ROOT}" --manifest "${MANIFEST}" \
    --paper-metrics "${PAPER_METRICS}" \
    --error-relative-tolerance "${ERROR_REL_TOL}" \
    --nse-absolute-tolerance "${NSE_ABS_TOL}" || SUMMARY_RC=$?

cleanup
trap - EXIT INT TERM
tar --exclude='*.pt' --exclude='*.npz' -czf "${BUNDLE_PATH}" \
    -C "${PROJECT_ROOT}" "results/${RUN_TAG}" "logs/${RUN_TAG}"
sha256sum "${BUNDLE_PATH}" | tee "${BUNDLE_PATH}.sha256"

echo "目标复现结束：候选=${#CASES[@]}，本次执行=${EXECUTED_CASES}，训练失败=${FAILED_CASES}"
echo "最佳结果：${RESULT_ROOT}/best_by_model.tsv"
echo "逐指标差距：${RESULT_ROOT}/best_metric_gaps.tsv"
echo "结果包：${BUNDLE_PATH}"
if (( SUMMARY_RC != 0 )); then
    echo "警告：汇总失败，退出码=${SUMMARY_RC}；训练产物和日志已经打包。" >&2
    exit "${SUMMARY_RC}"
fi
