#!/usr/bin/env bash
# One restartable queue for checkpoint semantics, optimizer, stages and seed robustness.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_ID="${GPU_ID:-6}"
RUN_TAG="${RUN_TAG:-que_final_reproduction_20260903}"
BASE_SEED="${SEED:-20240604}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-8192}"
RESULT_ROOT="${PROJECT_ROOT}/results/${RUN_TAG}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_TAG}"
BUNDLE_PATH="${BUNDLE_PATH:-${PROJECT_ROOT}/../${RUN_TAG}_compact.tar.gz}"
STATUS_TSV="${LOG_ROOT}/diagnostic_status.tsv"
MANIFEST_TSV="${LOG_ROOT}/case_manifest.tsv"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec 9>"${LOG_ROOT}/run.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "错误：同一最终复现队列已经在运行。" >&2
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
sources = (Path(dma_wdf.__file__).resolve(), Path(inspect.getsourcefile(MSCMNetM)).resolve())
for source in sources:
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
    echo "错误：环境或源码导入预检失败。" >&2
    sed -n '1,240p' "${LOG_ROOT}/environment_and_git.txt" >&2
    exit 1
fi

nvidia-smi --id="${GPU_ID}" \
    --query-gpu=timestamp,index,temperature.gpu,power.draw,memory.used,memory.free,utilization.gpu \
    --format=csv -l 60 >"${LOG_ROOT}/gpu_monitor.csv" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "${MONITOR_PID}" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# case, candidate, phase, model, seed, batch, mode, zero, share, optimizer, decay, epochs, formal
CASES=(
    "m_res_b4_paper_adamw|m_res_b4|checkpoint_semantics|mscmnet_m|${BASE_SEED}|4|residual|1|0|adamw|0.01|6|1"
    "m_res_b4_e100_adamw|m_res_b4|checkpoint_semantics|mscmnet_m|${BASE_SEED}|4|residual|1|0|adamw|0.01|100|0"
    "m_res_b4_e100_adam0|m_res_b4|optimizer_semantics|mscmnet_m|${BASE_SEED}|4|residual|1|0|adam|0|100|0"
    "m_direct_b4_e100_adamw|m_direct_b4|correction_semantics|mscmnet_m|${BASE_SEED}|4|direct|0|0|adamw|0.01|100|0"
    "m_direct_b4_e100_adam0|m_direct_b4|optimizer_semantics|mscmnet_m|${BASE_SEED}|4|direct|0|0|adam|0|100|0"
    "w_res_b1_paper_adamw|w_res_b1|checkpoint_semantics|mscmnet_w|${BASE_SEED}|1|residual|1|0|adamw|0.01|11|1"
    "w_res_b1_e100_adamw|w_res_b1|checkpoint_semantics|mscmnet_w|${BASE_SEED}|1|residual|1|0|adamw|0.01|100|0"
    "w_res_b1_e100_adam0|w_res_b1|optimizer_semantics|mscmnet_w|${BASE_SEED}|1|residual|1|0|adam|0|100|0"
    "w_res_share001_b1_e100_adamw|w_res_share001_b1|share_semantics|mscmnet_w|${BASE_SEED}|1|residual|1|0.01|adamw|0.01|100|0"
    "w_direct_b1_e100_adamw|w_direct_b1|correction_semantics|mscmnet_w|${BASE_SEED}|1|direct|0|0|adamw|0.01|100|0"
    "wm_direct_b8_paper_adamw|wm_direct_b8|checkpoint_semantics|mscmnet_wm|${BASE_SEED}|8|direct|0|0|adamw|0.0001|55|1"
    "wm_direct_b8_e100_adamw|wm_direct_b8|checkpoint_semantics|mscmnet_wm|${BASE_SEED}|8|direct|0|0|adamw|0.0001|100|0"
    "wm_direct_share005_b8_e100_adamw|wm_direct_share005_b8|share_semantics|mscmnet_wm|${BASE_SEED}|8|direct|0|0.05|adamw|0.0001|100|0"
    "wm_res_share01_b8_e100_adamw|wm_res_share01_b8|correction_semantics|mscmnet_wm|${BASE_SEED}|8|residual|1|0.1|adamw|0.0001|100|0"
    "wm_direct_b8_e100_adam0|wm_direct_b8|optimizer_semantics|mscmnet_wm|${BASE_SEED}|8|direct|0|0|adam|0|100|0"
    "m_res_b4_seed2|m_res_b4|seed_robustness|mscmnet_m|$((BASE_SEED + 1))|4|residual|1|0|adamw|0.01|6|1"
    "m_res_b4_seed3|m_res_b4|seed_robustness|mscmnet_m|$((BASE_SEED + 2))|4|residual|1|0|adamw|0.01|6|1"
    "w_res_b1_seed2|w_res_b1|seed_robustness|mscmnet_w|$((BASE_SEED + 1))|1|residual|1|0|adamw|0.01|11|1"
    "w_res_b1_seed3|w_res_b1|seed_robustness|mscmnet_w|$((BASE_SEED + 2))|1|residual|1|0|adamw|0.01|11|1"
    "wm_direct_b8_seed2|wm_direct_b8|seed_robustness|mscmnet_wm|$((BASE_SEED + 1))|8|direct|0|0|adamw|0.0001|55|1"
    "wm_direct_b8_seed3|wm_direct_b8|seed_robustness|mscmnet_wm|$((BASE_SEED + 2))|8|direct|0|0|adamw|0.0001|55|1"
)

printf 'case\tcandidate\tphase\tmodel\tseed\tbatch_size\tcorrection_mode\tzero_init\tshare_weight\toptimizer\tweight_decay\tepochs\tformal_epoch\n' >"${MANIFEST_TSV}"
for spec in "${CASES[@]}"; do
    IFS='|' read -r case_name candidate phase model seed batch mode zero share optimizer decay epochs formal <<<"${spec}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${case_name}" "${candidate}" "${phase}" "${model}" "${seed}" "${batch}" \
        "${mode}" "${zero}" "${share}" "${optimizer}" "${decay}" "${epochs}" "${formal}" \
        >>"${MANIFEST_TSV}"
done

printf 'case\tmodel\tseed\tepochs\texit_code\tvalidation\tlog\n' >"${STATUS_TSV}"

is_complete() {
    local run_dir="$1" model="$2" seed="$3" epochs="$4" mode="$5" zero="$6" share="$7" optimizer="$8" decay="$9" formal="${10}"
    python - "${run_dir}" "${model}" "${seed}" "${epochs}" "${mode}" "${zero}" "${share}" "${optimizer}" "${decay}" "${formal}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

run = Path(sys.argv[1])
model, seed, epochs, mode = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
zero, share, optimizer, decay, formal = bool(int(sys.argv[6])), float(sys.argv[7]), sys.argv[8], float(sys.argv[9]), bool(int(sys.argv[10]))
names = (
    "status.json", "metrics.csv", "loss_curve.csv", "resolved_config.yaml",
    "predictions_common46.npz", "stage_predictions_common46.npz", "train_fit_stages_24h.npz",
)
required = [run / name for name in names]
if not all(path.is_file() for path in required):
    raise SystemExit(1)
status = json.loads(required[0].read_text(encoding="utf-8"))
metrics = pd.read_csv(required[1])
loss = pd.read_csv(required[2])
config = yaml.safe_load(required[3].read_text(encoding="utf-8"))
pred = np.load(required[4])
stages = np.load(required[5])
train = np.load(required[6])
mc = config["model"]
expected_share = share if model in {"mscmnet_w", "mscmnet_wm"} else 0.0
stage_keys = {"prediction_24h", "prediction_168h", "msnet_prediction_24h", "msnet_prediction_168h"}
if model in {"mscmnet_m", "mscmnet_w", "mscmnet_wm"}:
    stage_keys.update({"fc1_prediction_24h", "fc1_prediction_168h"})
if model in {"mscmnet_w", "mscmnet_wm"}:
    stage_keys.update({"predicted_daily_share_24h", "predicted_daily_share_168h"})
ok = (
    status.get("status") == "completed"
    and status.get("seed") == seed
    and status.get("formal_protocol") is formal
    and status.get("single_frozen_checkpoint_for_24h_and_168h") is True
    and status.get("stage_predictions_saved") is True
    and status.get("train_fit_stages_saved") is True
    and status.get("prediction_24h_shape") == [46, 24, 10]
    and status.get("prediction_168h_shape") == [46, 168, 10]
    and status.get("correction_mode") == mode
    and bool(status.get("zero_init_correction")) is zero
    and status.get("optimizer") == optimizer
    and math.isclose(float(status.get("effective_weight_decay")), decay)
    and math.isclose(float(status.get("fc2_share_supervision_weight")), expected_share)
    and len(loss) == epochs and int(loss["epoch"].iloc[-1]) == epochs
    and len(metrics) == 88 and metrics["value"].map(math.isfinite).all()
    and pred["y_pred_24h"].shape == (46, 24, 10)
    and pred["y_pred_168h"].shape == (46, 168, 10)
    and stage_keys.issubset(set(stages.files))
    and train["y_true_24h"].shape == (686, 24, 10)
    and train["prediction_24h"].shape == (686, 24, 10)
    and np.isfinite(stages["prediction_168h"]).all()
)
raise SystemExit(0 if ok else 1)
PY
}

for spec in "${CASES[@]}"; do
    IFS='|' read -r case_name candidate phase model seed batch mode zero share optimizer decay epochs formal <<<"${spec}"
    output_root="${RESULT_ROOT}/${case_name}"
    run_dir="${output_root}/${model}/seed_${seed}"
    log_path="${LOG_ROOT}/${case_name}.log"
    if is_complete "${run_dir}" "${model}" "${seed}" "${epochs}" "${mode}" "${zero}" "${share}" "${optimizer}" "${decay}" "${formal}"; then
        echo "跳过已完成任务: ${case_name}"
        printf '%s\t%s\t%s\t%s\t0\tPASS(existing)\t%s\n' "${case_name}" "${model}" "${seed}" "${epochs}" "${log_path}" >>"${STATUS_TSV}"
        continue
    fi
    overwrite_args=()
    if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then overwrite_args+=(--overwrite); fi
    zero_args=()
    if [[ "${zero}" == "1" ]]; then zero_args+=(--zero-init-correction); fi
    share_args=()
    if [[ "${model}" != "mscmnet_m" ]]; then share_args+=(--fc2-share-supervision-weight "${share}"); fi
    epoch_args=()
    if [[ "${formal}" == "0" ]]; then epoch_args+=(--max-epochs "${epochs}"); fi

    echo "开始任务: ${case_name} (${model}, seed=${seed}, epochs=${epochs})"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train/train_temporal_baselines.py \
        --model "${model}" --device cuda:0 --seed "${seed}" --normalization zscore \
        --optimizer "${optimizer}" --joint-weight-decay "${decay}" --batch-size "${batch}" \
        --cam-attention-update replace --cam-attention-scaling sqrt_dim \
        --cam-temporal-layout per_day_vectors --correction-mode "${mode}" \
        --output-root "${output_root}" "${zero_args[@]}" "${share_args[@]}" \
        "${epoch_args[@]}" "${overwrite_args[@]}" 2>&1 | tee "${log_path}"
    train_rc=${PIPESTATUS[0]}
    set -e
    validation="FAIL"
    if (( train_rc == 0 )) && is_complete "${run_dir}" "${model}" "${seed}" "${epochs}" "${mode}" "${zero}" "${share}" "${optimizer}" "${decay}" "${formal}"; then validation="PASS"; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${case_name}" "${model}" "${seed}" "${epochs}" "${train_rc}" "${validation}" "${log_path}" >>"${STATUS_TSV}"
done

set +e
python scripts/reproduce/summarize_que_final_reproduction.py \
    --result-root "${RESULT_ROOT}" --manifest "${MANIFEST_TSV}" \
    --paper-metrics configs/evaluation/mscmnet_paper_metrics.yaml \
    --paper-totals configs/evaluation/mscmnet_literature_totals.yaml
summary_rc=$?

python - "${RESULT_ROOT}" "${STATUS_TSV}" "${#CASES[@]}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root, status_path, expected = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
status = pd.read_csv(status_path, sep="\t")
if len(status) != expected or not status["validation"].str.startswith("PASS").all():
    raise SystemExit("Not every queued case passed validation")
expected_rows = expected * 2
for name in ("final_metric_summary.tsv", "stage_metric_summary.tsv", "train_only_calibration_summary.tsv"):
    table = pd.read_csv(root / name, sep="\t")
    if name == "final_metric_summary.tsv" and len(table) != expected_rows:
        raise SystemExit(f"Unexpected row count in {name}: {len(table)}")
    if table.empty:
        raise SystemExit(f"Empty summary: {name}")
PY
validation_rc=$?
set -e

cleanup
trap - EXIT INT TERM
tar --exclude='*.pt' -czf "${BUNDLE_PATH}" -C "${PROJECT_ROOT}" "results/${RUN_TAG}" "logs/${RUN_TAG}"
sha256sum "${BUNDLE_PATH}" | tee "${BUNDLE_PATH}.sha256"

failed="$(awk -F '\t' 'NR > 1 && $6 !~ /^PASS/ {count++} END {print count+0}' "${STATUS_TSV}")"
echo "最终复现队列结束：任务=${#CASES[@]}，失败=${failed}"
echo "汇总：${RESULT_ROOT}/final_metric_summary.tsv"
echo "阶段诊断：${RESULT_ROOT}/stage_metric_summary.tsv"
echo "训练集校准诊断：${RESULT_ROOT}/train_only_calibration_summary.tsv"
echo "紧凑结果包（不含 checkpoint）：${BUNDLE_PATH}"
if (( failed > 0 || summary_rc > 0 || validation_rc > 0 )); then exit 1; fi
