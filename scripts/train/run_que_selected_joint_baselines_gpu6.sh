#!/usr/bin/env bash
# Run the evidence-selected Que et al. joint temporal baselines on one GPU.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_ID="${GPU_ID:-6}"
RUN_TAG="${RUN_TAG:-que_selected_joint_baselines_20260901}"
SEED="${SEED:-20240604}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-8192}"
RESULT_ROOT="${PROJECT_ROOT}/results/${RUN_TAG}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_TAG}"
BUNDLE_PATH="${BUNDLE_PATH:-${PROJECT_ROOT}/../${RUN_TAG}.tar.gz}"
STATUS_TSV="${LOG_ROOT}/run_status.tsv"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec 9>"${LOG_ROOT}/run.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "错误：同一结果目录已有任务在运行。" >&2
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
from dma_wdf.models.mscmnet import ConvAttentionBlock

expected = (Path(os.environ["PROJECT_ROOT_PY"]) / "src").resolve()
for source in (Path(dma_wdf.__file__).resolve(), Path(inspect.getsourcefile(ConvAttentionBlock)).resolve()):
    if not source.is_relative_to(expected):
        raise SystemExit(f"Wrong dma_wdf source: {source}; expected below {expected}")
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
cleanup() { kill "${MONITOR_PID}" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

printf 'model\tepochs\tweight_decay\texit_code\tvalidation\tlog\n' >"${STATUS_TSV}"

is_complete() {
    local model="$1"
    local epochs="$2"
    local decay="$3"
    local run_dir="${RESULT_ROOT}/${model}/seed_${SEED}"
    python - "${run_dir}" "${epochs}" "${decay}" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

run = Path(sys.argv[1])
epochs, decay = int(sys.argv[2]), float(sys.argv[3])
required = [run / name for name in ("status.json", "metrics.csv", "loss_curve.csv", "resolved_config.yaml")]
if not all(path.is_file() for path in required):
    raise SystemExit(1)
status = json.loads(required[0].read_text(encoding="utf-8"))
metrics = pd.read_csv(required[1])
loss = pd.read_csv(required[2])
config = yaml.safe_load(required[3].read_text(encoding="utf-8"))
checkpoints = list(run.glob("checkpoint_*.pt"))
ok = (
    status.get("status") == "completed"
    and status.get("formal_protocol") is True
    and status.get("single_frozen_checkpoint_for_24h_and_168h") is True
    and status.get("optimizer") == "adamw"
    and math.isclose(float(status.get("effective_weight_decay", -1)), decay)
    and status.get("prediction_24h_shape") == [46, 24, 10]
    and status.get("prediction_168h_shape") == [46, 168, 10]
    and len(metrics) == 88
    and metrics["value"].map(math.isfinite).all()
    and len(loss) == epochs
    and len(checkpoints) == 1
    and config["cam"].get("attention_update") == "replace"
    and config["cam"].get("attention_scaling") == "sqrt_dim"
    and config["cam"].get("temporal_layout") == "per_day_vectors"
    and int(config["training"].get("batch_size")) == 8
)
raise SystemExit(0 if ok else 1)
PY
}

MODELS=(msnet mscmnet_m mscmnet_wm mscmnet_w)
EPOCHS=(99 6 55 11)
DECAYS=(0.1 0.01 0.0001 0.01)

for index in "${!MODELS[@]}"; do
    model="${MODELS[$index]}"
    epochs="${EPOCHS[$index]}"
    decay="${DECAYS[$index]}"
    log_path="${LOG_ROOT}/${model}.log"
    run_dir="${RESULT_ROOT}/${model}/seed_${SEED}"
    if is_complete "${model}" "${epochs}" "${decay}"; then
        echo "跳过已完成任务: ${model}"
        printf '%s\t%s\t%s\t0\tPASS(existing)\t%s\n' \
            "${model}" "${epochs}" "${decay}" "${log_path}" >>"${STATUS_TSV}"
        continue
    fi
    overwrite_args=()
    if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
        overwrite_args+=(--overwrite)
    fi
    echo "开始正式任务: model=${model}, epochs=${epochs}, weight_decay=${decay}"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train/train_temporal_baselines.py \
        --model "${model}" \
        --device cuda:0 \
        --seed "${SEED}" \
        --normalization zscore \
        --optimizer adamw \
        --joint-weight-decay "${decay}" \
        --cam-attention-update replace \
        --cam-attention-scaling sqrt_dim \
        --cam-temporal-layout per_day_vectors \
        --batch-size 8 \
        --output-root "${RESULT_ROOT}" \
        "${overwrite_args[@]}" \
        2>&1 | tee "${log_path}"
    train_rc=${PIPESTATUS[0]}
    set -e
    validation="FAIL"
    if (( train_rc == 0 )) && is_complete "${model}" "${epochs}" "${decay}"; then
        validation="PASS"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${model}" "${epochs}" "${decay}" "${train_rc}" "${validation}" "${log_path}" \
        >>"${STATUS_TSV}"
done

python - "${RESULT_ROOT}" "${SEED}" >"${RESULT_ROOT}/total_metrics.tsv" <<'PY'
import sys
from pathlib import Path

import pandas as pd

root, seed = Path(sys.argv[1]), sys.argv[2]
print("model\ttask\tMAE\tMAPE\tRMSE\tNSE")
for model in ("msnet", "mscmnet_m", "mscmnet_wm", "mscmnet_w"):
    path = root / model / f"seed_{seed}" / "metrics.csv"
    if not path.is_file():
        continue
    table = pd.read_csv(path)
    for task in ("24h", "168h"):
        rows = table[(table["task"] == task) & (table["series"] == "total")]
        values = dict(zip(rows["metric"], rows["value"]))
        print(f"{model}\t{task}\t{values['MAE']:.6f}\t{values['MAPE']:.6f}\t{values['RMSE']:.6f}\t{values['NSE']:.6f}")
PY

cleanup
trap - EXIT INT TERM
tar -czf "${BUNDLE_PATH}" -C "${PROJECT_ROOT}" "results/${RUN_TAG}" "logs/${RUN_TAG}"
sha256sum "${BUNDLE_PATH}" | tee "${BUNDLE_PATH}.sha256"

failed="$(awk -F '\t' 'NR > 1 && $5 !~ /^PASS/ {count++} END {print count+0}' "${STATUS_TSV}")"
echo "联合基线结束：失败任务=${failed}"
echo "汇总：${RESULT_ROOT}/total_metrics.tsv"
echo "结果包：${BUNDLE_PATH}"
if (( failed > 0 )); then exit 1; fi
