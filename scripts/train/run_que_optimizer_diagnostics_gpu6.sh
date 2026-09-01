#!/usr/bin/env bash
# Diagnose the optimizer/weight-decay semantics behind MSNet weight collapse.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_ID="${GPU_ID:-6}"
RUN_TAG="${RUN_TAG:-que_optimizer_diagnostics_20260901}"
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
import inspect
import torch
import dma_wdf
from dma_wdf.models.mscmnet import ConvAttentionBlock
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
} >"${LOG_ROOT}/environment_and_git.txt" 2>&1

nvidia-smi --id="${GPU_ID}" \
    --query-gpu=timestamp,index,temperature.gpu,power.draw,memory.used,memory.free,utilization.gpu \
    --format=csv -l 60 >"${LOG_ROOT}/gpu_monitor.csv" 2>&1 &
MONITOR_PID=$!
cleanup() {
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

printf 'case\toptimizer\tweight_decay\tattention_update\texit_code\tvalidation\tlog\n' \
    >"${STATUS_TSV}"

is_complete() {
    local run_dir="$1"
    local expected_optimizer="$2"
    local expected_decay="$3"
    local expected_attention="$4"
    python - "${run_dir}" "${expected_optimizer}" "${expected_decay}" "${expected_attention}" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

run_dir = Path(sys.argv[1])
expected_optimizer = sys.argv[2]
expected_decay = float(sys.argv[3])
expected_attention = sys.argv[4]
paths = [run_dir / name for name in ("status.json", "metrics.csv", "resolved_config.yaml")]
if not all(path.is_file() for path in paths):
    raise SystemExit(1)
status = json.loads(paths[0].read_text(encoding="utf-8"))
metrics = pd.read_csv(paths[1])
config = yaml.safe_load(paths[2].read_text(encoding="utf-8"))
ok = (
    status.get("status") == "completed"
    and status.get("formal_protocol") is True
    and status.get("optimizer") == expected_optimizer
    and math.isclose(status.get("effective_weight_decay", -1.0), expected_decay)
    and len(metrics) == 88
    and metrics["value"].map(math.isfinite).all()
    and config["cam"].get("attention_update") == expected_attention
)
raise SystemExit(0 if ok else 1)
PY
}

CASES=(
    "replace_adam_wd0|adam|0|replace"
    "replace_adam_wd1e4|adam|0.0001|replace"
    "replace_adam_wd1e2|adam|0.01|replace"
    "replace_adamw_wd1e1|adamw|0.1|replace"
    "residual_adam_wd0|adam|0|residual"
    "skip_final_adam_wd0|adam|0|skip_final"
)

for item in "${CASES[@]}"; do
    IFS='|' read -r case_name optimizer decay attention <<<"${item}"
    output_root="${RESULT_ROOT}/${case_name}"
    run_dir="${output_root}/msnet/seed_${SEED}"
    log_path="${LOG_ROOT}/${case_name}.log"
    if is_complete "${run_dir}" "${optimizer}" "${decay}" "${attention}"; then
        echo "跳过已完成任务: ${case_name}"
        printf '%s\t%s\t%s\t%s\t0\tPASS(existing)\t%s\n' \
            "${case_name}" "${optimizer}" "${decay}" "${attention}" "${log_path}" \
            >>"${STATUS_TSV}"
        continue
    fi
    overwrite_args=()
    if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
        overwrite_args+=(--overwrite)
    fi
    echo "开始诊断: ${case_name} optimizer=${optimizer} decay=${decay} attention=${attention}"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train/train_temporal_baselines.py \
        --model msnet \
        --device cuda:0 \
        --seed "${SEED}" \
        --normalization zscore \
        --optimizer "${optimizer}" \
        --joint-weight-decay "${decay}" \
        --cam-attention-update "${attention}" \
        --output-root "${output_root}" \
        "${overwrite_args[@]}" \
        2>&1 | tee "${log_path}"
    train_rc=${PIPESTATUS[0]}
    set -e
    validation="FAIL"
    if (( train_rc == 0 )) && is_complete "${run_dir}" "${optimizer}" "${decay}" "${attention}"; then
        validation="PASS"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${case_name}" "${optimizer}" "${decay}" "${attention}" \
        "${train_rc}" "${validation}" "${log_path}" >>"${STATUS_TSV}"
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
print(
    "case\toptimizer\tweight_decay\tattention_update\ttask\tfinal_loss\t"
    "MAE\tMAPE\tRMSE\tNSE\tmean_dma_temporal_std\t"
    "conv_norm\tattention_norm\tlstm_norm\tjoint_weight_norm\tjoint_bias_norm"
)
for case_root in sorted(path for path in root.iterdir() if path.is_dir()):
    run = case_root / "msnet" / f"seed_{seed}"
    config = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run / "metrics.csv")
    final_loss = pd.read_csv(run / "loss_curve.csv")["train_loss"].iloc[-1]
    predictions = np.load(run / "predictions_common46.npz")
    checkpoint = torch.load(
        run / "checkpoint_msnet.pt", map_location="cpu", weights_only=False
    )
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
    for task, key in (("24h", "y_pred_24h"), ("168h", "y_pred_168h")):
        total = metrics[(metrics["task"] == task) & (metrics["series"] == "total")]
        values = dict(zip(total["metric"], total["value"]))
        temporal_std = predictions[key].reshape(-1, 10).std(axis=0).mean()
        print(
            f"{case_root.name}\t{config['training']['optimizer']}\t"
            f"{config['training']['joint_weight_decay_override']}\t"
            f"{config['cam']['attention_update']}\t{task}\t{final_loss:.8f}\t"
            f"{values['MAE']:.6f}\t{values['MAPE']:.6f}\t{values['RMSE']:.6f}\t"
            f"{values['NSE']:.6f}\t{temporal_std:.6f}\t{norms['conv']:.6f}\t"
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
echo "优化器诊断结束：失败任务=${failed}"
echo "摘要：${RESULT_ROOT}/diagnostic_summary.tsv"
echo "结果包：${BUNDLE_PATH}"
if (( failed > 0 )); then
    exit 1
fi
