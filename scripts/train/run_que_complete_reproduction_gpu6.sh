#!/usr/bin/env bash
# Restartable, adaptive six-model reconstruction matrix for Que et al. (2024).

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_ID="${GPU_ID:-6}"
RUN_TAG="${RUN_TAG:-que_complete_reproduction_20260903}"
BASE_SEED="${SEED:-20240604}"
EXTRA_SEEDS="${EXTRA_SEEDS:-4}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-8192}"
RESULT_ROOT="${PROJECT_ROOT}/results/${RUN_TAG}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_TAG}"
BUNDLE_PATH="${BUNDLE_PATH:-${PROJECT_ROOT}/../${RUN_TAG}_compact.tar.gz}"
MANIFEST="${LOG_ROOT}/case_manifest.tsv"
STATUS="${LOG_ROOT}/case_status.tsv"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec 9>"${LOG_ROOT}/run.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "错误：同一完整复现队列已经在运行。" >&2
    exit 1
fi

GPU_ROW="$(nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits)"
GPU_FREE_MIB="$(printf '%s\n' "${GPU_ROW}" | awk -F',' '{gsub(/ /,"",$5); print $5}')"
echo "GPU 预检: ${GPU_ROW}"
if [[ ! "${GPU_FREE_MIB}" =~ ^[0-9]+$ ]] || (( GPU_FREE_MIB < MINIMUM_FREE_MIB )); then
    echo "错误：GPU ${GPU_ID} 可用显存不足 ${MINIMUM_FREE_MIB} MiB；请先停止旧队列。" >&2
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
sources = (Path(dma_wdf.__file__).resolve(), Path(inspect.getsourcefile(ConvAttentionBlock)).resolve())
for source in sources:
    if not source.is_relative_to(expected):
        raise SystemExit(f"Wrong dma_wdf source: {source}; expected below {expected}")
print({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "sources": [str(x) for x in sources]})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY
    nvidia-smi --id="${GPU_ID}"
} >"${LOG_ROOT}/environment_and_git.txt" 2>&1

nvidia-smi --id="${GPU_ID}" \
    --query-gpu=timestamp,index,temperature.gpu,power.draw,memory.used,memory.free,utilization.gpu \
    --format=csv -l 60 >"${LOG_ROOT}/gpu_monitor.csv" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "${MONITOR_PID}" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# case|candidate|phase|model|normalization|optimizer|decay|batch|loss|stride|epochs|mode|zero|share
# "paper" decay/epochs retain Supplementary Table S3 values.
CASES=(
  "gru_paper_aw_b8|gru_paper_aw_b8|screening|gru|zscore|adamw|paper|8|mse|24|paper|direct|0|0"
  "gru_paper_adam_b8|gru_paper_adam_b8|screening|gru|zscore|adam|paper|8|mse|24|paper|direct|0|0"
  "gru_adam0_b8|gru_adam0_b8|screening|gru|zscore|adam|0|8|mse|24|paper|direct|0|0"
  "gru_aw_b4|gru_aw_b4|screening|gru|zscore|adamw|paper|4|mse|24|paper|direct|0|0"
  "gru_aw_b16|gru_aw_b16|screening|gru|zscore|adamw|paper|16|mse|24|paper|direct|0|0"
  "gru_minmax_aw_b8|gru_minmax_aw_b8|screening|gru|minmax|adamw|paper|8|mse|24|paper|direct|0|0"
  "gru_stride6_aw_b8|gru_stride6_aw_b8|screening|gru|zscore|adamw|paper|8|mse|6|paper|direct|0|0"
  "gru_huber_aw_b8|gru_huber_aw_b8|screening|gru|zscore|adamw|paper|8|huber|24|paper|direct|0|0"
  "gru_e100_aw_b8|gru_e100_aw_b8|screening|gru|zscore|adamw|paper|8|mse|24|100|direct|0|0"
  "lstm_paper_aw_b8|lstm_paper_aw_b8|screening|lstm|zscore|adamw|paper|8|mse|24|paper|direct|0|0"
  "lstm_paper_adam_b8|lstm_paper_adam_b8|screening|lstm|zscore|adam|paper|8|mse|24|paper|direct|0|0"
  "lstm_adam0_b8|lstm_adam0_b8|screening|lstm|zscore|adam|0|8|mse|24|paper|direct|0|0"
  "lstm_aw_b4|lstm_aw_b4|screening|lstm|zscore|adamw|paper|4|mse|24|paper|direct|0|0"
  "lstm_aw_b16|lstm_aw_b16|screening|lstm|zscore|adamw|paper|16|mse|24|paper|direct|0|0"
  "lstm_minmax_aw_b8|lstm_minmax_aw_b8|screening|lstm|minmax|adamw|paper|8|mse|24|paper|direct|0|0"
  "lstm_stride6_aw_b8|lstm_stride6_aw_b8|screening|lstm|zscore|adamw|paper|8|mse|6|paper|direct|0|0"
  "lstm_huber_aw_b8|lstm_huber_aw_b8|screening|lstm|zscore|adamw|paper|8|huber|24|paper|direct|0|0"
  "lstm_e100_aw_b8|lstm_e100_aw_b8|screening|lstm|zscore|adamw|paper|8|mse|24|100|direct|0|0"
  "msnet_paper_aw_b8|msnet_paper_aw_b8|screening|msnet|zscore|adamw|paper|8|mse|24|paper|direct|0|0"
  "msnet_paper_adam_b8|msnet_paper_adam_b8|screening|msnet|zscore|adam|paper|8|mse|24|paper|direct|0|0"
  "msnet_adam0_b8|msnet_adam0_b8|screening|msnet|zscore|adam|0|8|mse|24|paper|direct|0|0"
  "msnet_aw_b4|msnet_aw_b4|screening|msnet|zscore|adamw|paper|4|mse|24|paper|direct|0|0"
  "msnet_aw_b16|msnet_aw_b16|screening|msnet|zscore|adamw|paper|16|mse|24|paper|direct|0|0"
  "msnet_minmax_aw_b8|msnet_minmax_aw_b8|screening|msnet|minmax|adamw|paper|8|mse|24|paper|direct|0|0"
  "msnet_stride6_aw_b8|msnet_stride6_aw_b8|screening|msnet|zscore|adamw|paper|8|mse|6|paper|direct|0|0"
  "msnet_huber_aw_b8|msnet_huber_aw_b8|screening|msnet|zscore|adamw|paper|8|huber|24|paper|direct|0|0"
  "msnet_mae_aw_b8|msnet_mae_aw_b8|screening|msnet|zscore|adamw|paper|8|mae|24|paper|direct|0|0"
  "m_direct_b8|m_direct_b8|screening|mscmnet_m|zscore|adamw|paper|8|mse|24|paper|direct|0|0"
  "m_direct_b4|m_direct_b4|screening|mscmnet_m|zscore|adamw|paper|4|mse|24|paper|direct|0|0"
  "m_res_b1|m_res_b1|screening|mscmnet_m|zscore|adamw|paper|1|mse|24|paper|residual|1|0"
  "m_res_b4|m_res_b4|screening|mscmnet_m|zscore|adamw|paper|4|mse|24|paper|residual|1|0"
  "m_res_b8|m_res_b8|screening|mscmnet_m|zscore|adamw|paper|8|mse|24|paper|residual|1|0"
  "m_res_adam0_b4|m_res_adam0_b4|screening|mscmnet_m|zscore|adam|0|4|mse|24|paper|residual|1|0"
  "m_res_minmax_b4|m_res_minmax_b4|screening|mscmnet_m|minmax|adamw|paper|4|mse|24|paper|residual|1|0"
  "m_res_stride6_b4|m_res_stride6_b4|screening|mscmnet_m|zscore|adamw|paper|4|mse|6|paper|residual|1|0"
  "m_res_huber_b4|m_res_huber_b4|screening|mscmnet_m|zscore|adamw|paper|4|huber|24|paper|residual|1|0"
  "m_res_e50_b4|m_res_e50_b4|screening|mscmnet_m|zscore|adamw|paper|4|mse|24|50|residual|1|0"
  "m_res_e100_b4|m_res_e100_b4|screening|mscmnet_m|zscore|adamw|paper|4|mse|24|100|residual|1|0"
  "wm_direct_b8|wm_direct_b8|screening|mscmnet_wm|zscore|adamw|paper|8|mse|24|paper|direct|0|0"
  "wm_res_b8|wm_res_b8|screening|mscmnet_wm|zscore|adamw|paper|8|mse|24|paper|residual|1|0"
  "wm_direct_share005|wm_direct_share005|screening|mscmnet_wm|zscore|adamw|paper|8|mse|24|paper|direct|0|0.05"
  "wm_res_share005|wm_res_share005|screening|mscmnet_wm|zscore|adamw|paper|8|mse|24|paper|residual|1|0.05"
  "wm_direct_b4|wm_direct_b4|screening|mscmnet_wm|zscore|adamw|paper|4|mse|24|paper|direct|0|0"
  "wm_direct_b16|wm_direct_b16|screening|mscmnet_wm|zscore|adamw|paper|16|mse|24|paper|direct|0|0"
  "wm_direct_adam0|wm_direct_adam0|screening|mscmnet_wm|zscore|adam|0|8|mse|24|paper|direct|0|0"
  "wm_direct_minmax|wm_direct_minmax|screening|mscmnet_wm|minmax|adamw|paper|8|mse|24|paper|direct|0|0"
  "wm_direct_stride6|wm_direct_stride6|screening|mscmnet_wm|zscore|adamw|paper|8|mse|6|paper|direct|0|0"
  "wm_direct_huber|wm_direct_huber|screening|mscmnet_wm|zscore|adamw|paper|8|huber|24|paper|direct|0|0"
  "wm_direct_e100|wm_direct_e100|screening|mscmnet_wm|zscore|adamw|paper|8|mse|24|100|direct|0|0"
  "w_direct_b8|w_direct_b8|screening|mscmnet_w|zscore|adamw|paper|8|mse|24|paper|direct|0|0"
  "w_direct_b1|w_direct_b1|screening|mscmnet_w|zscore|adamw|paper|1|mse|24|paper|direct|0|0"
  "w_res_b1|w_res_b1|screening|mscmnet_w|zscore|adamw|paper|1|mse|24|paper|residual|1|0"
  "w_res_b4|w_res_b4|screening|mscmnet_w|zscore|adamw|paper|4|mse|24|paper|residual|1|0"
  "w_res_b8|w_res_b8|screening|mscmnet_w|zscore|adamw|paper|8|mse|24|paper|residual|1|0"
  "w_res_share001|w_res_share001|screening|mscmnet_w|zscore|adamw|paper|1|mse|24|paper|residual|1|0.01"
  "w_res_share005|w_res_share005|screening|mscmnet_w|zscore|adamw|paper|1|mse|24|paper|residual|1|0.05"
  "w_res_share01|w_res_share01|screening|mscmnet_w|zscore|adamw|paper|1|mse|24|paper|residual|1|0.1"
  "w_res_adam0|w_res_adam0|screening|mscmnet_w|zscore|adam|0|1|mse|24|paper|residual|1|0"
  "w_res_minmax|w_res_minmax|screening|mscmnet_w|minmax|adamw|paper|1|mse|24|paper|residual|1|0"
  "w_res_stride6|w_res_stride6|screening|mscmnet_w|zscore|adamw|paper|1|mse|6|paper|residual|1|0"
  "w_res_huber|w_res_huber|screening|mscmnet_w|zscore|adamw|paper|1|huber|24|paper|residual|1|0"
  "w_res_e50|w_res_e50|screening|mscmnet_w|zscore|adamw|paper|1|mse|24|50|residual|1|0"
  "w_res_e100|w_res_e100|screening|mscmnet_w|zscore|adamw|paper|1|mse|24|100|residual|1|0"
)

HEADER='case\tcandidate\tphase\tmodel\tnormalization\toptimizer\tweight_decay\tbatch_size\tloss\ttrain_stride_hours\tepochs\tcorrection_mode\tzero_init\tshare_weight\tseed'
printf '%b\n' "${HEADER}" >"${MANIFEST}"
for spec in "${CASES[@]}"; do
    printf '%s\t%s\n' "${spec//|/$'\t'}" "${BASE_SEED}" >>"${MANIFEST}"
done
printf 'case\tphase\tmodel\tseed\texit_code\tvalidation\tlog\n' >"${STATUS}"

is_complete() {
    local run="$1" model="$2" seed="$3" stride="$4" loss="$5"
    python - "${run}" "${model}" "${seed}" "${stride}" "${loss}" <<'PY'
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

run, model, seed, stride, loss = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
required = [run / x for x in ("status.json", "metrics.csv", "loss_curve.csv", "predictions_common46.npz")]
if not all(x.is_file() for x in required): raise SystemExit(1)
status = json.loads(required[0].read_text(encoding="utf-8"))
metrics, curve, pred = pd.read_csv(required[1]), pd.read_csv(required[2]), np.load(required[3])
ok = (status.get("status") == "completed" and status.get("seed") == seed
      and status.get("model") == model and status.get("train_stride_hours") == stride
      and status.get("loss") == loss and status.get("single_frozen_checkpoint_for_24h_and_168h") is True
      and status.get("prediction_24h_shape") == [46, 24, 10]
      and status.get("prediction_168h_shape") == [46, 168, 10]
      and len(metrics) == 88 and metrics["value"].map(math.isfinite).all()
      and not curve.empty and pred["y_pred_24h"].shape == (46, 24, 10)
      and pred["y_pred_168h"].shape == (46, 168, 10))
raise SystemExit(0 if ok else 1)
PY
}

run_case() {
    local case_name="$1" candidate="$2" phase="$3" model="$4" norm="$5" optimizer="$6"
    local decay="$7" batch="$8" loss="$9" stride="${10}" epochs="${11}" mode="${12}"
    local zero="${13}" share="${14}" seed="${15}"
    local output_root="${RESULT_ROOT}/${case_name}" run="${output_root}/${model}/seed_${seed}"
    local log="${LOG_ROOT}/${case_name}.log"
    if is_complete "${run}" "${model}" "${seed}" "${stride}" "${loss}"; then
        echo "跳过已完成任务: ${case_name}"
        printf '%s\t%s\t%s\t%s\t0\tPASS(existing)\t%s\n' "${case_name}" "${phase}" "${model}" "${seed}" "${log}" >>"${STATUS}"
        return
    fi
    local args=(--model "${model}" --device cuda:0 --seed "${seed}" --normalization "${norm}"
      --optimizer "${optimizer}" --batch-size "${batch}" --loss "${loss}"
      --train-stride-hours "${stride}" --output-root "${output_root}")
    [[ "${epochs}" == paper ]] || args+=(--max-epochs "${epochs}")
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
    echo "开始任务: ${case_name} (${model}, seed=${seed})"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/train/train_temporal_baselines.py "${args[@]}" 2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
    set -e
    validation=FAIL
    if (( rc == 0 )) && is_complete "${run}" "${model}" "${seed}" "${stride}" "${loss}"; then validation=PASS; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${case_name}" "${phase}" "${model}" "${seed}" "${rc}" "${validation}" "${log}" >>"${STATUS}"
}

for spec in "${CASES[@]}"; do
    IFS='|' read -r case_name candidate phase model norm optimizer decay batch loss stride epochs mode zero share <<<"${spec}"
    run_case "${case_name}" "${candidate}" "${phase}" "${model}" "${norm}" "${optimizer}" "${decay}" "${batch}" "${loss}" "${stride}" "${epochs}" "${mode}" "${zero}" "${share}" "${BASE_SEED}"
done

python scripts/reproduce/summarize_que_complete_reproduction.py \
  --result-root "${RESULT_ROOT}" --manifest "${MANIFEST}" \
  --paper-metrics configs/evaluation/mscmnet_paper_metrics.yaml

python - "${RESULT_ROOT}/selected_candidates.tsv" "${LOG_ROOT}/selected_repeats.tsv" "${BASE_SEED}" "${EXTRA_SEEDS}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

selected, output, base, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
columns = ["case", "candidate", "phase", "model", "normalization", "optimizer", "weight_decay", "batch_size", "loss", "train_stride_hours", "epochs", "correction_mode", "zero_init", "share_weight", "seed"]
rows = []
for item in pd.read_csv(selected, sep="\t", dtype=str).fillna("").to_dict("records"):
    winner = item["case"]
    for offset in range(1, count + 1):
        seed = base + offset
        row = {key: item[key] for key in columns if key not in {"case", "candidate", "phase", "seed"}}
        row.update({"case": f"{winner}__seed_{seed}", "candidate": winner, "phase": "robustness", "seed": str(seed)})
        rows.append(row)
pd.DataFrame(rows, columns=columns).to_csv(output, sep="\t", index=False)
PY

tail -n +2 "${LOG_ROOT}/selected_repeats.tsv" >>"${MANIFEST}"
while IFS=$'\t' read -r case_name candidate phase model norm optimizer decay batch loss stride epochs mode zero share seed; do
    run_case "${case_name}" "${candidate}" "${phase}" "${model}" "${norm}" "${optimizer}" "${decay}" "${batch}" "${loss}" "${stride}" "${epochs}" "${mode}" "${zero}" "${share}" "${seed}"
done < <(tail -n +2 "${LOG_ROOT}/selected_repeats.tsv")

python scripts/reproduce/summarize_que_complete_reproduction.py \
  --result-root "${RESULT_ROOT}" --manifest "${MANIFEST}" \
  --paper-metrics configs/evaluation/mscmnet_paper_metrics.yaml

python - "${STATUS}" "${RESULT_ROOT}/selected_seed_robustness.tsv" <<'PY'
import sys
from pathlib import Path
import pandas as pd
status, robustness = pd.read_csv(sys.argv[1], sep="\t"), Path(sys.argv[2])
if not status["validation"].str.startswith("PASS").all(): raise SystemExit("One or more cases failed")
if status["model"].nunique() != 6: raise SystemExit("Not all six models were run")
if not robustness.is_file() or len(pd.read_csv(robustness, sep="\t")) != 6: raise SystemExit("Missing six-model robustness summary")
PY

cleanup
trap - EXIT INT TERM
tar --exclude='*.pt' --exclude='*.npz' -czf "${BUNDLE_PATH}" -C "${PROJECT_ROOT}" "results/${RUN_TAG}" "logs/${RUN_TAG}"
sha256sum "${BUNDLE_PATH}" | tee "${BUNDLE_PATH}.sha256"
echo "完整复现结束：筛选=${#CASES[@]}，追加种子=$((6 * EXTRA_SEEDS))，失败=0"
echo "论文差距：${RESULT_ROOT}/all_case_paper_gaps.tsv"
echo "入选配置：${RESULT_ROOT}/selected_candidates.tsv"
echo "五种子汇总：${RESULT_ROOT}/selected_seed_robustness.tsv"
echo "紧凑结果包：${BUNDLE_PATH}"
