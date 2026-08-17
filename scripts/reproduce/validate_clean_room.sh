#!/usr/bin/env bash
# ============================================================
# GitHub发布前：全新Conda环境 + 全新源码副本 + 10组从零训练验收
# ============================================================
#
# 该脚本模拟外部研究者的真实流程：
#   1. 只复制公开源码清单，不复制现有数据、图或训练结果；
#   2. 单独装入冻结checkpoint Release asset并校验；
#   3. 创建全新Conda prefix并安装精确依赖；
#   4. 执行不依赖生成数据的源码测试和冻结checkpoint结构验证；
#   5. 从wf4bwdf原始数据重新预处理并构图；
#   6. 数据生成后运行包含样本索引检查的完整测试套件；
#   7. 训练10组模型，全部完成后才读取Test并执行common-46；
#   8. 对照冻结指标、层级、DMA/逐日/Pearson表图；
#   9. 用新生成的数据和图重新推理10个冻结checkpoint。
#
# 默认不会覆盖任何已有目录，也不会删除创建的环境和结果。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
WORKSPACE="${PROJECT_ROOT}/../STaR-GNN-BWDF-cleanroom-${STAMP}"
ENV_PREFIX=""
FROZEN_RELEASE="${PROJECT_ROOT}/results/paper/frozen_v1"
DEVICE="cuda:0"
EVALUATION_DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --env-prefix)
            ENV_PREFIX="$2"
            shift 2
            ;;
        --frozen-release)
            FROZEN_RELEASE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --evaluation-device)
            EVALUATION_DEVICE="$2"
            shift 2
            ;;
        *)
            echo "未知参数：$1" >&2
            exit 2
            ;;
    esac
done

WORKSPACE="$(realpath -m "${WORKSPACE}")"
if [[ -z "${ENV_PREFIX}" ]]; then
    ENV_PREFIX="${WORKSPACE}/.conda-env"
else
    ENV_PREFIX="$(realpath -m "${ENV_PREFIX}")"
fi
FROZEN_RELEASE="$(realpath -m "${FROZEN_RELEASE}")"

CONTROL_ROOT="${PROJECT_ROOT}/results/cleanroom_validation"
CONTROL_DIR="${CONTROL_ROOT}/${STAMP}"
mkdir -p "${CONTROL_DIR}"
printf '%s\n' "${CONTROL_DIR}" > "${CONTROL_ROOT}/latest_run_dir.txt"
printf '%s\n' "RUNNING" > "${CONTROL_DIR}/STATUS"
printf '%s\n' "STARTING" > "${CONTROL_DIR}/CURRENT"
printf '%s\n' "${WORKSPACE}" > "${CONTROL_DIR}/WORKSPACE"
printf '%s\n' "${ENV_PREFIX}" > "${CONTROL_DIR}/ENV_PREFIX"

on_error() {
    local code=$?
    printf '%s\n' "FAILED exit_code=${code}" > "${CONTROL_DIR}/STATUS"
    printf '%s\n' "失败阶段：$(cat "${CONTROL_DIR}/CURRENT")" >&2
    exit "${code}"
}
trap on_error ERR

stage() {
    local text="$1"
    printf '%s\n' "${text}" > "${CONTROL_DIR}/CURRENT"
    echo
    echo "============================================================"
    echo "${text}"
    echo "============================================================"
}

run_clean() {
    conda run --no-capture-output --prefix "${ENV_PREFIX}" "$@"
}

test ! -e "${WORKSPACE}" || {
    echo "clean-room工作目录已存在，拒绝覆盖：${WORKSPACE}" >&2
    exit 1
}
test ! -e "${ENV_PREFIX}" || {
    echo "clean-room Conda prefix已存在，拒绝覆盖：${ENV_PREFIX}" >&2
    exit 1
}
command -v conda >/dev/null || {
    echo "未找到conda；请先source conda.sh。" >&2
    exit 1
}

stage "[1/11] 创建纯净公开源码副本并装入冻结Release asset"
python "${PROJECT_ROOT}/scripts/reproduce/prepare_clean_room.py" \
    --source "${PROJECT_ROOT}" \
    --destination "${WORKSPACE}" \
    --frozen-release "${FROZEN_RELEASE}"

stage "[2/11] 从environment.yml创建全新Conda环境"
cd "${WORKSPACE}"
conda env create \
    --prefix "${ENV_PREFIX}" \
    --file environment.yml
conda env export --prefix "${ENV_PREFIX}" --no-builds \
    > "${CONTROL_DIR}/environment_export.yml"
conda list --prefix "${ENV_PREFIX}" --explicit \
    > "${CONTROL_DIR}/conda_explicit.txt"

stage "[3/11] 全新环境的源码SHA、依赖和非数据工件测试"
run_clean bash scripts/reproduce/verify_source.sh
run_clean python scripts/reproduce/check_environment.py \
    > "${CONTROL_DIR}/environment_report.txt"
run_clean bash scripts/reproduce/smoke_test.sh --source-only

stage "[4/11] 冻结checkpoint离线结构、SHA、协议和指标验证"
run_clean python scripts/reproduce/verify_paper_release.py

stage "[5/11] 从原始BWDF数据预处理并仅用训练期重建Pearson图"
run_clean bash scripts/data/run_pipeline.sh
run_clean bash scripts/graph/run_graph_pipeline.sh

stage "[6/11] 处理数据生成后运行包含样本索引的完整测试套件"
run_clean bash scripts/reproduce/smoke_test.sh

stage "[7/11] 使用冻结论文参数从零训练并评价10组模型"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
run_clean python scripts/reproduce/reproduce.py \
    --skip-data \
    --device "${DEVICE}" \
    --evaluation-device "${EVALUATION_DEVICE}" \
    --seeds 0 \
    --output "${WORKSPACE}/results/paper/reproduction" \
    --control-dir "${CONTROL_DIR}"

stage "[8/11] 审计10组从零训练、Test、防泄漏和论文层级"
run_clean python scripts/reproduce/audit_from_scratch.py \
    --reproduction "${WORKSPACE}/results/paper/reproduction" \
    --frozen "${WORKSPACE}/results/paper/frozen_v1" \
    --output "${CONTROL_DIR}/from_scratch_audit"

stage "[9/11] 用从零生成的数据和图复推理10个冻结checkpoint"
run_clean python scripts/reproduce/verify_paper_release.py \
    --re-evaluate \
    --device "${EVALUATION_DEVICE}" \
    --reevaluation-absolute-tolerance 5e-4 \
    --reevaluation-relative-tolerance 5e-4 \
    --verification-output \
    "${WORKSPACE}/results/cleanroom_checkpoint_reevaluation"

stage "[10/11] 从冻结预测生成并审计论文表格和图件"
run_clean python scripts/reproduce/build_paper_tables.py \
    --input "${WORKSPACE}/results/paper/frozen_v1" \
    --output "${WORKSPACE}/paper/tables/literature" \
    --frozen-layout
run_clean python scripts/reproduce/build_detailed_test_artifacts.py \
    --release "${WORKSPACE}/results/paper/frozen_v1" \
    --output "${WORKSPACE}/paper"

run_clean python scripts/reproduce/audit_release_inventory.py \
    --release "${WORKSPACE}/results/paper/frozen_v1" \
    --require-paper-artifacts \
    --require-reevaluation \
    "${WORKSPACE}/results/cleanroom_checkpoint_reevaluation"

stage "[11/11] 复核公开源码未被运行过程修改"
run_clean bash scripts/reproduce/verify_source.sh

cat > "${CONTROL_DIR}/FINAL_REPORT.txt" <<EOF
STaR-GNN-BWDF clean-room从零复现：PASS
完成时间：$(date --iso-8601=seconds)
纯净工作目录：${WORKSPACE}
全新Conda环境：${ENV_PREFIX}
源码SHA与非数据工件测试：PASS
数据生成后的完整测试套件：PASS
冻结checkpoint离线验证：10/10
原始数据预处理：PASS
训练期Pearson图：PASS
从零训练：10/10
从零common-46 Test：10/10
从零指标与冻结结果对照：40/40
从零论文层级：PASS
冻结checkpoint基于新数据复推理：40/40
总体/消融/DMA/Day1-Day7/Pearson表图：PASS
Test参与训练或选参：否
EOF

printf '%s\n' "DONE" > "${CONTROL_DIR}/CURRENT"
printf '%s\n' "SUCCESS" > "${CONTROL_DIR}/STATUS"
trap - ERR

cat "${CONTROL_DIR}/FINAL_REPORT.txt"
