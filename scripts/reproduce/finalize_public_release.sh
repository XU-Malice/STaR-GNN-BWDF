#!/usr/bin/env bash
# ============================================================
# DCRNN/Base 唯一化后的公开仓库一键收口与验收（不重新训练）
# ============================================================
#
# 默认执行：
#   1. 删除已经合并进4份主文档的旧文档和两份重复 DCRNN 论文入口配置；
#   2. 保留 star_gnn/Base，删除冻结包中的 baselines/dcrnn；
#   3. 校验源码、环境和测试套件；
#   4. 用10个现有 checkpoint 重新执行 common-46 推理（40项指标）；
#   5. 重建总体/消融/DMA/Day1-Day7/Pearson 表图；
#   6. 审计 GitHub 发布边界并生成冻结 Release asset。
#
# 此脚本不调用任何训练入口。用法：
#   bash scripts/reproduce/finalize_public_release.sh --device cuda:0

set -euo pipefail

DEVICE="cuda:0"
PACKAGE=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-package)
            PACKAGE=false
            shift
            ;;
        --help|-h)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)
            echo "未知参数：$1" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

python - <<'PY'
from pathlib import Path
import tomllib

root = Path.cwd()
payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
if payload.get("project", {}).get("name") != "star-gnn-bwdf":
    raise SystemExit("ERROR：当前目录不是 STaR-GNN-BWDF，拒绝执行清理。")
print("项目身份：STaR-GNN-BWDF PASS")
PY

STAMP="$(date +%Y%m%d-%H%M%S)"
CONTROL_DIR="results/public_release_validation/${STAMP}"
REEVALUATION_DIR="${CONTROL_DIR}/checkpoint_reevaluation"
mkdir -p "${CONTROL_DIR}"
printf '%s\n' "${CONTROL_DIR}" > results/public_release_validation/latest_run_dir.txt
printf '%s\n' "RUNNING" > "${CONTROL_DIR}/STATUS"
printf '%s\n' "STARTING" > "${CONTROL_DIR}/CURRENT"

on_error() {
    local code=$?
    printf '%s\n' "FAILED exit_code=${code}" > "${CONTROL_DIR}/STATUS"
    echo "失败阶段：$(cat "${CONTROL_DIR}/CURRENT")" >&2
    exit "${code}"
}
trap on_error ERR

stage() {
    printf '%s\n' "$1" > "${CONTROL_DIR}/CURRENT"
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

stage "[1/8] 清理已合并文档与重复 DCRNN 论文入口"
rm -f \
    INSTALL_ON_SERVER_CN.md \
    configs/paper/dcrnn_24h.yaml \
    configs/paper/dcrnn_168h.yaml \
    docs/00_QUICKSTART_CN.md \
    docs/BASELINES_CN.md \
    docs/CHECKPOINT_VERIFICATION_CN.md \
    docs/CLEAN_ROOM_REPRODUCTION_CN.md \
    docs/CODE_FLOW_CN.md \
    docs/DATA.md \
    docs/EXPERIMENTS.md \
    docs/GITHUB_RELEASE_CHECKLIST_CN.md \
    docs/PAPER_ARTIFACTS_CN.md \
    docs/REPRODUCIBILITY.md \
    docs/RESULTS_PROVENANCE.md
echo "中文主文档收敛为4份：PASS"

stage "[2/8] 合并 DCRNN/Base 冻结工件（不训练）"
python scripts/reproduce/consolidate_dcrnn_base_release.py

stage "[3/8] 源码SHA、环境、语法与测试套件"
bash scripts/reproduce/verify_source.sh
python -m pip install -e . --no-deps
bash scripts/reproduce/smoke_test.sh --source-only

stage "[4/8] 10组冻结checkpoint、协议、指标与论文层级"
python scripts/reproduce/verify_paper_release.py

stage "[5/8] 10组checkpoint重新执行common-46推理"
python scripts/reproduce/verify_paper_release.py \
    --re-evaluate \
    --device "${DEVICE}" \
    --reevaluation-absolute-tolerance 5e-4 \
    --reevaluation-relative-tolerance 5e-4 \
    --verification-output "${REEVALUATION_DIR}"

stage "[6/8] 重建并审计论文表格和图件"
python scripts/reproduce/build_paper_tables.py \
    --input results/paper/frozen_v1 \
    --output paper/tables/literature \
    --frozen-layout
python scripts/reproduce/build_detailed_test_artifacts.py
python scripts/reproduce/audit_release_inventory.py \
    --require-paper-artifacts \
    --require-reevaluation "${REEVALUATION_DIR}"

stage "[7/8] 公开GitHub仓库结构、泄漏和大文件边界"
python scripts/reproduce/audit_public_repository.py \
    --require-frozen \
    --require-paper-artifacts \
    --output "${CONTROL_DIR}/repository_audit.json"

stage "[8/8] 生成GitHub Release checkpoint资产"
ASSET_STATUS="跳过"
if [[ "${PACKAGE}" == "true" ]]; then
    rm -f \
        dist/STaR-GNN-BWDF-frozen-v1.tar.gz \
        dist/STaR-GNN-BWDF-frozen-v1.tar.gz.sha256
    python scripts/reproduce/package_frozen_release.py
    ASSET_STATUS="PASS"
fi

cat > "${CONTROL_DIR}/FINAL_REPORT.txt" <<EOF
STaR-GNN-BWDF 公开发布收口验收：PASS
时间：$(date --iso-8601=seconds)
重新训练：未执行
DCRNN/Base唯一化：PASS
冻结checkpoint/predictions/test_summary：10/10/10
checkpoint common-46复推理：10/10
复推理指标：40/40（绝对与相对容差5e-4）
论文总体/消融/DMA/Day1-Day7/Pearson表图：PASS
中文主文档：4/4
公开仓库结构与发布边界：PASS
GitHub Release checkpoint资产：${ASSET_STATUS}
复推理目录：${PROJECT_ROOT}/${REEVALUATION_DIR}
论文工件目录：${PROJECT_ROOT}/paper
EOF

printf '%s\n' "DONE" > "${CONTROL_DIR}/CURRENT"
printf '%s\n' "SUCCESS" > "${CONTROL_DIR}/STATUS"
trap - ERR

echo
cat "${CONTROL_DIR}/FINAL_REPORT.txt"
