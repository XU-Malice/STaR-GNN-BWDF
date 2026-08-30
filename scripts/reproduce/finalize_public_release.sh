#!/usr/bin/env bash
# ============================================================
# STaR-GNN-BWDF 公开仓库一键收口与验收（不重新训练）
# ============================================================
#
# 执行：
#   1. 清理已弃用旧文档/重复 DCRNN 入口；
#   2. 保证 DCRNN/Base 冻结工件唯一；
#   3. 封存并验证当前 public-source SHA；
#   4. 校验 10 个冻结 checkpoint、协议与内部 aggregate 诊断；
#   5. 重新执行 10 组 common-46 推理；
#   6. 重建源表、submission tables、Main Fig. 1--7；
#   7. 审计 GitHub 发布边界；
#   8. 可选生成 GitHub Release checkpoint asset。
#
# 此脚本不调用训练入口。
#
# 用法：
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
            sed -n '2,26p' "$0"
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
    raise SystemExit("ERROR：当前目录不是 STaR-GNN-BWDF，拒绝执行收口。")
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

stage "[1/8] 清理弃用文档与重复 DCRNN 论文入口"
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
echo "弃用入口清理：PASS"

stage "[2/8] 合并 DCRNN/Base 冻结工件（不训练）"
python scripts/reproduce/consolidate_dcrnn_base_release.py

stage "[3/8] 封存源码 SHA、环境、语法与测试套件"
python scripts/reproduce/regenerate_source_checksums.py
bash scripts/reproduce/verify_source.sh
python -m pip install -e . --no-deps
bash scripts/reproduce/smoke_test.sh --source-only
python -m pytest tests/test_paper_release.py tests/test_paper_artifacts.py tests/test_submission_artifacts.py -q

stage "[4/8] 10组冻结 checkpoint、协议与内部 aggregate 诊断"
python scripts/reproduce/verify_paper_release.py

stage "[5/8] 10组 checkpoint 重新执行 common-46 推理"
python scripts/reproduce/verify_paper_release.py \
    --re-evaluate \
    --device "${DEVICE}" \
    --reevaluation-absolute-tolerance 5e-4 \
    --reevaluation-relative-tolerance 5e-4 \
    --verification-output "${REEVALUATION_DIR}"

stage "[6/8] 重建审计源表、submission tables 与 canonical figures"
python scripts/reproduce/build_paper_tables.py \
    --input results/paper/frozen_v1 \
    --output paper/tables/literature \
    --frozen-layout

# 保留 legacy aggregate-demand 诊断工件，但不再作为投稿图权威入口。
python scripts/reproduce/build_detailed_test_artifacts.py

python scripts/reproduce/render_submission_tables.py \
    --source-dir paper/tables/literature \
    --output-dir paper/tables/submission \
    --release results/paper/frozen_v1

python scripts/reproduce/render_submission_figures.py \
    --release results/paper/frozen_v1 \
    --overall-table paper/tables/literature/table_literature_comparison_common46.csv \
    --dma-table paper/tables/literature/table_all_models_dma.csv \
    --main-output paper/figures/submission \
    --audit-output paper/tables/manuscript/submission \
    --block-length 7 \
    --bootstrap-iterations 50000 \
    --bootstrap-seed 20260821

python scripts/reproduce/audit_release_inventory.py \
    --require-paper-artifacts \
    --require-reevaluation "${REEVALUATION_DIR}"

stage "[7/8] 公开 GitHub 结构、指标口径、submission 图表与大文件边界"
python scripts/reproduce/audit_public_repository.py \
    --require-frozen \
    --require-paper-artifacts \
    --output "${CONTROL_DIR}/repository_audit.json"

stage "[8/8] 生成 GitHub Release checkpoint 资产"
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
DCRNN/Base 唯一化：PASS
冻结 checkpoint/predictions/test_summary：10/10/10
checkpoint common-46 复推理：10/10
复推理指标：40/40（绝对与相对容差5e-4）
Main Table 1 overall：PASS
Main Table 2 factorial ablation：4 models / no STGCN / 30/32 PASS
Supplementary Tables S1--S3：PASS
Main Fig. 1--7：PASS
7-origin moving-block bootstrap：PASS
公开文档与发布边界：PASS
GitHub Release checkpoint资产：${ASSET_STATUS}
复推理目录：${PROJECT_ROOT}/${REEVALUATION_DIR}
论文工件目录：${PROJECT_ROOT}/paper
EOF

printf '%s\n' "DONE" > "${CONTROL_DIR}/CURRENT"
printf '%s\n' "SUCCESS" > "${CONTROL_DIR}/STATUS"
trap - ERR

echo
cat "${CONTROL_DIR}/FINAL_REPORT.txt"
