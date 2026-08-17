#!/usr/bin/env bash
# ============================================================
# 独立仓库发布前全面验收（不重新训练）
# ============================================================
#
# 依次验证：
#   1. 源码 SHA；2. 环境与安装；3. 完整导入10组冻结工件；
#   4. 物理文件数量；5. 完整测试套件；6. checkpoint/协议/防泄漏；
#   7. 10组GPU复推理；8. 总体、消融、DMA、Day1-Day7、Pearson表图；
#   9. 旧HPO/SGDR执行代码隔离。
#
# 这条流程用于 checkpoint 复验证，不重训10个模型。从原始数据重新训练使用：
#   bash scripts/reproduce/train_from_scratch.sh --device cuda:0 --seeds 0
#
# 用法：
#   bash scripts/reproduce/validate_everything.sh \
#       /path/to/DMA-WDF --device cuda:0

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "用法：$0 OLD_DMA_WDF_ROOT [--device DEVICE]" >&2
    exit 2
fi

SOURCE_ROOT="$1"
shift
DEVICE="cuda:0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
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

STAMP="$(date +%Y%m%d-%H%M%S)"
CONTROL_DIR="results/release_validation/${STAMP}"
REEVALUATION_DIR="${CONTROL_DIR}/checkpoint_reevaluation"
mkdir -p "${CONTROL_DIR}"
printf '%s\n' "${CONTROL_DIR}" > results/release_validation/latest_run_dir.txt
printf '%s\n' "RUNNING" > "${CONTROL_DIR}/STATUS"
printf '%s\n' "STARTING" > "${CONTROL_DIR}/CURRENT"

on_error() {
    local code=$?
    printf '%s\n' "FAILED exit_code=${code}" > "${CONTROL_DIR}/STATUS"
    printf '%s\n' "失败阶段：$(cat "${CONTROL_DIR}/CURRENT")" >&2
    exit "${code}"
}
trap on_error ERR

step() {
    local number="$1"
    shift
    printf '%s\n' "$*" > "${CONTROL_DIR}/CURRENT"
    echo
    echo "============================================================"
    echo "[${number}/9] $*"
    echo "============================================================"
}

step 1 "源码SHA-256完整性"
bash scripts/reproduce/verify_source.sh

step 2 "环境、可导入依赖与本地安装"
python scripts/reproduce/check_environment.py
python -m pip install -e . --no-deps

step 3 "原子导入10组checkpoint、Test预测、数据和Pearson图"
python scripts/reproduce/import_local_artifacts.py "${SOURCE_ROOT}"

step 4 "冻结工件物理数量与目录布局"
python scripts/reproduce/audit_release_inventory.py

step 5 "源码、配置、协议、防泄漏和模型测试"
bash scripts/reproduce/smoke_test.sh

step 6 "checkpoint哈希、元数据、common-46和论文层级"
python scripts/reproduce/verify_paper_release.py

step 7 "10组checkpoint重新执行common-46推理"
python scripts/reproduce/verify_paper_release.py \
    --re-evaluate \
    --device "${DEVICE}" \
    --reevaluation-absolute-tolerance 5e-4 \
    --reevaluation-relative-tolerance 5e-4 \
    --verification-output "${REEVALUATION_DIR}"

step 8 "生成并审计论文总体、消融、DMA、Day1-Day7与Pearson表图"
python scripts/reproduce/build_paper_tables.py \
    --input results/paper/frozen_v1 \
    --output paper/tables/literature \
    --frozen-layout
python scripts/reproduce/build_detailed_test_artifacts.py
python scripts/reproduce/audit_release_inventory.py \
    --require-paper-artifacts \
    --require-reevaluation "${REEVALUATION_DIR}"

step 9 "公开仓库旧HPO/SGDR执行代码隔离"
python - <<'PY'
from pathlib import Path
import re

roots = (
    Path("src"),
    Path("configs"),
    Path("tests"),
    Path("scripts/innovation"),
    Path("scripts/train"),
    Path("scripts/evaluate"),
)
pattern = re.compile(
    r"StateGuidedDailyRetrieval|use_sgdr|run_star_hparam|candidate_matrix"
)
matches = []
for root in roots:
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, start=1):
            if pattern.search(line):
                matches.append(f"{path}:{number}:{line.strip()}")
if matches:
    raise SystemExit(
        "公开源码仍含旧SGDR/HPO执行引用：\n" + "\n".join(matches)
    )
print("旧HPO/SGDR执行代码隔离：PASS（Python逐文件扫描）")
PY

cat > "${CONTROL_DIR}/FINAL_REPORT.txt" <<EOF
STaR-GNN-BWDF 全面验收：PASS
时间：$(date --iso-8601=seconds)
源码SHA：PASS
测试套件：PASS
冻结checkpoint：10/10
冻结predictions：10/10
冻结test_summary：10/10
common-46注册指标与层级：PASS
checkpoint重新推理：10/10
checkpoint复推理指标审计：40/40（绝对容差5e-4，相对容差5e-4）
总体/消融/DMA/Day1-Day7/Pearson表图：PASS
旧HPO/SGDR执行代码隔离：PASS
复推理目录：${PROJECT_ROOT}/${REEVALUATION_DIR}
论文工件目录：${PROJECT_ROOT}/paper
EOF

printf '%s\n' "DONE" > "${CONTROL_DIR}/CURRENT"
printf '%s\n' "SUCCESS" > "${CONTROL_DIR}/STATUS"
trap - ERR

echo
cat "${CONTROL_DIR}/FINAL_REPORT.txt"
