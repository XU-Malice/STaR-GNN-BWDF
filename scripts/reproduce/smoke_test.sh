#!/usr/bin/env bash
set -euo pipefail

SOURCE_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-only)
      SOURCE_ONLY=true
      shift
      ;;
    --help|-h)
      echo "用法：bash scripts/reproduce/smoke_test.sh [--source-only]"
      echo "  --source-only  跳过4项必须读取预处理样本索引文件的测试"
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

python scripts/reproduce/check_environment.py

python -m py_compile \
  scripts/reproduce/*.py \
  scripts/innovation/train_star_dcrnn.py \
  scripts/innovation/evaluate_star_dcrnn.py \
  scripts/train/train_model.py \
  scripts/evaluate/test_model.py

bash -n scripts/reproduce/*.sh scripts/data/*.sh scripts/graph/*.sh

python - <<'PY'
from pathlib import Path
import yaml

root = Path.cwd()
for path in sorted((root / "configs").rglob("*.yaml")):
    with path.open(encoding="utf-8") as handle:
        yaml.safe_load(handle)
print("YAML parse check: PASS")
PY

if [[ "${SOURCE_ONLY}" == "true" ]]; then
  # TestReadSampleIndexFiles 的4项检查必须等数据管道生成CSV后执行；
  # 这里只跳过这4项，test_evaluation_indices.py中的其余协议计算仍会运行。
  python -m pytest tests -q -k 'not TestReadSampleIndexFiles'
  echo "Repository source-only smoke test: PASS"
else
  python -m pytest tests -q
  echo "Repository full smoke test (including generated-data tests): PASS"
fi
