#!/usr/bin/env bash
# Consolidate completed historical results using a pinned tool outside the checkout.
set -Eeuo pipefail
trap 'rc=$?; echo "目录归档未完成：第 ${LINENO} 行，退出码=${rc}；请保留上方日志。" >&2; exit "$rc"' ERR
snapshot=${1:?需要完整工具提交号}
[[ "$snapshot" =~ ^[0-9a-f]{40}$ ]]
project_root=$(git rev-parse --show-toplevel)
cd "$project_root"
head_before=$(git rev-parse HEAD)
git cat-file -e "${snapshot}^{commit}"
parent="$HOME/projects/que_history_organization_tools"
tool_root="$parent/$snapshot"
mkdir -p "$parent" "$project_root/logs"
test ! -L "$parent"
case "$(realpath "$parent")/" in "$project_root/"*) echo '工具目录必须位于项目外部' >&2; exit 1;; esac
exec 9> "$project_root/logs/que_history_organization_install.lock"
flock -n 9 || { echo '已有目录归档安装在进行'; exit 0; }
python_path=$(conda run --no-capture-output -n bwdf311 python -c 'import sys; print(sys.executable)' | tail -n 1)
test -x "$python_path"
paths=(scripts/reproduce/organize_que_history.py scripts/reproduce/finalize_que_paper_baselines.py configs/evaluation/que_selected_total8_20260911.json docs/QUE_HISTORY_ORGANIZATION_CN.md tests/test_que_history_organization.py tests/test_que_paper_finalization.py)
for name in "${paths[@]}"; do git cat-file -e "$snapshot:$name"; done
if [[ ! -e "$tool_root" ]]; then
  temporary=$(mktemp -d "$parent/.export.XXXXXXXX")
  git archive "$snapshot" "${paths[@]}" | tar -x -C "$temporary"
  mv "$temporary" "$tool_root"
fi
test ! -L "$tool_root"
for name in "${paths[@]}"; do
  test -f "$tool_root/$name"
  test ! -L "$tool_root/$name"
  test "$(git rev-parse "$snapshot:$name")" = "$(git hash-object "$tool_root/$name")"
done
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
echo '核验目录归档工具并运行文件保留、移动及恢复检查。'
(cd "$tool_root"; "$python_path" -m pytest -q tests/test_que_history_organization.py tests/test_que_paper_finalization.py) \
  > "$project_root/logs/que_history_organization_preflight.log" 2>&1 || {
  tail -n 70 "$project_root/logs/que_history_organization_preflight.log"; exit 1;
}
test "$(git rev-parse HEAD)" = "$head_before"
if ! flock -n "$project_root/logs/que_history_organization.lock" true; then
  echo '已有目录归档流程在运行，本次不重复启动。'
  tail -n 15 "$project_root/logs/que_history_organization_launcher.log"
  exit 0
fi
nohup "$python_path" -u "$tool_root/scripts/reproduce/organize_que_history.py" \
  --project-root "$project_root" --execute \
  >> "$project_root/logs/que_history_organization_launcher.log" 2>&1 &
job_pid=$!
echo "$job_pid" > "$project_root/logs/que_history_organization_launcher.pid"
echo "历史目录归档 PID=$job_pid；复核固定模型后集中移动旧目录。"
echo "日志：$project_root/logs/que_history_organization_launcher.log"
sleep 2
if kill -0 "$job_pid" 2>/dev/null; then
  tail -n 15 "$project_root/logs/que_history_organization_launcher.log"
else
  wait "$job_pid" || { tail -n 50 "$project_root/logs/que_history_organization_launcher.log"; exit 1; }
  "$python_path" "$tool_root/scripts/reproduce/organize_que_history.py" --project-root "$project_root" --status
fi
