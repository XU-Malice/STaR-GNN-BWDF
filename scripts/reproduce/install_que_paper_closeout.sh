#!/usr/bin/env bash
# Export this pinned CPU-only retention tool outside the frozen training checkout.
set -Eeuo pipefail
trap 'rc=$?; echo "收尾操作未完成：第 ${LINENO} 行，退出码=${rc}；请保留上方日志。" >&2; exit "$rc"' ERR
snapshot=${1:?需要完整工具提交号}
[[ "$snapshot" =~ ^[0-9a-f]{40}$ ]]
project_root=$(git rev-parse --show-toplevel)
cd "$project_root"
head_before=$(git rev-parse HEAD)
git cat-file -e "${snapshot}^{commit}"
parent="$HOME/projects/que_paper_closeout_tools"
tool_root="$parent/$snapshot"
mkdir -p "$parent" "$project_root/logs"
test ! -L "$parent"
case "$(realpath "$parent")/" in "$project_root/"*) echo '工具目录必须位于项目外部' >&2; exit 1;; esac
exec 9> "$project_root/logs/que_paper_closeout_install.lock"
flock -n 9 || { echo '已有收尾安装在进行'; exit 0; }
python_path=$(conda run --no-capture-output -n bwdf311 python -c 'import sys; print(sys.executable)' | tail -n 1)
test -x "$python_path"
paths=(scripts/reproduce/finalize_que_paper_baselines.py configs/evaluation/que_selected_total8_20260911.json docs/QUE_PAPER_BASELINES_20260911_CN.md tests/test_que_paper_finalization.py)
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
echo '收尾工具已核验，运行文件保留及清理回归检查。'
(cd "$tool_root"; "$python_path" -m pytest -q tests/test_que_paper_finalization.py) \
  > "$project_root/logs/que_paper_closeout_preflight.log" 2>&1 || {
  tail -n 70 "$project_root/logs/que_paper_closeout_preflight.log"; exit 1;
}
test "$(git rev-parse HEAD)" = "$head_before"
if ! flock -n "$project_root/logs/que_paper_closeout.lock" true; then
  echo '已有保存清理流程在运行，本次不重复启动。'
  tail -n 15 "$project_root/logs/que_paper_closeout_launcher.log"
  exit 0
fi
nohup "$python_path" -u "$tool_root/scripts/reproduce/finalize_que_paper_baselines.py" \
  --project-root "$project_root" --execute \
  >> "$project_root/logs/que_paper_closeout_launcher.log" 2>&1 &
job_pid=$!
echo "$job_pid" > "$project_root/logs/que_paper_closeout_launcher.pid"
echo "保存与清理流程 PID=$job_pid；先保存核验，再删除未选权重。"
echo "日志：$project_root/logs/que_paper_closeout_launcher.log"
sleep 2
if kill -0 "$job_pid" 2>/dev/null; then
  tail -n 15 "$project_root/logs/que_paper_closeout_launcher.log"
else
  wait "$job_pid" || { tail -n 50 "$project_root/logs/que_paper_closeout_launcher.log"; exit 1; }
  "$python_path" "$tool_root/scripts/reproduce/finalize_que_paper_baselines.py" --project-root "$project_root" --status
fi
