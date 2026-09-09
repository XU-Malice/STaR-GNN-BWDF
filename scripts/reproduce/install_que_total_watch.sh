#!/usr/bin/env bash
# Run with `git show PIN:scripts/reproduce/install_que_total_watch.sh | bash -s -- PIN`.
# Only git objects, external tools, logs and observer output are written.
set -Eeuo pipefail

snapshot=${1:?Provide the pinned tools commit}
if [[ ! "$snapshot" =~ ^[0-9a-f]{40}$ ]]; then
  echo "错误：需要完整的40位工具提交号。" >&2
  exit 2
fi
project_root=$(git rev-parse --show-toplevel)
cd "$project_root"
head_before=$(git rev-parse HEAD)
git cat-file -e "${snapshot}^{commit}"

tool_parent="${QUE_WATCH_TOOL_PARENT:-$HOME/projects/que_total_watch_tools}"
output_root="$project_root/results/que_total_match_watch_20260909"
launcher_log="$project_root/logs/que_total_match_watch_launcher.log"
tool_root="$tool_parent/$snapshot"
files=(
  scripts/reproduce/watch_que_total_match.py
  scripts/reproduce/que_total_watch_evidence.py
  scripts/reproduce/que_total_watch_process.py
)

# Resolve symlinks before creating anything: exported tools must be outside repo.
tool_parent=$(realpath -m "$tool_parent")
case "$tool_parent/" in
  "$project_root/"*) echo "错误：监测工具必须放在训练项目之外。" >&2; exit 2 ;;
esac
tool_root="$tool_parent/$snapshot"
mkdir -p "$tool_parent" "$output_root" "$project_root/logs"
exec 9> "$output_root/install.lock"
if ! flock -n 9; then
  echo "另一个监测安装流程正在进行，本次不重复安装或覆盖PID记录。"
  exit 0
fi

if [[ -n "${QUE_WATCH_PYTHON:-}" ]]; then
  watcher_python="$QUE_WATCH_PYTHON"
else
  watcher_python=$(conda run --no-capture-output -n bwdf311 python -c 'import sys; print(sys.executable)' | tail -n 1)
fi
test -x "$watcher_python"
"$watcher_python" -c 'import numpy, yaml; print("监测环境已就绪；仅执行CPU审计，不加载训练模型。")'

temporary=""
trap 'if [[ -n "$temporary" && -d "$temporary" ]]; then rm -rf -- "$temporary"; fi' EXIT
if [[ ! -e "$tool_root" ]]; then
  temporary=$(mktemp -d "$tool_parent/.install.XXXXXXXX")
  git archive "$snapshot" "${files[@]}" | tar -x -C "$temporary"
  for name in "${files[@]}"; do
    test -f "$temporary/$name"
  done
  mv "$temporary" "$tool_root"
  temporary=""
fi
test -d "$tool_root"
test ! -L "$tool_root"

# Verify exact contents even if a prior install exists; never overwrite live tools.
for name in "${files[@]}"; do
  expected_blob=$(git rev-parse "$snapshot:$name")
  test ! -L "$tool_root/$name"
  actual_blob=$(git hash-object "$tool_root/$name")
  test "$actual_blob" = "$expected_blob"
done
test "$(git rev-parse HEAD)" = "$head_before"
watcher="$tool_root/scripts/reproduce/watch_que_total_match.py"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

if ! flock -n "$output_root/watch.lock" true; then
  echo "已有总体监测在运行，本次不重复启动。"
  "$watcher_python" "$watcher" --project-root "$project_root" --status
  exit 0
fi

echo "核验现有结果与训练源码；保持当前提交 $head_before。"
"$watcher_python" "$watcher" --project-root "$project_root" --once

nohup "$watcher_python" -u "$watcher" --project-root "$project_root" --watch --stop-on-success 9>&- >> "$launcher_log" 2>&1 &
watch_pid=$!
echo "$watch_pid" > "$project_root/logs/que_total_match_watch_launcher.pid"
sleep 2
if kill -0 "$watch_pid" 2>/dev/null; then
  echo "总体监测已启动：PID=$watch_pid；每60秒检查，全部达标后保存结果。自动停止是否可用见下方状态。"
else
  echo "监测进程已经退出，请查看下面的状态及日志。"
  tail -n 20 "$launcher_log"
  "$watcher_python" "$watcher" --project-root "$project_root" --status
  "$watcher_python" - "$output_root/watch_status.json" <<'PY'
import json, sys
status = json.load(open(sys.argv[1]))['status']
if status not in {'matched_and_preserved', 'matched_queue_already_exited', 'matched_and_queue_stopped',
                  'matched_stop_requested_cleanup_pending', 'queue_ended_without_all_total_matches'}:
    raise SystemExit('监测没有成功进入运行或完成状态：' + status)
PY
fi
"$watcher_python" "$watcher" --project-root "$project_root" --status
echo "监测日志：$launcher_log"
echo "训练工作区提交仍为：$(git rev-parse HEAD)"
