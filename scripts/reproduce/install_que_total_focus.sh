#!/usr/bin/env bash
# git show COMMIT:scripts/reproduce/install_que_total_focus.sh | bash -s -- COMMIT
set -Eeuo pipefail
snapshot=${1:?Provide the full pinned tool commit}
[[ "$snapshot" =~ ^[0-9a-f]{40}$ ]] || { echo "需要完整40位工具提交号" >&2; exit 2; }
project_root=$(git rev-parse --show-toplevel)
cd "$project_root"
head_before=$(git rev-parse HEAD)
git cat-file -e "${snapshot}^{commit}"
tool_parent=$(realpath -m "${QUE_FOCUS_TOOL_PARENT:-$HOME/projects/que_total_focus_tools}")
case "$tool_parent/" in "$project_root/"*) echo "工具必须安装在训练项目外部" >&2; exit 2;; esac
tool_root="$tool_parent/$snapshot"
output_root="$project_root/results/que_total_focus_20260910"
mkdir -p "$tool_parent" "$output_root" "$project_root/logs"
exec 9> "$output_root/install.lock"
flock -n 9 || { echo "已有安装流程，本次不重复启动。"; exit 0; }
if [[ -n "${QUE_FOCUS_PYTHON:-}" ]]; then
  campaign_python="$QUE_FOCUS_PYTHON"
else
  campaign_python=$(conda run --no-capture-output -n bwdf311 python -c 'import sys; print(sys.executable)' | tail -n 1)
fi
test -x "$campaign_python"
files=(src scripts configs tests pyproject.toml uv.lock SOURCE_CHECKSUMS.sha256)
temporary=""
trap 'if [[ -n "$temporary" && -d "$temporary" ]]; then rm -rf -- "$temporary"; fi' EXIT
if [[ ! -e "$tool_root" ]]; then
  temporary=$(mktemp -d "$tool_parent/.install.XXXXXXXX")
  git archive "$snapshot" "${files[@]}" | tar -x -C "$temporary"
  mv "$temporary" "$tool_root"
  temporary=""
fi
test ! -L "$tool_root"
# Verify the complete exported source inventory, including existing installations.
"$campaign_python" - "$project_root" "$tool_root" "$snapshot" "${files[@]}" <<'PY'
import json, pathlib, subprocess, sys
project, tool, commit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
raw = subprocess.check_output(['git','ls-tree','-rz',commit,'--',*sys.argv[4:]], cwd=project)
expected = {}
for item in raw.split(b'\0'):
    if not item: continue
    meta, name = item.split(b'\t',1)
    mode, kind, sha = meta.decode().split()
    if kind != 'blob' or mode not in ('100644','100755'): raise SystemExit('Unsupported exported source type')
    rel = name.decode(); path = tool / rel
    if path.is_symlink() or not path.is_file(): raise SystemExit('Missing source: '+rel)
    actual = subprocess.check_output(['git','hash-object',str(path)],cwd=project,text=True).strip()
    if actual != sha: raise SystemExit('Exported source changed: '+rel)
    expected[rel] = sha
actual_names = {str(p.relative_to(tool)) for d in ('src','scripts','configs','tests') for p in (tool/d).rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
if actual_names != {n for n in expected if n.split('/')[0] in ('src','scripts','configs','tests')}:
    raise SystemExit('Unexpected exported source files')
deployment = {'commit':commit,'source_blobs':expected,'installation':'external_full_source_export'}
path = tool / 'deployment.json'
if path.exists() and json.loads(path.read_text()) != deployment: raise SystemExit('Deployment manifest changed')
path.write_text(json.dumps(deployment,indent=2)+'\n')
PY
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
campaign="$tool_root/scripts/reproduce/run_que_total_focus.py"
if ! flock -n "$output_root/campaign.lock" true; then
  echo "已有总体定向搜索在运行，本次不重复启动。"
  "$campaign_python" "$campaign" --project-root "$project_root" --status
  exit 0
fi
echo "执行新流程CPU测试与命令检查；训练项目提交保持 $head_before。"
(
  cd "$tool_root"
  PYTHONPATH="$tool_root/src" "$campaign_python" -m pytest -q \
    tests/test_que_total_recurrent_search.py tests/test_que_total_objective.py \
    tests/test_que_total_followup_training.py tests/test_que_total_focus.py
) > "$output_root/installation_preflight.log" 2>&1 || {
  tail -n 60 "$output_root/installation_preflight.log"
  echo "CPU预检未通过，未启动新流程。" >&2
  exit 1
}
test "$(git rev-parse HEAD)" = "$head_before"
nohup "$campaign_python" -u "$campaign" --project-root "$project_root" --watch \
  9>&- >> "$project_root/logs/que_total_focus_launcher.log" 2>&1 &
campaign_pid=$!
echo "$campaign_pid" > "$project_root/logs/que_total_focus_launcher.pid"
sleep 2
if ! kill -0 "$campaign_pid" 2>/dev/null; then
  tail -n 40 "$project_root/logs/que_total_focus_launcher.log"
  "$campaign_python" "$campaign" --project-root "$project_root" --status
  exit 1
fi
echo "总体定向搜索已启动：PID=$campaign_pid"
echo "先执行CPU核验/组合搜索；旧队列结束并释放锁后，自动追加至多48组GRU/LSTM训练。"
echo "日志：$project_root/logs/que_total_focus_launcher.log"
"$campaign_python" "$campaign" --project-root "$project_root" --status
