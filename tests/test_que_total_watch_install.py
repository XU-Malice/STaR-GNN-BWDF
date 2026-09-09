"""Exercise the real installer while an older training checkout stays untouched."""
import os
import fcntl
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = (ROOT / "scripts/reproduce/install_que_total_watch.sh").read_text()
FAKE_WATCHER = '''import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--project-root',type=Path,required=True)
p.add_argument('--once',action='store_true')
p.add_argument('--watch',action='store_true')
p.add_argument('--status',action='store_true')
p.add_argument('--stop-on-success',action='store_true')
a=p.parse_args()
out=a.project_root/'results/que_total_match_watch_20260909'
out.mkdir(parents=True,exist_ok=True)
path=out/'watch_status.json'
if a.status:
    print(path.read_text())
else:
    path.write_text(json.dumps({'status':'snapshot_complete' if a.once else 'queue_ended_without_all_total_matches'}))
'''


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def commit(root, message):
    git(root, "add", ".")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", message)
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def checkout(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    git(project, "init", "-q")
    (project / "src").mkdir()
    (project / "src/training.py").write_text("TRAINING_BYTES_MUST_STAY_FIXED = True\n")
    old = commit(project, "live training")
    scripts = project / "scripts/reproduce"
    scripts.mkdir(parents=True)
    (scripts / "watch_que_total_match.py").write_text(FAKE_WATCHER)
    for name in ("que_total_watch_evidence.py", "que_total_watch_process.py"):
        (scripts / name).write_text("# Fixture helper, no training or signals\n")
    snapshot = commit(project, "new external tools")
    git(project, "checkout", "--detach", "-q", old)
    env = {**os.environ, "QUE_WATCH_TOOL_PARENT": str(tmp_path / "external tools"), "QUE_WATCH_PYTHON": sys.executable}
    return project, old, snapshot, env


def test_external_install_does_not_checkout_or_change_training_sources(checkout):
    project, old, snapshot, env = checkout
    result = subprocess.run(["bash", "-s", "--", snapshot], input=INSTALLER, cwd=project,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert git(project, "rev-parse", "HEAD") == old
    assert not (project / "scripts/reproduce/watch_que_total_match.py").exists()
    assert (project / "src/training.py").read_text() == "TRAINING_BYTES_MUST_STAY_FIXED = True\n"
    assert git(project, "diff", "--exit-code") == ""
    external = Path(env["QUE_WATCH_TOOL_PARENT"]) / snapshot / "scripts/reproduce/watch_que_total_match.py"
    assert external.read_text() == FAKE_WATCHER
    assert (project / "logs/que_total_match_watch_launcher.pid").exists()


def test_incomplete_tools_commit_does_not_launch(checkout):
    project, old, _, env = checkout
    result = subprocess.run(["bash", "-s", "--", old], input=INSTALLER, cwd=project,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert not (project / "logs/que_total_match_watch_launcher.pid").exists()
    assert git(project, "rev-parse", "HEAD") == old


def test_install_refuses_source_directory_as_tool_destination(checkout):
    project, old, snapshot, env = checkout
    env["QUE_WATCH_TOOL_PARENT"] = str(project / "scripts/watcher")
    result = subprocess.run(["bash", "-s", "--", snapshot], input=INSTALLER, cwd=project,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert not (project / "scripts/watcher").exists()
    assert git(project, "rev-parse", "HEAD") == old


def test_concurrent_install_cannot_overwrite_launcher_pid(checkout):
    project, old, snapshot, env = checkout
    out = project / "results/que_total_match_watch_20260909"
    out.mkdir(parents=True)
    (project / "logs").mkdir()
    pid = project / "logs/que_total_match_watch_launcher.pid"
    pid.write_text("12345\n")
    with (out / "install.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", "-s", "--", snapshot], input=INSTALLER, cwd=project,
                                env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0
    assert "不重复安装" in result.stdout
    assert pid.read_text() == "12345\n"
    assert git(project, "rev-parse", "HEAD") == old
