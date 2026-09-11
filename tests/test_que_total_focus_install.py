"""Exercise the real installer against small, real pinned Git repositories.

Only the expensive pytest preflight is stubbed through QUE_FOCUS_PYTHON.  Source
export, Git object verification, process launch, flock, and status are real.  The
exported fixture campaign owns its lock and never imports the training stack.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts/reproduce/install_que_total_focus.sh"
PREFLIGHT_TESTS = (
    "test_que_total_recurrent_search.py",
    "test_que_total_objective.py",
    "test_que_total_followup_training.py",
    "test_que_total_focus.py",
    "test_que_joint_closeout.py",
    "test_que_shared_closeout_stop.py",
    "test_que_training_source_compatibility.py",
)
REQUIRED_PATHS = (
    "src", "scripts", "configs", "tests", "pyproject.toml",
    "SOURCE_CHECKSUMS.sha256",
)

CAMPAIGN_STUB = r'''
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--project-root', type=Path, required=True)
parser.add_argument('--watch', action='store_true')
parser.add_argument('--status', action='store_true')
parser.add_argument('--output-root', type=Path)
parser.add_argument('--close-joint-first', action='store_true')
args = parser.parse_args()
out = args.output_root or args.project_root / 'results/que_total_focus_20260910'
status = out / 'campaign_status.json'
if args.status:
    print(status.read_text() if status.exists() else '{}')
else:
    assert args.watch
    with (out / 'campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status': 'fixture_running', 'pid': os.getpid(),
                 'tool_root': str(Path(__file__).resolve().parents[2]),
                 'close_joint_first': args.close_joint_first}
        with (out / 'fixture_starts.jsonl').open('a') as events:
            events.write(json.dumps(state) + '\n')
        status.write_text(json.dumps(state))
        time.sleep(60)
'''

PYTHON_WRAPPER = r'''
import json
import os
from pathlib import Path
import sys

if sys.argv[1:3] == ['-m', 'pytest']:
    assert sys.argv[3] == '-q'
    paths = sys.argv[4:]
    assert paths and all(Path(path).is_file() for path in paths)
    with Path(os.environ['QUE_INSTALL_TEST_CALLS']).open('a') as events:
        events.write(json.dumps({'cwd': os.getcwd(), 'tests': paths}) + '\n')
    print('fixture CPU preflight')
    raise SystemExit(int(os.environ.get('QUE_INSTALL_TEST_PREFLIGHT_EXIT', '0')))
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
'''


class Installation:
    def __init__(self, base: Path):
        self.project = base / "training"
        self.project.mkdir()
        self.tools = base / "external-tools"
        self.calls = base / "preflight_calls.jsonl"
        self.output = self.project / "results/que_total_focus_20260910"
        self.wrapper = base / "fixture-python"
        self.wrapper.write_text(f"#!{sys.executable}\n" + PYTHON_WRAPPER)
        self.wrapper.chmod(0o755)
        self.env = {
            **os.environ,
            "QUE_FOCUS_TOOL_PARENT": str(self.tools),
            "QUE_FOCUS_PYTHON": str(self.wrapper),
            "QUE_INSTALL_TEST_CALLS": str(self.calls),
        }
        self.git("init", "-q")
        self.git("config", "user.name", "Installer regression fixture")
        self.git("config", "user.email", "installer@example.invalid")
        files = {
            ".gitignore": "results/\nlogs/\n",
            "src/fixture.py": "VALUE = 'pinned source'\n",
            "configs/fixture.yaml": "fixture: true\n",
            "scripts/reproduce/install_que_total_focus.sh": INSTALLER.read_text(),
            "scripts/reproduce/run_que_total_focus.py": CAMPAIGN_STUB,
            "pyproject.toml": '[project]\nname = "installer-fixture"\nversion = "0.0.0"\n',
            "SOURCE_CHECKSUMS.sha256": "fixture inventory checked by Git blob hashes\n",
        }
        files.update({f"tests/{name}": "def test_fixture():\n    assert True\n"
                      for name in PREFLIGHT_TESTS})
        for name, text in files.items():
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.pin = self.commit()
        self.stopped_pids = set()

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=self.project, text=True, stderr=subprocess.PIPE,
        ).strip()

    def commit(self) -> str:
        self.git("add", "-A")
        self.git("commit", "-qm", "Fixture snapshot")
        return self.git("rev-parse", "HEAD")

    @property
    def tool_root(self) -> Path:
        return self.tools / self.pin

    def run(self, *options) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(INSTALLER), self.pin, *options], cwd=self.project, env=self.env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20,
        )

    def starts(self) -> list[dict]:
        return self.json_lines(self.output / "fixture_starts.jsonl")

    def preflights(self) -> list[dict]:
        return self.json_lines(self.calls)

    @staticmethod
    def json_lines(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def stop(self) -> None:
        # Only kill children whose PID was recorded by this fixture campaign.
        for event in self.starts():
            pid = event["pid"]
            assert event["tool_root"] == str(self.tool_root)
            if pid in self.stopped_pids:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.stopped_pids.add(pid)
        # The campaign lock disappearing is the relevant shutdown boundary.
        if (self.output / "campaign.lock").exists():
            for _ in range(100):
                result = subprocess.run(
                    ["flock", "-n", str(self.output / "campaign.lock"), "true"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if result.returncode == 0:
                    return
                time.sleep(0.02)
            raise AssertionError("Fixture campaign failed to release its lock")


@pytest.fixture
def installation(tmp_path):
    if not shutil.which("git") or not shutil.which("flock"):
        pytest.skip("The Bash installer requires git and flock")
    fixture = Installation(tmp_path)
    try:
        yield fixture
    finally:
        fixture.stop()


@pytest.mark.parametrize("lock_state", ["absent", "tracked", "untracked", "later_commit"])
def test_real_export_preflight_and_launch_use_pinned_tree(installation, lock_state):
    case = installation
    if lock_state != "absent":
        (case.project / "uv.lock").write_text("lock fixture\n")
    if lock_state == "tracked":
        case.pin = case.commit()
    elif lock_state == "later_commit":
        case.commit()  # This newer HEAD must not influence the selected pin.
    (case.project / "src/fixture.py").write_text("VALUE = 'uncommitted training edit'\n")
    head_before = case.git("rev-parse", "HEAD")
    status_before = case.git("status", "--porcelain")

    result = case.run()

    assert result.returncode == 0, result.stdout
    assert "fixture_running" in result.stdout
    assert not case.tool_root.is_relative_to(case.project)
    assert (case.tool_root / "src/fixture.py").read_text() == "VALUE = 'pinned source'\n"
    assert (case.tool_root / "uv.lock").exists() == (lock_state == "tracked")
    deployment = json.loads((case.tool_root / "deployment.json").read_text())
    expected = {}
    raw = subprocess.check_output(
        ["git", "ls-tree", "-rz", case.pin, "--", *REQUIRED_PATHS, "uv.lock"],
        cwd=case.project,
    )
    for entry in raw.split(b"\0"):
        if entry:
            metadata, name = entry.split(b"\t", 1)
            expected[name.decode()] = metadata.decode().split()[2]
    assert deployment["commit"] == case.pin
    assert deployment["source_blobs"] == expected
    for name, blob in expected.items():
        assert case.git("hash-object", str(case.tool_root / name)) == blob
    assert case.preflights() == [{
        "cwd": str(case.tool_root), "tests": [f"tests/{name}" for name in PREFLIGHT_TESTS],
    }]
    assert len(case.starts()) == 1
    assert case.starts()[0]["tool_root"] == str(case.tool_root)
    assert case.git("rev-parse", "HEAD") == head_before
    assert case.git("status", "--porcelain") == status_before


@pytest.mark.parametrize("missing", REQUIRED_PATHS)
def test_missing_required_pinned_path_fails_before_preflight_or_launch(installation, missing):
    case = installation
    path = case.project / missing
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    case.pin = case.commit()

    result = case.run()

    assert result.returncode != 0, result.stdout
    assert not case.preflights()
    assert not case.starts()
    assert not (case.project / "logs/que_total_focus_launcher.pid").exists()
    assert not case.tool_root.exists()
    assert not list(case.tools.glob(".install.*"))


def test_running_campaign_is_not_started_twice_and_export_can_be_reused(installation):
    case = installation
    first = case.run()
    assert first.returncode == 0, first.stdout
    manifest = (case.tool_root / "deployment.json").read_bytes()
    source_mtime = (case.tool_root / "src/fixture.py").stat().st_mtime_ns
    first_pid = case.starts()[0]["pid"]

    duplicate = case.run()
    assert duplicate.returncode == 0, duplicate.stdout
    assert "fixture_running" in duplicate.stdout
    assert len(case.starts()) == 1
    assert len(case.preflights()) == 1
    assert int((case.project / "logs/que_total_focus_launcher.pid").read_text()) == first_pid

    case.stop()
    restarted = case.run()
    assert restarted.returncode == 0, restarted.stdout
    assert len(case.starts()) == 2
    assert case.starts()[-1]["pid"] != first_pid
    assert len(case.preflights()) == 2
    assert (case.tool_root / "deployment.json").read_bytes() == manifest
    assert (case.tool_root / "src/fixture.py").stat().st_mtime_ns == source_mtime


@pytest.mark.parametrize("corruption", ["changed_blob", "unexpected_file", "manifest"])
def test_existing_export_corruption_is_rejected_before_another_launch(installation, corruption):
    case = installation
    first = case.run()
    assert first.returncode == 0, first.stdout
    case.stop()
    if corruption == "changed_blob":
        (case.tool_root / "src/fixture.py").write_text("altered export\n")
    elif corruption == "unexpected_file":
        (case.tool_root / "src/unexpected.py").write_text("extra file\n")
    else:
        (case.tool_root / "deployment.json").write_text('{}\n')

    result = case.run()

    assert result.returncode != 0, result.stdout
    assert len(case.starts()) == 1
    assert len(case.preflights()) == 1


def test_failed_cpu_preflight_never_launches_campaign(installation):
    case = installation
    case.env["QUE_INSTALL_TEST_PREFLIGHT_EXIT"] = "17"

    result = case.run()

    assert result.returncode != 0, result.stdout
    assert len(case.preflights()) == 1
    assert not case.starts()
    assert not (case.project / "logs/que_total_focus_launcher.pid").exists()


def test_joint_handoff_installs_separately_and_forwards_mode_once(installation):
    case = installation
    case.output = case.project / "results/que_recurrent_focus_20260910"
    result = case.run("--close-joint-first")
    assert result.returncode == 0, result.stdout
    assert case.starts()[0]["close_joint_first"] is True
    assert (case.project / "logs/que_recurrent_focus_launcher.pid").is_file()
    assert (case.project / "logs/que_recurrent_focus_launcher.log").is_file()
    assert not (case.project / "results/que_total_focus_20260910").exists()
    repeated = case.run("--close-joint-first")
    assert repeated.returncode == 0, repeated.stdout
    assert len(case.starts()) == 1
