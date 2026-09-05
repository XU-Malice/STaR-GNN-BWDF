from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stop_que_queue", ROOT / "scripts/train/stop_que_queue.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
NATIVE_PROC_GUARD = MODULE.require_native_proc

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc process control")


@pytest.fixture(autouse=True)
def same_namespace_proc(monkeypatch):
    """Adapt the test harness only when its /proc belongs to a parent namespace.

    Codex's test executor exposes host /proc while subprocess/kill PIDs are local.
    Production deliberately refuses that setup.  Translate only same-namespace
    processes here so the real child-tree signal tests can still exercise the
    termination logic; no other namespace's processes are ever signal targets.
    """
    proc_pid = int(Path("/proc/self/stat").read_text().split(" ", 1)[0])
    if proc_pid == os.getpid():
        return
    original_read = MODULE.read_process
    expected_namespace = os.readlink("/proc/self/ns/pid")

    def snapshot():
        local_by_host = {}
        host_processes = {}
        for directory in Path("/proc").iterdir():
            if not directory.name.isdecimal():
                continue
            try:
                if os.readlink(directory / "ns/pid") != expected_namespace:
                    continue
                statuses = (directory / "status").read_text().splitlines()
                local_pid = int(next(row for row in statuses if row.startswith("NSpid:")).split()[-1])
                process = original_read(int(directory.name))
                if process is not None:
                    local_by_host[process.pid] = local_pid
                    host_processes[process.pid] = process
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        return {
            local_by_host[pid]: MODULE.Process(
                local_by_host[pid], local_by_host.get(process.ppid, 0),
                process.start_ticks, process.uid, process.state, process.cwd,
                process.argv,
            )
            for pid, process in host_processes.items()
        }

    monkeypatch.setattr(MODULE, "require_native_proc", lambda: None)
    monkeypatch.setattr(MODULE, "read_process", lambda pid: snapshot().get(pid))
    monkeypatch.setattr(MODULE, "children_of", lambda pid: [p for p in snapshot().values() if p.ppid == pid and MODULE.is_alive(p)])


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for test child")


@pytest.fixture
def queue(tmp_path):
    script = tmp_path / "scripts/train/run_que_targeted_reproduction_gpu6.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\ntrap '' TERM\nsleep 300 &\necho $! > child.pid\nwait\n")
    pid_file = tmp_path / "logs/que_targeted_reproduction_launcher.pid"
    pid_file.parent.mkdir()
    process = subprocess.Popen(["bash", str(script.relative_to(tmp_path))], cwd=tmp_path, start_new_session=True)
    pid_file.write_text(str(process.pid))
    _wait_for(lambda: (tmp_path / "child.pid").is_file() and (tmp_path / "child.pid").read_text().strip())
    child = int((tmp_path / "child.pid").read_text())
    try:
        yield tmp_path, script, pid_file, process, child
    finally:
        # This test owns a newly allocated session/process group exclusively.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


def test_missing_or_finished_pid_is_noop(tmp_path):
    assert MODULE.stop_queue(project_root=tmp_path, pid_file=Path("missing.pid"), expected_script=Path("queue.sh"), execute=True) == 0
    process = subprocess.Popen(["true"])
    process.wait()
    pid_path = tmp_path / "queue.pid"
    pid_path.write_text(str(process.pid))
    assert MODULE.stop_queue(project_root=tmp_path, pid_file=pid_path, expected_script=Path("queue.sh"), execute=True) == 0


def test_dry_run_leaves_launcher_and_child_alive(queue):
    project, script, pid_file, process, child = queue
    messages = []
    assert MODULE.stop_queue(project_root=project, pid_file=pid_file, expected_script=script, report=messages.append) == 0
    assert process.poll() is None
    assert MODULE.is_alive(MODULE.read_process(child))
    assert any(f"PID={child}" in message for message in messages)
    assert MODULE.read_process(process.pid).state not in {"T", "t"}


def test_execute_stops_tree_but_not_unrelated_process(queue):
    project, script, pid_file, process, child = queue
    unrelated = subprocess.Popen(["sleep", "300"], cwd=project, start_new_session=True)
    try:
        assert MODULE.stop_queue(project_root=project, pid_file=pid_file, expected_script=script, execute=True, wait_seconds=0.1) == 0
        process.wait(timeout=3)
        _wait_for(lambda: not MODULE.is_alive(MODULE.read_process(child)))
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=3)


def test_mismatched_live_pid_refuses_without_signals(queue):
    project, script, pid_file, process, child = queue
    with pytest.raises(MODULE.SafetyError, match="not"):
        MODULE.stop_queue(project_root=project, pid_file=pid_file, expected_script=Path("different.sh"), execute=True)
    assert process.poll() is None
    assert MODULE.is_alive(MODULE.read_process(child))
    assert MODULE.read_process(process.pid).state not in {"T", "t"}


def test_pid_identity_change_never_signals_reused_pid(queue, monkeypatch):
    project, script, pid_file, process, child = queue
    original = MODULE.read_process(child)
    assert original is not None
    changed = MODULE.Process(original.pid, original.ppid, original.start_ticks + 1, original.uid, original.state, original.cwd, original.argv)
    monkeypatch.setattr(MODULE, "read_process", lambda pid: changed)
    sent = []
    monkeypatch.setattr(MODULE.os, "kill", lambda *args: sent.append(args))
    assert MODULE.signal_owned(original, signal.SIGTERM, project, set()) is False
    assert sent == []


def test_caller_pid_is_protected(tmp_path):
    pid_path = tmp_path / "queue.pid"
    pid_path.write_text(str(os.getpid()))
    with pytest.raises(MODULE.SafetyError, match="caller/ancestor/session"):
        MODULE.stop_queue(project_root=tmp_path, pid_file=pid_path, expected_script=Path("queue.sh"), execute=True)


def test_identification_failure_thaws_parent_without_terminating(queue, monkeypatch):
    project, script, pid_file, process, child = queue
    original_validate = MODULE.validate_owned

    def refuse_child(candidate, project_root, protected):
        if candidate.pid == child:
            raise MODULE.SafetyError("Synthetic child identity mismatch")
        original_validate(candidate, project_root, protected)

    monkeypatch.setattr(MODULE, "validate_owned", refuse_child)
    with pytest.raises(MODULE.SafetyError, match="child identity"):
        MODULE.stop_queue(project_root=project, pid_file=pid_file, expected_script=script, execute=True)
    assert process.poll() is None
    assert MODULE.is_alive(MODULE.read_process(child))
    _wait_for(lambda: MODULE.read_process(process.pid).state not in {"T", "t"})


def test_production_refuses_mismatched_proc_pid_namespace(monkeypatch):
    monkeypatch.setattr(MODULE.os, "getpid", lambda: -1)
    with pytest.raises(MODULE.SafetyError, match="PID namespace"):
        NATIVE_PROC_GUARD()
