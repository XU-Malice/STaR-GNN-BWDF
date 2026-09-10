from __future__ import annotations

from dataclasses import replace
import ctypes
import errno
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("que_test_closeout_stop", ROOT / "scripts/reproduce/stop_que_shared_after_closeout.py")
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
P = M.PROCESS


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project = tmp_path / "project"
    source = project / "results" / M.SOURCE_TAG
    source.mkdir(parents=True)
    (project / "logs").mkdir()
    (project / "logs/que_shared_gpu7_launcher.pid").write_text("123456789\n")
    (project / "logs/que_gpu_7.lock").touch()
    (source / "stage_c_manifest.json").write_text("{}")
    plans = {f"C_{model}_test": dict(case=f"C_{model}_test", model=model, stage="C", seed=20240604)
             for model in ("gru", "lstm", "msnet")}
    records = [dict(case=c["case"], model=c["model"], settings=c, technical_status="PASS", exit_code=0)
               for c in plans.values()]
    records[-1].update(technical_status="running", exit_code=None)
    queue = dict(status="running", current_stage="C", case_count=3, cases=records)
    (source / "queue_status.json").write_text(json.dumps(queue))
    archive = tmp_path / "preserved"
    archive.mkdir()
    report = dict(status="READY", project_root=str(project), source_results=str(source),
                  source_manifest_signature="abc", selected_models={k: {} for k in M.JOINT_MODELS},
                  archive_sha256="0" * 64)
    state = dict(signals=[], opened=[], closed=[], exited=False, checks=0, archive_checks=0)

    def fingerprint():
        state["checks"] += 1

    context = SimpleNamespace(project_root=project, result_root=source, manifest={"signature": "abc"},
                              _plans=lambda: plans, check_fingerprints=fingerprint)
    process = P.Process(pid=123456789, ppid=44, start_ticks=123, uid=os.getuid(), state="S", cwd=project,
                        executable=Path("/env/bin/python3.11"), pid_namespace=os.readlink("/proc/self/ns/pid"),
                        argv=("python", "-u", "scripts/train/run_que_comprehensive_reconstruction.py",
                              "--run-tag", source.name, "--gpu-id", "7", "--allow-shared-gpu"))
    state["process"] = process
    descriptor = 987654321
    monkeypatch.setattr(P, "require_native_proc", lambda: None)
    monkeypatch.setattr(P, "protected_pids", lambda: {1, 2, 3})
    monkeypatch.setattr(P, "read_process", lambda pid: state["process"])

    def opener(pid, flags):
        state["opened"].append((pid, flags))
        return descriptor

    def sender(fd, sig, info, flags):
        state["signals"].append((fd, sig, info, flags))
        state["exited"] = True
        queue["status"] = "interrupted"
        (source / "queue_status.json").write_text(json.dumps(queue))

    api = P._PidfdAPI(opener, sender, "test_boundary")
    monkeypatch.setattr(P, "_load_pidfd_api", lambda: api)
    original_close = os.close

    def close(fd):
        if fd == descriptor:
            state["closed"].append(fd)
        else:
            original_close(fd)

    monkeypatch.setattr(P.os, "close", close)
    original_select = P.select.select

    def poll(readable, writable, exceptional, timeout):
        if readable == [descriptor]:
            return ([descriptor] if state["exited"] else [], [], [])
        return original_select(readable, writable, exceptional, timeout)

    monkeypatch.setattr(P.select, "select", poll)

    def verified(path):
        assert path == archive
        state["archive_checks"] += 1
        return report

    return SimpleNamespace(project=project, source=source, archive=archive, report=report, context=context,
                           queue=queue, plans=plans, state=state, descriptor=descriptor, api=api, verified=verified)


def run(s, **kwargs):
    return M.stop_after_closeout(s.project, s.source, s.archive, validate_archive=s.verified,
                                context=s.context, wait_seconds=0, **kwargs)


def save_queue(s):
    (s.source / "queue_status.json").write_text(json.dumps(s.queue))


def test_archive_verified_before_single_bound_sigterm_and_no_other_signals(setup, monkeypatch):
    s = setup

    def forbidden(*args, **kwargs):
        raise AssertionError("No numeric PID, process-group, or GPU process signals")

    monkeypatch.setattr(M.os, "kill", forbidden)
    monkeypatch.setattr(M.os, "killpg", forbidden)
    result = run(s, execute=True)
    assert result["status"] == "stopped"
    assert s.state["signals"] == [(s.descriptor, signal.SIGTERM, None, 0)]
    assert s.state["archive_checks"] == 2
    assert s.state["checks"] == 2
    assert s.state["closed"] == [s.descriptor]


def test_dry_run_is_read_only_and_has_no_signal(setup):
    s = setup
    before = {p: p.read_bytes() for p in s.project.rglob("*") if p.is_file()}
    assert run(s)["status"] == "ready_to_stop"
    assert s.state["signals"] == []
    assert before == {p: p.read_bytes() for p in s.project.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", ["uid", "cwd", "argv", "gpu", "shared", "ancestor"])
def test_wrong_runner_identity_never_signalled(setup, monkeypatch, change):
    s = setup
    process = s.state["process"]
    if change == "uid":
        process = replace(process, uid=os.getuid() + 1)
    elif change == "cwd":
        process = replace(process, cwd=s.project.parent)
    elif change == "argv":
        process = replace(process, argv=("python", "-c", "print('not queue')"))
    elif change == "gpu":
        process = replace(process, argv=tuple("6" if v == "7" else v for v in process.argv))
    elif change == "shared":
        process = replace(process, argv=tuple(v for v in process.argv if v != "--allow-shared-gpu"))
    elif change == "ancestor":
        monkeypatch.setattr(P, "protected_pids", lambda: {process.pid})
    s.state["process"] = process
    with pytest.raises(M.SafetyError):
        run(s, execute=True)
    assert not s.state["signals"]


def test_identity_race_while_binding_refuses_and_closes_descriptor(setup, monkeypatch):
    s = setup
    original = s.state["process"]
    states = iter([original, replace(original, start_ticks=original.start_ticks + 1)])
    monkeypatch.setattr(P, "read_process", lambda pid: next(states))
    with pytest.raises(M.SafetyError, match="identity changed"):
        run(s, execute=True)
    assert not s.state["signals"]
    assert s.state["closed"] == [s.descriptor]


def test_identity_race_immediately_before_signal_refuses(setup, monkeypatch):
    s = setup
    original = s.state["process"]
    states = iter([original, original, replace(original, start_ticks=124)])
    monkeypatch.setattr(P, "read_process", lambda pid: next(states))
    with pytest.raises(M.SafetyError, match="identity changed"):
        run(s, execute=True)
    assert not s.state["signals"]


@pytest.mark.parametrize("status", ["running", "waiting_gpu", "", "FAIL_WITHOUT_EXIT"])
def test_pending_recurrent_refuses_before_process_binding(setup, status):
    s = setup
    record = s.queue["cases"][0]
    record.update(technical_status="FAIL" if status == "FAIL_WITHOUT_EXIT" else status, exit_code=None)
    save_queue(s)
    with pytest.raises(M.SafetyError, match="unfinished"):
        run(s, execute=True)
    assert not s.state["opened"] and not s.state["signals"]


def test_recorded_completed_failure_does_not_block_new_recurrent_work(setup):
    s = setup
    s.queue["cases"][0].update(technical_status="FAIL", exit_code=1)
    save_queue(s)
    result = run(s, execute=True)
    assert result["recurrent"]["gru"]["failed"] == 1
    assert result["status"] == "stopped"


@pytest.mark.parametrize("mutation", ["no_c", "count", "stage", "settings", "duplicate"])
def test_incomplete_or_changed_plan_blocks_stop(setup, mutation):
    s = setup
    if mutation == "no_c":
        (s.source / "stage_c_manifest.json").unlink()
    elif mutation == "count":
        s.queue["case_count"] += 1
    elif mutation == "stage":
        s.queue["current_stage"] = "B"
    elif mutation == "settings":
        s.queue["cases"][0]["settings"] = {}
    else:
        s.queue["cases"].append(s.queue["cases"][0])
    save_queue(s)
    with pytest.raises(M.SafetyError):
        run(s, execute=True)
    assert not s.state["opened"]


@pytest.mark.parametrize("key,value", [("status", "COPYING"), ("project_root", "/other"),
                                      ("source_results", "/other"), ("source_manifest_signature", "different"),
                                      ("selected_models", {"msnet": {}})])
def test_invalid_archive_binding_fails_before_signal(setup, key, value):
    s = setup
    s.report[key] = value
    with pytest.raises(M.SafetyError):
        run(s, execute=True)
    assert not s.state["opened"] and not s.state["signals"]


def test_archive_hash_failure_fails_before_signal(setup):
    s = setup

    def invalid(path):
        raise ValueError("checkpoint SHA256 mismatch")

    s.verified = invalid
    with pytest.raises(ValueError, match="SHA256"):
        run(s, execute=True)
    assert not s.state["opened"]


def test_archive_changed_after_binding_still_sends_no_signal(setup):
    s = setup
    original = s.verified

    def invalid_second(path):
        if s.state["archive_checks"]:
            raise ValueError("archive changed")
        return original(path)

    s.verified = invalid_second
    with pytest.raises(ValueError, match="changed"):
        run(s, execute=True)
    assert s.state["opened"] and not s.state["signals"]
    assert s.state["closed"] == [s.descriptor]


def test_already_finished_requires_terminal_status_and_released_lock(setup):
    s = setup
    s.state["process"] = None
    with pytest.raises(M.SafetyError, match="No verified live launcher"):
        run(s, execute=True)
    s.queue["status"] = "completed_with_failures"
    save_queue(s)
    assert run(s, execute=True)["status"] == "already_finished"
    assert not s.state["signals"]


def test_graceful_cleanup_can_outlast_bounded_observation(setup, monkeypatch):
    s = setup
    monkeypatch.setattr(M, "_gpu_lock_released", lambda p: False)
    result = run(s, execute=True)
    assert result["status"] == "stop_requested"
    assert len(s.state["signals"]) == 1


def test_existing_gpu_lock_probe_does_not_create_write_or_steal_lock(setup):
    s = setup
    path = s.project / "logs/que_gpu_7.lock"
    path.write_bytes(b"existing lock")
    with path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not M._gpu_lock_released(s.project)
    assert M._gpu_lock_released(s.project)
    assert path.read_bytes() == b"existing lock"
    path.unlink()
    with pytest.raises(M.SafetyError, match="absent"):
        M._gpu_lock_released(s.project)
    assert not path.exists()


def test_existing_wrapper_preferred_and_capability_absence_uses_raw(monkeypatch):
    exported = object()
    raw = object()
    monkeypatch.setattr(M, "_LOAD_EXPORTED_API", lambda: exported)
    monkeypatch.setattr(M, "_raw_pidfd_api", lambda: raw)
    assert M._load_pidfd_api() is exported

    def unavailable():
        raise M.PidfdUnavailable("wrappers absent")

    monkeypatch.setattr(M, "_LOAD_EXPORTED_API", unavailable)
    assert M._load_pidfd_api() is raw


@pytest.mark.parametrize("error", [errno.EPERM, errno.ENOSYS])
def test_failed_runtime_wrapper_is_not_retried(monkeypatch, error):
    def denied():
        raise OSError(error, "not allowed")

    def prohibited():
        raise AssertionError("Runtime errors must not trigger fallback")

    monkeypatch.setattr(M, "_LOAD_EXPORTED_API", denied)
    monkeypatch.setattr(M, "_raw_pidfd_api", prohibited)
    with pytest.raises(OSError) as caught:
        M._load_pidfd_api()
    assert caught.value.errno == error


class Syscall:
    def __init__(self, error=0):
        self.calls = []
        self.error = error
        self.restype = None

    def __call__(self, *args):
        values = [arg.value for arg in args]
        self.calls.append(values)
        if self.error:
            ctypes.set_errno(self.error)
            return -1
        return 15 if values[0] == 434 else 0


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_raw_pidfd_exact_syscall_abi_and_bound_descriptor(monkeypatch, arch):
    syscall = Syscall()
    monkeypatch.setattr(M.platform, "machine", lambda: arch)
    monkeypatch.setattr(M.sys, "platform", "linux")
    monkeypatch.setattr(M.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(syscall=syscall))
    api = M._raw_pidfd_api()
    assert api.open(123456789, 0) == 15
    api.send_signal(15, signal.SIGTERM, None, 0)
    assert syscall.calls == [[434, 123456789, 0], [424, 15, signal.SIGTERM, None, 0]]
    assert syscall.restype is ctypes.c_long


@pytest.mark.parametrize("arch", ["i686", "riscv64", "ppc64le", "unknown"])
def test_unknown_raw_syscall_architecture_fails_closed(monkeypatch, arch):
    monkeypatch.setattr(M.platform, "machine", lambda: arch)
    with pytest.raises(M.PidfdUnavailable, match="LP64 ABI"):
        M._raw_pidfd_api()


@pytest.mark.parametrize("error", [errno.EPERM, errno.ENOSYS])
def test_raw_syscall_runtime_refusal_is_propagated_once(monkeypatch, error):
    syscall = Syscall(error)
    monkeypatch.setattr(M.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(M.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(syscall=syscall))
    with pytest.raises(OSError) as caught:
        M._raw_pidfd_api().open(123456789, 0)
    assert caught.value.errno == error
    assert syscall.calls == [[434, 123456789, 0]]
