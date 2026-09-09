from __future__ import annotations

from dataclasses import replace
import ctypes
import errno
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "que_total_watch_process", ROOT / "scripts/reproduce/que_total_watch_process.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def queue(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "logs").mkdir()
    result = project / "results/que_comprehensive_reconstruction_shared_20260908"
    result.mkdir(parents=True)
    pid_file = project / "logs/que_shared_gpu7_launcher.pid"
    pid_file.write_text("123456789\n")
    process = MODULE.Process(
        pid=123456789, ppid=44, start_ticks=97531, uid=os.getuid(), state="S",
        cwd=project, executable=Path("/env/bin/python3.11"),
        pid_namespace=os.readlink("/proc/self/ns/pid"),
        argv=(
            "python", "-u", "scripts/train/run_que_comprehensive_reconstruction.py",
            "--run-tag", result.name, "--gpu-id", "7", "--allow-shared-gpu",
            "--shared-memory-limit-gib", "6", "--shared-headroom-gib", "2",
            "--reuse-from", "results/old", "--budget-hours", "0",
        ),
    )
    state = {"process": process, "signals": [], "closed": [], "opened": [], "exited": False}
    fd = 987654321
    monkeypatch.setattr(MODULE, "require_native_proc", lambda: None)
    monkeypatch.setattr(MODULE, "protected_pids", lambda: {1, 2, 3})
    monkeypatch.setattr(MODULE, "read_process", lambda pid: state["process"])
    def open_fd(pid, flags):
        state["opened"].append((pid, flags))
        return fd
    monkeypatch.setattr(MODULE.os, "pidfd_open", open_fd, raising=False)
    original_close = os.close
    def close_fd(number):
        if number == fd:
            state["closed"].append(number)
        else:
            original_close(number)
    monkeypatch.setattr(MODULE.os, "close", close_fd)
    original_select = MODULE.select.select
    def poll(readable, writable, exceptional, timeout):
        if readable == [fd]:
            assert writable == exceptional == [] and timeout == 0
            return ([fd] if state["exited"] else [], [], [])
        return original_select(readable, writable, exceptional, timeout)
    monkeypatch.setattr(MODULE.select, "select", poll)
    monkeypatch.setattr(
        MODULE.signal, "pidfd_send_signal",
        lambda number, sig, info, flags: state["signals"].append((number, sig, info, flags)),
        raising=False,
    )
    return project, result, pid_file, state, fd


def bind(queue):
    return MODULE.bind_queue(*queue[:3])


def test_binding_is_read_only_and_description_json_serializable(queue):
    handle = bind(queue)
    assert handle is not None
    metadata = json.loads(json.dumps(handle.describe()))
    assert metadata["gpu_id"] == "7"
    assert metadata["run_tag"] == queue[1].name
    assert metadata["start_ticks"] == 97531
    assert metadata["signal_mechanism"] == "linux_pidfd_SIGTERM_only"
    assert metadata["pidfd_backend"] == "cpython"
    assert queue[3]["signals"] == []
    assert not handle.exited()
    handle.close()
    handle.close()
    assert queue[3]["closed"] == [queue[4]]


def test_stop_only_uses_bound_pidfd_sigterm_and_is_idempotent(queue, monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("Numeric-PID and process-group signals are forbidden")
    monkeypatch.setattr(MODULE.os, "kill", prohibited)
    monkeypatch.setattr(MODULE.os, "killpg", prohibited)
    with bind(queue) as handle:
        assert handle.request_stop()
        assert not handle.request_stop()
        assert handle.describe()["stop_requested"]
    assert queue[3]["signals"] == [(queue[4], signal.SIGTERM, None, 0)]


def test_missing_pid_file_or_gone_process_returns_none(queue):
    queue[3]["process"] = None
    assert bind(queue) is None
    assert queue[3]["opened"] == []
    queue[2].unlink()
    assert bind(queue) is None


@pytest.mark.parametrize("value", ["", "1", "0", "-2", "123 456", "123\n456", "x", "9" * 65])
def test_invalid_pid_record_refused(queue, value):
    queue[2].write_text(value)
    with pytest.raises(MODULE.SafetyError):
        bind(queue)
    assert queue[3]["signals"] == []


def test_pid_record_outside_logs_refused(queue):
    outside = queue[0].parent / "foreign.pid"
    outside.write_text("123456789")
    with pytest.raises(MODULE.SafetyError, match="inside project/logs"):
        MODULE.bind_queue(queue[0], queue[1], outside)


def test_pid_record_symlink_escape_refused(queue):
    outside = queue[0].parent / "foreign.pid"
    outside.write_text("123456789")
    queue[2].unlink()
    queue[2].symlink_to(outside)
    with pytest.raises(MODULE.SafetyError, match="inside project/logs"):
        bind(queue)


@pytest.mark.parametrize("suffix", ["results/a/nested", "outside/tag", "results/..tag"])
def test_invalid_result_root_refused(queue, suffix):
    with pytest.raises(MODULE.SafetyError, match="project/results"):
        MODULE.bind_queue(queue[0], Path(suffix), queue[2])


@pytest.mark.parametrize("change", [
    {"uid": 98765},
    {"cwd": Path("/foreign/project")},
    {"executable": Path("/usr/bin/bash")},
    {"pid_namespace": "pid:[foreign]"},
])
def test_wrong_process_identity_refused_without_signal(queue, change):
    queue[3]["process"] = replace(queue[3]["process"], **change)
    with pytest.raises(MODULE.SafetyError):
        bind(queue)
    assert queue[3]["opened"] == []
    assert queue[3]["signals"] == []


@pytest.mark.parametrize("argv", [
    ("bash", "scripts/train/run_que_comprehensive_reconstruction.py"),
    ("python", "-c", "pass"),
    ("python", "-m", "scripts.train.run_que_comprehensive_reconstruction"),
    ("python", "-W", "ignore", "scripts/train/run_que_comprehensive_reconstruction.py"),
    ("python", "scripts/train/train_temporal_baselines.py"),
    ("python", "scripts/train/run_que_comprehensive_reconstruction.py"),
])
def test_non_direct_or_other_runner_is_refused(queue, argv):
    queue[3]["process"] = replace(queue[3]["process"], argv=argv)
    with pytest.raises(MODULE.SafetyError):
        bind(queue)
    assert queue[3]["signals"] == []


@pytest.mark.parametrize("tail", [
    ("--run-tag", "wrong", "--gpu-id", "7"),
    ("--run-tag", "TAG", "--gpu-id", "GPU-uuid"),
    ("--run-tag", "TAG", "--gpu-id", "7", "--run-tag", "TAG"),
    ("--run-tag", "TAG", "--gpu-id", "7", "--gpu-id=7"),
    ("--run-tag", "TAG", "--gpu-id", "7", "--status"),
    ("--run-tag", "TAG", "--gpu-id", "7", "--dry-run"),
    ("--run-tag", "TAG", "--gpu-id", "7", "--device", "cpu"),
    ("--run-tag", "TAG", "--gpu-id"),
    ("--run-tag", "TAG", "--gpu-id", "7", "--other-option", "x"),
    ("--run-tag", "TAG", "--gpu-id", "7", "--allow-shared-gpu=true"),
])
def test_ambiguous_wrong_or_nontraining_arguments_refused(queue, tail):
    tail = tuple(queue[1].name if word == "TAG" else word for word in tail)
    queue[3]["process"] = replace(queue[3]["process"], argv=queue[3]["process"].argv[:3] + tail)
    with pytest.raises(MODULE.SafetyError):
        bind(queue)


def test_absolute_script_and_equals_options_supported(queue):
    process = queue[3]["process"]
    argv = (
        "/env/bin/python3.11", "-u", "-B", str(queue[0] / MODULE._SCRIPT),
        "--run-tag=" + queue[1].name, "--gpu-id=7", "--allow-shared-gpu",
    )
    queue[3]["process"] = replace(process, argv=argv)
    with bind(queue) as handle:
        assert handle.describe()["gpu_id"] == "7"


def test_runner_script_symlink_escape_refused(queue):
    expected = queue[0] / MODULE._SCRIPT
    expected.parent.mkdir(parents=True)
    foreign = queue[0].parent / "foreign.py"
    foreign.write_text("pass\n")
    expected.symlink_to(foreign)
    with pytest.raises(MODULE.SafetyError, match="comprehensive queue script"):
        bind(queue)


def test_protected_ancestor_refused_before_private_process_inspection(queue, monkeypatch):
    monkeypatch.setattr(MODULE, "protected_pids", lambda: {123456789})
    monkeypatch.setattr(MODULE, "read_process", lambda pid: pytest.fail("Must not read ancestor cwd"))
    with pytest.raises(MODULE.SafetyError, match="ancestor"):
        bind(queue)


@pytest.mark.parametrize("field,value", [
    ("start_ticks", 97532),
    ("argv", ("python", "-c", "pass")),
    ("cwd", Path("/other")),
    ("uid", 98765),
    ("executable", Path("/different/bin/python3.12")),
])
def test_changed_identity_before_stop_refused(queue, field, value):
    with bind(queue) as handle:
        queue[3]["process"] = replace(queue[3]["process"], **{field: value})
        with pytest.raises(MODULE.SafetyError):
            handle.request_stop()
    assert queue[3]["signals"] == []


def test_changed_gpu_argument_before_stop_refused(queue):
    with bind(queue) as handle:
        process = queue[3]["process"]
        argv = tuple("6" if word == "7" else word for word in process.argv)
        queue[3]["process"] = replace(process, argv=argv)
        with pytest.raises(MODULE.SafetyError, match="identity changed"):
            handle.request_stop()
    assert queue[3]["signals"] == []


def test_reparenting_is_allowed(queue):
    with bind(queue) as handle:
        queue[3]["process"] = replace(queue[3]["process"], ppid=1, state="R")
        assert handle.request_stop()


def test_queue_exits_before_signal_noop(queue):
    with bind(queue) as handle:
        queue[3]["exited"] = True
        assert handle.exited()
        assert not handle.request_stop()
    assert queue[3]["signals"] == []


def test_process_gone_without_ready_poll_noop(queue):
    with bind(queue) as handle:
        queue[3]["process"] = None
        assert not handle.request_stop()
    assert queue[3]["signals"] == []


def test_pidfd_race_at_send_noop(queue, monkeypatch):
    def gone(*args):
        raise ProcessLookupError()
    monkeypatch.setattr(MODULE.signal, "pidfd_send_signal", gone)
    with bind(queue) as handle:
        assert not handle.request_stop()


def test_pidfd_permission_failure_no_fallback(queue, monkeypatch):
    def denied(*args):
        raise PermissionError("denied")
    monkeypatch.setattr(MODULE.signal, "pidfd_send_signal", denied)
    with bind(queue) as handle:
        with pytest.raises(MODULE.SafetyError, match="graceful queue shutdown"):
            handle.request_stop()


@pytest.mark.parametrize("attribute,owner", [("pidfd_open", "os"), ("pidfd_send_signal", "signal")])
def test_missing_pidfd_support_refuses(queue, monkeypatch, attribute, owner):
    monkeypatch.delattr(getattr(MODULE, owner), attribute)
    monkeypatch.setattr(MODULE.ctypes, "CDLL", lambda *args, **kwargs: object())
    with pytest.raises(MODULE.PidfdUnavailable, match="pidfd support"):
        bind(queue)
    assert queue[3]["signals"] == []
    assert queue[3]["opened"] == []


def test_pidfd_kernel_unsupported_refuses(queue, monkeypatch):
    def unsupported(*args):
        raise OSError(errno.ENOSYS, "unsupported")
    monkeypatch.setattr(MODULE.os, "pidfd_open", unsupported)
    with pytest.raises(MODULE.PidfdUnavailable, match="Cannot bind queue pidfd"):
        bind(queue)


class LibcSymbol:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *arguments):
        return self.implementation(*arguments)


@pytest.fixture
def libc(queue, monkeypatch):
    state = queue[3]
    calls = {"open": [], "signal": [], "loads": [], "open_errno": 0, "signal_errno": 0}

    def open_fd(pid, flags):
        calls["open"].append((pid, flags))
        if calls["open_errno"]:
            ctypes.set_errno(calls["open_errno"])
            return -1
        return queue[4]

    def send(fd, sig, info, flags):
        calls["signal"].append((fd, sig, info, flags))
        if calls["signal_errno"]:
            ctypes.set_errno(calls["signal_errno"])
            return -1
        state["signals"].append((fd, sig, info, flags))
        return 0

    class Library:
        pidfd_open = LibcSymbol(open_fd)
        pidfd_send_signal = LibcSymbol(send)

    library = Library()

    def load(name, **kwargs):
        calls["loads"].append((name, kwargs))
        return library

    monkeypatch.setattr(MODULE.ctypes, "CDLL", load)
    return library, calls


@pytest.mark.parametrize("missing", [("open",), ("signal",), ("open", "signal")])
def test_missing_python_wrappers_use_same_pidfd_libc_api(queue, libc, monkeypatch, missing):
    library, calls = libc
    if "open" in missing:
        monkeypatch.delattr(MODULE.os, "pidfd_open")
    if "signal" in missing:
        monkeypatch.delattr(MODULE.signal, "pidfd_send_signal")

    def prohibited(*args, **kwargs):
        pytest.fail("No numeric PID or process-group signals may be sent")

    monkeypatch.setattr(MODULE.os, "kill", prohibited)
    monkeypatch.setattr(MODULE.os, "killpg", prohibited)
    with bind(queue) as handle:
        assert queue[3]["signals"] == []
        metadata = handle.describe()
        assert "libc.pidfd_" in metadata["pidfd_backend"]
        assert handle.request_stop()
        assert not handle.request_stop()
    assert queue[3]["signals"] == [(queue[4], signal.SIGTERM, None, 0)]
    assert queue[3]["closed"] == [queue[4]]
    assert calls["loads"] == [(None, {"use_errno": True})]
    if "open" in missing:
        assert calls["open"] == [(123456789, 0)]
        assert library.pidfd_open.argtypes == [ctypes.c_int, ctypes.c_uint]
        assert library.pidfd_open.restype is ctypes.c_int
    else:
        assert calls["open"] == []
    if "signal" in missing:
        assert calls["signal"] == [(queue[4], signal.SIGTERM, None, 0)]
        assert library.pidfd_send_signal.argtypes == [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        assert library.pidfd_send_signal.restype is ctypes.c_int
    else:
        assert calls["signal"] == []


def test_available_python_wrappers_do_not_load_libc(queue, libc):
    with bind(queue) as handle:
        assert handle.request_stop()
    assert libc[1]["loads"] == []


@pytest.mark.parametrize("operation", ["open", "signal"])
@pytest.mark.parametrize("error", [errno.EPERM, errno.EACCES, errno.ENOSYS])
def test_python_kernel_error_never_retries_through_libc(queue, libc, monkeypatch, operation, error):
    def fail(*args):
        raise OSError(error, "denied or unavailable")
    if operation == "open":
        monkeypatch.setattr(MODULE.os, "pidfd_open", fail)
    else:
        monkeypatch.setattr(MODULE.signal, "pidfd_send_signal", fail)
    error_type = MODULE.PidfdUnavailable if error == errno.ENOSYS else MODULE.SafetyError
    with pytest.raises(error_type) as raised:
        with bind(queue) as handle:
            handle.request_stop()
    if error != errno.ENOSYS:
        assert not isinstance(raised.value, MODULE.PidfdUnavailable)
    assert libc[1]["loads"] == []
    assert queue[3]["signals"] == []
    assert queue[3]["closed"] == ([queue[4]] if operation == "signal" else [])


@pytest.mark.parametrize("operation", ["open", "signal"])
@pytest.mark.parametrize("error", [errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EINVAL])
def test_libc_errno_is_preserved_without_numeric_fallback(queue, libc, monkeypatch, operation, error):
    monkeypatch.delattr(MODULE.os, "pidfd_open")
    monkeypatch.delattr(MODULE.signal, "pidfd_send_signal")
    libc[1][operation + "_errno"] = error
    error_type = MODULE.PidfdUnavailable if error == errno.ENOSYS else MODULE.SafetyError
    with pytest.raises(error_type) as raised:
        with bind(queue) as handle:
            handle.request_stop()
    assert raised.value.__cause__.errno == error
    if error != errno.ENOSYS:
        assert not isinstance(raised.value, MODULE.PidfdUnavailable)
    assert queue[3]["signals"] == []
    assert queue[3]["closed"] == ([queue[4]] if operation == "signal" else [])
    assert len(libc[1]["open"]) == 1
    assert len(libc[1]["signal"]) == (1 if operation == "signal" else 0)


@pytest.mark.parametrize("operation", ["open", "signal"])
def test_libc_process_exit_race_returns_noop(queue, libc, monkeypatch, operation):
    monkeypatch.delattr(MODULE.os, "pidfd_open")
    monkeypatch.delattr(MODULE.signal, "pidfd_send_signal")
    libc[1][operation + "_errno"] = errno.ESRCH
    if operation == "open":
        assert bind(queue) is None
    else:
        with bind(queue) as handle:
            assert not handle.request_stop()
    assert queue[3]["signals"] == []
    assert queue[3]["closed"] == ([queue[4]] if operation == "signal" else [])


def test_libc_backend_still_refuses_changed_identity_and_closes_fd(queue, libc, monkeypatch):
    monkeypatch.delattr(MODULE.os, "pidfd_open")
    monkeypatch.delattr(MODULE.signal, "pidfd_send_signal")
    with bind(queue) as handle:
        queue[3]["process"] = replace(queue[3]["process"], start_ticks=9999999)
        with pytest.raises(MODULE.SafetyError, match="identity changed"):
            handle.request_stop()
    assert libc[1]["signal"] == []
    assert queue[3]["closed"] == [queue[4]]


def test_libc_loading_unavailable_is_explicit_capability_error(queue, monkeypatch):
    monkeypatch.delattr(MODULE.os, "pidfd_open")
    def fail(*args, **kwargs):
        raise OSError("No dynamic loader")
    monkeypatch.setattr(MODULE.ctypes, "CDLL", fail)
    with pytest.raises(MODULE.PidfdUnavailable, match="pidfd support is unavailable"):
        bind(queue)
    assert queue[3]["opened"] == queue[3]["signals"] == []


def test_process_exits_during_pidfd_open(queue, monkeypatch):
    def gone(*args):
        raise ProcessLookupError()
    monkeypatch.setattr(MODULE.os, "pidfd_open", gone)
    assert bind(queue) is None


def test_process_changes_during_pidfd_open_closes_descriptor(queue, monkeypatch):
    def reused(*args):
        queue[3]["process"] = replace(queue[3]["process"], start_ticks=99999)
        return queue[4]
    monkeypatch.setattr(MODULE.os, "pidfd_open", reused)
    with pytest.raises(MODULE.SafetyError, match="changed while opening"):
        bind(queue)
    assert queue[3]["closed"] == [queue[4]]


def test_process_gone_after_open_closes_descriptor(queue, monkeypatch):
    def gone(*args):
        queue[3]["process"] = None
        return queue[4]
    monkeypatch.setattr(MODULE.os, "pidfd_open", gone)
    assert bind(queue) is None
    assert queue[3]["closed"] == [queue[4]]


def test_closed_handle_cannot_stop(queue):
    handle = bind(queue)
    handle.close()
    with pytest.raises(MODULE.SafetyError, match="closed"):
        handle.request_stop()


def test_stat_parser_handles_spaces_and_parentheses():
    text = "45 (python odd ) name) " + " ".join(["S", "12"] + ["0"] * 17 + ["891"])
    assert MODULE._stat_fields(text) == (45, "S", 12, 891)


def test_parent_reader_uses_stat_only(monkeypatch):
    original = Path.read_text
    stat_text = "45 (sshd) " + " ".join(["S", "12"] + ["0"] * 17 + ["891"])
    def read(path, *args, **kwargs):
        if str(path).startswith("/proc/"):
            assert str(path) == "/proc/45/stat"
            return stat_text
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    assert MODULE.read_parent_pid(45) == 12


def test_namespace_guard_refuses_host_proc(monkeypatch):
    original = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/stat":
            return f"{os.getpid() + 100000} (python) S"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(MODULE.SafetyError, match="PID namespace differs"):
        MODULE.require_native_proc()
