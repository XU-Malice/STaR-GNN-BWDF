"""Bind to one owned comprehensive Que runner and request graceful shutdown.

This standalone, standard-library-only helper is intended to be deployed outside
the live training checkout.  It never scans GPUs, signals children or process
groups, or escalates to SIGKILL.  The bound Python runner handles its own children
and final archive in its existing SIGTERM/finally path.  Linux pidfds are required;
there is deliberately no numeric-PID signal fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import os
from pathlib import Path
import re
import select
import signal
import stat as stat_module
import sys
from typing import Callable


class SafetyError(RuntimeError):
    """The requested queue cannot be identified or signalled safely."""


class PidfdUnavailable(SafetyError):
    """Pidfd APIs are absent; a caller may continue without automatic stopping."""


@dataclass(frozen=True)
class _PidfdAPI:
    open: Callable[[int, int], int]
    send_signal: Callable[[int, int, None, int], object]
    backend: str


def _load_pidfd_api() -> _PidfdAPI:
    """Use existing Python APIs, or exported libc equivalents when absent.

    A Python build can omit these wrappers despite a supporting host libc and
    kernel.  Exported libc functions invoke the same pidfd interfaces; never
    retry a denied/unsupported syscall through another backend.  Signatures
    follow glibc's sys/pidfd.h; no numeric syscall identifiers are used.
    """
    opener = getattr(os, "pidfd_open", None)
    sender = getattr(signal, "pidfd_send_signal", None)
    if callable(opener) and callable(sender):
        return _PidfdAPI(opener, sender, "cpython")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:
        raise PidfdUnavailable(f"Linux pidfd support is unavailable: {exc}") from exc

    def exported(name: str, argument_types: list) -> Callable:
        try:
            function = getattr(libc, name)
        except AttributeError as exc:
            raise PidfdUnavailable(
                f"Linux pidfd support is unavailable: {name} is absent from Python and libc; "
                "numeric-PID signals are disabled"
            ) from exc
        function.argtypes = argument_types
        function.restype = ctypes.c_int

        def call(*arguments: object) -> int:
            # Keep libc alive through this closure and read its thread-local
            # errno immediately.  OSError creates PermissionError/ESRCH types.
            _ = libc
            ctypes.set_errno(0)
            result = function(*arguments)
            if result < 0:
                error = ctypes.get_errno() or errno.EIO
                raise OSError(error, f"{name}: {os.strerror(error)}")
            return result

        return call

    backends = []
    if not callable(opener):
        opener = exported("pidfd_open", [ctypes.c_int, ctypes.c_uint])
        backends.append("libc.pidfd_open")
    else:
        backends.append("cpython.pidfd_open")
    if not callable(sender):
        sender = exported(
            "pidfd_send_signal", [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        )
        backends.append("libc.pidfd_send_signal")
    else:
        backends.append("cpython.pidfd_send_signal")
    return _PidfdAPI(opener, sender, "+".join(backends))


_PYTHON_NAME = re.compile(r"python(?:[0-9]+(?:\.[0-9]+)*)?\Z")
_RUN_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SCRIPT = Path("scripts/train/run_que_comprehensive_reconstruction.py")
_VALUE_OPTIONS = {
    "--run-tag", "--gpu-id", "--device", "--minimum-free-mib",
    "--shared-memory-limit-gib", "--shared-headroom-gib", "--gpu-poll-seconds",
    "--shared-resource-retries", "--time-budget-hours", "--budget-hours",
    "--data-dir", "--paper-config", "--reuse-from",
}
_FLAG_OPTIONS = {"--allow-shared-gpu"}


def require_native_proc() -> None:
    """Refuse host-mounted /proc when numeric PIDs belong to another namespace."""
    if sys.platform != "linux" or os.getuid() != os.geteuid():
        raise SafetyError("Run this watcher directly on Linux as the training user, without setuid")
    try:
        proc_pid = int(Path("/proc/self/stat").read_text().split(" ", 1)[0])
    except (OSError, ValueError) as exc:
        raise SafetyError(f"Cannot verify native /proc: {exc}") from exc
    if proc_pid != os.getpid():
        raise SafetyError("/proc PID namespace differs from the watcher; run directly on the training server")


def _stat_fields(text: str) -> tuple[int, str, int, int]:
    try:
        pid = int(text.split(" ", 1)[0])
        fields = text[text.rfind(")") + 2:].split()
        return pid, fields[0], int(fields[1]), int(fields[19])
    except (ValueError, IndexError) as exc:
        raise SafetyError("Malformed /proc process stat") from exc


def read_parent_pid(pid: int) -> int | None:
    """Read metadata only, including for SSH/sudo ancestors owned by others."""
    try:
        recorded_pid, _, ppid, _ = _stat_fields((Path("/proc") / str(pid) / "stat").read_text())
        if recorded_pid != pid:
            raise SafetyError("Ancestor /proc PID mismatch")
        return ppid
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise SafetyError(f"Cannot inspect ancestor metadata for PID {pid}: {exc}") from exc


def protected_pids() -> set[int]:
    protected = {1, os.getpid(), os.getppid(), os.getsid(0)}
    current = os.getpid()
    seen: set[int] = set()
    while current and current not in seen:
        seen.add(current)
        protected.add(current)
        if current == 1:
            break
        current = read_parent_pid(current)
    return protected


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    start_ticks: int
    uid: int
    state: str
    cwd: Path | None
    argv: tuple[str, ...]
    executable: Path | None
    pid_namespace: str


def read_process(pid: int) -> Process | None:
    """Inspect only the requested PID, verifying ownership before cwd/cmdline."""
    directory = Path("/proc") / str(pid)
    try:
        first = _stat_fields((directory / "stat").read_text())
        if first[0] != pid:
            raise SafetyError(f"/proc identity mismatch for PID {pid}")
        if first[1] in {"Z", "X", "x"}:
            return None
        uid = directory.stat().st_uid
        if uid != os.getuid():
            raise SafetyError(f"PID {pid} is not owned by the current user")
        status = (directory / "status").read_text()
        uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), "")
        uids = tuple(int(value) for value in uid_line.split()[1:])
        if len(uids) != 4 or any(value != os.getuid() for value in uids):
            raise SafetyError(f"PID {pid} does not have the current user's complete UID identity")
        namespace = os.readlink(directory / "ns/pid")
        if namespace != os.readlink("/proc/self/ns/pid"):
            raise SafetyError(f"PID {pid} belongs to a different PID namespace")
        cwd = (directory / "cwd").resolve(strict=True)
        executable = (directory / "exe").resolve(strict=True)
        raw_argv = (directory / "cmdline").read_bytes()
        argv = tuple(part.decode("utf-8", errors="strict") for part in raw_argv.rstrip(b"\0").split(b"\0"))
        second = _stat_fields((directory / "stat").read_text())
        if first[0] != second[0] or first[3] != second[3]:
            raise SafetyError(f"PID {pid} identity changed during inspection")
        if second[1] in {"Z", "X", "x"}:
            return None
        return Process(pid, second[2], second[3], uid, second[1], cwd, argv, executable, namespace)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, ValueError, UnicodeError) as exc:
        raise SafetyError(f"Cannot inspect queue PID {pid}: {exc}") from exc


def _queue_arguments(process: Process, project: Path) -> dict[str, str | bool]:
    argv = process.argv
    if not argv or not _PYTHON_NAME.fullmatch(Path(argv[0]).name):
        raise SafetyError(f"PID {process.pid} is not a direct Python runner")
    if process.executable is None or not _PYTHON_NAME.fullmatch(process.executable.name):
        raise SafetyError(f"PID {process.pid} executable is not Python")
    index = 1
    while index < len(argv) and argv[index] in {"-u", "-B"}:
        index += 1
    if index >= len(argv) or argv[index].startswith("-"):
        raise SafetyError("Only direct Python [-u] [-B] SCRIPT invocation is supported; no -c or -m")
    script = Path(argv[index])
    script = script if script.is_absolute() else project / script
    expected_script = (project / _SCRIPT).resolve()
    if not expected_script.is_relative_to(project) or script.resolve() != expected_script:
        raise SafetyError(f"PID {process.pid} does not run the comprehensive queue script")
    values: dict[str, str | bool] = {}
    index += 1
    while index < len(argv):
        item = argv[index]
        name, separator, inline = item.partition("=")
        if name in values:
            raise SafetyError(f"Ambiguous duplicate queue option: {name}")
        if name in _FLAG_OPTIONS and not separator:
            values[name] = True
        elif name in _VALUE_OPTIONS:
            if separator:
                value = inline
            else:
                index += 1
                if index >= len(argv):
                    raise SafetyError(f"Missing queue option value: {name}")
                value = argv[index]
            if not value or value.startswith("--"):
                raise SafetyError(f"Invalid queue option value: {name}")
            values[name] = value
        else:
            raise SafetyError(f"Unsupported queue option: {item}")
        index += 1
    if "--run-tag" not in values or "--gpu-id" not in values:
        raise SafetyError("Queue must identify an explicit --run-tag and --gpu-id")
    gpu_id = values["--gpu-id"]
    if not isinstance(gpu_id, str) or not re.fullmatch(r"[0-9]+", gpu_id):
        raise SafetyError("Queue GPU must be a numeric physical index")
    if values.get("--device", "cuda:0") != "cuda:0":
        raise SafetyError("The selected queue is not a cuda:0 training queue")
    return values


def _validate_identity(process: Process, project: Path, result_root: Path) -> dict[str, str | bool]:
    if process.pid in protected_pids():
        raise SafetyError(f"Refusing caller, ancestor or session leader PID {process.pid}")
    if process.uid != os.getuid() or process.cwd != project:
        raise SafetyError(f"PID {process.pid} ownership or project cwd does not match")
    if process.pid_namespace != os.readlink("/proc/self/ns/pid"):
        raise SafetyError(f"PID {process.pid} belongs to a different PID namespace")
    values = _queue_arguments(process, project)
    if values["--run-tag"] != result_root.name:
        raise SafetyError(f"PID {process.pid} run tag does not match {result_root.name}")
    return values


def _same_identity(first: Process, second: Process) -> bool:
    # Reparenting and process state transitions are expected during a long run.
    return (
        first.pid, first.start_ticks, first.uid, first.cwd, first.argv,
        first.executable, first.pid_namespace,
    ) == (
        second.pid, second.start_ticks, second.uid, second.cwd, second.argv,
        second.executable, second.pid_namespace,
    )


class QueueHandle:
    """A Linux pidfd bound to one complete queue identity; call close() finally."""

    def __init__(self, process: Process, project: Path, result_root: Path, pid_file: Path,
                 pidfd: int, arguments: dict[str, str | bool], api: _PidfdAPI):
        self._process = process
        self._project = project
        self._result_root = result_root
        self._pid_file = pid_file
        self._pidfd: int | None = pidfd
        self._arguments = dict(arguments)
        self._api = api
        self._stop_requested = False

    def describe(self) -> dict[str, object]:
        return {
            "pid": self._process.pid,
            "start_ticks": self._process.start_ticks,
            "uid": self._process.uid,
            "cwd": str(self._project),
            "executable": str(self._process.executable),
            "argv": list(self._process.argv),
            "pid_namespace": self._process.pid_namespace,
            "run_tag": self._result_root.name,
            "gpu_id": self._arguments["--gpu-id"],
            "result_root": str(self._result_root),
            "pid_file": str(self._pid_file),
            "signal_mechanism": "linux_pidfd_SIGTERM_only",
            "pidfd_backend": self._api.backend,
            "stop_requested": self._stop_requested,
        }

    def exited(self) -> bool:
        if self._pidfd is None:
            raise SafetyError("Queue handle is closed")
        try:
            return bool(select.select([self._pidfd], [], [], 0)[0])
        except OSError as exc:
            raise SafetyError(f"Cannot query queue pidfd: {exc}") from exc

    def request_stop(self) -> bool:
        """Send one SIGTERM after checking identity again; return False if gone."""
        require_native_proc()
        if self.exited() or self._stop_requested:
            return False
        current = read_process(self._process.pid)
        if current is None:
            return False
        _validate_identity(current, self._project, self._result_root)
        if not _same_identity(self._process, current):
            raise SafetyError("Bound queue identity changed; no signal sent")
        if self.exited():
            return False
        try:
            self._api.send_signal(self._pidfd, signal.SIGTERM, None, 0)
        except ProcessLookupError:
            return False
        except OSError as exc:
            if exc.errno == errno.ENOSYS:
                raise PidfdUnavailable(
                    f"Could not request graceful queue shutdown: kernel pidfd signalling is unavailable: {exc}"
                ) from exc
            raise SafetyError(f"Could not request graceful queue shutdown: {exc}") from exc
        self._stop_requested = True
        return True

    def close(self) -> None:
        if self._pidfd is not None:
            descriptor, self._pidfd = self._pidfd, None
            os.close(descriptor)

    def __enter__(self) -> QueueHandle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def bind_queue(project_root: Path, result_root: Path, pid_file: Path) -> QueueHandle | None:
    """Bind a live owned queue, or return None if its PID record/process is gone.

    The result must be project/results/<simple-run-tag> and the PID record must
    reside within project/logs.  Active wrong identities and unsupported pidfds
    fail closed with SafetyError.  No signal is sent by this function.
    """
    require_native_proc()
    try:
        project = project_root.resolve(strict=True)
        result = (project / result_root).resolve()
        pid_path = (project / pid_file).resolve()
        if result.parent != project / "results" or not _RUN_TAG.fullmatch(result.name):
            raise SafetyError("Results must be project/results/<simple-run-tag>")
        if not pid_path.is_relative_to(project / "logs"):
            raise SafetyError("Queue PID record must be inside project/logs")
        try:
            info = pid_path.stat()
        except FileNotFoundError:
            return None
        if not stat_module.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > 64:
            raise SafetyError("PID record must be a small regular file owned by the current user")
        text = pid_path.read_text().strip()
        if not re.fullmatch(r"[0-9]+", text) or int(text) <= 1:
            raise SafetyError("PID record must contain one non-system PID")
        pid = int(text)
        # Protect ancestors before reading private cwd/cmdline metadata.
        if pid in protected_pids():
            raise SafetyError(f"Refusing caller, ancestor or session leader PID {pid}")
        process = read_process(pid)
        if process is None:
            return None
        arguments = _validate_identity(process, project, result)
        api = _load_pidfd_api()
        try:
            descriptor = api.open(pid, 0)
        except ProcessLookupError:
            return None
        except OSError as exc:
            if exc.errno == errno.ENOSYS:
                raise PidfdUnavailable(f"Cannot bind queue pidfd: kernel pidfd support is unavailable: {exc}") from exc
            raise SafetyError(f"Cannot bind queue pidfd; no signals will be sent: {exc}") from exc
        try:
            current = read_process(pid)
            if current is None:
                os.close(descriptor)
                return None
            _validate_identity(current, project, result)
            if not _same_identity(process, current):
                raise SafetyError("Queue identity changed while opening pidfd")
            return QueueHandle(process, project, result, pid_path, descriptor, arguments, api)
        except BaseException:
            os.close(descriptor)
            raise
    except SafetyError:
        raise
    except (OSError, ValueError) as exc:
        raise SafetyError(f"Cannot bind queue: {exc}") from exc
