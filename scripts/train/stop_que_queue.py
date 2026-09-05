#!/usr/bin/env python3
"""Stop one identified legacy Que queue without touching other GPU jobs.

This Linux-only helper defaults to a read-only dry run.  The old Bash queue's
TERM trap does not terminate its children, so execute mode freezes the verified
launcher and its owned descendants before terminating them.  It never signals a
process group, a GPU-wide PID list, or an unverified PID from a stale PID file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import sys
import time
from typing import Callable


class SafetyError(RuntimeError):
    """The requested process cannot be safely identified."""


def require_native_proc() -> None:
    """Never mix a host-mounted /proc PID with a container-local kill PID."""
    proc_pid = int(Path("/proc/self/stat").read_text().split(" ", 1)[0])
    if proc_pid != os.getpid():
        raise SafetyError("/proc PID namespace differs from this process; run this helper directly on the training server")


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    start_ticks: int
    uid: int
    state: str
    cwd: Path | None
    argv: tuple[str, ...]


def read_process(pid: int) -> Process | None:
    directory = Path("/proc") / str(pid)
    try:
        stat = (directory / "stat").read_text()
        # The command field is parenthesized and may contain spaces or ')'.
        fields = stat[stat.rfind(")") + 2 :].split()
        uid = directory.stat().st_uid
        argv = tuple(
            part.decode(errors="replace")
            for part in (directory / "cmdline").read_bytes().split(b"\0")
            if part
        )
        try:
            cwd = (directory / "cwd").resolve(strict=True)
        except FileNotFoundError:
            cwd = None  # Zombies have no cwd/cmdline.
        return Process(pid, int(fields[1]), int(fields[19]), uid, fields[0], cwd, argv)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as exc:
        raise SafetyError(f"Cannot inspect PID {pid}: {exc}") from exc


def is_alive(process: Process | None) -> bool:
    return process is not None and process.state not in {"Z", "X", "x"}


def protected_pids() -> set[int]:
    """Protect this helper, its invoking ancestors, and its session leader."""
    protected = {os.getpid(), os.getppid(), os.getsid(0), 1}
    current = read_process(os.getpid())
    seen: set[int] = set()
    while current is not None and current.pid not in seen:
        seen.add(current.pid)
        protected.add(current.pid)
        if current.ppid <= 0:
            break
        current = read_process(current.ppid)
    return protected


def validate_owned(process: Process, project: Path, protected: set[int]) -> None:
    if process.pid in protected:
        raise SafetyError(f"Refusing to signal the caller/ancestor/session leader PID {process.pid}")
    if process.uid != os.getuid():
        raise SafetyError(f"PID {process.pid} is not owned by the current user")
    if process.cwd != project:
        raise SafetyError(f"PID {process.pid} cwd is {process.cwd}, not {project}")


def validate_launcher(process: Process, project: Path, expected: Path, protected: set[int]) -> None:
    validate_owned(process, project, protected)
    if len(process.argv) != 2 or Path(process.argv[0]).name != "bash":
        raise SafetyError(f"PID {process.pid} is not the expected 'bash SCRIPT' launcher: {process.argv}")
    actual_script = Path(process.argv[1])
    if not actual_script.is_absolute():
        actual_script = project / actual_script
    if actual_script.resolve() != expected:
        raise SafetyError(f"PID {process.pid} runs {actual_script}, not {expected}")


def children_of(pid: int) -> list[Process]:
    children: list[Process] = []
    child_pids: set[int] = set()
    try:
        for thread in (Path("/proc") / str(pid) / "task").iterdir():
            try:
                child_pids.update(int(value) for value in (thread / "children").read_text().split())
            except FileNotFoundError:
                continue  # A thread that exited cannot fork another child.
    except FileNotFoundError:
        return []
    except PermissionError as exc:
        raise SafetyError(f"Cannot inspect children of PID {pid}: {exc}") from exc
    for child_pid in child_pids:
        candidate = read_process(child_pid)
        if is_alive(candidate) and candidate.ppid == pid:
            children.append(candidate)
    return sorted(children, key=lambda child: child.pid)


def current_owned(process: Process, project: Path, protected: set[int]) -> Process | None:
    current = read_process(process.pid)
    if not is_alive(current) or current.start_ticks != process.start_ticks:
        return None
    validate_owned(current, project, protected)
    return current


def signal_owned(process: Process, sig: signal.Signals, project: Path, protected: set[int]) -> bool:
    if current_owned(process, project, protected) is None:
        return False
    try:
        os.kill(process.pid, sig)
        return True
    except ProcessLookupError:
        return False


def stop_queue(
    *,
    project_root: Path,
    pid_file: Path,
    expected_script: Path,
    execute: bool = False,
    wait_seconds: float = 5.0,
    report: Callable[[str], None] = print,
) -> int:
    require_native_proc()
    project = project_root.resolve(strict=True)
    pid_path = (project / pid_file).resolve()
    expected = (project / expected_script).resolve()
    if not pid_path.is_relative_to(project) or not expected.is_relative_to(project):
        raise SafetyError("PID file and expected script must be inside the selected project")
    if not 0.0 <= wait_seconds <= 5.0:
        raise SafetyError("wait_seconds must be between 0 and 5")
    if not pid_path.is_file():
        report(f"NOOP: PID file does not exist: {pid_path}")
        return 0
    text = pid_path.read_text().strip()
    if not text.isdecimal() or int(text) <= 1:
        raise SafetyError(f"PID file is not a valid non-system PID: {pid_path}")
    launcher = read_process(int(text))
    if not is_alive(launcher):
        report(f"NOOP: recorded launcher PID {text} has already exited; no processes signalled")
        return 0
    protected = protected_pids()
    validate_launcher(launcher, project, expected, protected)

    targets: dict[int, Process] = {}
    frozen_here: list[Process] = []

    def collect(process: Process) -> None:
        if process.pid in targets:
            return
        current = current_owned(process, project, protected)
        if current is None:
            return
        if execute:
            already_stopped = current.state in {"T", "t"}
            if not signal_owned(process, signal.SIGSTOP, project, protected):
                return
            if not already_stopped:
                frozen_here.append(process)
            deadline = time.monotonic() + 0.5
            while True:
                current = current_owned(process, project, protected)
                if current is None:
                    return
                if current.state in {"T", "t"}:
                    break
                if time.monotonic() >= deadline:
                    raise SafetyError(f"PID {process.pid} did not stop; no processes terminated")
                time.sleep(0.01)
        targets[process.pid] = process
        # A stopped parent cannot create another child while we recurse.
        for child in children_of(process.pid):
            collect(child)

    try:
        collect(launcher)
        stopped_launcher = current_owned(launcher, project, protected)
        if stopped_launcher is not None:
            validate_launcher(stopped_launcher, project, expected, protected)
    except BaseException:
        # Identification failure is not authority to terminate a partial tree.
        for process in reversed(frozen_here):
            signal_owned(process, signal.SIGCONT, project, protected)
        raise

    mode = "EXECUTE" if execute else "DRY_RUN"
    for process in targets.values():
        report(f"{mode}: PID={process.pid} start_ticks={process.start_ticks} cwd={process.cwd} argv={process.argv!r}")
    if not execute:
        report("Dry run only; no signals sent. Use --execute to stop these verified targets.")
        return 0

    descendants = [process for pid, process in targets.items() if pid != launcher.pid]
    for process in reversed(descendants):
        signal_owned(process, signal.SIGTERM, project, protected)
    for process in reversed(descendants):
        signal_owned(process, signal.SIGCONT, project, protected)
    # Do not run the legacy launcher's broken TERM trap or allow it to start
    # another training case.  Its children were identified while it was frozen.
    signal_owned(launcher, signal.SIGKILL, project, protected)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if not any(current_owned(process, project, protected) is not None for process in targets.values()):
            break
        time.sleep(0.05)
    for process in reversed(list(targets.values())):
        signal_owned(process, signal.SIGKILL, project, protected)
    report(f"Stopped verified queue PID {launcher.pid}; targeted {len(targets)} owned processes. No result files deleted.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_project = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--pid-file", type=Path, default=Path("logs/que_targeted_reproduction_launcher.pid"))
    parser.add_argument("--expected-script", type=Path, default=Path("scripts/train/run_que_targeted_reproduction_gpu6.sh"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="Default: inspect without sending signals")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This helper requires Linux /proc")
    # A Ctrl-C/TERM in the middle of the short freeze/terminate transaction must
    # not strand the queue in SIGSTOP.  Deliver pending signals after cleanup.
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        return stop_queue(project_root=args.project_root, pid_file=args.pid_file, expected_script=args.expected_script, execute=args.execute)
    except (SafetyError, OSError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


if __name__ == "__main__":
    raise SystemExit(main())
