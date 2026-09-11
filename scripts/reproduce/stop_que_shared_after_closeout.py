"""Close one verified shared queue after its joint-model evidence is preserved.

The only signal this helper can send is SIGTERM through a descriptor bound to
the owned comprehensive Python runner. It never signals numeric PIDs, children,
process groups, or GPU occupants. Existing runner cleanup remains responsible
for its training child and archive. The live training checkout is read-only.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load required process helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# An isolated instance keeps capability adaptation local to this closeout tool.
# It does not change the running observer or its installed process helper.
PROCESS = _module("que_closeout_owned_process", Path(__file__).with_name("que_total_watch_process.py"))
SafetyError = PROCESS.SafetyError
PidfdUnavailable = PROCESS.PidfdUnavailable
_LOAD_EXPORTED_API = PROCESS._load_pidfd_api
SOURCE_TAG = "que_comprehensive_reconstruction_shared_20260908"
JOINT_MODELS = {"msnet", "mscmnet_m", "mscmnet_wm", "mscmnet_w"}
TERMINAL = {"completed", "completed_with_failures", "interrupted", "failed", "paused_time_budget"}


def _raw_pidfd_api() -> Any:
    """Provide missing wrappers for the standard Linux 64-bit syscall ABI.

    Raw wrappers are only selected if Python/libc wrappers are absent. A denied
    or unsupported runtime syscall is never retried using another mechanism.
    The x32 ABI and unknown architectures are deliberately unsupported.
    """
    machine = platform.machine().lower()
    if (sys.platform != "linux" or machine not in {"x86_64", "amd64", "aarch64", "arm64"}
            or ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_long) != 8):
        raise PidfdUnavailable("Raw pidfd wrappers require Linux x86_64 or aarch64 LP64 ABI")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
    except (OSError, AttributeError) as exc:
        raise PidfdUnavailable(f"Linux syscall wrapper is unavailable: {exc}") from exc
    syscall.restype = ctypes.c_long

    def call(number: int, *arguments: Any) -> int:
        _ = libc
        ctypes.set_errno(0)
        result = syscall(ctypes.c_long(number), *arguments)
        if result < 0:
            error = ctypes.get_errno() or errno.EIO
            raise OSError(error, f"pidfd syscall {number}: {os.strerror(error)}")
        return int(result)

    def opener(pid: int, flags: int = 0) -> int:
        return call(434, ctypes.c_int(pid), ctypes.c_uint(flags))

    def sender(descriptor: int, signum: int, info: None, flags: int) -> int:
        if info is not None:
            raise SafetyError("Only a null siginfo pointer is supported")
        return call(424, ctypes.c_int(descriptor), ctypes.c_int(signum),
                    ctypes.c_void_p(None), ctypes.c_uint(flags))

    return PROCESS._PidfdAPI(opener, sender, f"linux_{machine}_pidfd_syscalls")


def _load_pidfd_api() -> Any:
    try:
        return _LOAD_EXPORTED_API()
    except PidfdUnavailable:
        return _raw_pidfd_api()


PROCESS._load_pidfd_api = _load_pidfd_api


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SafetyError(f"Expected a JSON object: {path}")
    return value


def check_no_pending_recurrent(context: Any, queue: dict[str, Any]) -> dict[str, Any]:
    """Require the complete frozen A/B/C plan and finished GRU/LSTM attempts."""
    context.check_fingerprints()
    if not (context.result_root / "stage_c_manifest.json").is_file():
        raise SafetyError("Stage C is not frozen; future recurrent candidates are still possible")
    if queue.get("current_stage") != "C":
        raise SafetyError("Only a queue that has reached the complete Stage C plan can close out")
    plans = context._plans()
    if queue.get("case_count") != len(plans):
        raise SafetyError("Queue count does not match the complete frozen A/B/C plan")
    records: dict[str, Any] = {}
    for record in queue.get("cases", []):
        name = record.get("case")
        if name not in plans or name in records:
            raise SafetyError("Queue records contain an unknown or duplicate frozen candidate")
        if record.get("model") != plans[name]["model"] or record.get("settings") != plans[name]:
            raise SafetyError("Queue record settings differ from the frozen candidate")
        records[name] = record
    counts = {model: {"planned": 0, "passed": 0, "failed": 0} for model in ("gru", "lstm")}
    pending = []
    for name, case in plans.items():
        model = case["model"]
        if model not in counts:
            continue
        counts[model]["planned"] += 1
        record = records.get(name, {})
        status = str(record.get("technical_status", ""))
        if status.startswith("PASS") and record.get("exit_code") == 0:
            counts[model]["passed"] += 1
        elif status == "FAIL" and isinstance(record.get("exit_code"), int):
            counts[model]["failed"] += 1
        else:
            pending.append(name)
    if any(not count["planned"] for count in counts.values()):
        raise SafetyError("The frozen plan must include both GRU and LSTM")
    if pending:
        raise SafetyError(f"Recurrent candidates remain unfinished; queue was not stopped: {pending[:8]}")
    return counts


def _verified_archive(callback: Callable[[Path], dict[str, Any]], archive_root: Path,
                      context: Any) -> dict[str, Any]:
    report = callback(archive_root)
    if not isinstance(report, dict) or report.get("status") != "READY":
        raise SafetyError("Joint closeout archive is not verified READY")
    if Path(report.get("project_root", "")).resolve() != context.project_root.resolve():
        raise SafetyError("Joint closeout archive belongs to a different project")
    if Path(report.get("source_results", "")).resolve() != context.result_root.resolve():
        raise SafetyError("Joint closeout archive belongs to a different source queue")
    if report.get("source_manifest_signature") != context.manifest["signature"]:
        raise SafetyError("Joint closeout archive has a different frozen source manifest")
    selected = report.get("selected_models", {})
    if not isinstance(selected, (dict, list)) or set(selected) != JOINT_MODELS:
        raise SafetyError("Closeout must preserve all four joint models before stopping")
    return report


def _gpu_lock_released(project_root: Path) -> bool:
    """Inspect the existing GPU lock without creating or writing any file."""
    path = project_root / "logs/que_gpu_7.lock"
    if not path.is_file() or path.is_symlink():
        raise SafetyError("Original GPU 7 lock is absent or not a regular file")
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
        return True


def stop_after_closeout(project_root: Path, source_results: Path, archive_root: Path, *,
                        validate_archive: Callable[[Path], dict[str, Any]], context: Any = None,
                        execute: bool = False, wait_seconds: float = 30) -> dict[str, Any]:
    """Verify preserved evidence, then optionally stop only its owned runner.

    ``validate_archive`` must re-read the READY report and verify every copied
    file and archive hash, not return an unchecked cached report. This helper
    additionally binds that verified report to this exact source queue. A
    ``stop_requested`` response means graceful cleanup is still in progress;
    callers must wait for the old lock before starting replacement training.
    """
    project, source, archive = map(lambda p: Path(p).resolve(),
                                   (project_root, source_results, archive_root))
    if source != project / "results" / SOURCE_TAG:
        raise SafetyError("Only the explicitly selected original shared GPU 7 queue may be stopped")
    if archive == source or archive.is_relative_to(source):
        raise SafetyError("Closeout evidence must be separate from the running source queue")
    if not 0 <= wait_seconds <= 30:
        raise SafetyError("Graceful-stop observation must be bounded to 0 through 30 seconds")
    if context is None:
        evidence = _module("que_closeout_stop_evidence", Path(__file__).with_name("que_total_watch_evidence.py"))
        context = evidence.Context(project, source)
    if context.project_root.resolve() != project or context.result_root.resolve() != source:
        raise SafetyError("Source verification context does not match the requested queue")
    report = _verified_archive(validate_archive, archive, context)
    queue = _read(source / "queue_status.json")
    recurrent = check_no_pending_recurrent(context, queue)
    pid_file = project / "logs/que_shared_gpu7_launcher.pid"
    handle = PROCESS.bind_queue(project, source, pid_file)
    result = dict(status="ready_to_stop", execute=bool(execute), recurrent=recurrent,
                  archive_root=str(archive), archive_sha256=report.get("archive_sha256"),
                  source_status=queue.get("status"), no_result_files_deleted=True)
    if handle is None:
        if queue.get("status") in TERMINAL and _gpu_lock_released(project):
            return dict(result, status="already_finished")
        raise SafetyError("No verified live launcher, but source status or GPU lock is not terminal")
    with handle:
        identity = handle.describe()
        if identity.get("gpu_id") != "7" or "--allow-shared-gpu" not in identity.get("argv", []):
            raise SafetyError("Bound runner is not the explicitly authorized shared GPU 7 queue")
        result["process"] = identity
        if not execute:
            return result
        # Recheck the whole archive and source plan immediately before the
        # handle rechecks process identity and sends descriptor-bound SIGTERM.
        _verified_archive(validate_archive, archive, context)
        check_no_pending_recurrent(context, _read(source / "queue_status.json"))
        requested = handle.request_stop()
        result.update(status="stop_requested", signal_requested=requested)
        deadline = time.monotonic() + wait_seconds
        while True:
            current = _read(source / "queue_status.json")
            if handle.exited() and current.get("status") in TERMINAL and _gpu_lock_released(project):
                return dict(result, status="stopped" if requested else "already_finished",
                            source_status=current.get("status"))
            if time.monotonic() >= deadline:
                return dict(result, source_status=current.get("status"))
            time.sleep(min(.25, max(0, deadline - time.monotonic())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-results", type=Path)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        api = _module("que_closeout_archive_verifier", Path(__file__).with_name("archive_que_joint_closeout.py"))
        result = stop_after_closeout(args.project_root,
            args.source_results or args.project_root / "results" / SOURCE_TAG,
            args.archive_root, validate_archive=api.verify_ready, execute=args.execute)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (SafetyError, ValueError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
