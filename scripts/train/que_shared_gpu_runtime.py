"""Opt-in GPU sharing without changing numerical training code or other jobs.

The limit controls PyTorch's caching allocator, not every CUDA allocation made
by the process. A separate free-memory margin accommodates its CUDA context and
other allocations. Existing GPU processes are only observed, never signalled.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER = PROJECT_ROOT / "scripts/train/train_temporal_baselines.py"
RESOURCE_RETRY_EXIT = 75


class GpuWaitExpired(RuntimeError):
    """The queue's wall-clock budget expired while waiting for GPU memory."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def query_gpu(gpu_id: str) -> dict[str, Any]:
    """Return one physical GPU and its compute processes; reject ambiguous data."""
    if not re.fullmatch(r"[0-9]+", str(gpu_id)):
        raise ValueError("GPU ID must be one numeric physical index")
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=uuid,name,memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    rows = list(csv.reader(result.stdout.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeError(f"Ambiguous GPU memory query: {result.stdout!r}")
    gpu_uuid, name, free_text, total_text = (value.strip() for value in rows[0])
    if not gpu_uuid.startswith("GPU-") or not name or not free_text.isdecimal() or not total_text.isdecimal():
        raise RuntimeError(f"Invalid GPU memory query: {result.stdout!r}")
    free_mib, total_mib = int(free_text), int(total_text)
    if total_mib <= 0 or not 0 <= free_mib <= total_mib:
        raise RuntimeError(f"Invalid GPU free/total memory: {free_mib}/{total_mib}")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    processes = []
    for row in csv.reader(result.stdout.strip().splitlines()):
        if len(row) != 3:
            raise RuntimeError(f"Invalid GPU process query: {result.stdout!r}")
        uuid, pid, process_name = (value.strip() for value in row)
        if not uuid.startswith("GPU-") or not pid.isdecimal() or int(pid) <= 0 or not process_name:
            raise RuntimeError(f"Invalid GPU process entry: {row!r}")
        if uuid == gpu_uuid:
            processes.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": process_name})
    return {"gpu_id": str(gpu_id), "gpu_uuid": gpu_uuid, "name": name,
            "free_mib": free_mib, "total_mib": total_mib, "processes": processes,
            "timestamp": _timestamp()}


def wait_for_memory(gpu_id: str, minimum_free_mib: int, poll_seconds: float,
                    deadline: float, on_wait: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Wait with bounded sleeps; query errors fail rather than waiting forever."""
    if minimum_free_mib <= 0 or not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError("A positive memory threshold and polling interval <=60 seconds are required")
    if math.isnan(deadline):
        raise ValueError("GPU waiting deadline must not be NaN")
    while True:
        if time.monotonic() >= deadline:
            raise GpuWaitExpired("Time budget expired while waiting for shared GPU memory")
        snapshot = query_gpu(gpu_id)
        if time.monotonic() >= deadline:
            raise GpuWaitExpired("Time budget expired during shared GPU inspection")
        if snapshot["free_mib"] >= minimum_free_mib:
            return snapshot
        snapshot.update(minimum_free_mib=minimum_free_mib, poll_seconds=poll_seconds)
        on_wait(snapshot)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GpuWaitExpired("Time budget expired while waiting for shared GPU memory")
        time.sleep(min(poll_seconds, remaining))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-limit-gib", type=float, required=True)
    parser.add_argument("--headroom-gib", type=float, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0:
        parser.error("memory-limit-gib must be finite and positive")
    if not math.isfinite(args.headroom_gib) or args.headroom_gib < 1:
        parser.error("headroom-gib must be finite and at least 1 GiB")
    if args.trainer_args[:1] != ["--"]:
        parser.error("trainer arguments must follow --")
    args.trainer_args = args.trainer_args[1:]
    if args.trainer_args.count("--device") != 1:
        parser.error("shared GPU training requires exactly one --device cuda:0")
    index = args.trainer_args.index("--device")
    if index + 1 >= len(args.trainer_args) or args.trainer_args[index + 1] != "cuda:0":
        parser.error("shared GPU training requires --device cuda:0")
    return args


def wrap_command(command: list[str], memory_limit_gib: float,
                 headroom_gib: float, report_path: Path) -> list[str]:
    """Wrap only this repository's existing direct trainer command."""
    if len(command) < 4 or command[1] != "-u" or not all(isinstance(value, str) for value in command):
        raise ValueError("Expected a direct python -u trainer.py command")
    target = Path(command[2])
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if target.resolve() != TRAINER.resolve():
        raise ValueError("Shared GPU wrapper accepts only the unchanged temporal-baseline trainer")
    options = ["--memory-limit-gib", str(memory_limit_gib), "--headroom-gib", str(headroom_gib),
               "--report", str(report_path), "--", *command[3:]]
    parse_arguments(options)
    return [command[0], "-u", str(Path(__file__).resolve()), *options]


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", visible):
        raise RuntimeError("Shared runtime requires exactly one physical GPU in CUDA_VISIBLE_DEVICES")
    report: dict[str, Any] = {
        "status": "initializing", "started_utc": _timestamp(), "cuda_visible_devices": visible,
        "memory_limit_gib": args.memory_limit_gib, "headroom_gib": args.headroom_gib,
        "limit_scope": "pytorch_caching_allocator_only", "training_arguments": args.trainer_args,
        "training_script": str(TRAINER), "other_processes_signalled": False,
    }
    _write_report(args.report, report)
    # Match the unchanged trainer's default before this wrapper initializes
    # CUDA. Preserve an explicit environment setting, exactly as the trainer.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    initialized = False
    old_argv = sys.argv
    exit_code = 1
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in shared GPU runtime")
        torch.cuda.set_device(0)
        initialized = True
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        # Parent checks limit + full margin before context initialization. This
        # second check allows up to 1 GiB of that margin for our own context,
        # retaining at least 1 GiB margin when launched with the default 2 GiB.
        post_context_margin = max(1.0, args.headroom_gib - 1.0)
        required_bytes = int((args.memory_limit_gib + post_context_margin) * 1024**3)
        limit_bytes = int(args.memory_limit_gib * 1024**3)
        report.update(free_after_context_gib=free_bytes / 1024**3,
                      total_gib=total_bytes / 1024**3,
                      minimum_free_after_context_gib=required_bytes / 1024**3)
        if total_bytes <= 0 or limit_bytes >= total_bytes:
            raise RuntimeError("Shared allocator limit must be smaller than total GPU memory")
        if free_bytes < required_bytes:
            report.update(status="resource_wait", error="Free GPU memory fell below the shared-runtime threshold")
            exit_code = RESOURCE_RETRY_EXIT
            return exit_code
        torch.cuda.set_per_process_memory_fraction(limit_bytes / total_bytes, device=0)
        report.update(status="running", allocator_fraction=limit_bytes / total_bytes)
        _write_report(args.report, report)
        sys.argv = [str(TRAINER), *args.trainer_args]
        try:
            runpy.run_path(str(TRAINER), run_name="__main__")
        except SystemExit as exc:
            code = exc.code
            exit_code = 0 if code is None else code if isinstance(code, int) else 1
            if exit_code != 0:
                report.update(status="failed", error=f"Trainer exited with code {code!r}")
                return exit_code
        exit_code = 0
        report["status"] = "completed"
        return exit_code
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status="resource_oom", error=str(exc))
        exit_code = RESOURCE_RETRY_EXIT
        return exit_code
    except RuntimeError as exc:
        # This exact guard belongs to the unchanged trainer's resource preflight.
        if re.fullmatch(r"cuda:0 has [0-9.]+ GiB free; [0-9.]+ GiB is required\.", str(exc)):
            report.update(status="resource_wait", error=str(exc))
            exit_code = RESOURCE_RETRY_EXIT
            return exit_code
        report.update(status="failed", error=f"RuntimeError: {exc}")
        raise
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        sys.argv = old_argv
        if initialized:
            for name, function in (("peak_allocated_gib", "max_memory_allocated"),
                                   ("peak_reserved_gib", "max_memory_reserved")):
                try:
                    report[name] = getattr(torch.cuda, function)(0) / 1024**3
                except Exception:
                    report[name] = None
        report.update(finished_utc=_timestamp(), exit_code=exit_code)
        _write_report(args.report, report)
        print(json.dumps({"shared_gpu_runtime": report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
