#!/usr/bin/env python
"""Launch one model's 24 h and 168 h tasks on two eligible GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.utils.config import (  # noqa: E402
    load_config_with_inheritance,
)
from dma_wdf.utils.device import (  # noqa: E402
    query_visible_gpus,
    select_gpu_statuses,
)


def _thresholds(runtime: dict[str, Any]) -> dict[str, float]:
    return {
        "minimum_free_memory_mib": float(
            runtime["minimum_free_memory_mib"]
        ),
        "maximum_used_memory_mib": float(
            runtime["maximum_used_memory_mib"]
        ),
        "maximum_gpu_utilization_percent": float(
            runtime["maximum_gpu_utilization_percent"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one baseline's 24h and 168h tasks on two GPUs."
        )
    )
    parser.add_argument(
        "--model",
        choices=["dcrnn", "stgcn"],
        default="dcrnn",
    )
    parser.add_argument("--seed-24h", type=int, default=0)
    parser.add_argument("--seed-168h", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    configs = {
        task: load_config_with_inheritance(
            PROJECT_ROOT,
            PROJECT_ROOT
            / "configs"
            / "train"
            / f"{args.model}_{task}.yaml",
        )
        for task in ["24h", "168h"]
    }
    threshold_24h = _thresholds(configs["24h"]["runtime"])
    threshold_168h = _thresholds(configs["168h"]["runtime"])
    if threshold_24h != threshold_168h:
        raise ValueError(
            "Parallel launcher requires identical runtime thresholds."
        )

    # Select both cards before creating either subprocess.
    statuses = query_visible_gpus()
    selected = select_gpu_statuses(
        statuses,
        count=2,
        **threshold_24h,
    )
    assignments = {
        "24h": selected[0],
        "168h": selected[1],
    }
    print(
        json.dumps(
            {
                "status": "resources_reserved_for_launch",
                "assignments": {
                    task: status.state_dict()
                    for task, status in assignments.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    log_dir = (
        PROJECT_ROOT
        / "results"
        / "training"
        / args.model
        / "parallel_logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    seeds = {
        "24h": args.seed_24h,
        "168h": args.seed_168h,
    }
    processes: list[tuple[str, subprocess.Popen, Any, Path]] = []
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    for task in ["24h", "168h"]:
        log_path = log_dir / f"{task}_seed_{seeds[task]}.log"
        handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train" / "train_model.py"),
            "--model",
            args.model,
            "--task",
            task,
            "--seed",
            str(seeds[task]),
            "--device",
            f"cuda:{assignments[task].logical_index}",
        ]
        if args.overwrite:
            command.append("--overwrite")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((task, process, handle, log_path))
        print(
            f"started task={task} pid={process.pid} "
            f"device=cuda:{assignments[task].logical_index} "
            f"log={log_path}",
            flush=True,
        )

    failures: list[dict[str, Any]] = []
    for task, process, handle, log_path in processes:
        return_code = process.wait()
        handle.close()
        print(
            f"finished task={task} return_code={return_code} "
            f"log={log_path}",
            flush=True,
        )
        if return_code != 0:
            failures.append(
                {
                    "task": task,
                    "return_code": return_code,
                    "log": str(log_path),
                }
            )
    if failures:
        print(json.dumps({"failures": failures}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
