#!/usr/bin/env python
"""Remove the resumable checkpoint after a baseline run completed safely.

During training, ``checkpoint_last.pt`` is overwritten every epoch so an
interrupted run can be resumed.  A successfully completed run only needs
``checkpoint_best.pt`` for formal evaluation.  This command validates the
run metadata and checkpoint identity before deleting exactly the redundant
``checkpoint_last.pt`` file.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def finalize_completed_run(run_dir: Path) -> dict[str, Any]:
    """Validate a completed run, then remove its redundant last checkpoint."""
    run_dir = run_dir.resolve()
    summary_path = run_dir / "training_summary.json"
    best_path = run_dir / "checkpoint_best.pt"
    last_path = run_dir / "checkpoint_last.pt"

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Training summary does not exist: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise ValueError(
            "Refusing cleanup because training_summary status is not "
            f"'completed': {summary.get('status')!r}."
        )
    if int(summary.get("epochs_completed", 0)) <= 0:
        raise ValueError("Refusing cleanup: epochs_completed is not positive.")
    if not best_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint does not exist: {best_path}"
        )

    best = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )
    model_name = str(best.get("model_name", "")).lower()
    if model_name not in {"dcrnn", "stgcn"}:
        raise ValueError(
            "Best checkpoint model_name must be dcrnn or stgcn."
        )
    expected_kind = f"dma_wdf_{model_name}_training_checkpoint"
    if best.get("kind") != expected_kind:
        raise ValueError(
            "Best checkpoint kind/model mismatch: "
            f"{best.get('kind')!r} != {expected_kind!r}."
        )
    summary_model = str(
        summary.get("model", model_name)
    ).lower()
    if summary_model != model_name:
        raise ValueError(
            "Best checkpoint/summary model mismatch: "
            f"{model_name!r} != {summary_model!r}."
        )
    expected = {
        "task_name": str(summary["task"]),
        "horizon": int(summary["horizon"]),
        "seed": int(summary["seed"]),
        "best_epoch": int(summary["best_epoch"]),
    }
    for key, expected_value in expected.items():
        if best.get(key) != expected_value:
            raise ValueError(
                f"Best checkpoint/summary mismatch for {key}: "
                f"{best.get(key)!r} != {expected_value!r}."
            )

    removed = False
    if last_path.exists():
        if not last_path.is_file():
            raise ValueError(
                f"Refusing cleanup because target is not a file: {last_path}"
            )
        last = torch.load(
            last_path,
            map_location="cpu",
            weights_only=False,
        )
        for key in ["kind", "task_name", "horizon", "seed"]:
            if last.get(key) != best.get(key):
                raise ValueError(
                    f"Last/best checkpoint mismatch for {key}: "
                    f"{last.get(key)!r} != {best.get(key)!r}."
                )
        last_path.unlink()
        removed = True

    summary["last_checkpoint"] = None
    summary["last_checkpoint_removed_after_success"] = True
    _atomic_write_json(summary, summary_path)

    return {
        "status": "completed",
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "model": model_name,
        "best_epoch": int(summary["best_epoch"]),
        "last_checkpoint_removed": removed,
        "last_checkpoint_exists": last_path.exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Safely remove checkpoint_last.pt from a completed "
            "DCRNN or STGCN run."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize_completed_run(args.run_dir),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
