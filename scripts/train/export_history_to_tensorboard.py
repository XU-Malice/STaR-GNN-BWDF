#!/usr/bin/env python
"""Convert an existing history.csv into TensorBoard event files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.training.engine import (  # noqa: E402
    export_history_to_tensorboard,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a completed/current DCRNN history to TensorBoard."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="Archive a non-empty TensorBoard directory before export.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"History does not exist: {history_path}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "tensorboard"
    )
    archived: Path | None = None
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.archive_existing:
            raise FileExistsError(
                f"TensorBoard directory is not empty: {output_dir}. "
                "Use --archive-existing to preserve and replace it."
            )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archived = output_dir.with_name(
            f"{output_dir.name}.backup-{timestamp}"
        )
        shutil.move(str(output_dir), str(archived))

    history_frame = pd.read_csv(history_path)
    history = history_frame.to_dict(orient="records")
    export_history_to_tensorboard(
        history=history,
        log_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "history": str(history_path),
                "epochs": len(history),
                "tensorboard_log_dir": str(output_dir),
                "archived_existing": (
                    None if archived is None else str(archived)
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
