#!/usr/bin/env python
"""Train one additive STaR-DCRNN factorial variant."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.data.dcrnn_dataset import (  # noqa: E402
    prepare_dcrnn_training_data,
)
from dma_wdf.models.star_dcrnn import (  # noqa: E402
    FORMAL_VARIANTS,
    build_star_dcrnn_model,
)
from dma_wdf.training.engine import set_reproducible_seed  # noqa: E402
from dma_wdf.training.star_engine import train_star_dcrnn  # noqa: E402
from dma_wdf.utils.config import (  # noqa: E402
    load_config_with_inheritance,
)
from dma_wdf.utils.device import resolve_training_device  # noqa: E402


def _default_config(task: str) -> Path:
    return (
        PROJECT_ROOT / "configs" / "train" / f"star_dcrnn_{task}.yaml"
    )


def _default_output(variant: str, task: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / "innovations"
        / "star_dcrnn_fadpr"
        / variant
        / task
        / f"seed_{seed}"
    )


def _prepare_output(output_dir: Path, *, overwrite: bool) -> Path | None:
    if not output_dir.exists() or not any(output_dir.iterdir()):
        output_dir.mkdir(parents=True, exist_ok=True)
        return None
    if not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --resume or --overwrite."
        )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = output_dir.with_name(
        f"{output_dir.name}.backup-{timestamp}"
    )
    suffix = 1
    while backup.exists():
        backup = output_dir.with_name(
            f"{output_dir.name}.backup-{timestamp}-{suffix}"
        )
        suffix += 1
    shutil.move(str(output_dir), str(backup))
    output_dir.mkdir(parents=True, exist_ok=True)
    return backup


def _progress(row: dict[str, Any]) -> None:
    print(
        "variant={variant} epoch={epoch:03d} step={global_step:05d} "
        "train={train_loss_normalized_mae:.6f} "
        "state={train_state_loss:.6f} "
        "total={train_total_loss:.6f} "
        "val={validation_loss_normalized_mae:.6f} "
        "val_MAE={validation_MAE:.6f} "
        "tf={teacher_forcing_ratio_mean:.4f} "
        "time={epoch_compute_seconds:.1f}s "
        "best={improved}".format(**row),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="star_dcrnn")
    parser.add_argument(
        "--variant",
        choices=list(FORMAL_VARIANTS),
        required=True,
    )
    parser.add_argument("--task", choices=["24h", "168h"], required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    args = parser.parse_args()
    if args.model != "star_dcrnn":
        parser.error("--model must be star_dcrnn.")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive.")

    config_path = (args.config or _default_config(args.task)).resolve()
    config = load_config_with_inheritance(PROJECT_ROOT, config_path)
    if str(config["model"]["name"]) != "star_dcrnn":
        raise ValueError("Model config must name star_dcrnn.")
    expected_horizon = 24 if args.task == "24h" else 168
    if int(config["task"]["horizon"]) != expected_horizon:
        raise ValueError("Task/config horizon mismatch.")
    if int(args.seed) not in [int(v) for v in config["training"]["seeds"]]:
        raise ValueError("Requested seed is outside configured seeds.")

    selection = resolve_training_device(
        args.device or str(config["runtime"]["device"]),
        runtime_config=config["runtime"],
    )
    print(
        json.dumps(
            {
                "model": "star_dcrnn",
                "variant": args.variant,
                "device_selection": selection.state_dict(),
                "cuda_visible_device_count": torch.cuda.device_count(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _default_output(args.variant, args.task, args.seed)
    )
    resume_path: Path | None = None
    archived: Path | None = None
    if args.resume:
        resume_path = output_dir / "checkpoint_last.pt"
        if not resume_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {resume_path}")
    else:
        archived = _prepare_output(output_dir, overwrite=args.overwrite)

    training_data = prepare_dcrnn_training_data(
        project_root=PROJECT_ROOT,
        config=config,
        data_dir=args.data_dir,
    )
    set_reproducible_seed(args.seed)
    model = build_star_dcrnn_model(
        config,
        project_root=PROJECT_ROOT,
        input_dim=training_data.input_dim,
        future_exog_dim=training_data.future_exog_dim,
        horizon=expected_horizon,
        history_hours=int(config["task"]["history_hours"]),
        variant=args.variant,
        device=selection.device,
    )
    result = train_star_dcrnn(
        config=config,
        model=model,
        training_data=training_data,
        variant=args.variant,
        device=selection.device,
        output_dir=output_dir,
        seed=args.seed,
        runtime_metadata={
            "config_path": str(config_path),
            "variant": args.variant,
            "archived_output": None if archived is None else str(archived),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "debug_max_epochs": args.max_epochs,
            "debug_max_train_batches": args.max_train_batches,
        },
        resume_checkpoint=resume_path,
        max_epochs_override=args.max_epochs,
        max_train_batches=args.max_train_batches,
        progress_callback=_progress,
    )
    print(json.dumps(result.state_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
