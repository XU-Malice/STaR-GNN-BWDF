#!/usr/bin/env python
"""Unified model-training entry point for DCRNN and STGCN baselines.

Examples
--------
Formal automatic-GPU run::

    python scripts/train/train_model.py --model dcrnn --task 24h --seed 0

Explicit logical GPU::

    python scripts/train/train_model.py --model stgcn --task 168h --seed 0 \
        --device cuda:1

Resume the same run directory::

    python scripts/train/train_model.py --task 24h --seed 0 --resume

CPU is accepted only when explicitly requested for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Required by PyTorch deterministic algorithms for CUDA matrix operations.
# It must be present before the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.data.dcrnn_dataset import (  # noqa: E402
    prepare_dcrnn_training_data,
)
from dma_wdf.data.forecast_dataset import (  # noqa: E402
    prepare_forecast_training_data,
)
from dma_wdf.models.dcrnn import build_dcrnn_model  # noqa: E402
from dma_wdf.models.stgcn import build_stgcn_model  # noqa: E402
from dma_wdf.training.engine import (  # noqa: E402
    set_reproducible_seed,
    train_dcrnn,
)
from dma_wdf.training.stgcn_engine import train_stgcn  # noqa: E402
from dma_wdf.utils.config import (  # noqa: E402
    load_config_with_inheritance,
)
from dma_wdf.utils.device import (  # noqa: E402
    resolve_training_device,
)


def _default_config(model: str, task: str) -> Path:
    return (
        PROJECT_ROOT
        / "configs"
        / "train"
        / f"{model}_{task}.yaml"
    )


def _resolve_output_dir(
    *,
    config: dict[str, Any],
    seed: int,
    override: Path | None,
) -> Path:
    if override is not None:
        return override.resolve()
    base = Path(config["output"]["base_dir"])
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return (base / f"seed_{int(seed)}").resolve()


def _prepare_new_output(
    output_dir: Path,
    *,
    overwrite: bool,
) -> Path | None:
    """Refuse accidental overwrite or move the old run to a backup."""
    if not output_dir.exists() or not any(output_dir.iterdir()):
        output_dir.mkdir(parents=True, exist_ok=True)
        return None
    if not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --resume to continue it or --overwrite to archive it."
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
        "epoch={epoch:03d} step={global_step:05d} "
        "train={train_loss_normalized_mae:.6f} "
        "val={validation_loss_normalized_mae:.6f} "
        "val_MAE={validation_MAE:.6f} "
        "tf={teacher_forcing_ratio_mean:.4f} "
        "time={epoch_compute_seconds:.1f}s "
        "val_time={validation_seconds:.1f}s "
        "best={improved}".format(**row),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a DCRNN or STGCN baseline for the BWDF "
            "24h or 168h task."
        )
    )
    parser.add_argument(
        "--model",
        choices=["dcrnn", "stgcn"],
        default="dcrnn",
        help="Model-specific implementation behind the shared CLI.",
    )
    parser.add_argument(
        "--task",
        choices=["24h", "168h"],
        required=True,
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="auto, cpu, or a PyTorch logical index such as cuda:0.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume output_dir/checkpoint_last.pt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Archive an existing run directory before starting.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Debug override; formal runs should use the YAML value.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Debug override for smoke tests.",
    )
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive.")

    config_path = (
        args.config
        or _default_config(args.model, args.task)
    ).resolve()
    config = load_config_with_inheritance(PROJECT_ROOT, config_path)
    expected_horizon = 24 if args.task == "24h" else 168
    if str(config["model"]["name"]).lower() != args.model:
        raise ValueError(
            f"Model/config mismatch: --model {args.model}, "
            f"config model={config['model']['name']}."
        )
    if int(config["task"]["horizon"]) != expected_horizon:
        raise ValueError(
            f"Task/config horizon mismatch: --task {args.task}, "
            f"config horizon={config['task']['horizon']}."
        )
    if bool(config["runtime"].get("allow_cpu_fallback", False)):
        raise ValueError(
            "Formal BWDF training requires allow_cpu_fallback=false."
        )
    configured_seeds = [
        int(value)
        for value in config["training"]["seeds"]
    ]
    if int(args.seed) not in configured_seeds:
        raise ValueError(
            f"seed={args.seed} is outside the configured experiment "
            f"seeds {configured_seeds}."
        )

    requested_device = args.device or str(
        config["runtime"]["device"]
    )
    selection = resolve_training_device(
        requested_device,
        runtime_config=config["runtime"],
    )
    print(
        json.dumps(
            {
                "device_selection": selection.state_dict(),
                "cuda_visible_device_count": (
                    torch.cuda.device_count()
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if selection.explicit_cpu:
        print(
            "WARNING: explicit CPU debug run; this is not a formal "
            "GPU experiment.",
            flush=True,
        )

    output_dir = _resolve_output_dir(
        config=config,
        seed=args.seed,
        override=args.output_dir,
    )
    resume_path: Path | None = None
    archived_output: Path | None = None
    if args.resume:
        resume_path = output_dir / "checkpoint_last.pt"
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resume_path}"
            )
    else:
        archived_output = _prepare_new_output(
            output_dir,
            overwrite=args.overwrite,
        )

    data_builder = (
        prepare_dcrnn_training_data
        if args.model == "dcrnn"
        else prepare_forecast_training_data
    )
    training_data = data_builder(
        project_root=PROJECT_ROOT,
        config=config,
        data_dir=args.data_dir,
    )
    # Model parameters must be initialised only after the run seed is set.
    set_reproducible_seed(args.seed)
    if args.model == "dcrnn":
        model = build_dcrnn_model(
            config,
            project_root=PROJECT_ROOT,
            input_dim=training_data.input_dim,
            future_exog_dim=training_data.future_exog_dim,
            horizon=int(config["task"]["horizon"]),
            device=selection.device,
        )
    else:
        model = build_stgcn_model(
            config,
            project_root=PROJECT_ROOT,
            input_dim=training_data.input_dim,
            future_exog_dim=training_data.future_exog_dim,
            horizon=int(config["task"]["horizon"]),
            history_hours=int(config["task"]["history_hours"]),
            device=selection.device,
        )
    runtime_metadata = {
        "device_selection": selection.state_dict(),
        "config_path": str(config_path),
        "data_dir_override": (
            None if args.data_dir is None else str(args.data_dir.resolve())
        ),
        "output_archived_before_run": (
            None
            if archived_output is None
            else str(archived_output)
        ),
        "debug_max_epochs_override": args.max_epochs,
        "debug_max_train_batches": args.max_train_batches,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    trainer = train_dcrnn if args.model == "dcrnn" else train_stgcn
    result = trainer(
        config=config,
        model=model,
        training_data=training_data,
        device=selection.device,
        output_dir=output_dir,
        seed=args.seed,
        runtime_metadata=runtime_metadata,
        resume_checkpoint=resume_path,
        max_epochs_override=args.max_epochs,
        max_train_batches=args.max_train_batches,
        progress_callback=_progress,
    )
    print(
        json.dumps(
            result.state_dict(),
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
