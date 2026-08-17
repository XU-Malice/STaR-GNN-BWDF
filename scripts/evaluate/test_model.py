#!/usr/bin/env python
"""Unified DCRNN/STGCN test entry point with paper comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.evaluation.dcrnn_evaluator import (  # noqa: E402
    run_dcrnn_checkpoint_evaluation,
)
from dma_wdf.evaluation.stgcn_evaluator import (  # noqa: E402
    run_stgcn_checkpoint_evaluation,
)


def _default_checkpoint(
    model: str,
    task: str,
    seed: int,
) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / "baselines"
        / model
        / task
        / f"seed_{seed}"
        / "checkpoint_best.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test a best DCRNN or STGCN checkpoint on all BWDF "
            "protocols and compare common_46 metrics with "
            "MSCMNet_W/WM."
        )
    )
    parser.add_argument(
        "--model",
        choices=["dcrnn", "stgcn"],
        default="dcrnn",
    )
    parser.add_argument("--task", choices=["24h", "168h"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--paper-reference",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "evaluation"
            / "mscmnet_paper_metrics.yaml"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Evaluation defaults to CPU to avoid occupying a training GPU; "
            "use a visible logical device such as cuda:0 when desired."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    checkpoint = (
        args.checkpoint
        if args.checkpoint is not None
        else _default_checkpoint(
            args.model,
            args.task,
            args.seed,
        )
    ).resolve()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else checkpoint.parent / "evaluation"
    ).resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable.")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    evaluator = (
        run_dcrnn_checkpoint_evaluation
        if args.model == "dcrnn"
        else run_stgcn_checkpoint_evaluation
    )
    summary = evaluator(
        project_root=PROJECT_ROOT,
        task=args.task,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        paper_reference_path=args.paper_reference,
        device=device,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
