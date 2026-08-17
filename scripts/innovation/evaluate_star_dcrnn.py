#!/usr/bin/env python
"""Evaluate one frozen STaR-DCRNN checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.evaluation.star_evaluator import (  # noqa: E402
    run_star_checkpoint_evaluation,
)
from dma_wdf.models.star_dcrnn import FORMAL_VARIANTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="star_dcrnn")
    parser.add_argument(
        "--variant", choices=list(FORMAL_VARIANTS), required=True
    )
    parser.add_argument("--task", choices=["24h", "168h"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
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
    args = parser.parse_args()
    if args.model != "star_dcrnn":
        parser.error("--model must be star_dcrnn.")
    run_dir = (
        PROJECT_ROOT
        / "results"
        / "innovations"
        / "star_dcrnn_fadpr"
        / args.variant
        / args.task
        / f"seed_{args.seed}"
    )
    checkpoint = (args.checkpoint or run_dir / "checkpoint_best.pt").resolve()
    output_dir = (args.output_dir or run_dir / "evaluation").resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable.")
    summary = run_star_checkpoint_evaluation(
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
