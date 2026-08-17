#!/usr/bin/env python
"""Build and smoke-test DCRNN without training it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.models.dcrnn import build_dcrnn_model
from dma_wdf.utils.config import read_yaml


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": int(sum(value.numel() for value in model.parameters())),
        "trainable": int(
            sum(
                value.numel()
                for value in model.parameters()
                if value.requires_grad
            )
        ),
    }


def validate(
    *,
    config_path: Path,
    horizons: list[int],
    input_dim: int,
    future_exog_dim: int,
    history_hours: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    models = []
    records: list[dict[str, Any]] = []

    for horizon in horizons:
        model = build_dcrnn_model(
            config,
            project_root=PROJECT_ROOT,
            input_dim=input_dim,
            future_exog_dim=future_exog_dim,
            horizon=horizon,
            device=device,
        )
        model.eval()
        x = torch.randn(
            batch_size,
            history_hours,
            model.num_nodes,
            input_dim,
            device=device,
        )
        future = (
            torch.randn(
                batch_size,
                horizon,
                model.num_nodes,
                future_exog_dim,
                device=device,
            )
            if future_exog_dim > 0
            else None
        )
        with torch.no_grad():
            output = model(
                x,
                x_future_exog=future,
            )
        expected_shape = (batch_size, horizon, model.num_nodes)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"Horizon {horizon}: output shape "
                f"{tuple(output.shape)} != {expected_shape}."
            )
        if not torch.isfinite(output).all():
            raise RuntimeError(
                f"Horizon {horizon}: output contains NaN/Inf."
            )

        metadata = model.model_metadata()
        records.append(
            {
                "horizon": horizon,
                "output_shape": list(output.shape),
                "output_finite": True,
                "parameters": _parameter_counts(model),
                "metadata": metadata,
            }
        )
        models.append(model)

    if len(models) > 1:
        first = models[0]
        for model in models[1:]:
            if not torch.equal(first.random_walk, model.random_walk):
                raise RuntimeError(
                    "Task models did not load the same random_walk."
                )
            first_graph = first.model_metadata()["graph"]
            graph = model.model_metadata()["graph"]
            if (
                first_graph["artifact_sha256"]
                != graph["artifact_sha256"]
            ):
                raise RuntimeError(
                    "Task models did not load the same graph artifact."
                )

    return {
        "all_passed": True,
        "config_path": str(config_path),
        "device": str(device),
        "history_hours": history_hours,
        "input_dim": input_dim,
        "future_exog_dim": future_exog_dim,
        "shared_graph_across_horizons": True,
        "models": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate DCRNN architecture only."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model" / "dcrnn.yaml",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[24, 168],
    )
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--future-exog-dim", type=int, default=7)
    parser.add_argument("--history-hours", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "model_validation"
            / "dcrnn_validation.json"
        ),
    )
    args = parser.parse_args()

    report = validate(
        config_path=args.config.resolve(),
        horizons=[int(value) for value in args.horizons],
        input_dim=int(args.input_dim),
        future_exog_dim=int(args.future_exog_dim),
        history_hours=int(args.history_hours),
        batch_size=int(args.batch_size),
        device=torch.device(args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
