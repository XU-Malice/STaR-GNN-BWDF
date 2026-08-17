#!/usr/bin/env python
"""Build and smoke-test STGCN without training or reading targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.models.stgcn import build_stgcn_model
from dma_wdf.utils.config import read_yaml


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": int(
            sum(value.numel() for value in model.parameters())
        ),
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
        model = build_stgcn_model(
            config,
            project_root=PROJECT_ROOT,
            input_dim=input_dim,
            future_exog_dim=future_exog_dim,
            horizon=horizon,
            history_hours=history_hours,
            device=device,
        )
        model.eval()
        x_past = torch.randn(
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
        with torch.inference_mode():
            output = model(
                x_past,
                x_future_exog=future,
                teacher_forcing_ratio=0.0,
            )
        expected = (batch_size, horizon, model.num_nodes)
        if tuple(output.shape) != expected:
            raise RuntimeError(
                f"Horizon {horizon}: output shape "
                f"{tuple(output.shape)} != {expected}."
            )
        if not torch.isfinite(output).all():
            raise RuntimeError(
                f"Horizon {horizon}: output contains NaN/Inf."
            )
        records.append(
            {
                "horizon": horizon,
                "output_shape": list(output.shape),
                "output_finite": True,
                "parameters": _parameter_counts(model),
                "metadata": model.model_metadata(),
            }
        )
        models.append(model)

    first = models[0]
    for model in models[1:]:
        if not torch.equal(
            first.chebyshev_supports,
            model.chebyshev_supports,
        ):
            raise RuntimeError(
                "Task models did not load identical graph supports."
            )
        if (
            first.model_metadata()["graph"]["artifact_sha256"]
            != model.model_metadata()["graph"]["artifact_sha256"]
        ):
            raise RuntimeError(
                "Task models did not load the same graph artifact."
            )
    return {
        "all_passed": True,
        "model": "stgcn",
        "config_path": str(config_path),
        "device": str(device),
        "history_hours": history_hours,
        "input_dim": input_dim,
        "future_exog_dim": future_exog_dim,
        "shared_graph_across_horizons": True,
        "target_passed_to_model": False,
        "models": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate the STGCN architecture only."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT / "configs" / "model" / "stgcn.yaml"
        ),
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[24, 168],
    )
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--future-exog-dim", type=int, default=7)
    parser.add_argument("--history-hours", type=int, default=672)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "model_validation"
            / "stgcn_validation.json"
        ),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA validation was requested but CUDA is unavailable."
        )
    report = validate(
        config_path=args.config.resolve(),
        horizons=[int(value) for value in args.horizons],
        input_dim=int(args.input_dim),
        future_exog_dim=int(args.future_exog_dim),
        history_hours=int(args.history_hours),
        batch_size=int(args.batch_size),
        device=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
