#!/usr/bin/env python
"""Validate shared BWDF tensors against one model's contract."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.data.forecast_dataset import (  # noqa: E402
    prepare_forecast_test_data,
    prepare_forecast_training_data,
)
from dma_wdf.models.dcrnn import build_dcrnn_model  # noqa: E402
from dma_wdf.models.stgcn import build_stgcn_model  # noqa: E402
from dma_wdf.utils.config import (  # noqa: E402
    load_config_with_inheritance,
)


def _check(
    records: list[dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
    passed: bool,
) -> None:
    records.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def validate_task(
    *,
    model_name: str,
    task: str,
    config_path: Path,
    data_dir: Path | None,
) -> dict[str, Any]:
    """Build shared tensors and the selected model without training."""
    config = load_config_with_inheritance(
        PROJECT_ROOT,
        config_path,
    )
    training = prepare_forecast_training_data(
        project_root=PROJECT_ROOT,
        config=config,
        data_dir=data_dir,
    )
    testing = prepare_forecast_test_data(
        project_root=PROJECT_ROOT,
        config=config,
        demand_scaler=training.demand_scaler,
        weather_scaler=training.weather_scaler,
        data_dir=data_dir,
    )
    builder = (
        build_dcrnn_model
        if model_name == "dcrnn"
        else build_stgcn_model
    )
    builder_kwargs = {
        "config": config,
        "project_root": PROJECT_ROOT,
        "input_dim": training.input_dim,
        "future_exog_dim": training.future_exog_dim,
        "horizon": int(config["task"]["horizon"]),
        "device": "cpu",
    }
    if model_name == "stgcn":
        builder_kwargs["history_hours"] = int(
            config["task"]["history_hours"]
        )
    model = builder(**builder_kwargs)

    split = config["split"]
    features = config["features"]
    horizon = int(config["task"]["horizon"])
    nodes = int(config["model"]["num_nodes"])
    history = int(config["task"]["history_hours"])
    checks: list[dict[str, Any]] = []
    expected_shapes = {
        "fit_x_past": (
            int(split["expected_fit_samples"]),
            history,
            nodes,
            int(features["expected_input_dim"]),
        ),
        "fit_y": (
            int(split["expected_fit_samples"]),
            horizon,
            nodes,
        ),
        "validation_x_past": (
            int(split["validation_samples"]),
            history,
            nodes,
            int(features["expected_input_dim"]),
        ),
        "test_x_past": (
            int(split["expected_test_candidates"]),
            history,
            nodes,
            int(features["expected_input_dim"]),
        ),
        "test_y": (
            int(split["expected_test_candidates"]),
            horizon,
            nodes,
        ),
    }
    observed_shapes = {
        "fit_x_past": tuple(training.fit.x_past.shape),
        "fit_y": tuple(training.fit.y_scaled.shape),
        "validation_x_past": tuple(
            training.validation.x_past.shape
        ),
        "test_x_past": tuple(testing.test.x_past.shape),
        "test_y": tuple(testing.test.y_scaled.shape),
    }
    for name, expected in expected_shapes.items():
        observed = observed_shapes[name]
        _check(
            checks,
            name,
            list(observed),
            list(expected),
            observed == expected,
        )
    arrays = {
        "fit_x_finite": training.fit.x_past,
        "fit_y_finite": training.fit.y_scaled,
        "validation_x_finite": training.validation.x_past,
        "validation_y_finite": training.validation.y_scaled,
        "test_x_finite": testing.test.x_past,
        "test_y_finite": testing.test.y_scaled,
    }
    for name, array in arrays.items():
        finite = bool(np.isfinite(array).all())
        _check(checks, name, finite, True, finite)

    first_validation = pd.Timestamp(
        training.metadata["first_validation_origin"]
    )
    scaler_end = pd.Timestamp(training.demand_scaler.fit_end)
    _check(
        checks,
        "scaler_fit_before_validation",
        {
            "fit_end": str(scaler_end),
            "first_validation_origin": str(first_validation),
        },
        "fit_end < first_validation_origin",
        scaler_end < first_validation,
    )
    _check(
        checks,
        "fit_validation_labels_do_not_overlap",
        training.metadata["labels_overlap"],
        False,
        training.metadata["labels_overlap"] is False,
    )
    _check(
        checks,
        "fixed_graph_fit_rows",
        training.metadata["graph_fit_rows"],
        17136,
        training.metadata["graph_fit_rows"] == 17136,
    )
    _check(
        checks,
        "model_input_dim_matches_data",
        model.input_dim,
        training.input_dim,
        model.input_dim == training.input_dim,
    )
    _check(
        checks,
        "model_future_exog_dim_matches_data",
        model.future_exog_dim,
        training.future_exog_dim,
        model.future_exog_dim == training.future_exog_dim,
    )
    expected_protocols = {
        "operational": int(
            split["expected_operational_samples"]
        ),
        "strict_within_test": int(
            split["expected_strict_samples"]
        ),
        "common_46": int(split["expected_common_samples"]),
    }
    observed_protocols = {
        name: int(len(indices))
        for name, indices in testing.protocol_indices.items()
    }
    for name, expected in expected_protocols.items():
        _check(
            checks,
            f"protocol_{name}",
            observed_protocols[name],
            expected,
            observed_protocols[name] == expected,
        )
    return {
        "model": model_name,
        "task": task,
        "all_passed": all(record["passed"] for record in checks),
        "config_path": str(config_path),
        "input_dim": training.input_dim,
        "future_exog_dim": training.future_exog_dim,
        "samples": {
            "development": training.metadata[
                "development_samples"
            ],
            "fit": training.fit.num_samples,
            "purge": len(training.purged_forecast_starts),
            "validation": training.validation.num_samples,
            "test": testing.test.num_samples,
        },
        "protocol_counts": observed_protocols,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate shared BWDF tensors for DCRNN or STGCN."
        )
    )
    parser.add_argument(
        "--model",
        choices=["dcrnn", "stgcn"],
        default="dcrnn",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["24h", "168h"],
        default=["24h", "168h"],
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (
            PROJECT_ROOT
            / "results"
            / "training"
            / args.model
            / "data_validation"
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    passed = True
    for task in args.tasks:
        config_path = (
            PROJECT_ROOT
            / "configs"
            / "train"
            / f"{args.model}_{task}.yaml"
        )
        result = validate_task(
            model_name=args.model,
            task=task,
            config_path=config_path,
            data_dir=args.data_dir,
        )
        output = output_dir / f"{task}.json"
        output.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "model": args.model,
                    "task": task,
                    "all_passed": result["all_passed"],
                    "samples": result["samples"],
                    "protocol_counts": result["protocol_counts"],
                    "report": str(output),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        passed = passed and bool(result["all_passed"])
        del result
        gc.collect()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
