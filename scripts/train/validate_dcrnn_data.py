#!/usr/bin/env python
"""Validate DCRNN data contracts before any long training run."""

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

from dma_wdf.data.dcrnn_dataset import (  # noqa: E402
    prepare_dcrnn_test_data,
    prepare_dcrnn_training_data,
)
from dma_wdf.models.dcrnn import build_dcrnn_model  # noqa: E402
from dma_wdf.utils.config import load_config_with_inheritance  # noqa: E402


def _record(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
    passed: bool,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def validate_task(
    *,
    task: str,
    config_path: Path,
    data_dir: Path | None,
) -> dict[str, Any]:
    config = load_config_with_inheritance(PROJECT_ROOT, config_path)
    training = prepare_dcrnn_training_data(
        project_root=PROJECT_ROOT,
        config=config,
        data_dir=data_dir,
    )
    testing = prepare_dcrnn_test_data(
        project_root=PROJECT_ROOT,
        config=config,
        demand_scaler=training.demand_scaler,
        weather_scaler=training.weather_scaler,
        data_dir=data_dir,
    )
    model = build_dcrnn_model(
        config,
        project_root=PROJECT_ROOT,
        input_dim=training.input_dim,
        future_exog_dim=training.future_exog_dim,
        horizon=int(config["task"]["horizon"]),
        device="cpu",
    )

    split = config["split"]
    features = config["features"]
    horizon = int(config["task"]["horizon"])
    checks: list[dict[str, Any]] = []
    _record(
        checks,
        "input_dim",
        training.input_dim,
        int(features["expected_input_dim"]),
        training.input_dim == int(features["expected_input_dim"]),
    )
    _record(
        checks,
        "future_exog_dim",
        training.future_exog_dim,
        int(features["expected_future_exog_dim"]),
        training.future_exog_dim
        == int(features["expected_future_exog_dim"]),
    )
    _record(
        checks,
        "fit_samples",
        training.fit.num_samples,
        int(split["expected_fit_samples"]),
        training.fit.num_samples == int(split["expected_fit_samples"]),
    )
    _record(
        checks,
        "validation_samples",
        training.validation.num_samples,
        int(split["validation_samples"]),
        training.validation.num_samples == int(split["validation_samples"]),
    )
    _record(
        checks,
        "purge_samples",
        len(training.purged_forecast_starts),
        int(split["purge_samples"]),
        len(training.purged_forecast_starts)
        == int(split["purge_samples"]),
    )
    _record(
        checks,
        "test_candidates",
        testing.test.num_samples,
        int(split["expected_test_candidates"]),
        testing.test.num_samples == int(split["expected_test_candidates"]),
    )

    expected_shapes = {
        "fit_x_past": (
            int(split["expected_fit_samples"]),
            int(config["task"]["history_hours"]),
            int(config["model"]["num_nodes"]),
            int(features["expected_input_dim"]),
        ),
        "fit_y": (
            int(split["expected_fit_samples"]),
            horizon,
            int(config["model"]["num_nodes"]),
        ),
        "validation_x_past": (
            int(split["validation_samples"]),
            int(config["task"]["history_hours"]),
            int(config["model"]["num_nodes"]),
            int(features["expected_input_dim"]),
        ),
        "test_x_past": (
            int(split["expected_test_candidates"]),
            int(config["task"]["history_hours"]),
            int(config["model"]["num_nodes"]),
            int(features["expected_input_dim"]),
        ),
        "test_y": (
            int(split["expected_test_candidates"]),
            horizon,
            int(config["model"]["num_nodes"]),
        ),
    }
    observed_shapes = {
        "fit_x_past": tuple(training.fit.x_past.shape),
        "fit_y": tuple(training.fit.y_scaled.shape),
        "validation_x_past": tuple(training.validation.x_past.shape),
        "test_x_past": tuple(testing.test.x_past.shape),
        "test_y": tuple(testing.test.y_scaled.shape),
    }
    for name, expected in expected_shapes.items():
        observed = observed_shapes[name]
        _record(
            checks,
            name,
            list(observed),
            list(expected),
            observed == expected,
        )

    for name, array in [
        ("fit_x_finite", training.fit.x_past),
        ("fit_y_finite", training.fit.y_scaled),
        ("validation_x_finite", training.validation.x_past),
        ("validation_y_finite", training.validation.y_scaled),
        ("test_x_finite", testing.test.x_past),
        ("test_y_finite", testing.test.y_scaled),
    ]:
        finite = bool(np.isfinite(array).all())
        _record(checks, name, finite, True, finite)

    scaler_before_validation = (
        pd.Timestamp(training.demand_scaler.fit_end)
        < pd.Timestamp(training.metadata["first_validation_origin"])
    )
    _record(
        checks,
        "scaler_fit_before_validation",
        {
            "fit_end": training.demand_scaler.fit_end,
            "first_validation_origin": training.metadata[
                "first_validation_origin"
            ],
        },
        "fit_end < first_validation_origin",
        scaler_before_validation,
    )
    _record(
        checks,
        "fit_validation_labels_do_not_overlap",
        training.metadata["labels_overlap"],
        False,
        training.metadata["labels_overlap"] is False,
    )
    _record(
        checks,
        "fixed_graph_fit_rows",
        training.metadata["graph_fit_rows"],
        17136,
        training.metadata["graph_fit_rows"] == 17136,
    )
    _record(
        checks,
        "model_input_dim_matches_data",
        model.input_dim,
        training.input_dim,
        model.input_dim == training.input_dim,
    )
    _record(
        checks,
        "model_future_exog_dim_matches_data",
        model.future_exog_dim,
        training.future_exog_dim,
        model.future_exog_dim == training.future_exog_dim,
    )
    protocol_expected = {
        "operational": int(split["expected_operational_samples"]),
        "strict_within_test": int(split["expected_strict_samples"]),
        "common_46": int(split["expected_common_samples"]),
    }
    protocol_observed = {
        name: int(len(indices))
        for name, indices in testing.protocol_indices.items()
    }
    for name, expected in protocol_expected.items():
        _record(
            checks,
            f"protocol_{name}",
            protocol_observed[name],
            expected,
            protocol_observed[name] == expected,
        )

    all_passed = all(item["passed"] for item in checks)
    return {
        "task": task,
        "all_passed": all_passed,
        "config_path": str(config_path),
        "input_dim": training.input_dim,
        "future_exog_dim": training.future_exog_dim,
        "samples": {
            "development": training.metadata["development_samples"],
            "fit": training.fit.num_samples,
            "purge": len(training.purged_forecast_starts),
            "validation": training.validation.num_samples,
            "test": testing.test.num_samples,
        },
        "protocol_counts": protocol_observed,
        "scalers": {
            "demand": training.demand_scaler.state_dict(),
            "weather": training.weather_scaler.state_dict(),
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate DCRNN data construction for 24h and/or 168h."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["24h", "168h"],
        default=["24h", "168h"],
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "training"
        / "dcrnn"
        / "data_validation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for task in args.tasks:
        config_path = (
            PROJECT_ROOT / "configs" / "train" / f"dcrnn_{task}.yaml"
        )
        result = validate_task(
            task=task,
            config_path=config_path,
            data_dir=args.data_dir,
        )
        output_path = args.output_dir / f"{task}.json"
        output_path.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "task": task,
                    "all_passed": result["all_passed"],
                    "samples": result["samples"],
                    "protocol_counts": result["protocol_counts"],
                    "report": str(output_path),
                },
                indent=2,
            )
        )
        del result
        gc.collect()

    if not all(result["all_passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
