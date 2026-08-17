"""Checkpoint-only STGCN inference using the established BWDF protocols."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dma_wdf.data.forecast_dataset import (
    ZScoreScaler,
    prepare_forecast_test_data,
)
from dma_wdf.evaluation.dcrnn_evaluator import (
    _predict_all,
    _sha256,
    _validate_checkpoint_graph_identity,
    evaluate_aggregate_total_predictions,
    evaluate_predictions,
)
from dma_wdf.evaluation.paper_comparison import (
    compare_named_model_metrics_to_paper,
    load_paper_reference,
    write_named_model_comparison_report,
)
from dma_wdf.models.stgcn import build_stgcn_model


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    expected_task: str,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    expected_name = (
        "single_step_24h"
        if expected_task == "24h"
        else "multi_step_168h"
    )
    expected_horizon = 24 if expected_task == "24h" else 168
    expected = {
        "kind": "dma_wdf_stgcn_training_checkpoint",
        "model_name": "stgcn",
        "task_name": expected_name,
        "horizon": expected_horizon,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Checkpoint mismatch for {key}: "
                f"{checkpoint.get(key)!r} != {value!r}."
            )
    return checkpoint


def _validate_stgcn_graph_details(
    *,
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
) -> None:
    checkpoint_graph = checkpoint["model_metadata"]["graph"]
    current_graph = model.model_metadata()["graph"]
    for field in (
        "source_artifact_normalization",
        "chebyshev_order",
        "lambda_max",
    ):
        if field not in checkpoint_graph or field not in current_graph:
            raise ValueError(
                f"STGCN graph metadata is missing {field!r}."
            )
        if field == "lambda_max":
            mismatch = abs(
                float(checkpoint_graph[field])
                - float(current_graph[field])
            )
            if mismatch > 1.0e-12:
                raise ValueError(
                    "Checkpoint/current STGCN graph mismatch for "
                    f"{field}: error={mismatch}."
                )
        elif checkpoint_graph[field] != current_graph[field]:
            raise ValueError(
                "Checkpoint/current STGCN graph mismatch for "
                f"{field}: {checkpoint_graph[field]!r} != "
                f"{current_graph[field]!r}."
            )


def run_stgcn_checkpoint_evaluation(
    *,
    project_root: Path,
    task: str,
    checkpoint_path: Path,
    output_dir: Path,
    paper_reference_path: Path,
    device: torch.device,
    batch_size: int = 16,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one STGCN checkpoint without changing DCRNN evaluation."""
    if task not in {"24h", "168h"}:
        raise ValueError("task must be '24h' or '168h'.")
    project_root = project_root.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_checkpoint(
        checkpoint_path,
        device=device,
        expected_task=task,
    )
    config = checkpoint["resolved_config"]
    demand_scaler = ZScoreScaler.from_state_dict(
        checkpoint["demand_scaler"]
    )
    weather_scaler = ZScoreScaler.from_state_dict(
        checkpoint["weather_scaler"]
    )
    test_data = prepare_forecast_test_data(
        project_root=project_root,
        config=config,
        demand_scaler=demand_scaler,
        weather_scaler=weather_scaler,
        data_dir=data_dir,
    )
    model = build_stgcn_model(
        config,
        project_root=project_root,
        input_dim=len(checkpoint["input_feature_names"]),
        future_exog_dim=len(
            checkpoint["future_exog_feature_names"]
        ),
        horizon=int(checkpoint["horizon"]),
        history_hours=int(config["task"]["history_hours"]),
        device=device,
    )
    graph_identity = _validate_checkpoint_graph_identity(
        checkpoint=checkpoint,
        model=model,
    )
    _validate_stgcn_graph_details(
        checkpoint=checkpoint,
        model=model,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    prediction_scaled = _predict_all(
        model=model,
        x_past=test_data.test.x_past,
        future_exog=test_data.test.future_exog,
        device=device,
        batch_size=batch_size,
    )
    prediction_raw = demand_scaler.inverse_transform(
        prediction_scaled
    )
    truth_raw = test_data.test.y_raw
    dma_letters = list("ABCDEFGHIJ")
    if len(test_data.dma_columns) != len(dma_letters):
        raise ValueError(
            "Paper comparison requires exactly ten DMAs."
        )

    metrics_by_protocol: dict[str, str] = {}
    aggregate_paths: dict[str, str] = {}
    for protocol, indices in test_data.protocol_indices.items():
        metrics = evaluate_predictions(
            y_true=truth_raw[indices],
            y_pred=prediction_raw[indices],
            dma_names=dma_letters,
        )
        path = output_dir / f"metrics_{protocol}.csv"
        metrics.to_csv(path, index=False, float_format="%.10f")
        metrics_by_protocol[protocol] = str(path)
        aggregate = evaluate_aggregate_total_predictions(
            y_true=truth_raw[indices],
            y_pred=prediction_raw[indices],
        )
        aggregate_path = (
            output_dir
            / f"metrics_aggregate_total_{protocol}.csv"
        )
        aggregate.to_csv(
            aggregate_path,
            index=False,
            float_format="%.10f",
        )
        aggregate_paths[protocol] = str(aggregate_path)

    common_indices = test_data.protocol_indices["common_46"]
    if len(common_indices) != 46:
        raise ValueError(
            "Publisher comparison requires exactly 46 sequences."
        )
    common_metrics = pd.read_csv(
        output_dir / "metrics_common_46.csv"
    )
    reference = load_paper_reference(paper_reference_path)
    comparison = compare_named_model_metrics_to_paper(
        task=task,
        model_name="stgcn",
        model_metrics=common_metrics,
        reference=reference,
    )
    comparison_path = output_dir / "paper_comparison.csv"
    comparison.to_csv(
        comparison_path,
        index=False,
        float_format="%.10f",
    )
    report_path = output_dir / "paper_comparison_report.md"
    comparison_summary = write_named_model_comparison_report(
        comparison=comparison,
        task=task,
        model_name="stgcn",
        reference=reference,
        output_path=report_path,
    )
    (
        output_dir / "paper_comparison_summary.json"
    ).write_text(
        json.dumps(
            comparison_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prediction_path = output_dir / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        y_true=truth_raw,
        y_pred=prediction_raw,
        forecast_starts=np.asarray(
            test_data.test.forecast_starts
        ),
        operational_indices=(
            test_data.protocol_indices["operational"]
        ),
        strict_within_test_indices=(
            test_data.protocol_indices["strict_within_test"]
        ),
        common_46_indices=common_indices,
        dma_columns=np.asarray(test_data.dma_columns),
        dma_letters=np.asarray(dma_letters),
    )
    summary = {
        "status": "completed",
        "model": "stgcn",
        "task": task,
        "horizon": int(checkpoint["horizon"]),
        "seed": int(checkpoint["seed"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "graph_identity": graph_identity,
        "device": str(device),
        "decoder": "direct_future_conditioned",
        "teacher_forcing_ratio": 0.0,
        "test_candidates": int(test_data.test.num_samples),
        "protocol_counts": {
            name: int(len(indices))
            for name, indices in test_data.protocol_indices.items()
        },
        "paper_comparison_protocol": "common_46",
        "paper_reference": str(paper_reference_path.resolve()),
        "paper_reference_sha256": (
            reference["_metadata"]["sha256"]
        ),
        "paper_metric_conventions": {
            "total_MAE": "sum_of_A_to_J_dma_mae",
            "total_MAPE_RMSE_NSE": (
                "metric_on_hourly_sum_of_A_to_J_demand"
            ),
        },
        "metrics": metrics_by_protocol,
        "aggregate_total_metrics": aggregate_paths,
        "paper_comparison_csv": str(comparison_path),
        "paper_comparison_report": str(report_path),
        "predictions": str(prediction_path),
        "test_targets_used_for_training_or_selection": False,
    }
    (output_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
