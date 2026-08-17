"""Checkpoint-only DCRNN inference and BWDF metric generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dma_wdf.data.dcrnn_dataset import (
    ZScoreScaler,
    prepare_dcrnn_test_data,
)
from dma_wdf.data.metrics import compute_metrics
from dma_wdf.evaluation.paper_comparison import (
    compare_metrics_to_paper,
    load_paper_reference,
    write_paper_comparison_report,
)
from dma_wdf.models.dcrnn import build_dcrnn_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_predictions(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dma_names: list[str],
) -> pd.DataFrame:
    """Return publisher-table A-J plus total metrics.

    The MSCMNet supplementary tables use a mixed ``total`` convention:

    - total MAE is the sum of the ten DMA-level MAE values;
    - total MAPE, RMSE, and NSE are calculated from the hourly total-demand
      series obtained by summing the ten DMA values first.

    The MAE convention is verified directly from all six models in both
    supplementary MAE tables: every displayed total equals the sum of A-J
    up to the published three-decimal rounding.
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"Prediction shape mismatch: {pred.shape} != {true.shape}.")
    if true.ndim != 3:
        raise ValueError("Predictions must have shape (samples,horizon,nodes).")
    if true.shape[2] != len(dma_names):
        raise ValueError("DMA-name count does not match prediction node count.")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Predictions/targets contain NaN or Inf.")

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(dma_names):
        rows.append(
            {
                "entity": str(name),
                **compute_metrics(true[:, :, index], pred[:, :, index]),
            }
        )
    total_metrics = compute_metrics(
        true.sum(axis=2),
        pred.sum(axis=2),
    )
    total_metrics["MAE"] = float(
        sum(float(row["MAE"]) for row in rows)
    )
    rows.append({"entity": "total", **total_metrics})
    return pd.DataFrame(rows, columns=["entity", "MAE", "MAPE", "RMSE", "NSE"])


def evaluate_aggregate_total_predictions(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return pure total-demand metrics for operational interpretation.

    Unlike the publisher table's mixed ``total`` row, every metric here is
    computed after summing A-J at each sample-hour.  This keeps the operational
    aggregate-demand MAE available without comparing it to the publisher's
    sum-of-DMA-MAE value.
    """
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"Prediction shape mismatch: {pred.shape} != {true.shape}.")
    if true.ndim != 3:
        raise ValueError("Predictions must have shape (samples,horizon,nodes).")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Predictions/targets contain NaN or Inf.")
    return pd.DataFrame(
        [
            {
                "entity": "aggregate_total",
                **compute_metrics(
                    true.sum(axis=2),
                    pred.sum(axis=2),
                ),
            }
        ],
        columns=["entity", "MAE", "MAPE", "RMSE", "NSE"],
    )


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    expected_task: str,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    expected_name = "single_step_24h" if expected_task == "24h" else "multi_step_168h"
    expected_horizon = 24 if expected_task == "24h" else 168
    expected = {
        "kind": "dma_wdf_dcrnn_training_checkpoint",
        "model_name": "dcrnn",
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


def _validate_checkpoint_graph_identity(
    *,
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Verify the checkpoint and current project use the same frozen graph.

    This check must run before ``load_state_dict``.  The random-walk matrix is
    a persistent model buffer, so loading the checkpoint first could silently
    replace the matrix constructed from the current graph artifact.
    """
    checkpoint_metadata = checkpoint.get("model_metadata")
    if not isinstance(checkpoint_metadata, dict):
        raise ValueError("Checkpoint is missing model_metadata.")
    checkpoint_graph = checkpoint_metadata.get("graph")
    if not isinstance(checkpoint_graph, dict):
        raise ValueError("Checkpoint is missing model_metadata.graph.")

    current_metadata = model.model_metadata()
    current_graph = current_metadata.get("graph")
    if not isinstance(current_graph, dict):
        raise ValueError("Current model is missing graph metadata.")

    # artifact_path is intentionally excluded: the same immutable artifact
    # may live at a different absolute path on another machine.
    identity_fields = (
        "artifact_sha256",
        "demand_sha256",
        "graph_method",
        "corr_threshold",
        "normalization",
        "matrix_key",
        "fit_start",
        "fit_end",
        "fit_rows",
        "node_names",
        "dma_columns",
    )
    for field in identity_fields:
        if field not in checkpoint_graph:
            raise ValueError(
                f"Checkpoint graph metadata is missing {field!r}."
            )
        if field not in current_graph:
            raise ValueError(
                f"Current graph metadata is missing {field!r}."
            )
        if checkpoint_graph[field] != current_graph[field]:
            raise ValueError(
                "Checkpoint/current graph mismatch for "
                f"{field}: {checkpoint_graph[field]!r} != "
                f"{current_graph[field]!r}."
            )

    return {
        "verified": True,
        "artifact_sha256": str(
            checkpoint_graph["artifact_sha256"]
        ),
        "demand_sha256": str(checkpoint_graph["demand_sha256"]),
        "fit_start": str(checkpoint_graph["fit_start"]),
        "fit_end": str(checkpoint_graph["fit_end"]),
        "fit_rows": int(checkpoint_graph["fit_rows"]),
    }


def _predict_all(
    *,
    model: torch.nn.Module,
    x_past: np.ndarray,
    future_exog: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dataset = TensorDataset(
        torch.from_numpy(x_past),
        torch.from_numpy(future_exog),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, future_batch in loader:
            x_batch = x_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            future_batch = future_batch.to(
                device,
                non_blocking=device.type == "cuda",
            )
            prediction = model(
                x_batch,
                x_future_exog=future_batch,
                teacher_forcing_ratio=0.0,
            )
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Model produced NaN/Inf predictions.")
            predictions.append(prediction.cpu().numpy())
    if not predictions:
        raise RuntimeError("Test loader produced no batches.")
    return np.concatenate(predictions, axis=0).astype(np.float32)


def run_dcrnn_checkpoint_evaluation(
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
    """Evaluate one best checkpoint and write all protocol artifacts."""
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
    demand_scaler = ZScoreScaler.from_state_dict(checkpoint["demand_scaler"])
    weather_scaler = ZScoreScaler.from_state_dict(checkpoint["weather_scaler"])
    test_data = prepare_dcrnn_test_data(
        project_root=project_root,
        config=config,
        demand_scaler=demand_scaler,
        weather_scaler=weather_scaler,
        data_dir=data_dir,
    )
    model = build_dcrnn_model(
        config,
        project_root=project_root,
        input_dim=len(checkpoint["input_feature_names"]),
        future_exog_dim=len(checkpoint["future_exog_feature_names"]),
        horizon=int(checkpoint["horizon"]),
        device=device,
    )
    graph_identity = _validate_checkpoint_graph_identity(
        checkpoint=checkpoint,
        model=model,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    prediction_scaled = _predict_all(
        model=model,
        x_past=test_data.test.x_past,
        future_exog=test_data.test.future_exog,
        device=device,
        batch_size=batch_size,
    )
    prediction_raw = demand_scaler.inverse_transform(prediction_scaled)
    truth_raw = test_data.test.y_raw
    dma_letters = list("ABCDEFGHIJ")
    if len(test_data.dma_columns) != len(dma_letters):
        raise ValueError("Paper comparison requires exactly ten DMAs.")

    metrics_by_protocol: dict[str, str] = {}
    aggregate_total_metrics_by_protocol: dict[str, str] = {}
    for protocol, indices in test_data.protocol_indices.items():
        metrics = evaluate_predictions(
            y_true=truth_raw[indices],
            y_pred=prediction_raw[indices],
            dma_names=dma_letters,
        )
        path = output_dir / f"metrics_{protocol}.csv"
        metrics.to_csv(path, index=False, float_format="%.10f")
        metrics_by_protocol[protocol] = str(path)
        aggregate_total = evaluate_aggregate_total_predictions(
            y_true=truth_raw[indices],
            y_pred=prediction_raw[indices],
        )
        aggregate_path = (
            output_dir / f"metrics_aggregate_total_{protocol}.csv"
        )
        aggregate_total.to_csv(
            aggregate_path,
            index=False,
            float_format="%.10f",
        )
        aggregate_total_metrics_by_protocol[protocol] = str(
            aggregate_path
        )

    common_indices = test_data.protocol_indices["common_46"]
    if len(common_indices) != 46:
        raise ValueError("Publisher comparison requires exactly 46 sequences.")
    common_metrics = pd.read_csv(output_dir / "metrics_common_46.csv")
    reference = load_paper_reference(paper_reference_path)
    comparison = compare_metrics_to_paper(
        task=task,
        dcrnn_metrics=common_metrics,
        reference=reference,
    )
    comparison_path = output_dir / "paper_comparison.csv"
    comparison.to_csv(comparison_path, index=False, float_format="%.10f")
    report_path = output_dir / "paper_comparison_report.md"
    comparison_summary = write_paper_comparison_report(
        comparison=comparison,
        task=task,
        reference=reference,
        output_path=report_path,
    )
    (output_dir / "paper_comparison_summary.json").write_text(
        json.dumps(comparison_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    np.savez_compressed(
        output_dir / "predictions.npz",
        y_true=truth_raw,
        y_pred=prediction_raw,
        forecast_starts=np.asarray(test_data.test.forecast_starts),
        operational_indices=test_data.protocol_indices["operational"],
        strict_within_test_indices=(
            test_data.protocol_indices["strict_within_test"]
        ),
        common_46_indices=common_indices,
        dma_columns=np.asarray(test_data.dma_columns),
        dma_letters=np.asarray(dma_letters),
    )
    summary = {
        "status": "completed",
        "model": "dcrnn",
        "task": task,
        "horizon": int(checkpoint["horizon"]),
        "seed": int(checkpoint["seed"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "graph_identity": graph_identity,
        "device": str(device),
        "teacher_forcing_ratio": 0.0,
        "test_candidates": int(test_data.test.num_samples),
        "protocol_counts": {
            name: int(len(indices))
            for name, indices in test_data.protocol_indices.items()
        },
        "paper_comparison_protocol": "common_46",
        "paper_reference": str(paper_reference_path.resolve()),
        "paper_reference_sha256": reference["_metadata"]["sha256"],
        "paper_metric_conventions": {
            "total_MAE": "sum_of_A_to_J_dma_mae",
            "total_MAPE_RMSE_NSE": (
                "metric_on_hourly_sum_of_A_to_J_demand"
            ),
        },
        "metrics": metrics_by_protocol,
        "aggregate_total_metrics": (
            aggregate_total_metrics_by_protocol
        ),
        "paper_comparison_csv": str(comparison_path),
        "paper_comparison_report": str(report_path),
        "predictions": str(output_dir / "predictions.npz"),
        "test_targets_used_for_training_or_selection": False,
    }
    (output_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
