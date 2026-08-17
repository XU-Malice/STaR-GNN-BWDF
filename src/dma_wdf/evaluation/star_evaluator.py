"""Checkpoint evaluation for the predeclared STaR-DCRNN variants."""

from __future__ import annotations

import hashlib
import json
import time
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
from dma_wdf.evaluation.dcrnn_evaluator import (
    _validate_checkpoint_graph_identity,
    evaluate_aggregate_total_predictions,
    evaluate_predictions,
)
from dma_wdf.evaluation.paper_comparison import (
    compare_metrics_to_paper,
    load_paper_reference,
    write_paper_comparison_report,
)
from dma_wdf.models.star_dcrnn import (
    FORMAL_VARIANTS,
    STaRDCRNN,
    STaRForwardDetails,
    build_star_dcrnn_model,
)
from dma_wdf.training.star_engine import CHECKPOINT_KIND


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predict_with_diagnostics(
    *,
    model: STaRDCRNN,
    x_past: np.ndarray,
    future_exog: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_past),
            torch.from_numpy(future_exog),
        ),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    predictions: list[np.ndarray] = []
    attention: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    future_mean_daily: list[np.ndarray] = []
    future_std_daily: list[np.ndarray] = []
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_start = time.perf_counter()
    with torch.inference_mode():
        for x_batch, future_batch in loader:
            details = model(
                x_batch.to(device),
                x_future_exog=future_batch.to(device),
                teacher_forcing_ratio=0.0,
                return_details=True,
            )
            if not isinstance(details, STaRForwardDetails):
                raise TypeError("STaR model did not return diagnostics.")
            if not torch.isfinite(details.prediction).all():
                raise FloatingPointError("Non-finite predictions.")
            predictions.append(details.prediction.cpu().numpy())
            if details.fa_dpr is not None:
                attention.append(
                    details.fa_dpr.attention_weights.cpu().numpy()
                )
                gates.append(details.fa_dpr.gate.cpu().numpy())
            if details.dssn_sasr is not None:
                future_mean_daily.append(
                    details.dssn_sasr.future_mean_daily.cpu().numpy()
                )
                future_std_daily.append(
                    torch.exp(
                        details.dssn_sasr.future_log_std_daily
                    ).cpu().numpy()
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start
    diagnostics: dict[str, np.ndarray] = {}
    if attention:
        diagnostics["attention_weights"] = np.concatenate(attention, axis=0)
        diagnostics["fusion_gate"] = np.concatenate(gates, axis=0)
    if future_mean_daily:
        diagnostics["future_mean_daily_scaled"] = np.concatenate(
            future_mean_daily,
            axis=0,
        )
        diagnostics["future_std_daily_scaled"] = np.concatenate(
            future_std_daily,
            axis=0,
        )
        assert model.dssn_sasr is not None
        diagnostics["alpha_mean"] = (
            model.dssn_sasr.restorer.alpha_mean.detach().cpu().numpy()
        )
        diagnostics["alpha_std"] = (
            model.dssn_sasr.restorer.alpha_std.detach().cpu().numpy()
        )
    return (
        np.concatenate(predictions, axis=0).astype(np.float32),
        diagnostics,
        float(inference_seconds),
    )


def run_star_checkpoint_evaluation(
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
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    expected = {
        "kind": CHECKPOINT_KIND,
        "model_name": "star_dcrnn",
        "horizon": 24 if task == "24h" else 168,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Checkpoint mismatch for {key}: "
                f"{checkpoint.get(key)!r} != {value!r}."
            )
    variant = str(checkpoint["variant"])
    if variant not in FORMAL_VARIANTS:
        raise ValueError(f"Unknown formal variant {variant!r}.")

    config = checkpoint["resolved_config"]
    demand_scaler = ZScoreScaler.from_state_dict(checkpoint["demand_scaler"])
    weather_scaler = ZScoreScaler.from_state_dict(
        checkpoint["weather_scaler"]
    )
    test_data = prepare_dcrnn_test_data(
        project_root=project_root,
        config=config,
        demand_scaler=demand_scaler,
        weather_scaler=weather_scaler,
        data_dir=data_dir,
    )
    model = build_star_dcrnn_model(
        config,
        project_root=project_root,
        input_dim=len(checkpoint["input_feature_names"]),
        future_exog_dim=len(checkpoint["future_exog_feature_names"]),
        horizon=int(checkpoint["horizon"]),
        history_hours=int(config["task"]["history_hours"]),
        variant=variant,
        device=device,
    )
    graph_identity = _validate_checkpoint_graph_identity(
        checkpoint=checkpoint,
        model=model,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    prediction_scaled, diagnostics, inference_seconds = (
        _predict_with_diagnostics(
            model=model,
            x_past=test_data.test.x_past,
            future_exog=test_data.test.future_exog,
            device=device,
            batch_size=batch_size,
        )
    )
    prediction_raw = demand_scaler.inverse_transform(prediction_scaled)
    truth_raw = test_data.test.y_raw
    output_dir.mkdir(parents=True, exist_ok=True)
    dma_names = list("ABCDEFGHIJ")

    metric_paths: dict[str, str] = {}
    aggregate_paths: dict[str, str] = {}
    for protocol, indices in test_data.protocol_indices.items():
        metrics = evaluate_predictions(
            y_true=truth_raw[indices],
            y_pred=prediction_raw[indices],
            dma_names=dma_names,
        )
        metric_path = output_dir / f"metrics_{protocol}.csv"
        metrics.to_csv(metric_path, index=False, float_format="%.10f")
        metric_paths[protocol] = str(metric_path)
        aggregate = evaluate_aggregate_total_predictions(
            y_true=truth_raw[indices],
            y_pred=prediction_raw[indices],
        )
        aggregate_path = (
            output_dir / f"metrics_aggregate_total_{protocol}.csv"
        )
        aggregate.to_csv(
            aggregate_path,
            index=False,
            float_format="%.10f",
        )
        aggregate_paths[protocol] = str(aggregate_path)

    common_indices = test_data.protocol_indices["common_46"]
    if len(common_indices) != 46:
        raise ValueError("Paper comparison requires exactly common_46.")
    reference = load_paper_reference(paper_reference_path)
    comparison = compare_metrics_to_paper(
        task=task,
        dcrnn_metrics=pd.read_csv(output_dir / "metrics_common_46.csv"),
        reference=reference,
    )
    if "model" in comparison.columns:
        comparison.loc[
            comparison["model"].astype(str).str.lower() == "dcrnn",
            "model",
        ] = f"STaR-DCRNN ({variant})"
    comparison_path = output_dir / "paper_comparison.csv"
    comparison.to_csv(comparison_path, index=False, float_format="%.10f")
    report_path = output_dir / "paper_comparison_report.md"
    write_paper_comparison_report(
        comparison=comparison,
        task=task,
        reference=reference,
        output_path=report_path,
    )

    np.savez_compressed(
        output_dir / "predictions.npz",
        prediction=prediction_raw,
        target=truth_raw,
        operational_indices=test_data.protocol_indices["operational"],
        strict_within_test_indices=test_data.protocol_indices[
            "strict_within_test"
        ],
        common_46_indices=test_data.protocol_indices["common_46"],
    )
    diagnostics_path: Path | None = None
    if diagnostics:
        diagnostics_path = output_dir / "mechanism_diagnostics.npz"
        np.savez_compressed(diagnostics_path, **diagnostics)

    summary = {
        "status": "completed",
        "model": "star_dcrnn",
        "variant": variant,
        "task": task,
        "horizon": int(checkpoint["horizon"]),
        "seed": int(checkpoint["seed"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "graph_identity": graph_identity,
        "device": str(device),
        "teacher_forcing_ratio": 0.0,
        "parameters_total": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "inference_seconds": inference_seconds,
        "inference_milliseconds_per_sample": float(
            1000.0 * inference_seconds / len(test_data.test.x_past)
        ),
        "paper_comparison_protocol": "common_46",
        "paper_reference": str(paper_reference_path.resolve()),
        "paper_reference_sha256": _sha256(paper_reference_path.resolve()),
        "paper_metric_conventions": dict(
            reference["paper"]["metric_conventions"]
        ),
        "protocol_counts": {
            name: int(len(indices))
            for name, indices in test_data.protocol_indices.items()
        },
        "metrics": metric_paths,
        "aggregate_total_metrics": aggregate_paths,
        "paper_comparison_csv": str(comparison_path),
        "paper_comparison_report": str(report_path),
        "mechanism_diagnostics": (
            None if diagnostics_path is None else str(diagnostics_path)
        ),
        "predictions": str(output_dir / "predictions.npz"),
        "test_targets_used_for_training_or_selection": False,
    }
    (output_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
