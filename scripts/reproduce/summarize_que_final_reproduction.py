#!/usr/bin/env python
"""Summarize the final Que et al. correction/checkpoint diagnostic matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


METRICS = ("MAE", "MAPE", "RMSE", "NSE")


def compute_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    denominator = np.maximum(np.abs(truth.astype(np.float64)), 1.0e-6)
    residual = float(np.square(error).sum())
    centered = truth.astype(np.float64) - float(truth.mean())
    total = float(np.square(centered).sum())
    return {
        "MAE": float(np.abs(error).mean()),
        "MAPE": float((np.abs(error) / denominator).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "NSE": float("nan") if total <= 1.0e-12 else float(1.0 - residual / total),
    }


def total_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    values = compute_metrics(truth.sum(axis=2), prediction.sum(axis=2))
    values["MAE"] = float(
        sum(
            compute_metrics(truth[:, :, index], prediction[:, :, index])["MAE"]
            for index in range(truth.shape[2])
        )
    )
    return values


def fit_calibration(
    truth: np.ndarray, prediction: np.ndarray, method: str
) -> tuple[np.ndarray, np.ndarray]:
    truth_flat = truth.reshape(-1, truth.shape[-1]).astype(np.float64)
    pred_flat = prediction.reshape(-1, prediction.shape[-1]).astype(np.float64)
    if method == "intercept":
        slope = np.ones(truth.shape[-1], dtype=np.float64)
        intercept = (truth_flat - pred_flat).mean(axis=0)
        return slope, intercept
    if method != "affine":
        raise ValueError(method)
    pred_centered = pred_flat - pred_flat.mean(axis=0)
    denominator = np.square(pred_centered).sum(axis=0)
    numerator = (pred_centered * (truth_flat - truth_flat.mean(axis=0))).sum(axis=0)
    slope = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 1.0e-12,
    )
    # Constrain an unstable in-sample fit while preserving transparent provenance.
    slope = np.clip(slope, 0.25, 4.0)
    intercept = truth_flat.mean(axis=0) - slope * pred_flat.mean(axis=0)
    return slope, intercept


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paper-metrics", type=Path, required=True)
    parser.add_argument("--paper-totals", type=Path, required=True)
    args = parser.parse_args()

    root = args.result_root.resolve()
    manifest = pd.read_csv(args.manifest, sep="\t", dtype={"seed": int})
    paper = yaml.safe_load(args.paper_metrics.read_text(encoding="utf-8"))["tasks"]
    paper_totals = yaml.safe_load(args.paper_totals.read_text(encoding="utf-8"))["tasks"]
    final_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    share_rows: list[dict[str, object]] = []

    for item in manifest.to_dict("records"):
        case = str(item["case"])
        model = str(item["model"])
        seed = int(item["seed"])
        run = root / case / model / f"seed_{seed}"
        required = (
            run / "predictions_common46.npz",
            run / "stage_predictions_common46.npz",
            run / "train_fit_stages_24h.npz",
            run / "metrics.csv",
            run / "status.json",
            run / "resolved_config.yaml",
        )
        if not all(path.is_file() for path in required):
            continue
        predictions = np.load(required[0])
        stages = np.load(required[1])
        train = np.load(required[2])
        metrics = pd.read_csv(required[3])
        config = yaml.safe_load(required[5].read_text(encoding="utf-8"))
        display = str(config["model"]["display_name"])

        for horizon in ("24h", "168h"):
            truth = predictions[f"y_true_{horizon}"]
            final = predictions[f"y_pred_{horizon}"]
            values = total_metrics(truth, final)
            reference = paper_totals[horizon].get(display)
            row: dict[str, object] = {**item, "task": horizon}
            row.update(values)
            if reference is not None:
                relative = [abs(values[key] - float(reference[key])) / abs(float(reference[key])) for key in METRICS]
                row["mean_paper_relative_error"] = float(np.mean(relative))
                row["max_paper_relative_error"] = float(np.max(relative))
            else:
                row["mean_paper_relative_error"] = np.nan
                row["max_paper_relative_error"] = np.nan

            paper_series = paper[horizon].get(display)
            if paper_series is not None and all(letter in paper_series for letter in "ABCDEFGHIJ"):
                for metric in METRICS:
                    observed = np.asarray([
                        metrics.loc[
                            (metrics["task"] == horizon)
                            & (metrics["series"] == letter)
                            & (metrics["metric"] == metric),
                            "value",
                        ].item()
                        for letter in "ABCDEFGHIJ"
                    ])
                    expected = np.asarray([paper_series[letter][metric] for letter in "ABCDEFGHIJ"])
                    row[f"dma_{metric}_mare"] = float(np.mean(np.abs(observed - expected) / np.maximum(np.abs(expected), 1.0e-12)))
                    row[f"dma_{metric}_correlation"] = correlation(observed, expected)
            final_rows.append(row)

            for key in ("msnet_prediction", "fc1_prediction", "prediction"):
                array_key = f"{key}_{horizon}"
                if array_key not in stages:
                    continue
                stage_values = total_metrics(truth, stages[array_key])
                stage_rows.append({**item, "task": horizon, "stage": key, **stage_values})

            train_truth = train["y_true_24h"]
            train_prediction = train["prediction_24h"]
            for method in ("intercept", "affine"):
                slope, intercept = fit_calibration(train_truth, train_prediction, method)
                calibrated = final * slope.reshape(1, 1, -1) + intercept.reshape(1, 1, -1)
                calibration_rows.append(
                    {
                        **item,
                        "task": horizon,
                        "calibration": method,
                        "fit_source": "training_fit_24h_only",
                        "slope_min": float(slope.min()),
                        "slope_max": float(slope.max()),
                        "intercept_sum": float(intercept.sum()),
                        **total_metrics(truth, calibrated),
                    }
                )

            share_key = f"predicted_daily_share_{horizon}"
            if share_key in stages:
                predicted_share = stages[share_key]
                if horizon == "24h":
                    actual_daily = truth.sum(axis=1, keepdims=True)
                else:
                    actual_daily = truth.reshape(truth.shape[0], 7, 24, truth.shape[2]).sum(axis=2)
                actual_share = actual_daily / np.maximum(actual_daily.sum(axis=-1, keepdims=True), 1.0e-6)
                if predicted_share.ndim == 2:
                    predicted_share = predicted_share[:, None, :]
                share_rows.append(
                    {
                        **item,
                        "task": horizon,
                        "share_mae": float(np.abs(predicted_share - actual_share).mean()),
                        "share_correlation": correlation(predicted_share.ravel(), actual_share.ravel()),
                        "share_sum_max_error": float(np.abs(predicted_share.sum(axis=-1) - 1.0).max()),
                    }
                )

    outputs = {
        "final_metric_summary.tsv": final_rows,
        "stage_metric_summary.tsv": stage_rows,
        "train_only_calibration_summary.tsv": calibration_rows,
        "share_forecast_summary.tsv": share_rows,
    }
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(root / name, sep="\t", index=False)

    if final_rows:
        final = pd.DataFrame(final_rows)
        seed_rows = final[final["phase"] == "seed_robustness"]
        if not seed_rows.empty:
            numeric = [*METRICS, "mean_paper_relative_error", "max_paper_relative_error"]
            grouped = seed_rows.groupby(["candidate", "model", "task"])[numeric].agg(["mean", "std", "min", "max"])
            grouped.columns = ["_".join(column) for column in grouped.columns]
            grouped.reset_index().to_csv(root / "seed_robustness_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
