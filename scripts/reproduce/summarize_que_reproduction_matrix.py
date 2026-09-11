#!/usr/bin/env python
"""Validate and summarize a queued Que et al. reproduction matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


EXPECTED_METRIC_ROWS = 2 * 11 * 4
EXPECTED_PREDICTION_SHAPES = {
    "y_true_24h": (46, 24, 10),
    "y_pred_24h": (46, 24, 10),
    "y_true_168h": (46, 168, 10),
    "y_pred_168h": (46, 168, 10),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_name(result_root: Path, run_dir: Path) -> str:
    return run_dir.relative_to(result_root).as_posix()


def _validate_predictions(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return ["missing predictions_common46.npz"]
    with np.load(path, allow_pickle=False) as arrays:
        for key, expected in EXPECTED_PREDICTION_SHAPES.items():
            if key not in arrays:
                errors.append(f"missing prediction array {key}")
                continue
            actual = tuple(int(value) for value in arrays[key].shape)
            if actual != expected:
                errors.append(f"{key} shape {actual}, expected {expected}")
            if not np.isfinite(arrays[key]).all():
                errors.append(f"{key} contains non-finite values")
    return errors


def summarize(result_root: Path, summary_dir: Path) -> int:
    summary_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []

    status_paths = sorted(result_root.glob("**/seed_*/status.json"))
    for status_path in status_paths:
        run_dir = status_path.parent
        case = _case_name(result_root, run_dir)
        status = _read_json(status_path)
        config_path = run_dir / "resolved_config.yaml"
        config = (
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        training = config.get("training", {})
        cam = config.get("cam", {})
        normalization = training.get("normalization")
        channels = cam.get("channel_sizes")
        channel_label = "-".join(str(value) for value in channels or [])

        inventory_rows.append(
            {
                "case": case,
                "model": status.get("model"),
                "normalization": normalization,
                "cam_channels": channel_label,
                "status": status.get("status"),
                "formal_protocol": status.get("formal_protocol"),
                "elapsed_seconds": status.get("elapsed_seconds"),
                "git_commit": status.get("git_commit"),
                "device": status.get("device"),
                "prediction_24h_shape": json.dumps(
                    status.get("prediction_24h_shape")
                ),
                "prediction_168h_shape": json.dumps(
                    status.get("prediction_168h_shape")
                ),
            }
        )

        errors: list[str] = []
        if status.get("status") != "completed":
            errors.append("status is not completed")
        if status.get("formal_protocol") is not True:
            errors.append("formal_protocol is not true")
        if not config_path.is_file():
            errors.append("missing resolved_config.yaml")

        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.is_file():
            errors.append("missing metrics.csv")
        else:
            metrics = pd.read_csv(metrics_path)
            if len(metrics) != EXPECTED_METRIC_ROWS:
                errors.append(
                    f"metrics rows {len(metrics)}, expected {EXPECTED_METRIC_ROWS}"
                )
            if "value" not in metrics or not np.isfinite(metrics["value"]).all():
                errors.append("metric values are missing or non-finite")
            total = metrics.loc[metrics["series"] == "total"].copy()
            if len(total) != 8:
                errors.append(f"total metric rows {len(total)}, expected 8")
            for row in total.to_dict(orient="records"):
                paper_value = row.get("paper_value")
                value = row.get("value")
                absolute_gap = (
                    float(value) - float(paper_value)
                    if pd.notna(value) and pd.notna(paper_value)
                    else np.nan
                )
                relative_gap = (
                    100.0 * absolute_gap / abs(float(paper_value))
                    if pd.notna(paper_value) and float(paper_value) != 0.0
                    else np.nan
                )
                metric_rows.append(
                    {
                        "case": case,
                        "model": status.get("model"),
                        "normalization": normalization,
                        "cam_channels": channel_label,
                        "task": row.get("task"),
                        "metric": row.get("metric"),
                        "value": value,
                        "paper_value": paper_value,
                        "local_minus_paper": absolute_gap,
                        "relative_gap_pct": relative_gap,
                    }
                )

        loss_path = run_dir / "loss_curve.csv"
        if not loss_path.is_file():
            errors.append("missing loss_curve.csv")
        else:
            losses = pd.read_csv(loss_path)
            if losses.empty or "train_loss" not in losses:
                errors.append("loss_curve.csv is empty or malformed")
            else:
                group_column = "dma" if "dma" in losses else None
                groups = losses.groupby(group_column) if group_column else [("joint", losses)]
                for scope, group in groups:
                    best_index = group["train_loss"].idxmin()
                    loss_rows.append(
                        {
                            "case": case,
                            "model": status.get("model"),
                            "normalization": normalization,
                            "cam_channels": channel_label,
                            "scope": scope,
                            "epochs": int(len(group)),
                            "best_epoch": int(losses.loc[best_index, "epoch"]),
                            "minimum_train_loss": float(
                                losses.loc[best_index, "train_loss"]
                            ),
                            "final_train_loss": float(group.iloc[-1]["train_loss"]),
                        }
                    )

        errors.extend(_validate_predictions(run_dir / "predictions_common46.npz"))
        validations.append(
            {
                "case": case,
                "passed": not errors,
                "errors": errors,
            }
        )

    pd.DataFrame(inventory_rows).to_csv(
        summary_dir / "run_inventory.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(
        summary_dir / "total_metrics_vs_paper.csv", index=False
    )
    pd.DataFrame(loss_rows).to_csv(
        summary_dir / "loss_summary.csv", index=False
    )
    report = {
        "result_root": str(result_root.resolve()),
        "runs_found": len(status_paths),
        "runs_passed": sum(item["passed"] for item in validations),
        "runs_failed_validation": sum(not item["passed"] for item in validations),
        "all_discovered_runs_passed": bool(validations)
        and all(item["passed"] for item in validations),
        "runs": validations,
    }
    (summary_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["all_discovered_runs_passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--summary-dir", type=Path, default=None)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    summary_dir = (
        args.summary_dir.resolve()
        if args.summary_dir is not None
        else result_root / "_summary"
    )
    raise SystemExit(summarize(result_root, summary_dir))


if __name__ == "__main__":
    main()
