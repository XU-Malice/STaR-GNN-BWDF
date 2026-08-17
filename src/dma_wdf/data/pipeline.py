#!/usr/bin/env python
"""Full data build pipeline for the MSCMNet paper protocol.

Orchestrates every preprocessing step from raw ``wf4bwdf`` data to
model-ready hourly feature tables.

Usage::

    python -m dma_wdf.data.pipeline \\
        --config configs/paper_split.yaml \\
        --output-dir data/processed/data_build/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dma_wdf.data.loader import load_raw_dataset, select_period
from dma_wdf.quality.report import (
    feature_summary,
    missing_rows,
    profile_table,
    quality_check_row,
    write_markdown_report,
    write_quality_checks_json,
)
from dma_wdf.utils.config import parse_timestamp, read_yaml
from dma_wdf.data.interpolation import interpolate_by_splits
from dma_wdf.data.metrics import compute_metrics
from dma_wdf.data.outlier_detection import (
    apply_iqr_thresholds,
    fit_iqr_thresholds,
)
from dma_wdf.data.sliding_window import build_sample_index, daily_starts
from dma_wdf.data.temporal_features import build_temporal_features
from dma_wdf.data.weather_features import rename_weather


def run_data_build(
    *,
    root: Path,
    config_path: Path,
    preprocessing_config_path: Path,
    output_dir: Path,
    wf4bwdf_repo: Path | None = None,
) -> dict[str, Any]:
    """Execute the full MSCMNet paper-protocol data build pipeline.

    Steps
    -----
    1. Load raw data via ``wf4bwdf.load_complete_dataset()``.
    2. Select the paper period (2021-01-01 to 2023-03-05).
    3. Interpolate demand and weather (time-based, both directions).
    4. Rename weather columns to paper names.
    5. Build temporal features from calendar.
    6. Save processed parquet files and profiling artifacts.
    7. Build sample indices for each task.
    8. Run metric sanity checks (perfect prediction, train-mean).
    9. Run quality checks and write reports.

    Args:
        root: Project root directory.
        config_path: Path to the ``paper_split.yaml`` configuration.
        output_dir: Directory for all output artifacts.
        wf4bwdf_repo: Optional path to a local wf4bwdf clone.  If
            provided, its ``src`` directory is added to ``sys.path``.

    Returns:
        Status dictionary with ``output_dir``, ``all_passed``, and
        summary statistics.
    """
    # --- 0. Setup ---
    cfg = read_yaml(config_path)
    preprocessing_cfg = read_yaml(preprocessing_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if wf4bwdf_repo:
        wf4bwdf_src = (wf4bwdf_repo / "src").resolve()
        sys.path.insert(0, str(wf4bwdf_src))

    # --- 1. Load raw data ---
    dataset = load_raw_dataset()
    inflows = dataset["dma_inflows"].copy()
    weather = dataset["weather"].copy()
    calendar = dataset["calendar"].copy()
    properties = dataset["dma_properties"].copy()
    tz = inflows.index.tz

    # --- 2. Select paper period ---
    paper_start = parse_timestamp(cfg["paper_period"]["start"], tz)
    paper_end = parse_timestamp(cfg["paper_period"]["end"], tz)
    train_start = parse_timestamp(cfg["split"]["train_start"], tz)
    train_end = parse_timestamp(cfg["split"]["train_end_inclusive"], tz)
    test_start = parse_timestamp(cfg["split"]["test_start"], tz)
    test_end = parse_timestamp(cfg["split"]["test_end_inclusive"], tz)

    inflows_period_raw = select_period(inflows, paper_start, paper_end)
    weather_period_raw = select_period(weather, paper_start, paper_end)
    calendar_period = select_period(calendar, paper_start, paper_end)

    # Profile raw tables.
    raw_profile = [
        profile_table("dma-properties", properties),
        profile_table("dma-inflows_raw_full", inflows),
        profile_table("weather_raw_full", weather),
        profile_table("calendar_raw_full", calendar),
    ]

    # --- 3. Leakage-safe interpolation and demand outlier handling ---
    # Train and test are processed independently, so a test value cannot
    # fill a missing value at the end of the training partition.
    split_ranges = {
        "train": (train_start, train_end),
        "test": (test_start, test_end),
    }
    interpolation_cfg = preprocessing_cfg["interpolation"]
    inflows_interpolated, demand_interpolation_profile = interpolate_by_splits(
        inflows_period_raw,
        split_ranges,
        method=str(interpolation_cfg["demand"]),
        limit_direction=str(interpolation_cfg["limit_direction"]),
        require_full_coverage=True,
        require_no_nan=bool(interpolation_cfg.get("require_no_nan", True)),
    )
    weather_period, weather_interpolation_profile = interpolate_by_splits(
        weather_period_raw,
        split_ranges,
        method=str(interpolation_cfg["weather"]),
        limit_direction=str(interpolation_cfg["limit_direction"]),
        require_full_coverage=True,
        require_no_nan=bool(interpolation_cfg.get("require_no_nan", True)),
    )

    # Fit IQR fences on training rows only, freeze them, and apply the same
    # fences to both train and test. Outlier interpolation also stays inside
    # each temporal partition.
    outlier_cfg = preprocessing_cfg["outliers"]
    threshold_source = str(outlier_cfg["threshold_source"])
    if threshold_source != "train":
        raise ValueError(
            "Leakage-safe pipeline requires outliers.threshold_source='train'."
        )
    train_demand_for_fit = inflows_interpolated.loc[
        (inflows_interpolated.index >= train_start)
        & (inflows_interpolated.index <= train_end)
    ]
    iqr_thresholds = fit_iqr_thresholds(
        train_demand_for_fit,
        multiplier=float(outlier_cfg["iqr_multiplier"]),
        threshold_source=threshold_source,
    )
    inflows_period, demand_outlier_profile, demand_outlier_mask = (
        apply_iqr_thresholds(
            inflows_interpolated,
            iqr_thresholds,
            method=str(outlier_cfg["method"]),
            limit_direction=str(outlier_cfg["limit_direction"]),
            split_ranges=split_ranges,
            require_no_nan=bool(outlier_cfg.get("require_no_nan", True)),
        )
    )

    # --- 4. Rename weather ---
    weather_clean = rename_weather(weather_period, cfg["features"]["weather"])
    weather_columns = list(weather_clean.columns)

    # --- 5. Build temporal features ---
    temporal = build_temporal_features(calendar_period)
    temporal_columns = [c for c in temporal.columns if c != "time_idx"]

    # --- 6. Combine and profile ---
    combined_features = pd.concat(
        [
            inflows_period.add_prefix("demand__"),
            weather_clean.add_prefix("weather__"),
            temporal.add_prefix("temporal__"),
        ],
        axis=1,
    )

    train_mask = (inflows_period.index >= train_start) & (inflows_period.index <= train_end)
    test_mask = (inflows_period.index >= test_start) & (inflows_period.index <= test_end)
    train_rows = int(train_mask.sum())
    test_rows = int(test_mask.sum())

    data_profile = raw_profile + [
        profile_table("dma-inflows_paper_period", inflows_period),
        profile_table("weather_paper_period", weather_clean),
        profile_table("calendar_paper_period", calendar_period),
        profile_table("temporal_features", temporal),
        profile_table("combined_hourly_features", combined_features),
    ]
    pd.DataFrame(data_profile).to_csv(output_dir / "data_profile.csv", index=False)

    # Missing value report.
    missing = []
    missing.extend(missing_rows("dma-inflows", inflows_period_raw, inflows_period))
    missing.extend(missing_rows("weather", weather_period_raw, weather_period))
    pd.DataFrame(missing).to_csv(output_dir / "missing_profile.csv", index=False)

    # Feature summaries.
    fsummary = []
    fsummary.extend(feature_summary("demand", inflows_period))
    fsummary.extend(feature_summary("weather", weather_clean))
    fsummary.extend(feature_summary("temporal", temporal))
    pd.DataFrame(fsummary).to_csv(output_dir / "feature_summary.csv", index=False)

    # --- 7. Save parquet artifacts ---
    inflows_period.to_parquet(output_dir / "demand_hourly.parquet")
    inflows_interpolated.to_parquet(
        output_dir / "demand_interpolated_before_outliers.parquet"
    )
    demand_outlier_mask.to_parquet(output_dir / "demand_outlier_mask.parquet")
    weather_clean.to_parquet(output_dir / "weather_hourly.parquet")
    temporal.to_parquet(output_dir / "temporal_hourly.parquet")
    combined_features.to_parquet(output_dir / "combined_hourly_features.parquet")
    properties.to_csv(output_dir / "dma_properties.csv")
    iqr_thresholds.reset_index().to_csv(
        output_dir / "demand_iqr_thresholds.csv",
        index=False,
    )
    demand_outlier_profile.to_csv(
        output_dir / "demand_outlier_profile.csv",
        index=False,
    )
    interpolation_profile = pd.concat(
        [
            demand_interpolation_profile.assign(table="demand"),
            weather_interpolation_profile.assign(table="weather"),
        ],
        ignore_index=True,
    )
    interpolation_profile.to_csv(
        output_dir / "interpolation_split_profile.csv",
        index=False,
    )

    # --- 8. Build sample indices ---
    sample_shape_summary: list[dict[str, Any]] = []
    for task_name, task_cfg in cfg["tasks"].items():
        sample_index = build_sample_index(
            paper_start=paper_start,
            paper_end=paper_end,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            horizon_hours=int(task_cfg["horizon_hours"]),
            stride_hours=int(task_cfg["stride_hours"]),
            max_history_weeks=int(task_cfg["max_history_weeks_for_indexing"]),
            tz=tz,
        )
        sample_index.to_csv(output_dir / f"sample_index_{task_name}.csv", index=False)
        sample_shape_summary.append(
            {
                "task": task_name,
                "horizon_hours": int(task_cfg["horizon_hours"]),
                "all_samples": int(len(sample_index)),
                "train_samples": int((sample_index["split"] == "train").sum()),
                "test_samples": int((sample_index["split"] == "test").sum()),
                "boundary_or_unused_samples": int(
                    (sample_index["split"] == "boundary_or_unused").sum()
                ),
                "per_sample_target_shape": (
                    f"{int(task_cfg['horizon_hours'])}x{len(inflows_period.columns)}"
                ),
                "max_history_shape": (
                    f"{int(task_cfg['max_history_weeks_for_indexing']) * 7 * 24}"
                    f"x{len(inflows_period.columns)}"
                ),
            }
        )
    pd.DataFrame(sample_shape_summary).to_csv(
        output_dir / "sample_shape_summary.csv", index=False
    )

    # Split summary.
    split_summary = {
        "paper_period_start": str(inflows_period.index.min()),
        "paper_period_end": str(inflows_period.index.max()),
        "paper_period_rows": int(len(inflows_period)),
        "train_start": str(inflows_period.loc[train_mask].index.min()) if train_rows else "",
        "train_end": str(inflows_period.loc[train_mask].index.max()) if train_rows else "",
        "train_rows": train_rows,
        "test_start": str(inflows_period.loc[test_mask].index.min()) if test_rows else "",
        "test_end": str(inflows_period.loc[test_mask].index.max()) if test_rows else "",
        "test_rows": test_rows,
        "train_fraction": float(
            train_rows / (train_rows + test_rows)
        ) if (train_rows + test_rows) else float("nan"),
        "test_fraction": float(
            test_rows / (train_rows + test_rows)
        ) if (train_rows + test_rows) else float("nan"),
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )

    # --- 9. Metric sanity checks ---
    test_values = inflows_period.loc[test_mask].to_numpy(dtype=float)
    perfect_metrics = compute_metrics(test_values, test_values)
    train_mean = inflows_period.loc[train_mask].mean(axis=0).to_numpy(dtype=float)
    mean_prediction = np.tile(train_mean.reshape(1, -1), (test_values.shape[0], 1))
    mean_metrics = compute_metrics(test_values, mean_prediction)
    metric_rows = []
    for name, mets in [
        ("perfect_prediction", perfect_metrics),
        ("train_mean_prediction", mean_metrics),
    ]:
        for metric, value in mets.items():
            metric_rows.append({"case": name, "metric": metric, "value": value})
    pd.DataFrame(metric_rows).to_csv(output_dir / "metric_sanity_check.csv", index=False)

    # --- 10. Quality checks ---
    quality_checks = [
        quality_check_row(
            "dma_count",
            len(inflows_period.columns),
            int(cfg["features"]["demand"]["expected_dma_count"]),
        ),
        quality_check_row(
            "paper_period_rows",
            len(inflows_period),
            int(cfg["paper_period"]["expected_rows"]),
        ),
        quality_check_row(
            "train_rows", train_rows, int(cfg["split"]["expected_train_rows"])
        ),
        quality_check_row(
            "test_rows", test_rows, int(cfg["split"]["expected_test_rows"])
        ),
        quality_check_row(
            "weather_feature_count",
            len(weather_clean.columns),
            len(cfg["features"]["weather"]),
        ),
        quality_check_row(
            "temporal_feature_count",
            len(temporal_columns),
            len(cfg["features"]["temporal"]),
        ),
        quality_check_row(
            "missing_after_interpolation",
            int(
                inflows_period.isna().sum().sum()
                + weather_clean.isna().sum().sum()
            ),
            0,
        ),
        quality_check_row(
            "iqr_threshold_source",
            sorted(iqr_thresholds["threshold_source"].unique().tolist()),
            ["train"],
        ),
        quality_check_row(
            "iqr_threshold_fit_rows",
            sorted(iqr_thresholds["fit_rows"].unique().tolist()),
            [train_rows],
        ),
        quality_check_row(
            "outlier_processed_missing",
            int(inflows_period.isna().sum().sum()),
            0,
        ),
        quality_check_row(
            "metric_perfect_prediction",
            json.dumps(perfect_metrics),
            "MAE=0, RMSE=0, NSE=1, MAPE=0",
            passed=(
                perfect_metrics.get("MAE", 999) == 0
                and perfect_metrics.get("RMSE", 999) == 0
                and perfect_metrics.get("NSE", -999) == 1
            ),
        ),
    ]
    pd.DataFrame(quality_checks).to_csv(output_dir / "quality_checks.csv", index=False)
    all_passed = all(bool(row["passed"]) for row in quality_checks)
    write_quality_checks_json(output_dir / "quality_checks.json", quality_checks, all_passed)

    # --- 11. Write report ---
    sample_lines = []
    for row in sample_shape_summary:
        sample_lines.append(
            f"| {row['task']} | {row['horizon_hours']} | "
            f"{row['all_samples']} | {row['train_samples']} | "
            f"{row['test_samples']} | {row['per_sample_target_shape']} | "
            f"{row['max_history_shape']} |"
        )
    qc_lines = []
    for row in quality_checks:
        qc_lines.append(
            f"| {row['check']} | {row['passed']} | "
            f"{row.get('observed', '')} | {row.get('expected', '')} |"
        )

    sections = {
        "Split": (
            "| split | rows | start | end |\n"
            "|---|---:|---|---|\n"
            f"| paper_period | {split_summary['paper_period_rows']} | "
            f"{split_summary['paper_period_start']} | "
            f"{split_summary['paper_period_end']} |\n"
            f"| train | {split_summary['train_rows']} | "
            f"{split_summary['train_start']} | "
            f"{split_summary['train_end']} |\n"
            f"| test | {split_summary['test_rows']} | "
            f"{split_summary['test_start']} | "
            f"{split_summary['test_end']} |"
        ),
        "Sample Shapes": (
            "| task | horizon_hours | all | train | test | "
            "target_shape | max_history_shape |\n"
            "|---|---:|---:|---:|---:|---|---|\n"
            + "\n".join(sample_lines)
        ),
        "Metric Sanity": (
            "MAPE is reported as a fraction, matching the supplementary S1 tables.\n"
        ),
        "Quality Checks": (
            "| check | passed | observed | expected |\n"
            "|---|---|---:|---:|\n"
            + "\n".join(qc_lines)
        ),
    }
    write_markdown_report(
        output_dir / "data_build_report.md",
        "MSCMNet Data Build Report",
        sections,
    )

    status = {
        "output_dir": str(output_dir),
        "all_passed": all_passed,
        "split_summary": split_summary,
        "sample_shape_summary": sample_shape_summary,
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, indent=2, default=str), encoding="utf-8"
    )

    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the MSCMNet paper-protocol dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent,
        help="Project root directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to paper_split.yaml.",
    )
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=None,
        help="Path to preprocessing.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for processed data.",
    )
    parser.add_argument(
        "--wf4bwdf-repo",
        type=Path,
        default=None,
        help="Optional path to a local wf4bwdf clone.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = args.config or root / "configs" / "data" / "paper_split.yaml"
    preprocessing_config_path = (
        args.preprocessing_config
        or root / "configs" / "data" / "preprocessing.yaml"
    )
    cfg = read_yaml(config_path)
    output_dir = args.output_dir or root / cfg["outputs"]["data_build_dir"]

    result = run_data_build(
        root=root,
        config_path=config_path,
        preprocessing_config_path=preprocessing_config_path,
        output_dir=output_dir,
        wf4bwdf_repo=args.wf4bwdf_repo,
    )

    print(json.dumps({"output_dir": str(output_dir), "all_passed": result["all_passed"]}, indent=2))


if __name__ == "__main__":
    main()