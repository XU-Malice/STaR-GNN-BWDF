#!/usr/bin/env python
"""Validate preprocessing artifacts and leakage guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dma_wdf.data.outlier_detection import fit_iqr_thresholds
from dma_wdf.utils.config import parse_timestamp, read_yaml


def _record(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    observed: object,
    expected: object,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def validate(root: Path, data_dir: Path) -> dict[str, object]:
    split_cfg = read_yaml(root / "configs" / "data" / "paper_split.yaml")
    checks: list[dict[str, object]] = []

    required = [
        "demand_hourly.parquet",
        "demand_interpolated_before_outliers.parquet",
        "demand_outlier_mask.parquet",
        "weather_hourly.parquet",
        "temporal_hourly.parquet",
        "demand_iqr_thresholds.csv",
        "demand_outlier_profile.csv",
        "interpolation_split_profile.csv",
        "quality_checks.csv",
        "sample_index_single_step_24h.csv",
        "sample_index_multi_step_168h.csv",
    ]
    for name in required:
        _record(
            checks,
            f"artifact:{name}",
            (data_dir / name).is_file(),
            "exists" if (data_dir / name).is_file() else "missing",
            "exists",
        )

    if not all(bool(row["passed"]) for row in checks):
        return {"all_passed": False, "checks": checks}

    demand = pd.read_parquet(data_dir / "demand_hourly.parquet")
    demand_before = pd.read_parquet(
        data_dir / "demand_interpolated_before_outliers.parquet"
    )
    outlier_mask = pd.read_parquet(data_dir / "demand_outlier_mask.parquet")
    weather = pd.read_parquet(data_dir / "weather_hourly.parquet")
    temporal = pd.read_parquet(data_dir / "temporal_hourly.parquet")
    thresholds = pd.read_csv(data_dir / "demand_iqr_thresholds.csv").set_index(
        "dma_column"
    )
    outlier_profile = pd.read_csv(data_dir / "demand_outlier_profile.csv")
    interpolation_profile = pd.read_csv(
        data_dir / "interpolation_split_profile.csv"
    )

    _record(checks, "paper_rows", len(demand) == 19056, len(demand), 19056)
    _record(checks, "dma_count", demand.shape[1] == 10, demand.shape[1], 10)
    for name, frame in [
        ("demand", demand),
        ("weather", weather),
        ("temporal", temporal),
    ]:
        missing = int(frame.isna().sum().sum())
        _record(checks, f"{name}_missing", missing == 0, missing, 0)

    sources = sorted(thresholds["threshold_source"].astype(str).unique().tolist())
    _record(
        checks,
        "threshold_source",
        sources == ["train"],
        sources,
        ["train"],
    )
    fit_rows = sorted(thresholds["fit_rows"].astype(int).unique().tolist())
    expected_train_rows = int(split_cfg["split"]["expected_train_rows"])
    _record(
        checks,
        "threshold_fit_rows",
        fit_rows == [expected_train_rows],
        fit_rows,
        [expected_train_rows],
    )

    tz = demand_before.index.tz
    train_start = parse_timestamp(split_cfg["split"]["train_start"], tz)
    train_end = parse_timestamp(split_cfg["split"]["train_end_inclusive"], tz)
    train_before = demand_before.loc[
        (demand_before.index >= train_start)
        & (demand_before.index <= train_end)
    ]
    recomputed = fit_iqr_thresholds(
        train_before,
        multiplier=float(thresholds["multiplier"].iloc[0]),
        threshold_source="train",
    )
    numeric_fields = ["q1", "q3", "iqr", "lower", "upper"]
    max_difference = float(
        np.max(
            np.abs(
                thresholds.loc[recomputed.index, numeric_fields].to_numpy(float)
                - recomputed[numeric_fields].to_numpy(float)
            )
        )
    )
    _record(
        checks,
        "thresholds_recomputed_from_train",
        max_difference <= 1.0e-10,
        max_difference,
        "<=1e-10",
    )

    remaining_after = int(interpolation_profile["missing_after"].sum())
    _record(
        checks,
        "split_interpolation_missing_after",
        remaining_after == 0,
        remaining_after,
        0,
    )
    mask_total = int(outlier_mask.to_numpy(dtype=bool).sum())
    profile_total = int(outlier_profile["outliers"].sum())
    _record(
        checks,
        "outlier_mask_matches_profile",
        mask_total == profile_total,
        mask_total,
        profile_total,
    )

    sample_24 = pd.read_csv(data_dir / "sample_index_single_step_24h.csv")
    sample_168 = pd.read_csv(data_dir / "sample_index_multi_step_168h.csv")
    train_24 = int((sample_24["split"] == "train").sum())
    train_168 = int((sample_168["split"] == "train").sum())
    _record(checks, "train_samples_24h", train_24 == 686, train_24, 686)
    _record(checks, "train_samples_168h", train_168 == 680, train_168, 680)

    pipeline_quality = pd.read_csv(data_dir / "quality_checks.csv")
    pipeline_passed = bool(pipeline_quality["passed"].astype(bool).all())
    _record(
        checks,
        "pipeline_quality_checks",
        pipeline_passed,
        pipeline_passed,
        True,
    )

    return {
        "all_passed": all(bool(row["passed"]) for row in checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/data_build"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir = (
        args.data_dir
        if args.data_dir.is_absolute()
        else (root / args.data_dir)
    )
    report = validate(root, data_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if not bool(report["all_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
