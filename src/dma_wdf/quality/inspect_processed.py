#!/usr/bin/env python
"""Inspect processed (post-pipeline) data artifacts.

This is a **standalone** script — run it after the preprocessing
pipeline to verify that all output files exist, have the correct
shapes, and contain no NaN values.

Usage::

    python -m dma_wdf.quality.inspect_processed --data-dir data/processed/data_build/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Expected values from the paper protocol.
EXPECTED = {
    "paper_period_rows": 19056,
    "train_rows": 17136,
    "test_rows": 1920,
    "dma_count": 10,
    "weather_count": 4,
    "temporal_count": 5,
    # Sample index counts (ALL samples, before within-test history filtering).
    "train_samples_24h": 686,
    "test_samples_24h": 80,
    "train_samples_168h": 680,
    "test_samples_168h": 74,
    # Eval-index sequences (with 4-week within-test history): 46.
    "test_eval_sequences": 46,
}


def check_file_exists(path: Path, label: str) -> dict[str, Any]:
    """Check that a file exists and is non-empty."""
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    return {
        "check": f"{label}_exists",
        "passed": exists and size > 0,
        "observed": f"{path} ({size} bytes)" if exists else "missing",
        "expected": "non-empty file",
    }


def check_parquet(path: Path, label: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Read a parquet file and run basic checks."""
    checks: list[dict[str, Any]] = []
    if not path.exists():
        checks.append(
            {
                "check": f"{label}_exists",
                "passed": False,
                "observed": "missing",
                "expected": str(path),
            }
        )
        return pd.DataFrame(), checks

    df = pd.read_parquet(path)
    checks.append(
        {
            "check": f"{label}_exists",
            "passed": True,
            "observed": f"{len(df)} rows × {len(df.columns)} cols",
            "expected": "non-empty parquet",
        }
    )
    nan_count = int(df.isna().sum().sum())
    checks.append(
        {
            "check": f"{label}_no_nan",
            "passed": nan_count == 0,
            "observed": nan_count,
            "expected": 0,
        }
    )
    return df, checks


def inspect_processed(
    data_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Read all parquet/CSV artifacts from a data_build output and profile them.

    Args:
        data_dir: Path to the data build output directory.
        output_dir: If provided, write reports here.

    Returns:
        Status dictionary with all check results.
    """
    all_checks: list[dict[str, Any]] = []

    # 1. Check expected files exist.
    expected_files = [
        "demand_hourly.parquet",
        "weather_hourly.parquet",
        "temporal_hourly.parquet",
        "combined_hourly_features.parquet",
        "dma_properties.csv",
    ]
    for fname in expected_files:
        all_checks.append(check_file_exists(data_dir / fname, fname.replace(".", "_")))

    # 2. Read and validate demand.
    demand, d_checks = check_parquet(data_dir / "demand_hourly.parquet", "demand")
    all_checks.extend(d_checks)
    if not demand.empty:
        all_checks.append(
            {
                "check": "demand_dma_count",
                "passed": len(demand.columns) == EXPECTED["dma_count"],
                "observed": len(demand.columns),
                "expected": EXPECTED["dma_count"],
            }
        )
        all_checks.append(
            {
                "check": "demand_paper_period_rows",
                "passed": len(demand) == EXPECTED["paper_period_rows"],
                "observed": len(demand),
                "expected": EXPECTED["paper_period_rows"],
            }
        )

    # 3. Read and validate weather.
    weather, w_checks = check_parquet(data_dir / "weather_hourly.parquet", "weather")
    all_checks.extend(w_checks)
    if not weather.empty:
        all_checks.append(
            {
                "check": "weather_column_count",
                "passed": len(weather.columns) == EXPECTED["weather_count"],
                "observed": len(weather.columns),
                "expected": EXPECTED["weather_count"],
            }
        )

    # 4. Read and validate temporal.
    temporal, t_checks = check_parquet(data_dir / "temporal_hourly.parquet", "temporal")
    all_checks.extend(t_checks)

    # 5. Value range sanity.
    if not demand.empty:
        for col in demand.columns:
            col_min = float(demand[col].min())
            col_max = float(demand[col].max())
            all_checks.append(
                {
                    "check": f"demand_{col}_non_negative",
                    "passed": col_min >= 0,
                    "observed": f"min={col_min:.2f}, max={col_max:.2f}",
                    "expected": "min >= 0",
                }
            )

    # 6. Check sample index files if they exist.
    sample_expectations = {
        "sample_index_single_step_24h.csv": {
            "train": EXPECTED["train_samples_24h"],
            "test": EXPECTED["test_samples_24h"],
        },
        "sample_index_multi_step_168h.csv": {
            "train": EXPECTED["train_samples_168h"],
            "test": EXPECTED["test_samples_168h"],
        },
    }
    for sf, exp in sample_expectations.items():
        sf_path = data_dir / sf
        if sf_path.exists():
            sample_df = pd.read_csv(sf_path)
            train_count = int((sample_df["split"] == "train").sum())
            test_count = int((sample_df["split"] == "test").sum())
            all_checks.append(
                {
                    "check": f"{sf}_train",
                    "passed": train_count == exp["train"],
                    "observed": train_count,
                    "expected": exp["train"],
                }
            )
            all_checks.append(
                {
                    "check": f"{sf}_test",
                    "passed": test_count == exp["test"],
                    "observed": test_count,
                    "expected": exp["test"],
                }
            )
    # 7. Summary.
    all_passed = all(check["passed"] for check in all_checks)
    status = {
        "status": "completed",
        "data_dir": str(data_dir),
        "all_checks_passed": all_passed,
        "checks": all_checks,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_checks).to_csv(output_dir / "processed_quality_checks.csv", index=False)
        (output_dir / "status.json").write_text(
            json.dumps(status, indent=2, default=str), encoding="utf-8"
        )
        print(f"Reports written to {output_dir}")

    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect processed data artifacts."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the data_build output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for reports.",
    )
    args = parser.parse_args()

    result = inspect_processed(data_dir=args.data_dir, output_dir=args.output_dir)

    # Print summary.
    passed = sum(1 for c in result["checks"] if c["passed"])
    total = len(result["checks"])
    print(f"\nQuality checks: {passed}/{total} passed")
    for c in result["checks"]:
        if not c["passed"]:
            print(f"  FAIL: {c['check']} — observed={c['observed']}, expected={c['expected']}")

    if result["all_checks_passed"]:
        print("\n✓ All quality checks passed.")


if __name__ == "__main__":
    main()
