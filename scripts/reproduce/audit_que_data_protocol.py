#!/usr/bin/env python
"""CPU-only audit of unmodified Que-paper inputs, splits and evaluation truths.

Table 1/2 values are reference observations, never normalization/calibration
targets. Statistical disagreements are advisory; invalid input provenance,
missing/nonfinite observations and a different split fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dma_wdf.data.mscmnet_dataset import load_paper_data
from dma_wdf.data.reproduction_metrics import array_sha256
from dma_wdf.data.sliding_window import make_eval_index, target_24h, target_168h

SOURCE = {
    "doi": "10.1016/j.wroa.2024.100269",
    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11605409/",
    "tables": "Main article Table 1 (DMA means) and Table 2 (weather statistics/counts)",
    "reference_transcription": "Published values, unchanged; one number without a split applies to both splits.",
}
DMA_LETTERS = tuple("ABCDEFGHIJ")
DMA_COLUMNS = tuple(f"DMA {index}" for index in range(1, 11))
TEMPORAL_COLUMNS = ("hour", "time_zone_standard", "weekday", "holiday", "day_of_week")
EXPECTED_ROWS = {"train": 17136, "test": 1920}
EXPECTED_BOUNDS = {
    "train_start": "2021-01-01T00:00:00+01:00",
    "train_end": "2022-12-15T23:00:00+01:00",
    "test_start": "2022-12-16T00:00:00+01:00",
    "test_end": "2023-03-05T23:00:00+01:00",
}
DEMAND_MEANS = {
    "train": (8.25, 9.60, 4.35, 33.05, 78.13, 8.02, 24.85, 20.59, 20.36, 26.34),
    "test": (6.57, 9.56, 2.94, 32.18, 81.66, 10.46, 26.37, 23.92, 23.89, 24.15),
}
WEATHER_REFERENCE = {
    "rainfall_depth": {
        "unit": "mm", "train": (0.08, 0.76, 0.0, 45.70), "test": (0.09, 0.53, 0.0, 12.50),
    },
    "air_temperature": {
        "unit": "degC", "train": (16.10, 7.34, -2.10, 35.40), "test": (8.31, 2.72, -0.20, 14.60),
    },
    "air_humidity": {
        "unit": "%", "train": (64.22, 15.19, 17.0, 100.0), "test": (74.02, 16.63, 27.0, 99.0),
    },
    "windspeed": {
        "unit": "km/h", "train": (14.00, 12.29, 1.0, 77.0), "test": (16.42, 17.90, 1.0, 77.0),
    },
}


def validate_paper_frames(
    frames: dict[str, pd.DataFrame], bounds: dict[str, pd.Timestamp]
) -> dict[str, int]:
    """Validate the fixed split and aligned finite hourly observations."""
    for name, expected in EXPECTED_BOUNDS.items():
        if name not in bounds or pd.Timestamp(bounds[name]) != pd.Timestamp(expected):
            raise ValueError(f"Invalid paper split boundary {name}: expected {expected}.")
    reference_index = pd.date_range(bounds["train_start"], bounds["test_end"], freq="h")
    for name in ("demand", "weather", "temporal"):
        frame = frames[name]
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(f"{name} must have a timezone-aware hourly DatetimeIndex.")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{name} contains duplicate or unordered timestamps.")
        if len(frame) != len(reference_index) or not np.array_equal(frame.index.asi8, reference_index.asi8):
            raise ValueError(f"{name} must contain all 19056 consecutive hourly paper rows.")
        if not frame.columns.is_unique:
            raise ValueError(f"{name} contains duplicate feature columns.")
    required = {
        "demand": DMA_COLUMNS, "weather": tuple(WEATHER_REFERENCE),
        "temporal": TEMPORAL_COLUMNS,
    }
    for name, columns in required.items():
        missing = set(columns).difference(frames[name].columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}.")
        values = frames[name].loc[:, list(columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains missing/nonfinite observations; masking is forbidden.")
    counts = {}
    for split, expected in EXPECTED_ROWS.items():
        selected = frames["demand"].loc[bounds[f"{split}_start"]:bounds[f"{split}_end"]]
        counts[split] = len(selected)
        if counts[split] != expected:
            raise ValueError(f"{split} requires {expected} rows, got {counts[split]}.")
    return counts


def statistic_row(
    *, stage: str, split: str, feature: str, unit: str,
    statistic: str, actual: float, reference: float | None,
) -> dict[str, Any]:
    """Difference is diagnostic, not a quality threshold or target transformation."""
    return {
        "stage": stage, "split": split, "feature": feature, "unit": unit,
        "statistic": statistic, "actual": float(actual), "paper_reference": reference,
        "actual_minus_paper": None if reference is None else float(actual - reference),
        "absolute_relative_difference": None if reference in (None, 0) else float(abs(actual - reference) / abs(reference)),
    }


def audit_frames(
    frames: dict[str, pd.DataFrame], bounds: dict[str, pd.Timestamp],
    *, pre_outlier_demand: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return descriptive evidence without changing frames or choosing parameters."""
    counts = validate_paper_frames(frames, bounds)
    records = []
    stages = {"processed": frames["demand"]}
    if pre_outlier_demand is not None:
        validate_paper_frames({**frames, "demand": pre_outlier_demand}, bounds)
        stages["interpolated_before_outliers"] = pre_outlier_demand
    for split, expected_count in EXPECTED_ROWS.items():
        for stage, demand in stages.items():
            selected = demand.loc[bounds[f"{split}_start"]:bounds[f"{split}_end"]]
            for index, (letter, column) in enumerate(zip(DMA_LETTERS, DMA_COLUMNS)):
                records.append(statistic_row(
                    stage=stage, split=split, feature=f"DMA {letter}", unit="L/s",
                    statistic="mean", actual=float(selected[column].mean()),
                    reference=DEMAND_MEANS[split][index],
                ))
        weather = frames["weather"].loc[bounds[f"{split}_start"]:bounds[f"{split}_end"]]
        for column, reference in WEATHER_REFERENCE.items():
            values = weather[column].to_numpy(dtype=np.float64)
            stats = {
                "mean": float(values.mean()), "std_sample_ddof1": float(values.std(ddof=1)),
                "min": float(values.min()), "max": float(values.max()),
                "std_population_ddof0": float(values.std(ddof=0)), "count": float(len(values)),
            }
            expected = dict(zip(("mean", "std_sample_ddof1", "min", "max"), reference[split]))
            expected["count"] = expected_count
            for statistic, actual in stats.items():
                records.append(statistic_row(
                    stage="processed", split=split, feature=column,
                    unit="rows" if statistic == "count" else str(reference["unit"]),
                    statistic=statistic, actual=actual, reference=expected.get(statistic),
                ))
    eval_index = make_eval_index(
        test_start=bounds["test_start"], test_end=bounds["test_end"],
        max_history_weeks=4, horizon_hours=168, stride_hours=24,
        tz=frames["demand"].index.tz,
    )
    if len(eval_index) != 46:
        raise ValueError(f"Expected 46 common evaluation origins, got {len(eval_index)}.")
    starts = pd.DatetimeIndex(eval_index["forecast_start"])
    demand = frames["demand"].loc[:, list(DMA_COLUMNS)]
    targets = {
        "24h": np.stack([target_24h(demand, start) for start in starts]),
        "168h": np.stack([target_168h(demand, start) for start in starts]),
    }
    if not np.array_equal(targets["24h"], targets["168h"][:, :24]):
        raise ValueError("24h truth must equal first day of common 168h truth.")
    truths = {}
    for task, values in targets.items():
        total = values.astype(np.float64).sum(axis=2)
        truths[task] = {
            "shape": list(values.shape), "dtype": str(values.dtype),
            "array_sha256": array_sha256(values),
            "total_truth_population_variance": float(total.var(ddof=0)),
            "total_truth_origin_population_variances": total.var(axis=1, ddof=0).tolist(),
            "total_truth_mean": float(total.mean()),
            "dma_truth_population_variances": values.astype(np.float64).var(axis=(0, 1)).tolist(),
            "variance_unit": "(L/s)^2",
        }
    report = {
        "status": "VALID_INPUTS_STATISTICS_ADVISORY", "source": SOURCE,
        "data_modified": False, "statistics_are_acceptance_targets": False,
        "row_counts": counts, "total_rows": len(frames["demand"]),
        "timezone": str(frames["demand"].index.tz),
        "configured_bounds": {key: str(value) for key, value in bounds.items()},
        "notes": [
            "Paper text has an overlapping December-15 boundary; 17136/1920 counts define the nonoverlapping split used here.",
            "Published descriptive means may precede outlier exclusion; compare both stages when pre-outlier data is available.",
            "Paper does not specify the standard-deviation divisor. Both sample and population values are reported; reference shown beside sample is not proof of ddof=1.",
            "A statistical difference is evidence to investigate, not authority to rescale observations or modify test labels.",
            "Truth variance pools overlapping evaluation windows exactly as current pooled metrics do; unique calendar hours have different weighting.",
        ],
        "common_evaluation": {
            "sequence_count": len(starts), "history_weeks": 4, "stride_hours": 24,
            "first_forecast": str(starts[0]), "last_forecast": str(starts[-1]),
            "forecast_starts": [str(start) for start in starts], "truths": truths,
        },
        "statistics": records,
    }
    return report, eval_index


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/data_build")
    parser.add_argument("--split-config", type=Path, default=ROOT / "configs/data/paper_split.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        frames, _, bounds = load_paper_data(
            data_dir=args.data_dir, split_config_path=args.split_config, require_audit=True,
        )
        pre_path = args.data_dir / "demand_interpolated_before_outliers.parquet"
        report, eval_index = audit_frames(
            frames, bounds,
            pre_outlier_demand=pd.read_parquet(pre_path) if pre_path.is_file() else None,
        )
        inputs = [args.split_config, *(
            args.data_dir / name for name in (
                "demand_hourly.parquet", "weather_hourly.parquet", "temporal_hourly.parquet",
                "quality_checks.json", "demand_iqr_thresholds.csv", "interpolation_split_profile.csv",
            )
        )]
        if pre_path.is_file():
            inputs.append(pre_path)
        report["input_files"] = [
            {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in inputs
        ]
        pd.DataFrame(report["statistics"]).to_csv(args.output_root / "paper_data_statistics.csv", index=False)
        eval_index.to_csv(args.output_root / "common_eval_index.csv", index=False)
        (args.output_root / "paper_data_statistics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        print(json.dumps({"status": report["status"], "row_counts": report["row_counts"],
                          "test_sequences": 46, "output_root": str(args.output_root)}), flush=True)
        return 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        failure = {"status": "INVALID_INPUTS", "error": f"{type(exc).__name__}: {exc}", "data_modified": False}
        (args.output_root / "paper_data_statistics_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
