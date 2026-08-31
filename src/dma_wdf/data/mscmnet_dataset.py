"""Leakage-safe sample construction for the six Que et al. baselines."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dma_wdf.data.builders import build_demand_only_samples
from dma_wdf.data.loader import read_parquet_artifacts
from dma_wdf.data.sliding_window import (
    build_branch_array,
    make_eval_index,
    make_train_starts,
    slice_hours,
    target_24h,
    target_168h,
)
from dma_wdf.utils.config import parse_timestamp, read_yaml


@dataclass(frozen=True)
class JointTemporalSamples:
    """Raw arrays for one jointly-trained MSNet-family model."""

    train_branches: tuple[np.ndarray, ...]
    test_branches: tuple[np.ndarray, ...]
    branch_feature_columns: tuple[tuple[str, ...], ...]
    y_train_24h: np.ndarray
    y_test_24h: np.ndarray
    y_test_168h: np.ndarray
    future_train: np.ndarray
    future_test: np.ndarray
    fc2_train: np.ndarray | None
    fc2_test: np.ndarray | None
    fc2_share_target_train: np.ndarray | None
    train_forecast_starts: pd.DatetimeIndex
    test_forecast_starts: pd.DatetimeIndex


def validate_leakage_safe_data_build(
    data_dir: Path,
    *,
    expected_train_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Fail closed unless preprocessing provenance proves train-only fitting."""
    data_dir = Path(data_dir)
    required = {
        "quality": data_dir / "quality_checks.json",
        "thresholds": data_dir / "demand_iqr_thresholds.csv",
        "split_interpolation": data_dir / "interpolation_split_profile.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Formal temporal-baseline training requires preprocessing audit "
            f"artifacts; missing: {missing}. Re-run scripts/data/run_pipeline.sh."
        )

    quality = json.loads(required["quality"].read_text(encoding="utf-8"))
    if not bool(quality.get("all_passed", False)):
        raise ValueError("Data-build quality_checks.json reports a failed check.")

    thresholds = pd.read_csv(required["thresholds"])
    if thresholds.empty or "threshold_source" not in thresholds.columns:
        raise ValueError("IQR threshold provenance is missing or empty.")
    sources = set(thresholds["threshold_source"].astype(str).str.lower())
    if sources != {"train"}:
        raise ValueError(
            "IQR thresholds must be fitted only on training rows; observed "
            f"threshold_source={sorted(sources)}."
        )
    if expected_train_end is not None and "fit_end" in thresholds.columns:
        fit_end = pd.to_datetime(thresholds["fit_end"], utc=True, errors="raise")
        boundary = pd.Timestamp(expected_train_end).tz_convert("UTC")
        if bool((fit_end > boundary).any()):
            raise ValueError("At least one IQR threshold was fitted beyond train_end.")

    interpolation = pd.read_csv(required["split_interpolation"])
    if interpolation.empty or "split" not in interpolation.columns:
        raise ValueError("Split-aware interpolation provenance is missing.")
    observed_splits = set(interpolation["split"].astype(str).str.lower())
    if not {"train", "test"}.issubset(observed_splits):
        raise ValueError(
            "Interpolation audit must contain independent train and test rows."
        )
    return {
        "quality_all_passed": True,
        "iqr_threshold_source": "train",
        "iqr_fit_rows": sorted(thresholds["fit_rows"].astype(int).unique().tolist()),
        "interpolation_splits": sorted(observed_splits),
    }


def load_paper_data(
    *,
    data_dir: Path,
    split_config_path: Path,
    require_audit: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, pd.Timestamp]]:
    """Load processed frames and parsed paper split bounds."""
    frames = read_parquet_artifacts(Path(data_dir))
    split_config = read_yaml(Path(split_config_path))
    tz = frames["demand"].index.tz
    bounds = {
        "train_start": parse_timestamp(split_config["split"]["train_start"], tz),
        "train_end": parse_timestamp(
            split_config["split"]["train_end_inclusive"], tz
        ),
        "test_start": parse_timestamp(split_config["split"]["test_start"], tz),
        "test_end": parse_timestamp(
            split_config["split"]["test_end_inclusive"], tz
        ),
    }
    if require_audit:
        validate_leakage_safe_data_build(
            Path(data_dir), expected_train_end=bounds["train_end"]
        )
    return frames, split_config, bounds


def _feature_frame(
    *,
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    temporal: pd.DataFrame,
    dma_column: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    output = pd.DataFrame(index=demand.index)
    for column in columns:
        if column == "own_dma_demand":
            output[column] = demand[dma_column].astype(float)
        elif column in weather.columns:
            output[column] = weather[column].astype(float)
        elif column in temporal.columns:
            output[column] = temporal[column].astype(float)
        else:
            raise KeyError(f"Unknown temporal-baseline feature {column!r}.")
    return output


def forecast_day_features(
    *,
    weather: pd.DataFrame,
    temporal: pd.DataFrame,
    starts: Sequence[pd.Timestamp],
    columns: Sequence[str],
    day_offset: int = 0,
) -> np.ndarray:
    """Return known forecast-day weather/time features, never future demand."""
    columns = tuple(str(column) for column in columns)
    if not columns:
        return np.zeros((len(starts), 24, 0), dtype=np.float32)
    exogenous = pd.concat([weather, temporal], axis=1)
    missing = set(columns).difference(exogenous.columns)
    if missing:
        raise KeyError(f"Missing forecast-day features: {sorted(missing)}.")
    rows = [
        slice_hours(
            exogenous.loc[:, list(columns)],
            pd.Timestamp(start) + pd.Timedelta(days=int(day_offset)),
            24,
        )
        for start in starts
    ]
    return np.stack(rows).astype(np.float32)


def daily_share_history(
    *,
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    starts: Sequence[pd.Timestamp],
    dma_columns: Sequence[str],
    history_days: int,
    include_temperature: bool,
    temperature_column: str = "air_temperature",
) -> np.ndarray:
    """Build FC2 inputs from historical daily DMA shares and temperature."""
    history_days = int(history_days)
    if history_days <= 0:
        raise ValueError("history_days must be positive.")
    rows: list[np.ndarray] = []
    for forecast_start in starts:
        day_rows: list[np.ndarray] = []
        for day_back in range(history_days, 0, -1):
            day_start = pd.Timestamp(forecast_start) - pd.Timedelta(days=day_back)
            hourly = slice_hours(demand.loc[:, list(dma_columns)], day_start, 24)
            totals = hourly.sum(axis=0)
            shares = totals / max(float(totals.sum()), 1.0e-6)
            if include_temperature:
                temperature = slice_hours(
                    weather.loc[:, [temperature_column]], day_start, 24
                ).reshape(-1)
                shares = np.concatenate(
                    [shares, [float(temperature.max()), float(temperature.min())]]
                )
            day_rows.append(shares.astype(np.float32))
        rows.append(np.stack(day_rows))
    return np.stack(rows).astype(np.float32)


def daily_share_targets(target: np.ndarray) -> np.ndarray:
    """Return next-day DMA demand shares from a ``(S,24,N)`` target."""
    if target.ndim != 3:
        raise ValueError("target must have shape (S,24,N).")
    daily = target.sum(axis=1)
    denominator = np.maximum(daily.sum(axis=1, keepdims=True), 1.0e-6)
    return (daily / denominator).astype(np.float32)


def build_joint_temporal_samples(
    *,
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    temporal: pd.DataFrame,
    bounds: dict[str, pd.Timestamp],
    dma_columns: Sequence[str],
    branch_features: Sequence[str],
    input_weeks: Sequence[int],
    future_features: Sequence[str] = (),
    fc2_history_days: int | None = None,
    fc2_include_temperature: bool = False,
    max_history_weeks: int = 4,
    expected_train_samples: int = 686,
    expected_test_sequences: int = 46,
) -> JointTemporalSamples:
    """Build common-origin arrays for MSNet and all MSCMNet variants."""
    dma_columns = tuple(str(value) for value in dma_columns)
    input_weeks = tuple(int(value) for value in input_weeks)
    if len(dma_columns) != 10 or len(input_weeks) != len(dma_columns):
        raise ValueError("Joint models require ten DMA columns and ten histories.")
    if max(input_weeks) > int(max_history_weeks):
        raise ValueError("A branch history exceeds the common history buffer.")
    tz = demand.index.tz
    train_starts = make_train_starts(
        train_start=bounds["train_start"],
        train_end=bounds["train_end"],
        input_weeks=int(max_history_weeks),
        target_hours=24,
        tz=tz,
    )
    eval_index = make_eval_index(
        test_start=bounds["test_start"],
        test_end=bounds["test_end"],
        max_history_weeks=int(max_history_weeks),
        horizon_hours=168,
        stride_hours=24,
        tz=tz,
    )
    test_starts = pd.DatetimeIndex(eval_index["forecast_start"])
    if len(train_starts) != int(expected_train_samples):
        raise ValueError(
            f"Expected {expected_train_samples} joint train samples, "
            f"got {len(train_starts)}."
        )
    if len(test_starts) != int(expected_test_sequences):
        raise ValueError(
            f"Expected {expected_test_sequences} test sequences, "
            f"got {len(test_starts)}."
        )

    train_branches: list[np.ndarray] = []
    test_branches: list[np.ndarray] = []
    feature_names: list[tuple[str, ...]] = []
    for dma_column, weeks in zip(dma_columns, input_weeks):
        features = _feature_frame(
            demand=demand,
            weather=weather,
            temporal=temporal,
            dma_column=dma_column,
            columns=branch_features,
        )
        train_branches.append(
            build_branch_array(features, train_starts, weeks * 7)
        )
        test_branches.append(
            build_branch_array(features, test_starts, weeks * 7)
        )
        feature_names.append(tuple(features.columns))

    selected_demand = demand.loc[:, list(dma_columns)]
    y_train = np.stack(
        [target_24h(selected_demand, start) for start in train_starts]
    ).astype(np.float32)
    y_test_24 = np.stack(
        [target_24h(selected_demand, start) for start in test_starts]
    ).astype(np.float32)
    y_test_168 = np.stack(
        [target_168h(selected_demand, start) for start in test_starts]
    ).astype(np.float32)
    future_train = forecast_day_features(
        weather=weather,
        temporal=temporal,
        starts=train_starts,
        columns=future_features,
    )
    future_test = forecast_day_features(
        weather=weather,
        temporal=temporal,
        starts=test_starts,
        columns=future_features,
    )

    fc2_train: np.ndarray | None = None
    fc2_test: np.ndarray | None = None
    share_target: np.ndarray | None = None
    if fc2_history_days is not None:
        fc2_train = daily_share_history(
            demand=selected_demand,
            weather=weather,
            starts=train_starts,
            dma_columns=dma_columns,
            history_days=fc2_history_days,
            include_temperature=fc2_include_temperature,
        )
        fc2_test = daily_share_history(
            demand=selected_demand,
            weather=weather,
            starts=test_starts,
            dma_columns=dma_columns,
            history_days=fc2_history_days,
            include_temperature=fc2_include_temperature,
        )
        share_target = daily_share_targets(y_train)

    return JointTemporalSamples(
        train_branches=tuple(train_branches),
        test_branches=tuple(test_branches),
        branch_feature_columns=tuple(feature_names),
        y_train_24h=y_train,
        y_test_24h=y_test_24,
        y_test_168h=y_test_168,
        future_train=future_train,
        future_test=future_test,
        fc2_train=fc2_train,
        fc2_test=fc2_test,
        fc2_share_target_train=share_target,
        train_forecast_starts=train_starts,
        test_forecast_starts=test_starts,
    )


def build_independent_temporal_samples(
    *,
    demand: pd.DataFrame,
    bounds: dict[str, pd.Timestamp],
    dma_column: str,
    input_weeks: int,
) -> dict[str, np.ndarray]:
    """Build demand-only arrays for one independent GRU/LSTM model."""
    return build_demand_only_samples(
        demand=demand,
        dma_column=str(dma_column),
        input_weeks=int(input_weeks),
        train_start=bounds["train_start"],
        train_end=bounds["train_end"],
        test_start=bounds["test_start"],
        test_end=bounds["test_end"],
        tz=demand.index.tz,
    )


__all__ = [
    "JointTemporalSamples",
    "build_independent_temporal_samples",
    "build_joint_temporal_samples",
    "daily_share_history",
    "daily_share_targets",
    "forecast_day_features",
    "load_paper_data",
    "validate_leakage_safe_data_build",
]
