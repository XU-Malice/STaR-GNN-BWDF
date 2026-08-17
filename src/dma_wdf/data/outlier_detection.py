"""IQR outlier handling with fit/apply separation.

The key rule is that thresholds may be fitted on training data and then
reused unchanged for validation/test data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from dma_wdf.data.interpolation import (
    interpolate_by_splits,
    interpolate_series,
    interpolate_time,
)


_THRESHOLD_COLUMNS = {"lower", "upper"}


def compute_thresholds(
    series: pd.Series,
    multiplier: float = 1.5,
) -> dict[str, float]:
    """Compute Tukey IQR fences for one Series."""
    if multiplier < 0:
        raise ValueError("multiplier must be non-negative.")
    values = series.astype(float).dropna()
    if values.empty:
        raise ValueError(f"Cannot fit IQR thresholds for empty column {series.name!r}.")

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": q1 - multiplier * iqr,
        "upper": q3 + multiplier * iqr,
    }


def detect_outliers(
    series: pd.Series,
    lower: float,
    upper: float,
) -> pd.Series:
    """Return a Boolean mask for values outside inclusive IQR fences."""
    return ((series < lower) | (series > upper)).fillna(False)


def clip_outliers(
    series: pd.Series,
    lower: float,
    upper: float,
) -> tuple[pd.Series, pd.Series]:
    """Clip values and return ``(processed, outlier_mask)``."""
    mask = detect_outliers(series, lower, upper)
    return series.clip(lower=lower, upper=upper).astype(float), mask


def interpolate_outliers(
    series: pd.Series,
    lower: float,
    upper: float,
    *,
    limit_direction: str = "both",
) -> tuple[pd.Series, pd.Series]:
    """Replace outliers with NaN and return ``(processed, mask)``."""
    mask = detect_outliers(series, lower, upper)
    masked = series.astype(float).mask(mask)
    return interpolate_series(masked, limit_direction=limit_direction), mask


def fit_iqr_thresholds(
    demand_fit: pd.DataFrame,
    *,
    multiplier: float = 1.5,
    threshold_source: str = "train",
) -> pd.DataFrame:
    """Fit one pair of fixed IQR fences per DMA column."""
    if demand_fit.empty:
        raise ValueError("demand_fit must contain at least one row.")
    rows: list[dict[str, Any]] = []
    for column in demand_fit.columns:
        stats = compute_thresholds(demand_fit[column], multiplier=multiplier)
        rows.append(
            {
                "dma_column": column,
                **stats,
                "multiplier": float(multiplier),
                "threshold_source": threshold_source,
                "fit_start": str(demand_fit.index.min()),
                "fit_end": str(demand_fit.index.max()),
                "fit_rows": int(len(demand_fit)),
            }
        )
    return pd.DataFrame(rows).set_index("dma_column")


def _validate_threshold_table(
    demand: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    missing_fields = _THRESHOLD_COLUMNS.difference(thresholds.columns)
    if missing_fields:
        raise ValueError(
            f"thresholds is missing fields: {sorted(missing_fields)}."
        )
    missing_dmas = set(demand.columns).difference(thresholds.index)
    if missing_dmas:
        raise ValueError(
            f"thresholds is missing DMA columns: {sorted(missing_dmas)}."
        )


def apply_iqr_thresholds(
    demand: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    method: str = "interpolate",
    limit_direction: str = "both",
    split_ranges: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    require_no_nan: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply fixed thresholds without recomputing them on later data."""
    if method not in {"interpolate", "clip"}:
        raise ValueError("method must be either 'interpolate' or 'clip'.")
    _validate_threshold_table(demand, thresholds)

    numeric = demand.astype(float)
    mask = pd.DataFrame(False, index=demand.index, columns=demand.columns)
    for column in demand.columns:
        row = thresholds.loc[column]
        mask[column] = detect_outliers(
            numeric[column],
            float(row["lower"]),
            float(row["upper"]),
        )

    if method == "clip":
        processed = numeric.copy()
        for column in demand.columns:
            row = thresholds.loc[column]
            processed[column], _ = clip_outliers(
                numeric[column],
                float(row["lower"]),
                float(row["upper"]),
            )
    else:
        masked = numeric.mask(mask)
        if split_ranges is None:
            processed = interpolate_time(
                masked,
                limit_direction=limit_direction,
            )
        else:
            processed, _ = interpolate_by_splits(
                masked,
                split_ranges,
                limit_direction=limit_direction,
                require_full_coverage=True,
                require_no_nan=require_no_nan,
            )

    if require_no_nan and bool(processed.isna().any().any()):
        missing = processed.isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(f"Outlier preprocessing left missing values: {missing}")

    profile_rows: list[dict[str, Any]] = []
    for column in demand.columns:
        threshold = thresholds.loc[column]
        before = numeric[column].to_numpy(dtype=float)
        after = processed[column].to_numpy(dtype=float)
        changed = ~np.isclose(before, after, equal_nan=True)
        profile_rows.append(
            {
                "dma_column": column,
                "q1": float(threshold.get("q1", np.nan)),
                "q3": float(threshold.get("q3", np.nan)),
                "iqr": float(threshold.get("iqr", np.nan)),
                "lower": float(threshold["lower"]),
                "upper": float(threshold["upper"]),
                "outliers": int(mask[column].sum()),
                "processed_nan_count": int(processed[column].isna().sum()),
                "changed_points": int(changed.sum()),
                "threshold_source": str(
                    threshold.get("threshold_source", "unknown")
                ),
                "outlier_count": int(mask[column].sum()),
                "changed_count": int(changed.sum()),
                "method": method,
            }
        )

    return processed, pd.DataFrame(profile_rows), mask


def preprocess_series(
    series: pd.Series,
    *,
    lower: float,
    upper: float,
    method: str = "interpolate",
) -> tuple[pd.Series, pd.Series]:
    """Apply supplied thresholds; preserves the existing public API."""
    if method == "clip":
        return clip_outliers(series, lower, upper)
    if method == "interpolate":
        return interpolate_outliers(series, lower, upper)
    raise ValueError(
        f"Unsupported outlier method: {method!r}. "
        "Use 'interpolate' or 'clip'."
    )


def preprocess_demand(
    demand: pd.DataFrame,
    multiplier: float = 1.5,
    method: str = "interpolate",
    *,
    fit_demand: pd.DataFrame | None = None,
    thresholds: pd.DataFrame | None = None,
    threshold_source: str = "full",
    limit_direction: str = "both",
    split_ranges: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit/apply IQR preprocessing while preserving the old public API.

    Existing calls such as ``preprocess_demand(demand, multiplier=1.5)``
    still work. Leakage-safe code should pass ``fit_demand=train_demand``
    and ``threshold_source="train"``.
    """
    if thresholds is not None and fit_demand is not None:
        raise ValueError("Pass either thresholds or fit_demand, not both.")
    if thresholds is None:
        if threshold_source == "train" and fit_demand is None:
            raise ValueError(
                "threshold_source='train' requires fit_demand=train_demand."
            )
        fit_frame = demand if fit_demand is None else fit_demand
        thresholds = fit_iqr_thresholds(
            fit_frame,
            multiplier=multiplier,
            threshold_source=threshold_source,
        )

    processed, profile, _ = apply_iqr_thresholds(
        demand,
        thresholds,
        method=method,
        limit_direction=limit_direction,
        split_ranges=split_ranges,
    )
    return processed, profile