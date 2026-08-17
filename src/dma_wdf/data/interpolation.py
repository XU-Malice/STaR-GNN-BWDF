"""Leakage-safe interpolation helpers for hourly time series."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


_VALID_METHODS = {"time", "linear"}
_VALID_DIRECTIONS = {"forward", "backward", "both"}


def _validate_time_index(
    obj: pd.DataFrame | pd.Series,
    *,
    name: str,
) -> None:
    """Validate the index assumptions required by time interpolation."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a pandas DatetimeIndex.")
    if not obj.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in increasing time order.")
    if not obj.index.is_unique:
        raise ValueError(f"{name} index must not contain duplicate timestamps.")


def _validate_options(method: str, limit_direction: str) -> None:
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unsupported interpolation method {method!r}; "
            f"expected one of {sorted(_VALID_METHODS)}."
        )
    if limit_direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"Unsupported limit_direction {limit_direction!r}; "
            f"expected one of {sorted(_VALID_DIRECTIONS)}."
        )


def interpolate_time(
    frame: pd.DataFrame,
    limit_direction: str = "both",
    *,
    method: str = "time",
) -> pd.DataFrame:
    """Interpolate a time-indexed DataFrame without mutating the input."""
    _validate_time_index(frame, name="frame")
    _validate_options(method, limit_direction)
    return frame.astype(float).interpolate(
        method=method,
        limit_direction=limit_direction,
    )


def interpolate_series(
    series: pd.Series,
    limit_direction: str = "both",
    *,
    method: str = "time",
) -> pd.Series:
    """Interpolate a time-indexed Series without mutating the input."""
    _validate_time_index(series, name="series")
    _validate_options(method, limit_direction)
    return series.astype(float).interpolate(
        method=method,
        limit_direction=limit_direction,
    )


def interpolate_by_splits(
    frame: pd.DataFrame,
    split_ranges: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
    *,
    method: str = "time",
    limit_direction: str = "both",
    require_full_coverage: bool = True,
    require_no_nan: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interpolate each temporal partition independently.

    This prevents a value in a later partition (for example, test) from
    being used to fill a missing value in an earlier partition (train).

    Returns:
        ``(interpolated, profile)`` where ``profile`` contains one row per
        split with row counts and missing-value counts before/after.
    """
    _validate_time_index(frame, name="frame")
    _validate_options(method, limit_direction)
    if not split_ranges:
        raise ValueError("split_ranges must contain at least one partition.")

    result = frame.astype(float).copy()
    coverage = pd.Series(0, index=frame.index, dtype="int64")
    profile_rows: list[dict[str, Any]] = []

    for split_name, (start, end) in split_ranges.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts > end_ts:
            raise ValueError(
                f"Split {split_name!r} has start later than end: "
                f"{start_ts} > {end_ts}."
            )

        mask = (frame.index >= start_ts) & (frame.index <= end_ts)
        if not bool(mask.any()):
            raise ValueError(
                f"Split {split_name!r} contains no rows in the supplied frame."
            )
        coverage.loc[mask] += 1
        if bool((coverage > 1).any()):
            raise ValueError("split_ranges must not overlap.")

        before = frame.loc[mask]
        after = interpolate_time(
            before,
            method=method,
            limit_direction=limit_direction,
        )
        result.loc[mask, :] = after
        profile_rows.append(
            {
                "split": split_name,
                "start": str(before.index.min()),
                "end": str(before.index.max()),
                "rows": int(len(before)),
                "missing_before": int(before.isna().sum().sum()),
                "missing_after": int(after.isna().sum().sum()),
            }
        )

    if require_full_coverage and bool((coverage != 1).any()):
        uncovered = int((coverage == 0).sum())
        raise ValueError(
            f"split_ranges do not cover the complete frame; "
            f"{uncovered} row(s) are uncovered."
        )

    if require_no_nan and bool(result.isna().any().any()):
        missing = result.isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(
            "Interpolation left missing values. This usually means that an "
            f"entire split/column is missing: {missing}"
        )

    return result, pd.DataFrame(profile_rows)