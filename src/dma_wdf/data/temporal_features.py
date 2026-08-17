"""Temporal feature engineering for hourly water demand data.

Constructs the 5 temporal features used by the MSCMNet paper:
hour, time_zone_standard (DST), weekday, holiday, day_of_week.
A sixth auxiliary feature ``time_idx`` (sequential index) is
also provided for models that need it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def boolish_to_int(series: pd.Series) -> pd.Series:
    """Convert boolean-like values to 0/1 integers.

    Handles: actual bools, strings ("True"/"False", "Yes"/"No",
    "Holiday"/"Workday", "1"/"0"), or numeric values.

    Args:
        series: A Series that may contain boolean-like values.

    Returns:
        Integer Series with 0 or 1 values.  If conversion fails,
        the original series is returned unchanged.
    """
    if series.dtype == bool:
        return series.astype(int)
    lowered = series.astype(str).str.lower()
    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "holiday": 1,
        "workday": 0,
        "1": 1,
        "0": 0,
    }
    mapped = lowered.map(mapping)
    if mapped.notna().all():
        return mapped.astype(int)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int)
    return series


def build_temporal_features(
    calendar: pd.DataFrame,
    include_time_idx: bool = True,
) -> pd.DataFrame:
    """Build temporal features from a calendar DataFrame.

    Constructs the following features from the DatetimeIndex
    and optional calendar columns:

    ====================  ======  =============================
    Feature               Type    Description
    ====================  ======  =============================
    ``hour``              int     0–23
    ``time_zone_standard`` int     1 if DST (CEST), 0 if standard
    ``weekday``           int     1 if Mon–Fri, 0 if Sat–Sun
    ``holiday``           int     1 if holiday, 0 otherwise
    ``day_of_week``       int     0=Mon … 6=Sun
    ``time_idx``          int     Sequential 0, 1, 2, …
    ====================  ======  =============================

    Args:
        calendar: DataFrame with a DatetimeIndex.  May optionally
            contain ``"CEST"`` and ``"Holiday"`` columns.
        include_time_idx: If True, include the sequential ``time_idx``
            column.

    Returns:
        DataFrame with the same index as ``calendar``.
    """
    idx = calendar.index
    temporal = pd.DataFrame(index=idx)
    temporal["hour"] = idx.hour.astype(int)

    # Daylight saving time flag.
    if "CEST" in calendar.columns:
        temporal["time_zone_standard"] = boolish_to_int(calendar["CEST"])
    else:
        temporal["time_zone_standard"] = idx.map(
            lambda ts: int(bool(ts.dst())) if ts.tzinfo else 0
        )

    # Weekday (Mon–Fri) vs weekend flag.
    temporal["weekday"] = (idx.dayofweek < 5).astype(int)

    # Holiday flag.
    if "Holiday" in calendar.columns:
        temporal["holiday"] = boolish_to_int(calendar["Holiday"])
    else:
        temporal["holiday"] = 0

    # Day of week (0=Monday … 6=Sunday).
    temporal["day_of_week"] = idx.dayofweek.astype(int)

    # Sequential time index (auxiliary).
    if include_time_idx:
        temporal["time_idx"] = np.arange(len(temporal), dtype=int)

    return temporal
