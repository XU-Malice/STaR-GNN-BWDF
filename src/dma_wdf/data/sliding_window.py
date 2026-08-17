"""Sliding window construction for hourly time series.

All forecast windows use a daily stride (one forecast per midnight).
History and future windows are extracted from time-indexed DataFrames
using ``slice_hours``, which validates that every expected row exists.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def daily_starts(
    start: pd.Timestamp,
    end: pd.Timestamp,
    tz: Any = None,
) -> pd.DatetimeIndex:
    """Generate daily midnight timestamps between ``start`` and ``end``.

    If ``start`` is already past its own midnight, the first entry is
    the **next** midnight.  This ensures every returned timestamp is a
    valid forecast start date (forecasts begin at midnight in the paper
    protocol).

    Args:
        start: Earliest timestamp to consider.
        end: Latest timestamp to consider.
        tz: Timezone for the generated timestamps.

    Returns:
        DatetimeIndex of daily midnight timestamps.
    """
    start_midnight = pd.Timestamp(start.date()).tz_localize(tz)
    if start > start_midnight:
        start_midnight = start_midnight + pd.Timedelta(days=1)
    end_midnight = pd.Timestamp(end.date()).tz_localize(tz)
    return pd.date_range(start_midnight, end_midnight, freq="D", tz=tz)


def slice_hours(
    df: pd.DataFrame | pd.Series,
    start: pd.Timestamp,
    hours: int,
) -> np.ndarray:
    """Extract ``hours`` consecutive rows from a time-indexed object.

    Args:
        df: DataFrame or Series with a DatetimeIndex.
        start: First timestamp (inclusive).
        hours: Number of hourly rows to extract.

    Returns:
        NumPy float32 array of shape ``(hours, N_cols)`` for DataFrame
        or ``(hours,)`` for Series.

    Raises:
        ValueError: If the expected number of rows is not found.
    """
    end = start + pd.Timedelta(hours=hours - 1)
    values = df.loc[(df.index >= start) & (df.index <= end)].to_numpy()
    if len(values) != hours:
        raise ValueError(
            f"Expected {hours} rows from {start} to {end}, got {len(values)}"
        )
    return values.astype(np.float32)


def slice_window(
    series: pd.Series,
    start: pd.Timestamp,
    hours: int,
) -> np.ndarray:
    """Extract ``hours`` consecutive values from a univariate Series.

    Convenience wrapper around ``slice_hours`` that always returns a
    1-D array.

    Args:
        series: Time-indexed Series.
        start: First timestamp (inclusive).
        hours: Number of hourly values to extract.

    Returns:
        1-D float32 NumPy array of length ``hours``.
    """
    return slice_hours(series, start, hours)


def build_windows(
    frame: pd.DataFrame,
    starts: pd.DatetimeIndex,
    history_hours: int,
    future_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build past and future sliding windows for all forecast starts.

    For each ``forecast_start`` in ``starts``:

    - **Past window**: ``[forecast_start - history_hours, forecast_start - 1h]``
    - **Future window**: ``[forecast_start, forecast_start + future_hours - 1h]``

    Args:
        frame: Time-indexed DataFrame with feature columns.
        starts: Forecast start timestamps.
        history_hours: Length of the historical window (past).
        future_hours: Length of the prediction window (future).

    Returns:
        Tuple ``(past_array, future_array)``:
        - ``past_array``: shape ``(len(starts), history_hours, N_features)``
        - ``future_array``: shape ``(len(starts), future_hours, N_features)``
    """
    past_rows: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    for forecast_start in starts:
        history_start = forecast_start - pd.Timedelta(hours=history_hours)
        past_rows.append(slice_hours(frame, history_start, history_hours))
        future_rows.append(slice_hours(frame, forecast_start, future_hours))
    return (
        np.stack(past_rows).astype(np.float32),
        np.stack(future_rows).astype(np.float32),
    )


def build_branch_array(
    features: pd.DataFrame,
    starts: pd.DatetimeIndex,
    history_days: int,
) -> np.ndarray:
    """Build per-DMA branch tensors with daily-grid layout.

    Each forecast start yields a block of shape
    ``(history_days, 24, N_features)``, preserving the day-hour
    structure for CNN-based models (MSNet / MSCMNet family).

    Args:
        features: Time-indexed feature DataFrame.
        starts: Forecast start timestamps.
        history_days: Number of days of history to include.

    Returns:
        Array of shape ``(len(starts), history_days, 24, N_features)``.
    """
    hours = int(history_days) * 24
    rows: list[np.ndarray] = []
    for forecast_start in starts:
        history_start = forecast_start - pd.Timedelta(hours=hours)
        window = slice_hours(features, history_start, hours)
        rows.append(window.reshape(history_days, 24, features.shape[1]))
    return np.stack(rows).astype(np.float32)


def target_24h(
    demand: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> np.ndarray:
    """Extract a 24-hour target window for all DMAs.

    Args:
        demand: Time-indexed DataFrame of DMA inflows (multi-column).
        forecast_start: Start of the forecast horizon.

    Returns:
        Array of shape ``(24, N_dmas)``.
    """
    return slice_hours(demand, forecast_start, 24).reshape(24, demand.shape[1])


def target_168h(
    demand: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> np.ndarray:
    """Extract a 168-hour (7-day) target window for all DMAs.

    Args:
        demand: Time-indexed DataFrame of DMA inflows (multi-column).
        forecast_start: Start of the forecast horizon.

    Returns:
        Array of shape ``(168, N_dmas)``.
    """
    return slice_hours(demand, forecast_start, 168).reshape(168, demand.shape[1])


def build_sample_index(
    *,
    paper_start: pd.Timestamp,
    paper_end: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    horizon_hours: int,
    stride_hours: int,
    max_history_weeks: int,
    tz: Any,
) -> pd.DataFrame:
    """Create a DataFrame describing every forecast sample in the paper protocol.

    Each row represents one forecast sequence with:

    - ``forecast_start``, ``forecast_end``
    - ``history_start``, ``history_end``
    - ``horizon_hours``, ``stride_hours``, ``max_history_weeks``
    - ``split``: ``"train"``, ``"test"``, or ``"boundary_or_unused"``

    The first valid forecast start is
    ``paper_start + max_history_weeks * 7 * 24 hours``.

    Args:
        paper_start: Start of the paper period.
        paper_end: End of the paper period.
        train_end: End of the training period (inclusive).
        test_start: Start of the test period.
        test_end: End of the test period (inclusive).
        horizon_hours: Forecast horizon in hours.
        stride_hours: Stride between consecutive forecasts.
        max_history_weeks: Maximum history length in weeks.
        tz: Timezone.

    Returns:
        DataFrame with one row per forecast sequence.
    """
    history_hours = max_history_weeks * 7 * 24
    first_forecast = paper_start + pd.Timedelta(hours=history_hours)
    starts = daily_starts(
        first_forecast,
        paper_end - pd.Timedelta(hours=horizon_hours - 1),
        tz,
    )
    rows: list[dict[str, Any]] = []
    for forecast_start in starts:
        forecast_end = forecast_start + pd.Timedelta(hours=horizon_hours - 1)
        if forecast_end <= train_end:
            split = "train"
        elif forecast_start >= test_start and forecast_end <= test_end:
            split = "test"
        else:
            split = "boundary_or_unused"
        rows.append(
            {
                "forecast_start": forecast_start,
                "forecast_end": forecast_end,
                "history_start": forecast_start - pd.Timedelta(hours=history_hours),
                "history_end": forecast_start - pd.Timedelta(hours=1),
                "horizon_hours": horizon_hours,
                "stride_hours": stride_hours,
                "max_history_weeks": max_history_weeks,
                "split": split,
            }
        )
    return pd.DataFrame(rows)


def make_eval_index(
    *,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    max_history_weeks: int = 4,
    horizon_hours: int = 168,
    stride_hours: int = 24,
    tz: Any = None,
) -> pd.DataFrame:
    """Build the 46-sequence test evaluation index.

    The paper reports 46 test evaluation sequences:
    ``80 test days - 28 history days - 7 horizon days + 1 = 46``.

    Args:
        test_start: Start of the test period.
        test_end: End of the test period (inclusive).
        max_history_weeks: Within-test history buffer in weeks.
        horizon_hours: Multi-step horizon (168 = 7 days).
        stride_hours: Stride between sequences (24 = daily).
        tz: Timezone.

    Returns:
        DataFrame with columns: ``sequence_id``, ``forecast_start``,
        ``forecast_end_24h``, ``forecast_end_168h``,
        ``history_start_max_4w``, ``history_end``, ``stride_hours``.
    """
    history_days = max_history_weeks * 7
    first_forecast = test_start + pd.Timedelta(days=history_days)
    last_forecast = test_end - pd.Timedelta(hours=horizon_hours - 1)
    starts = daily_starts(first_forecast, last_forecast, tz)
    rows: list[dict[str, Any]] = []
    for i, forecast_start in enumerate(starts):
        history_start = forecast_start - pd.Timedelta(days=history_days)
        rows.append(
            {
                "sequence_id": i,
                "forecast_start": forecast_start,
                "forecast_end_24h": forecast_start + pd.Timedelta(hours=23),
                "forecast_end_168h": forecast_start + pd.Timedelta(hours=167),
                "history_start_max_4w": history_start,
                "history_end": forecast_start - pd.Timedelta(hours=1),
                "stride_hours": stride_hours,
            }
        )
    return pd.DataFrame(rows)


def make_train_starts(
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    input_weeks: int,
    target_hours: int = 24,
    tz: Any = None,
) -> pd.DatetimeIndex:
    """Generate training forecast start dates for a specific DMA/model.

    The first valid start is offset by ``input_weeks * 7 * 24`` hours
    from ``train_start``.

    Args:
        train_start: Start of the training period.
        train_end: End of the training period (inclusive).
        input_weeks: Number of weeks of history required.
        target_hours: Forecast target length (last forecast must fit).
        tz: Timezone.

    Returns:
        DatetimeIndex of daily midnight forecast start timestamps.
    """
    offset_hours = input_weeks * 7 * 24
    return daily_starts(
        train_start + pd.Timedelta(hours=offset_hours),
        train_end - pd.Timedelta(hours=target_hours - 1),
        tz,
    )


def combine_past(
    demand_past: np.ndarray,
    weather_past: np.ndarray,
    time_past: np.ndarray,
) -> np.ndarray:
    """Combine multi-modal past features into a 4-D array.

    Weather and temporal features are concatenated along the feature
    axis, then expanded across the DMA dimension, then concatenated
    with demand as the last channel.

    Args:
        demand_past: shape ``(N, hours, N_dmas)`` — past demand.
        weather_past: shape ``(N, hours, N_weather)`` — past weather.
        time_past: shape ``(N, hours, N_time)`` — past temporal features.

    Returns:
        Array of shape ``(N, hours, N_dmas, 1 + N_weather + N_time)``.
    """
    exog = np.concatenate([weather_past, time_past], axis=2)
    exog = np.repeat(exog[:, :, None, :], demand_past.shape[2], axis=2)
    return np.concatenate([demand_past[..., None], exog], axis=3).astype(np.float32)
