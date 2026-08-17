"""Tests for leakage-aware DCRNN dataset construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dma_wdf.data.dcrnn_dataset import (
    TEMPORAL_FEATURE_ORDER,
    ZScoreScaler,
    _build_window_subset,
    build_evaluation_protocol_indices,
    encode_temporal_features,
    split_development_index,
)


def _sample_index(
    *,
    count: int,
    horizon: int,
    start: str = "2021-01-29",
) -> pd.DataFrame:
    starts = pd.date_range(
        start,
        periods=count,
        freq="D",
        tz="Europe/Rome",
    )
    return pd.DataFrame(
        {
            "forecast_start": starts,
            "forecast_end": starts
            + pd.Timedelta(hours=horizon - 1),
            "history_start": starts - pd.Timedelta(hours=672),
            "history_end": starts - pd.Timedelta(hours=1),
            "horizon_hours": horizon,
            "stride_hours": 24,
            "split": "train",
        }
    )


def test_temporal_encoding_has_fixed_seven_feature_order() -> None:
    index = pd.date_range(
        "2022-01-03",
        periods=24,
        freq="h",
        tz="Europe/Rome",
    )
    temporal = pd.DataFrame(
        {
            "hour": np.arange(24),
            "time_zone_standard": 1,
            "weekday": 1,
            "holiday": 0,
            "day_of_week": 0,
            "time_idx": np.arange(24),
        },
        index=index,
    )
    encoded = encode_temporal_features(temporal)
    assert list(encoded.columns) == TEMPORAL_FEATURE_ORDER
    assert encoded.shape == (24, 7)
    assert encoded.loc[index[0], "hour_sin"] == pytest.approx(0.0)
    assert encoded.loc[index[0], "hour_cos"] == pytest.approx(1.0)
    assert np.isfinite(encoded.to_numpy()).all()


def test_scaler_round_trip_and_state_round_trip() -> None:
    index = pd.date_range(
        "2021-01-01",
        periods=12,
        freq="h",
        tz="Europe/Rome",
    )
    frame = pd.DataFrame(
        {
            "A": np.arange(12, dtype=float),
            "B": np.arange(12, dtype=float) * 2.0,
        },
        index=index,
    )
    scaler = ZScoreScaler.fit(
        frame,
        feature_names=["A", "B"],
        fit_start=index[0],
        fit_end=index[7],
    )
    restored = ZScoreScaler.from_state_dict(scaler.state_dict())
    values = frame.to_numpy(dtype=np.float32)
    assert np.allclose(
        restored.inverse_transform(restored.transform(values)),
        values,
        atol=1.0e-6,
    )
    assert restored.fit_rows == 8
    assert restored.fit_end == str(index[7])


def test_validation_changes_do_not_change_fit_scaler() -> None:
    index = pd.date_range(
        "2021-01-01",
        periods=20,
        freq="h",
        tz="Europe/Rome",
    )
    original = pd.DataFrame(
        {"A": np.arange(20, dtype=float)},
        index=index,
    )
    changed = original.copy()
    changed.loc[index[10]:, "A"] *= 1000.0
    first = ZScoreScaler.fit(
        original,
        feature_names=["A"],
        fit_start=index[0],
        fit_end=index[9],
    )
    second = ZScoreScaler.fit(
        changed,
        feature_names=["A"],
        fit_start=index[0],
        fit_end=index[9],
    )
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.std, second.std)


def test_24h_development_split_is_618_0_68() -> None:
    index = _sample_index(count=686, horizon=24)
    fit, purge, validation = split_development_index(
        index,
        validation_samples=68,
        purge_samples=0,
    )
    assert (len(fit), len(purge), len(validation)) == (618, 0, 68)
    assert fit.iloc[-1]["forecast_end"] < validation.iloc[0][
        "forecast_start"
    ]


def test_168h_development_split_is_606_6_68() -> None:
    index = _sample_index(count=680, horizon=168)
    fit, purge, validation = split_development_index(
        index,
        validation_samples=68,
        purge_samples=6,
    )
    assert (len(fit), len(purge), len(validation)) == (606, 6, 68)
    assert fit.iloc[-1]["forecast_end"] < validation.iloc[0][
        "forecast_start"
    ]


def test_168h_without_purge_is_rejected() -> None:
    index = _sample_index(count=680, horizon=168)
    with pytest.raises(ValueError, match="overlap"):
        split_development_index(
            index,
            validation_samples=68,
            purge_samples=0,
        )


@pytest.mark.parametrize(
    ("horizon", "test_count", "strict_count"),
    [(24, 80, 52), (168, 74, 46)],
)
def test_evaluation_protocol_counts(
    horizon: int,
    test_count: int,
    strict_count: int,
) -> None:
    test_start = pd.Timestamp(
        "2022-12-16 00:00:00",
        tz="Europe/Rome",
    )
    test_end = pd.Timestamp(
        "2023-03-05 23:00:00",
        tz="Europe/Rome",
    )
    starts = pd.date_range(
        test_start,
        periods=test_count,
        freq="D",
        tz="Europe/Rome",
    )
    index = pd.DataFrame(
        {
            "forecast_start": starts,
            "forecast_end": starts
            + pd.Timedelta(hours=horizon - 1),
        }
    )
    protocols = build_evaluation_protocol_indices(
        index,
        official_test_start=test_start,
        official_test_end=test_end,
        history_hours=672,
        timezone=test_start.tz,
    )
    assert len(protocols["operational"]) == test_count
    assert len(protocols["strict_within_test"]) == strict_count
    assert len(protocols["common_46"]) == 46


def test_small_window_build_matches_dcrnn_contract() -> None:
    index = pd.date_range(
        "2021-01-01",
        periods=80,
        freq="h",
        tz="Europe/Rome",
    )
    demand_columns = [f"DMA {i}" for i in range(1, 11)]
    weather_columns = ["rain", "temperature", "humidity", "wind"]
    demand = pd.DataFrame(
        np.arange(80 * 10, dtype=np.float32).reshape(80, 10),
        index=index,
        columns=demand_columns,
    )
    weather = pd.DataFrame(
        np.arange(80 * 4, dtype=np.float32).reshape(80, 4),
        index=index,
        columns=weather_columns,
    )
    temporal_raw = pd.DataFrame(
        {
            "hour": index.hour,
            "day_of_week": index.dayofweek,
            "time_zone_standard": 1,
            "weekday": (index.dayofweek < 5).astype(int),
            "holiday": 0,
        },
        index=index,
    )
    temporal = encode_temporal_features(temporal_raw)
    starts = pd.DatetimeIndex([index[16], index[32]])
    sample_index = pd.DataFrame({"forecast_start": starts})
    demand_scaler = ZScoreScaler.fit(
        demand,
        feature_names=demand_columns,
        fit_start=index[0],
        fit_end=index[63],
    )
    weather_scaler = ZScoreScaler.fit(
        weather,
        feature_names=weather_columns,
        fit_start=index[0],
        fit_end=index[63],
    )
    subset = _build_window_subset(
        demand=demand,
        weather=weather,
        temporal=temporal,
        sample_index=sample_index,
        history_hours=8,
        horizon=24,
        demand_scaler=demand_scaler,
        weather_scaler=weather_scaler,
        num_nodes=10,
    )
    assert subset.x_past.shape == (2, 8, 10, 12)
    assert subset.y_scaled.shape == (2, 24, 10)
    assert subset.future_exog.shape == (2, 24, 10, 7)
    assert np.isfinite(subset.x_past).all()


def test_task_configs_lock_expected_counts() -> None:
    root = Path(__file__).resolve().parent.parent
    expected = {
        "24h": (686, 618, 0, 68, 80, 52, 46),
        "168h": (680, 606, 6, 68, 74, 46, 46),
    }
    for task, values in expected.items():
        config = yaml.safe_load(
            (root / "configs" / "train" / f"dcrnn_{task}.yaml").read_text(
                encoding="utf-8"
            )
        )
        split = config["split"]
        observed = (
            split["expected_development_samples"],
            split["expected_fit_samples"],
            split["purge_samples"],
            split["validation_samples"],
            split["expected_test_candidates"],
            split["expected_strict_samples"],
            split["expected_common_samples"],
        )
        assert observed == values
