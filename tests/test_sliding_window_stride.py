from __future__ import annotations

import pandas as pd
import pytest

from dma_wdf.data.sliding_window import make_train_starts


def test_training_stride_is_explicit_and_preserves_daily_default() -> None:
    kwargs = {
        "train_start": pd.Timestamp("2020-01-01 00:00:00"),
        "train_end": pd.Timestamp("2020-01-20 23:00:00"),
        "input_weeks": 1,
        "target_hours": 24,
    }
    daily = make_train_starts(**kwargs)
    six_hourly = make_train_starts(**kwargs, stride_hours=6)
    assert len(daily) == 13
    assert len(six_hourly) == 49
    assert daily[0] == six_hourly[0]
    assert daily[-1] == six_hourly[-1]
    assert six_hourly[1] - six_hourly[0] == pd.Timedelta(hours=6)


def test_training_stride_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_train_starts(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-01-20 23:00:00"),
            input_weeks=1,
            stride_hours=0,
        )
