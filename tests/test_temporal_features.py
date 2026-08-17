"""Tests for temporal feature engineering."""

import pandas as pd
import numpy as np

from dma_wdf.data.temporal_features import boolish_to_int, build_temporal_features


class TestBoolishToInt:
    def test_bool_series(self):
        s = pd.Series([True, False, True])
        result = boolish_to_int(s)
        assert result.tolist() == [1, 0, 1]

    def test_string_series(self):
        s = pd.Series(["True", "False", "Yes", "No"])
        result = boolish_to_int(s)
        assert result.tolist() == [1, 0, 1, 0]

    def test_holiday_workday(self):
        s = pd.Series(["Holiday", "Workday", "Holiday"])
        result = boolish_to_int(s)
        assert result.tolist() == [1, 0, 1]


class TestBuildTemporalFeatures:
    def test_basic_features(self):
        idx = pd.date_range("2021-06-15", periods=48, freq="h", tz="Europe/Rome")
        calendar = pd.DataFrame(index=idx)
        calendar["CEST"] = True
        calendar["Holiday"] = False

        result = build_temporal_features(calendar, include_time_idx=True)

        assert "hour" in result.columns
        assert "time_zone_standard" in result.columns
        assert "weekday" in result.columns
        assert "holiday" in result.columns
        assert "day_of_week" in result.columns
        assert "time_idx" in result.columns

        # June 15, 2021 is a Tuesday → day_of_week = 1, weekday = 1.
        assert result["day_of_week"].iloc[0] == 1
        assert result["weekday"].iloc[0] == 1
        # CEST should be 1 (True → int).
        assert result["time_zone_standard"].iloc[0] == 1
        # hour should cycle 0-23.
        assert result["hour"].iloc[0] == 0
        assert result["hour"].iloc[23] == 23

    def test_without_time_idx(self):
        idx = pd.date_range("2021-06-15", periods=24, freq="h")
        calendar = pd.DataFrame(index=idx)
        result = build_temporal_features(calendar, include_time_idx=False)
        assert "time_idx" not in result.columns
        assert result.shape[1] == 5
