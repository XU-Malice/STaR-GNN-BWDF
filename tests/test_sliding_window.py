"""Tests for sliding window construction."""

import numpy as np
import pandas as pd
import pytest

from dma_wdf.data.sliding_window import (
    build_sample_index,
    daily_starts,
    make_eval_index,
    slice_hours,
    target_24h,
    target_168h,
)


class TestDailyStarts:
    def test_basic_range(self):
        start = pd.Timestamp("2021-01-01 00:00:00", tz="Europe/Rome")
        end = pd.Timestamp("2021-01-03 00:00:00", tz="Europe/Rome")
        result = daily_starts(start, end, tz="Europe/Rome")
        assert len(result) == 3
        assert result[0] == start
        assert result[-1] == end

    def test_start_after_midnight(self):
        start = pd.Timestamp("2021-01-01 12:00:00", tz="Europe/Rome")
        end = pd.Timestamp("2021-01-02 00:00:00", tz="Europe/Rome")
        result = daily_starts(start, end, tz="Europe/Rome")
        # Should advance to next midnight (Jan 2).
        assert result[0] == pd.Timestamp("2021-01-02 00:00:00", tz="Europe/Rome")


class TestSliceHours:
    def test_extract_window(self):
        idx = pd.date_range("2021-01-01", periods=48, freq="h")
        df = pd.DataFrame({"val": np.arange(48, dtype=float)}, index=idx)
        result = slice_hours(df, idx[0], 24)
        assert result.shape == (24, 1)
        assert result[0, 0] == 0
        assert result[-1, 0] == 23

    def test_insufficient_rows(self):
        idx = pd.date_range("2021-01-01", periods=10, freq="h")
        df = pd.DataFrame({"val": np.arange(10, dtype=float)}, index=idx)
        with pytest.raises(ValueError):
            slice_hours(df, idx[0], 24)


class TestTargetFunctions:
    def test_target_24h_shape(self):
        idx = pd.date_range("2021-01-01", periods=48, freq="h")
        demand = pd.DataFrame(
            {f"DMA {i}": np.random.randn(48) for i in range(1, 4)},
            index=idx,
        )
        result = target_24h(demand, idx[0])
        assert result.shape == (24, 3)

    def test_target_168h_shape(self):
        idx = pd.date_range("2021-01-01", periods=200, freq="h")
        demand = pd.DataFrame(
            {f"DMA {i}": np.random.randn(200) for i in range(1, 4)},
            index=idx,
        )
        result = target_168h(demand, idx[0])
        assert result.shape == (168, 3)


class TestBuildSampleIndex:
    def test_train_test_split(self):
        tz = "Europe/Rome"
        paper_start = pd.Timestamp("2021-01-01 00:00:00").tz_localize(tz)
        paper_end = pd.Timestamp("2021-02-28 23:00:00").tz_localize(tz)
        train_end = pd.Timestamp("2021-01-31 23:00:00").tz_localize(tz)
        test_start = pd.Timestamp("2021-02-01 00:00:00").tz_localize(tz)
        test_end = paper_end

        si = build_sample_index(
            paper_start=paper_start,
            paper_end=paper_end,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            horizon_hours=24,
            stride_hours=24,
            max_history_weeks=4,
            tz=tz,
        )
        assert "train" in si["split"].values
        assert "test" in si["split"].values
        # All samples should have valid splits.
        assert si["split"].isin(["train", "test", "boundary_or_unused"]).all()


class TestMakeEvalIndex:
    def test_46_sequences_for_paper_protocol(self):
        """With 80 test days, 28 history days, 7 horizon days: 80-28-7+1=46."""
        tz = "Europe/Rome"
        test_start = pd.Timestamp("2022-12-16 00:00:00").tz_localize(tz)
        test_end = pd.Timestamp("2023-03-05 23:00:00").tz_localize(tz)
        ei = make_eval_index(
            test_start=test_start,
            test_end=test_end,
            max_history_weeks=4,
            horizon_hours=168,
            tz=tz,
        )
        assert len(ei) == 46
        assert ei["sequence_id"].iloc[0] == 0
        assert ei["sequence_id"].iloc[-1] == 45
