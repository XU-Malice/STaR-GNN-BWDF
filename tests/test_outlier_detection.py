"""Tests for IQR-based outlier detection."""

import numpy as np
import pandas as pd
import pytest

from dma_wdf.data.outlier_detection import (
    clip_outliers,
    compute_thresholds,
    detect_outliers,
    interpolate_outliers,
    preprocess_demand,
    preprocess_series,
)


class TestComputeThresholds:
    def test_standard_tukey(self):
        """IQR 1.5 should produce standard Tukey fences."""
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        t = compute_thresholds(s, multiplier=1.5)
        assert t["q1"] == 3.25
        assert t["q3"] == 7.75
        assert t["iqr"] == 4.5
        assert t["lower"] == 3.25 - 1.5 * 4.5  # -3.5
        assert t["upper"] == 7.75 + 1.5 * 4.5  # 14.5

    def test_custom_multiplier(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        t = compute_thresholds(s, multiplier=3.0)
        assert t["lower"] == 3.25 - 3.0 * 4.5
        assert t["upper"] == 7.75 + 3.0 * 4.5


class TestDetectOutliers:
    def test_no_outliers(self):
        s = pd.Series([5, 6, 7, 8], dtype=float)
        mask = detect_outliers(s, lower=0, upper=100)
        assert not mask.any()

    def test_has_outliers(self):
        s = pd.Series([-999, 5, 6, 7, 999], dtype=float)
        mask = detect_outliers(s, lower=0, upper=100)
        assert mask.iloc[0]
        assert mask.iloc[-1]
        assert not mask.iloc[1:4].any()


class TestClipOutliers:
    def test_clip_bounds(self):
        s = pd.Series([-10, 5, 6, 7, 50], dtype=float)
        clipped, mask = clip_outliers(s, lower=0, upper=10)
        assert clipped.iloc[0] == 0
        assert clipped.iloc[-1] == 10
        assert mask.iloc[0] and mask.iloc[-1]


class TestInterpolateOutliers:
    def test_interpolate_inner_point(self):
        """Outlier in the middle should be interpolated from neighbors."""
        idx = pd.date_range("2021-01-01", periods=5, freq="h")
        s = pd.Series([5.0, 999.0, 6.0, 7.0, 8.0], index=idx)
        processed, mask = interpolate_outliers(s, lower=0, upper=10)
        # The outlier should be interpolated to ~5.5 (midpoint in time).
        assert mask.iloc[1]
        assert 5.0 < processed.iloc[1] < 6.0


class TestPreprocessDemand:
    def test_returns_profile(self):
        idx = pd.date_range("2021-01-01", periods=100, freq="h")
        demand = pd.DataFrame(
            {"DMA 1": np.sin(np.linspace(0, 4 * np.pi, 100)) + 5},
            index=idx,
        )
        processed, profile = preprocess_demand(demand, multiplier=1.5)
        assert len(profile) == 1
        assert "q1" in profile.columns
        assert "outliers" in profile.columns
        assert processed.shape == demand.shape
