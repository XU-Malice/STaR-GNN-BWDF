"""Tests for forecast evaluation metrics."""

import numpy as np
import pytest

from dma_wdf.data.metrics import compute_metrics, mae, mape, nse, rmse


class TestMetrics:
    def test_mae_perfect(self):
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        assert mae(y, y) == 0.0

    def test_mae_positive(self):
        assert mae([1, 2], [0, 0]) == 1.5

    def test_rmse_perfect(self):
        y = np.array([1, 2, 3], dtype=float)
        assert rmse(y, y) == 0.0

    def test_rmse_value(self):
        # For [1,2] vs [0,0]: sqrt(mean([1^2, 2^2])) = sqrt(2.5) ≈ 1.5811
        result = rmse([1, 2], [0, 0])
        assert abs(result - np.sqrt(2.5)) < 1e-6

    def test_nse_perfect(self):
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        assert nse(y, y) == 1.0

    def test_nse_mean_prediction(self):
        # Predicting the mean should give NSE ≈ 0.
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        mean_pred = np.full_like(y, y.mean())
        result = nse(y, mean_pred)
        assert abs(result) < 1e-10

    def test_mape_perfect(self):
        y = np.array([1, 2, 3], dtype=float)
        assert mape(y, y) == 0.0

    def test_mape_as_fraction(self):
        # MAPE should be returned as fraction, not percent.
        y_true = np.array([10, 20], dtype=float)
        y_pred = np.array([11, 22], dtype=float)  # 10% error on both.
        result = mape(y_true, y_pred)
        assert abs(result - 0.1) < 1e-6

    def test_compute_metrics_keys(self):
        mets = compute_metrics([1, 2, 3], [1, 2, 3])
        assert set(mets.keys()) == {"MAE", "MAPE", "RMSE", "NSE"}

    def test_nan_handling(self):
        """Metrics should filter NaN values."""
        y_true = np.array([1, 2, np.nan, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        result = mae(y_true, y_pred)
        # Only 4 finite pairs: errors are 0, 0, 0, 0 → MAE = 0.
        assert result == 0.0
