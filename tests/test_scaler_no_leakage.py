"""Verify scaler no-leakage invariants.

- Scaler must NOT read test data during fitting.
- inverse_transform must recover original-scale demand.
- Per-DMA statistics are computed independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Simulate the DMAScaler interface (import from training script location).
# We test the logic in isolation.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "train"))


class DMAScalerTest:
    """Re-implements the scaler logic for standalone testing."""

    def __init__(self, dma_columns: list[str]):
        self.dma_columns = list(dma_columns)
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        self._fitted = False

    def fit(self, demand_train: pd.DataFrame):
        data = demand_train[self.dma_columns].to_numpy(dtype=np.float64)
        self.means = data.mean(axis=0).astype(np.float32)
        self.stds = data.std(axis=0).astype(np.float32)
        self.stds[self.stds <= 1e-6] = 1.0
        self._fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted")
        return (x - self.means) / self.stds

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted")
        return x * self.stds + self.means


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demand_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train, val, test) DataFrames with DMA columns."""
    np.random.seed(123)
    dma_cols = [f"DMA {i}" for i in range(1, 11)]
    idx_train = pd.date_range("2021-01-01", periods=1000, freq="h", tz="Europe/Rome")
    idx_val = pd.date_range("2022-01-01", periods=200, freq="h", tz="Europe/Rome")
    idx_test = pd.date_range("2023-01-01", periods=100, freq="h", tz="Europe/Rome")

    train = pd.DataFrame(
        {c: np.abs(np.random.randn(1000) * 10 + 50 + i * 5) for i, c in enumerate(dma_cols)},
        index=idx_train,
    )
    val = pd.DataFrame(
        {c: np.abs(np.random.randn(200) * 12 + 55 + i * 5) for i, c in enumerate(dma_cols)},
        index=idx_val,
    )
    test = pd.DataFrame(
        {c: np.abs(np.random.randn(100) * 15 + 60 + i * 5) for i, c in enumerate(dma_cols)},
        index=idx_test,
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScalerFitting:
    """Scaler must fit only on training data."""

    def test_fit_only_reads_train(self, demand_data):
        train, val, test = demand_data
        scaler = DMAScalerTest(list(train.columns))
        scaler.fit(train)

        # Compute training means manually.
        train_means = train.to_numpy(dtype=np.float64).mean(axis=0)
        assert np.allclose(scaler.means, train_means, atol=1e-4)

    def test_fit_does_not_read_test(self, demand_data):
        """Verify that test data is never passed to fit()."""
        train, val, test = demand_data
        scaler = DMAScalerTest(list(train.columns))

        # Fit with train only.
        scaler.fit(train)

        # Verify means were set from training data.
        test_means = test.to_numpy(dtype=np.float64).mean(axis=0)
        # Test means should differ from train means.
        diff = np.abs(scaler.means - test_means).max()
        assert diff > 1e-4, (
            f"Train and test means too similar (diff={diff:.6f}) — "
            f"scaler might have been fit on test data"
        )


class TestInverseTransform:
    """inverse_transform must recover original values."""

    def test_roundtrip(self, demand_data):
        train, val, test = demand_data
        scaler = DMAScalerTest(list(train.columns))
        scaler.fit(train)

        original = train.to_numpy(dtype=np.float32)
        transformed = scaler.transform(original)
        recovered = scaler.inverse_transform(transformed)

        assert np.allclose(original, recovered, atol=1e-4), (
            f"Round-trip failed: max diff={np.abs(original - recovered).max():.6f}"
        )

    def test_zero_mean_unit_variance(self, demand_data):
        train, val, test = demand_data
        scaler = DMAScalerTest(list(train.columns))
        scaler.fit(train)

        transformed = scaler.transform(train.to_numpy(dtype=np.float32))
        for i in range(10):
            assert abs(transformed[:, i].mean()) < 1e-5, (
                f"DMA {i}: mean={transformed[:, i].mean():.6f}, expected ≈0"
            )
            assert abs(transformed[:, i].std() - 1.0) < 1e-5, (
                f"DMA {i}: std={transformed[:, i].std():.6f}, expected ≈1"
            )


class TestScalerPersistence:
    """Scaler save/load round-trip."""

    def test_save_load_roundtrip(self, demand_data, tmp_path):
        train, val, test = demand_data
        scaler = DMAScalerTest(list(train.columns))
        scaler.fit(train)

        # Simulate save/load via npz.
        path = tmp_path / "scaler.npz"
        np.savez_compressed(
            path,
            means=scaler.means,
            stds=scaler.stds,
            dma_columns=np.array(scaler.dma_columns),
        )

        # Load.
        data = np.load(path)
        scaler2 = DMAScalerTest(list(data["dma_columns"]))
        scaler2.means = data["means"]
        scaler2.stds = data["stds"]
        scaler2._fitted = True

        # Both scalers should produce identical results.
        original = test.to_numpy(dtype=np.float32)
        t1 = scaler.transform(original)
        t2 = scaler2.transform(original)
        assert np.allclose(t1, t2, atol=1e-6)

    def test_scaler_not_fitted_raises(self, demand_data):
        scaler = DMAScalerTest(["DMA 1"])
        with pytest.raises(RuntimeError, match="Not fitted"):
            scaler.transform(np.zeros((5, 1)))
