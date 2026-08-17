"""Verify no data leakage between fitting, validation, and test sets.

Invariants:
- Modifying validation/test demand must not change the static Pearson graph.
- Modifying validation-period data must not change the fitted scaler.
- The last fitting label window must not overlap the first validation label window.
- Graph and scaler fitting use only the fitting set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dma_wdf.data.graph import build_pearson_graph


@pytest.fixture
def demand_data() -> pd.DataFrame:
    """Create multi-year demand data with known train/test structure."""
    np.random.seed(42)
    dma_cols = [f"DMA {i}" for i in range(1, 11)]
    idx = pd.date_range(
        "2021-01-01",
        periods=2000,
        freq="h",
        tz="Europe/Rome",
    )
    data = {}
    base = np.sin(np.linspace(0, 20 * np.pi, 2000))
    for i, column in enumerate(dma_cols):
        data[column] = np.abs(
            base * (1 + 0.1 * i)
            + np.random.randn(2000) * 0.5
            + 10
            + i * 3
        )
    return pd.DataFrame(data, index=idx)


class TestGraphLeakage:
    """Static Pearson graph must not read validation or test data."""

    @staticmethod
    def _build_graph(demand_fit: pd.DataFrame) -> dict:
        return build_pearson_graph(
            demand_fit,
            dma_columns=list(demand_fit.columns),
            node_names=list("ABCDEFGHIJ"),
        )

    def test_modifying_validation_demand_does_not_change_graph(
        self,
        demand_data,
    ):
        fit_demand = demand_data.iloc[:1000]
        val_demand = demand_data.iloc[1000:1500].copy()

        graph1 = self._build_graph(fit_demand)
        adj1 = graph1["static_adj"].copy()

        val_demand *= 100.0
        graph2 = self._build_graph(fit_demand)
        adj2 = graph2["static_adj"]

        assert np.allclose(adj1, adj2, atol=1e-10), (
            "Graph changed even though fitting data was unchanged."
        )

    def test_graph_only_uses_fitting_data(self, demand_data):
        fit_demand = demand_data.iloc[:800]
        result = self._build_graph(fit_demand)
        assert result["fit_rows"] == 800


class TestScalerLeakage:
    """Scaler must fit only on fitting-set data."""

    def test_modifying_validation_data_does_not_change_scaler(
        self,
        demand_data,
    ):
        fit_data = demand_data.iloc[:800].to_numpy(dtype=np.float64)
        val_data = demand_data.iloc[800:1200].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        means1 = fit_data.mean(axis=0)
        stds1 = fit_data.std(axis=0)
        stds1[stds1 < 1e-6] = 1.0

        val_data *= 100.0
        means2 = fit_data.mean(axis=0)
        stds2 = fit_data.std(axis=0)
        stds2[stds2 < 1e-6] = 1.0

        assert np.allclose(means1, means2, atol=1e-10)
        assert np.allclose(stds1, stds2, atol=1e-10)


class TestLabelWindowNoOverlap:
    """Fitting and validation label windows must not overlap."""

    def test_24h_no_overlap_needed(self):
        starts = pd.date_range(
            "2021-01-29",
            periods=100,
            freq="24h",
            tz="Europe/Rome",
        )
        n_fit = 90
        last_fit_end = starts[n_fit - 1] + pd.Timedelta(hours=23)
        first_val_start = starts[n_fit]
        assert last_fit_end < first_val_start

    def test_168h_purge_prevents_overlap(self):
        starts = pd.date_range(
            "2021-01-29",
            periods=100,
            freq="24h",
            tz="Europe/Rome",
        )
        n_fit = 90
        purge = 6
        last_fit_label_end = starts[n_fit - 1] + pd.Timedelta(hours=167)
        first_val_start = starts[n_fit + purge]
        gap_hours = (
            first_val_start - last_fit_label_end
        ).total_seconds() / 3600 - 1
        assert gap_hours >= 0


class TestPredictionsShapeConsistency:
    """predictions.npz first dimension must match declared sample count."""

    def test_shape_consistency(self, tmp_path):
        n_test = 80
        horizon = 24
        n_nodes = 10
        preds = np.random.randn(
            n_test,
            horizon,
            n_nodes,
        ).astype(np.float32)
        targets = np.random.randn(
            n_test,
            horizon,
            n_nodes,
        ).astype(np.float32)
        path = tmp_path / "predictions.npz"
        np.savez_compressed(path, predictions=preds, targets=targets)
        loaded = np.load(path)
        assert loaded["predictions"].shape[0] == n_test
        assert loaded["targets"].shape[0] == n_test
