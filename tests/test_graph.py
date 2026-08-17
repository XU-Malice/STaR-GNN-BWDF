"""Tests for the fixed positive Pearson graph."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dma_wdf.data.graph import (
    _random_walk_normalise,
    build_pearson_graph,
    compute_graph_metrics,
    compute_segment_stability,
    load_graph,
    moving_block_bootstrap,
    save_graph,
    validate_graph,
)


@pytest.fixture
def demand_10dma() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 672
    t = np.arange(rows, dtype=np.float64)
    base = (
        np.sin(2.0 * np.pi * t / 24.0)
        + 0.5 * np.sin(2.0 * np.pi * t / 168.0)
    )
    data: dict[str, np.ndarray] = {}
    data["DMA 1"] = base + 0.10 * rng.normal(size=rows)
    data["DMA 2"] = 0.8 * base + 0.10 * rng.normal(size=rows)
    data["DMA 3"] = 1.2 * base + 0.15 * rng.normal(size=rows)
    data["DMA 4"] = -base + 0.10 * rng.normal(size=rows)
    data["DMA 5"] = -0.7 * base + 0.15 * rng.normal(size=rows)
    for i in range(6, 11):
        data[f"DMA {i}"] = (
            (0.15 + 0.05 * i) * base
            + rng.normal(scale=0.6, size=rows)
        )
    index = pd.date_range(
        "2021-01-01",
        periods=rows,
        freq="h",
        tz="Europe/Rome",
    )
    return pd.DataFrame(data, index=index)


def _build(frame: pd.DataFrame) -> dict[str, object]:
    return build_pearson_graph(
        frame,
        dma_columns=[f"DMA {i}" for i in range(1, 11)],
        node_names=list("ABCDEFGHIJ"),
    )


def test_pearson_matches_numpy(demand_10dma: pd.DataFrame) -> None:
    graph = _build(demand_10dma)
    expected = np.corrcoef(
        demand_10dma.to_numpy(dtype=np.float64),
        rowvar=False,
    )
    assert np.allclose(graph["static_corr"], expected, atol=1.0e-12)


def test_shapes_and_contract(demand_10dma: pd.DataFrame) -> None:
    graph = _build(demand_10dma)
    assert graph["static_corr"].shape == (10, 10)
    assert graph["static_adj"].shape == (10, 10)
    assert graph["random_walk"].shape == (10, 10)
    assert graph["node_names"] == list("ABCDEFGHIJ")
    assert graph["dma_columns"] == [f"DMA {i}" for i in range(1, 11)]
    assert graph["fit_rows"] == 672
    assert graph["graph_method"] == "pearson_full_positive"
    assert graph["corr_threshold"] is None
    assert graph["negative_policy"] == "clip_zero"
    assert graph["self_loop_in_adjacency"] is False
    assert graph["static"] is True


def test_negative_clipped_without_absolute_value(
    demand_10dma: pd.DataFrame,
) -> None:
    graph = _build(demand_10dma)
    corr = graph["static_corr"]
    adj = graph["static_adj"]
    assert corr[0, 3] < 0.0
    assert adj[0, 3] == 0.0
    assert adj[3, 0] == 0.0


def test_no_threshold_keeps_weak_positive_edge() -> None:
    rows = 2000
    rng = np.random.default_rng(8)
    index = pd.date_range(
        "2021-01-01",
        periods=rows,
        freq="h",
        tz="Europe/Rome",
    )
    x = rng.normal(size=rows)
    data = {"DMA 1": x}
    data["DMA 2"] = 0.08 * x + rng.normal(size=rows)
    for i in range(3, 11):
        data[f"DMA {i}"] = rng.normal(size=rows)
    graph = _build(pd.DataFrame(data, index=index))
    correlation = float(graph["static_corr"][0, 1])
    assert 0.0 < correlation < 0.5
    assert graph["static_adj"][0, 1] == pytest.approx(correlation)


def test_adjacency_and_random_walk(demand_10dma: pd.DataFrame) -> None:
    graph = _build(demand_10dma)
    corr = graph["static_corr"]
    adj = graph["static_adj"]
    walk = graph["random_walk"]
    expected_adj = np.maximum(corr, 0.0)
    np.fill_diagonal(expected_adj, 0.0)
    assert np.allclose(adj, expected_adj)
    assert np.allclose(adj, adj.T)
    assert np.allclose(np.diag(adj), 0.0)
    assert np.all(adj >= 0.0)
    assert np.allclose(walk.sum(axis=1), 1.0)


def test_zero_degree_random_walk_is_safe() -> None:
    walk = _random_walk_normalise(np.zeros((10, 10), dtype=np.float64))
    assert np.isfinite(walk).all()
    assert np.allclose(walk, 0.0)


def test_validation_passes(demand_10dma: pd.DataFrame) -> None:
    graph = _build(demand_10dma)
    checks = validate_graph(
        graph,
        expected_fit_rows=672,
        expected_node_names=list("ABCDEFGHIJ"),
        expected_dma_columns=[f"DMA {i}" for i in range(1, 11)],
    )
    assert all(row["passed"] for row in checks), [
        row for row in checks if not row["passed"]
    ]


def test_save_load_roundtrip(
    demand_10dma: pd.DataFrame,
    tmp_path,
) -> None:
    graph = _build(demand_10dma)
    path = tmp_path / "graph.npz"
    save_graph(graph, path)
    loaded = load_graph(path)
    for key in ["static_corr", "static_adj", "random_walk"]:
        assert np.array_equal(graph[key], loaded[key])
    for key in [
        "node_names",
        "dma_columns",
        "fit_start",
        "fit_end",
        "fit_rows",
        "graph_method",
        "corr_threshold",
        "negative_policy",
        "self_loop_in_adjacency",
        "static",
        "normalization",
        "demand_sha256",
    ]:
        assert graph[key] == loaded[key]


def test_test_period_change_cannot_change_train_graph(
    demand_10dma: pd.DataFrame,
) -> None:
    train = demand_10dma.iloc[:500]
    test = demand_10dma.iloc[500:].copy()
    graph_a = _build(train)
    test.iloc[:, :] = test.to_numpy() * 1000.0
    graph_b = _build(train)
    assert np.array_equal(graph_a["static_corr"], graph_b["static_corr"])
    assert graph_a["demand_sha256"] == graph_b["demand_sha256"]


def test_overlapping_windows_are_not_graph_input(
    demand_10dma: pd.DataFrame,
) -> None:
    array = demand_10dma.to_numpy()
    overlapping = np.stack([array[:336], array[168:504]], axis=0)
    with pytest.raises(TypeError, match="DataFrame"):
        build_pearson_graph(overlapping)  # type: ignore[arg-type]


def test_time_gaps_rejected(demand_10dma: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="continuous hourly"):
        _build(demand_10dma.drop(demand_10dma.index[100]))


def test_metrics_and_delta2(demand_10dma: pd.DataFrame) -> None:
    metrics = compute_graph_metrics(_build(demand_10dma))
    assert metrics["possible_undirected_edges"] == 45
    assert 0 <= metrics["positive_undirected_edges"] <= 45
    assert 0.0 <= metrics["density"] <= 1.0
    assert np.isfinite(metrics["delta2"])
    assert len(metrics["node_metrics"]) == 10


def test_segment_stability_shape(demand_10dma: pd.DataFrame) -> None:
    result = compute_segment_stability(
        demand_10dma,
        dma_columns=list(demand_10dma.columns),
        n_segments=4,
    )
    assert result["similarity_matrix"].shape == (4, 4)
    assert len(result["segment_ranges"]) == 4


def test_bootstrap_is_reproducible(demand_10dma: pd.DataFrame) -> None:
    kwargs = {
        "dma_columns": list(demand_10dma.columns),
        "block_hours": 168,
        "n_resamples": 10,
        "seed": 7,
    }
    first = moving_block_bootstrap(demand_10dma, **kwargs)
    second = moving_block_bootstrap(demand_10dma, **kwargs)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 45
