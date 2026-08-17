"""Deterministic Pearson graph for BWDF multi-DMA forecasting.

The graph is built exactly once from the continuous, cleaned official
training period. It is shared by the 24 h and 168 h forecasting tasks.

Definition
----------
``R`` is the Pearson correlation matrix. The static adjacency is

    A_ij = R_ij  if i != j and R_ij > 0
           0     otherwise

No empirical correlation threshold, top-K pruning, absolute value, or
self-loop is used. The DCRNN support is the row-normalised random walk
matrix ``P = D^{-1} A``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPSILON: float = 1.0e-12
GRAPH_METHOD = "pearson_full_positive"
NEGATIVE_POLICY = "clip_zero"


def _check_hourly_frame(
    demand_train: pd.DataFrame,
    dma_columns: list[str],
) -> pd.DataFrame:
    """Validate and return ordered float64 training demand."""
    if not isinstance(demand_train, pd.DataFrame):
        raise TypeError("demand_train must be a pandas DataFrame.")
    if not isinstance(demand_train.index, pd.DatetimeIndex):
        raise TypeError("demand_train must use a DatetimeIndex.")
    if demand_train.index.tz is None:
        raise ValueError("demand_train index must be timezone-aware.")
    if not demand_train.index.is_monotonic_increasing:
        raise ValueError("demand_train index must be sorted.")
    if not demand_train.index.is_unique:
        raise ValueError("demand_train index contains duplicate timestamps.")
    if len(dma_columns) != 10 or len(set(dma_columns)) != 10:
        raise ValueError(
            f"Expected 10 unique DMA columns, got {len(dma_columns)}: "
            f"{dma_columns}"
        )
    missing = [column for column in dma_columns if column not in demand_train]
    if missing:
        raise ValueError(f"Missing DMA columns: {missing}")
    if len(demand_train) < 2:
        raise ValueError(
            f"Need at least 2 training rows, got {len(demand_train)}."
        )

    ordered = demand_train.loc[:, dma_columns].astype(np.float64)
    values = ordered.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"Training demand contains {bad} NaN/Inf value(s).")

    expected_index = pd.date_range(
        ordered.index[0],
        ordered.index[-1],
        freq="h",
        tz=ordered.index.tz,
    )
    if not ordered.index.equals(expected_index):
        raise ValueError(
            "Training demand must be a continuous hourly series with no gaps."
        )

    variance = values.var(axis=0)
    constant = [
        dma_columns[i]
        for i, value in enumerate(variance)
        if value <= EPSILON
    ]
    if constant:
        raise ValueError(f"Constant/near-constant DMA series: {constant}")
    return ordered


def _random_walk_normalise(adjacency: np.ndarray) -> np.ndarray:
    """Return the row random-walk matrix ``P = D^{-1} A``."""
    matrix = np.asarray(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square 2-D matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("adjacency contains NaN/Inf.")
    if np.any(matrix < -EPSILON):
        raise ValueError("random-walk adjacency must be non-negative.")

    degree = matrix.sum(axis=1)
    random_walk = np.zeros_like(matrix, dtype=np.float64)
    nonzero = degree > EPSILON
    random_walk[nonzero] = (
        matrix[nonzero] / degree[nonzero, np.newaxis]
    )
    return random_walk


def build_pearson_graph(
    demand_train: pd.DataFrame,
    *,
    dma_columns: list[str] | None = None,
    node_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build the fixed positive Pearson graph from continuous training data."""
    if not isinstance(demand_train, pd.DataFrame):
        raise TypeError("demand_train must be a pandas DataFrame.")

    columns = (
        list(dma_columns)
        if dma_columns is not None
        else list(demand_train.columns)
    )
    names = (
        list(node_names)
        if node_names is not None
        else list(columns)
    )
    if len(names) != len(columns) or len(set(names)) != len(names):
        raise ValueError("node_names must be unique and match dma_columns.")

    ordered = _check_hourly_frame(demand_train, columns)
    values = ordered.to_numpy(dtype=np.float64)

    static_corr = np.corrcoef(values, rowvar=False).astype(np.float64)
    static_corr = 0.5 * (static_corr + static_corr.T)
    np.fill_diagonal(static_corr, 1.0)
    if not np.isfinite(static_corr).all():
        raise ValueError("Pearson correlation contains NaN/Inf.")

    static_adj = np.maximum(static_corr, 0.0)
    np.fill_diagonal(static_adj, 0.0)
    static_adj = 0.5 * (static_adj + static_adj.T)
    random_walk = _random_walk_normalise(static_adj)

    demand_sha256 = hashlib.sha256(
        np.ascontiguousarray(values).tobytes()
    ).hexdigest()

    return {
        "node_names": names,
        "dma_columns": columns,
        "static_corr": static_corr,
        "static_adj": static_adj,
        "random_walk": random_walk,
        "fit_start": str(ordered.index.min()),
        "fit_end": str(ordered.index.max()),
        "fit_rows": int(len(ordered)),
        "graph_method": GRAPH_METHOD,
        "corr_threshold": None,
        "negative_policy": NEGATIVE_POLICY,
        "self_loop_in_adjacency": False,
        "static": True,
        "normalization": "random_walk",
        "demand_sha256": demand_sha256,
    }


def _record(
    checks: list[dict[str, Any]],
    check: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "check": check,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def validate_graph(
    graph: dict[str, Any],
    *,
    expected_nodes: int = 10,
    expected_fit_rows: int | None = None,
    expected_node_names: list[str] | None = None,
    expected_dma_columns: list[str] | None = None,
    atol: float = 1.0e-10,
) -> list[dict[str, Any]]:
    """Validate the mathematical and data-contract properties of a graph."""
    checks: list[dict[str, Any]] = []
    corr = np.asarray(graph["static_corr"], dtype=np.float64)
    adj = np.asarray(graph["static_adj"], dtype=np.float64)
    walk = np.asarray(graph["random_walk"], dtype=np.float64)
    names = [str(value) for value in graph["node_names"]]
    columns = [str(value) for value in graph["dma_columns"]]
    shape = (expected_nodes, expected_nodes)

    for name, matrix in [
        ("static_corr", corr),
        ("static_adj", adj),
        ("random_walk", walk),
    ]:
        _record(checks, f"{name}_shape", matrix.shape == shape, matrix.shape, shape)
        finite = bool(np.isfinite(matrix).all())
        _record(checks, f"{name}_finite", finite, finite, True)

    _record(
        checks,
        "node_names_length",
        len(names) == expected_nodes,
        len(names),
        expected_nodes,
    )
    _record(
        checks,
        "dma_columns_length",
        len(columns) == expected_nodes,
        len(columns),
        expected_nodes,
    )
    _record(
        checks,
        "node_names_unique",
        len(set(names)) == len(names),
        len(set(names)),
        len(names),
    )
    _record(
        checks,
        "dma_columns_unique",
        len(set(columns)) == len(columns),
        len(set(columns)),
        len(columns),
    )
    if expected_node_names is not None:
        _record(
            checks,
            "node_names_order",
            names == list(expected_node_names),
            names,
            list(expected_node_names),
        )
    if expected_dma_columns is not None:
        _record(
            checks,
            "dma_columns_order",
            columns == list(expected_dma_columns),
            columns,
            list(expected_dma_columns),
        )
    if expected_fit_rows is not None:
        _record(
            checks,
            "fit_rows",
            int(graph["fit_rows"]) == expected_fit_rows,
            int(graph["fit_rows"]),
            expected_fit_rows,
        )

    corr_symmetric = float(np.max(np.abs(corr - corr.T)))
    _record(
        checks,
        "static_corr_symmetric",
        corr_symmetric <= atol,
        corr_symmetric,
        f"<={atol}",
    )
    corr_diag = float(np.max(np.abs(np.diag(corr) - 1.0)))
    _record(
        checks,
        "static_corr_diagonal_one",
        corr_diag <= atol,
        corr_diag,
        f"<={atol}",
    )
    corr_range_ok = bool((corr >= -1.0 - atol).all() and (corr <= 1.0 + atol).all())
    _record(checks, "static_corr_range", corr_range_ok, [float(corr.min()), float(corr.max())], "[-1,1]")

    adj_symmetric = float(np.max(np.abs(adj - adj.T)))
    _record(
        checks,
        "static_adj_symmetric",
        adj_symmetric <= atol,
        adj_symmetric,
        f"<={atol}",
    )
    adj_diag = float(np.max(np.abs(np.diag(adj))))
    _record(checks, "static_adj_diagonal_zero", adj_diag <= atol, adj_diag, f"<={atol}")
    adj_min = float(adj.min())
    _record(checks, "static_adj_non_negative", adj_min >= -atol, adj_min, ">=0")

    expected_adj = np.maximum(corr, 0.0)
    np.fill_diagonal(expected_adj, 0.0)
    adj_rule_error = float(np.max(np.abs(adj - expected_adj)))
    _record(
        checks,
        "static_adj_matches_positive_pearson",
        adj_rule_error <= atol,
        adj_rule_error,
        f"<={atol}",
    )

    degree = adj.sum(axis=1)
    _record(
        checks,
        "no_zero_degree_nodes",
        bool((degree > EPSILON).all()),
        degree.tolist(),
        "all > 0",
    )
    row_sums = walk.sum(axis=1)
    row_error = float(np.max(np.abs(row_sums - 1.0)))
    _record(
        checks,
        "random_walk_rows_sum_one",
        row_error <= atol,
        row_error,
        f"<={atol}",
    )
    expected_walk = _random_walk_normalise(adj)
    walk_error = float(np.max(np.abs(walk - expected_walk)))
    _record(
        checks,
        "random_walk_matches_adjacency",
        walk_error <= atol,
        walk_error,
        f"<={atol}",
    )

    metadata_expectations = {
        "graph_method": GRAPH_METHOD,
        "corr_threshold": None,
        "negative_policy": NEGATIVE_POLICY,
        "self_loop_in_adjacency": False,
        "static": True,
        "normalization": "random_walk",
    }
    for key, expected in metadata_expectations.items():
        observed = graph.get(key)
        _record(checks, key, observed == expected, observed, expected)
    return checks


def compute_graph_metrics(graph: dict[str, Any]) -> dict[str, Any]:
    """Compute descriptive graph diagnostics without tuning the graph."""
    adj = np.asarray(graph["static_adj"], dtype=np.float64)
    walk = np.asarray(graph["random_walk"], dtype=np.float64)
    names = [str(value) for value in graph["node_names"]]
    n = len(names)
    upper = np.triu_indices(n, k=1)
    all_weights = adj[upper]
    positive_weights = all_weights[all_weights > EPSILON]
    weighted_degree = adj.sum(axis=1)
    degree = (adj > EPSILON).sum(axis=1)

    entropy = np.zeros(n, dtype=np.float64)
    for i in range(n):
        probabilities = walk[i]
        positive = probabilities[probabilities > EPSILON]
        entropy[i] = -float(np.sum(positive * np.log(positive)))
    effective_neighbors = np.exp(entropy)

    p2 = walk @ walk
    denominator = float(np.linalg.norm(walk, ord="fro"))
    delta2 = (
        float(np.linalg.norm(p2 - walk, ord="fro")) / denominator
        if denominator > EPSILON
        else float("nan")
    )

    def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
        if values.size == 0:
            return {
                "count": 0,
                "min": None,
                "q1": None,
                "median": None,
                "mean": None,
                "q3": None,
                "max": None,
                "std": None,
            }
        return {
            "count": int(values.size),
            "min": float(np.min(values)),
            "q1": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "q3": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        }

    node_metrics = []
    for i, name in enumerate(names):
        node_metrics.append(
            {
                "node_name": name,
                "dma_column": str(graph["dma_columns"][i]),
                "degree": int(degree[i]),
                "weighted_degree": float(weighted_degree[i]),
                "propagation_entropy": float(entropy[i]),
                "effective_neighbors": float(effective_neighbors[i]),
            }
        )

    return {
        "num_nodes": n,
        "possible_undirected_edges": int(n * (n - 1) // 2),
        "positive_undirected_edges": int(positive_weights.size),
        "density": float(positive_weights.size / max(n * (n - 1) // 2, 1)),
        "all_offdiagonal_adjacency_weights": _distribution(all_weights),
        "positive_edge_weights": _distribution(positive_weights),
        "delta2": delta2,
        "node_metrics": node_metrics,
    }


def compute_segment_stability(
    demand_train: pd.DataFrame,
    *,
    dma_columns: list[str],
    n_segments: int = 4,
) -> dict[str, Any]:
    """Compare upper-triangle Pearson weights across time segments."""
    ordered = _check_hourly_frame(demand_train, dma_columns)
    if n_segments < 2 or n_segments > len(ordered):
        raise ValueError("n_segments must be between 2 and the row count.")
    indices = np.array_split(np.arange(len(ordered)), n_segments)
    vectors: list[np.ndarray] = []
    ranges: list[dict[str, Any]] = []
    upper = np.triu_indices(len(dma_columns), k=1)
    for segment_id, idx in enumerate(indices):
        segment = ordered.iloc[idx]
        corr = np.corrcoef(
            segment.to_numpy(dtype=np.float64),
            rowvar=False,
        )
        vectors.append(corr[upper])
        ranges.append(
            {
                "segment": segment_id + 1,
                "start": str(segment.index.min()),
                "end": str(segment.index.max()),
                "rows": int(len(segment)),
            }
        )
    similarities = np.corrcoef(np.stack(vectors, axis=0))
    return {
        "n_segments": n_segments,
        "segment_ranges": ranges,
        "similarity_matrix": similarities,
    }


def moving_block_bootstrap(
    demand_train: pd.DataFrame,
    *,
    dma_columns: list[str],
    block_hours: int = 168,
    n_resamples: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Return synchronized moving-block bootstrap intervals for all edges."""
    ordered = _check_hourly_frame(demand_train, dma_columns)
    values = ordered.to_numpy(dtype=np.float64)
    n_rows, n_nodes = values.shape
    if block_hours < 2 or block_hours > n_rows:
        raise ValueError("block_hours must be in [2, number of rows].")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2.")

    blocks_needed = math.ceil(n_rows / block_hours)
    max_start = n_rows - block_hours
    rng = np.random.default_rng(seed)
    upper = np.triu_indices(n_nodes, k=1)
    boot = np.empty((n_resamples, len(upper[0])), dtype=np.float64)

    offsets = np.arange(block_hours, dtype=np.int64)
    for sample_id in range(n_resamples):
        starts = rng.integers(
            0,
            max_start + 1,
            size=blocks_needed,
        )
        indices = (starts[:, None] + offsets[None, :]).reshape(-1)[:n_rows]
        corr = np.corrcoef(values[indices], rowvar=False)
        boot[sample_id] = corr[upper]

    full_corr = np.corrcoef(values, rowvar=False)[upper]
    rows: list[dict[str, Any]] = []
    for edge_id, (i, j) in enumerate(zip(*upper)):
        samples = boot[:, edge_id]
        rows.append(
            {
                "node_i": dma_columns[i],
                "node_j": dma_columns[j],
                "full_pearson": float(full_corr[edge_id]),
                "bootstrap_median": float(np.median(samples)),
                "ci_2_5": float(np.quantile(samples, 0.025)),
                "ci_97_5": float(np.quantile(samples, 0.975)),
                "ci_width": float(
                    np.quantile(samples, 0.975)
                    - np.quantile(samples, 0.025)
                ),
                "positive_rate": float(np.mean(samples > 0.0)),
                "n_resamples": int(n_resamples),
                "block_hours": int(block_hours),
                "seed": int(seed),
            }
        )
    return pd.DataFrame(rows)


def save_graph(
    graph: dict[str, Any],
    artifact_path: Path,
) -> Path:
    """Save the model-facing graph artifact without pickle objects."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    threshold = (
        np.nan
        if graph.get("corr_threshold") is None
        else float(graph["corr_threshold"])
    )
    np.savez_compressed(
        artifact_path,
        node_names=np.asarray(graph["node_names"], dtype=str),
        dma_columns=np.asarray(graph["dma_columns"], dtype=str),
        static_corr=np.asarray(graph["static_corr"], dtype=np.float64),
        static_adj=np.asarray(graph["static_adj"], dtype=np.float64),
        random_walk=np.asarray(graph["random_walk"], dtype=np.float64),
        fit_start=np.asarray(str(graph["fit_start"])),
        fit_end=np.asarray(str(graph["fit_end"])),
        fit_rows=np.asarray(int(graph["fit_rows"]), dtype=np.int64),
        graph_method=np.asarray(str(graph["graph_method"])),
        corr_threshold=np.asarray(threshold, dtype=np.float64),
        negative_policy=np.asarray(str(graph["negative_policy"])),
        self_loop_in_adjacency=np.asarray(
            bool(graph["self_loop_in_adjacency"]),
            dtype=np.bool_,
        ),
        static=np.asarray(bool(graph["static"]), dtype=np.bool_),
        normalization=np.asarray(str(graph["normalization"])),
        demand_sha256=np.asarray(str(graph["demand_sha256"])),
    )
    return artifact_path


def load_graph(artifact_path: Path) -> dict[str, Any]:
    """Load a graph artifact using ``allow_pickle=False``."""
    with np.load(artifact_path, allow_pickle=False) as data:
        threshold_value = float(data["corr_threshold"].item())
        return {
            "node_names": data["node_names"].astype(str).tolist(),
            "dma_columns": data["dma_columns"].astype(str).tolist(),
            "static_corr": data["static_corr"].astype(np.float64),
            "static_adj": data["static_adj"].astype(np.float64),
            "random_walk": data["random_walk"].astype(np.float64),
            "fit_start": str(data["fit_start"].item()),
            "fit_end": str(data["fit_end"].item()),
            "fit_rows": int(data["fit_rows"].item()),
            "graph_method": str(data["graph_method"].item()),
            "corr_threshold": (
                None if np.isnan(threshold_value) else threshold_value
            ),
            "negative_policy": str(data["negative_policy"].item()),
            "self_loop_in_adjacency": bool(
                data["self_loop_in_adjacency"].item()
            ),
            "static": bool(data["static"].item()),
            "normalization": str(data["normalization"].item()),
            "demand_sha256": str(data["demand_sha256"].item()),
        }


def save_matrix_csvs(
    graph: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Save human-readable graph matrices and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [str(value) for value in graph["node_names"]]
    paths: dict[str, Path] = {}
    for filename, key in [
        ("static_corr.csv", "static_corr"),
        ("static_adj.csv", "static_adj"),
        ("random_walk.csv", "random_walk"),
    ]:
        path = output_dir / filename
        pd.DataFrame(
            graph[key],
            index=columns,
            columns=columns,
        ).to_csv(path, float_format="%.12f")
        paths[key] = path

    metadata = {
        key: value
        for key, value in graph.items()
        if key not in {"static_corr", "static_adj", "random_walk"}
    }
    metadata_path = output_dir / "graph_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    paths["metadata"] = metadata_path
    return paths


def plot_graph_diagnostics(
    graph: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Save correlation, adjacency, network, and weighted-degree plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
    except ImportError as exc:
        raise RuntimeError(
            "Graph plots require matplotlib. Install it with "
            "`python -m pip install matplotlib`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    names = [str(value) for value in graph["node_names"]]
    corr = np.asarray(graph["static_corr"], dtype=np.float64)
    adj = np.asarray(graph["static_adj"], dtype=np.float64)
    paths: dict[str, Path] = {}

    def _heatmap(
        matrix: np.ndarray,
        title: str,
        filename: str,
        *,
        cmap: str,
        norm: Any = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(
            matrix,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        ax.set_title(title)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths[filename] = path

    _heatmap(
        corr,
        "Training-period Pearson correlation",
        "static_corr_heatmap.png",
        cmap="RdBu_r",
        norm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
    )
    _heatmap(
        adj,
        "Positive Pearson static adjacency",
        "static_adj_heatmap.png",
        cmap="Blues",
        vmin=0.0,
        vmax=max(float(adj.max()), EPSILON),
    )

    angles = np.linspace(0.0, 2.0 * np.pi, len(names), endpoint=False)
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    fig, ax = plt.subplots(figsize=(8, 8))
    max_weight = max(float(adj.max()), EPSILON)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            weight = float(adj[i, j])
            if weight <= EPSILON:
                continue
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color="steelblue",
                linewidth=0.5 + 4.0 * weight / max_weight,
                alpha=0.15 + 0.75 * weight / max_weight,
                zorder=1,
            )
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=500,
        color="white",
        edgecolor="navy",
        linewidth=1.5,
        zorder=3,
    )
    for name, (x, y) in zip(names, positions):
        ax.text(x, y, name, ha="center", va="center", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("BWDF positive Pearson weighted graph")
    network_path = output_dir / "weighted_network.png"
    fig.tight_layout()
    fig.savefig(network_path, dpi=180)
    plt.close(fig)
    paths["weighted_network.png"] = network_path

    node_metrics = pd.DataFrame(metrics["node_metrics"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        node_metrics["node_name"],
        node_metrics["weighted_degree"],
        color="steelblue",
    )
    ax.set_xlabel("DMA")
    ax.set_ylabel("Weighted degree")
    ax.set_title("DMA weighted degree")
    fig.tight_layout()
    degree_path = output_dir / "weighted_degree.png"
    fig.savefig(degree_path, dpi=180)
    plt.close(fig)
    paths["weighted_degree.png"] = degree_path
    return paths


def build_graph_from_artifacts(
    data_dir: Path,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    *,
    dma_columns: list[str],
    node_names: list[str],
) -> dict[str, Any]:
    """Build Pearson graph from the continuous cleaned demand artifact."""
    demand = pd.read_parquet(data_dir / "demand_hourly.parquet")
    demand_train = demand.loc[
        (demand.index >= train_start)
        & (demand.index <= train_end),
        dma_columns,
    ]
    return build_pearson_graph(
        demand_train,
        dma_columns=dma_columns,
        node_names=node_names,
    )


def build_spearman_graph(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Removed API guard: Spearman is not part of the final BWDF graph."""
    raise RuntimeError(
        "build_spearman_graph() has been removed. Build the shared graph with "
        "build_pearson_graph() or load bwdf_pearson_static_graph.npz."
    )
