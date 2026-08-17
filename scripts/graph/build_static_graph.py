#!/usr/bin/env python
"""Build the shared BWDF Pearson static graph from cleaned training demand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dma_wdf.data.graph import (
    build_pearson_graph,
    compute_graph_metrics,
    plot_graph_diagnostics,
    save_graph,
    save_matrix_csvs,
    validate_graph,
)
from dma_wdf.utils.config import parse_timestamp, read_yaml


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def build(root: Path, config_path: Path) -> dict[str, object]:
    cfg = read_yaml(config_path)
    source_cfg = cfg["source"]
    graph_cfg = cfg["graph"]
    output_cfg = cfg["output"]
    validation_cfg = cfg["validation"]

    if graph_cfg != {
        "method": "pearson_full_positive",
        "corr_threshold": None,
        "negative_policy": "clip_zero",
        "self_loop_in_adjacency": False,
        "static": True,
        "normalization": "random_walk",
    }:
        raise ValueError(
            "Graph configuration differs from the fixed BWDF Pearson scheme."
        )

    data_dir = _resolve(root, source_cfg["data_dir"])
    demand_path = data_dir / source_cfg["demand_file"]
    split_cfg = read_yaml(_resolve(root, source_cfg["split_config"]))
    features_cfg = read_yaml(_resolve(root, source_cfg["features_config"]))

    demand = pd.read_parquet(demand_path)
    tz = demand.index.tz
    train_start = parse_timestamp(split_cfg["split"]["train_start"], tz)
    train_end = parse_timestamp(
        split_cfg["split"]["train_end_inclusive"],
        tz,
    )
    dma_columns = list(split_cfg["features"]["demand"]["dma_columns"])
    dma_letters = list(split_cfg["features"]["demand"]["dma_letters"])

    mapping = features_cfg["dma_mapping"]
    mapped_columns = [mapping[letter] for letter in dma_letters]
    if mapped_columns != dma_columns:
        raise ValueError(
            "features.yaml dma_mapping order does not match paper_split.yaml."
        )

    demand_train = demand.loc[
        (demand.index >= train_start)
        & (demand.index <= train_end),
        dma_columns,
    ]
    graph = build_pearson_graph(
        demand_train,
        dma_columns=dma_columns,
        node_names=dma_letters,
    )
    checks = validate_graph(
        graph,
        expected_nodes=int(validation_cfg["expected_nodes"]),
        expected_fit_rows=int(validation_cfg["expected_fit_rows"]),
        expected_node_names=dma_letters,
        expected_dma_columns=dma_columns,
        atol=float(validation_cfg["atol"]),
    )
    failed = [row for row in checks if not row["passed"]]
    if failed:
        raise RuntimeError(
            "Built graph failed validation:\n"
            + json.dumps(failed, indent=2, ensure_ascii=False)
        )

    artifact_path = _resolve(root, output_cfg["artifact_path"])
    diagnostics_dir = _resolve(root, output_cfg["diagnostics_dir"])
    save_graph(graph, artifact_path)
    save_matrix_csvs(graph, diagnostics_dir)
    metrics = compute_graph_metrics(graph)
    (diagnostics_dir / "graph_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(metrics["node_metrics"]).to_csv(
        diagnostics_dir / "node_metrics.csv",
        index=False,
    )
    plot_graph_diagnostics(graph, metrics, diagnostics_dir)
    (diagnostics_dir / "build_checks.json").write_text(
        json.dumps(
            {
                "all_passed": True,
                "checks": checks,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    status = {
        "all_passed": True,
        "artifact_path": str(artifact_path),
        "diagnostics_dir": str(diagnostics_dir),
        "fit_start": graph["fit_start"],
        "fit_end": graph["fit_end"],
        "fit_rows": graph["fit_rows"],
        "node_names": graph["node_names"],
        "dma_columns": graph["dma_columns"],
        "graph_method": graph["graph_method"],
        "corr_threshold": graph["corr_threshold"],
        "positive_undirected_edges": metrics["positive_undirected_edges"],
        "possible_undirected_edges": metrics["possible_undirected_edges"],
        "delta2": metrics["delta2"],
    }
    (diagnostics_dir / "build_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/graph/pearson_static.yaml"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = _resolve(root, args.config)
    status = build(root, config_path)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
