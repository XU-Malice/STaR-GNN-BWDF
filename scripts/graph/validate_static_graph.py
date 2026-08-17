#!/usr/bin/env python
"""Independently validate the saved BWDF Pearson static graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dma_wdf.data.graph import (
    build_pearson_graph,
    compute_graph_metrics,
    compute_segment_stability,
    load_graph,
    moving_block_bootstrap,
    validate_graph,
)
from dma_wdf.utils.config import parse_timestamp, read_yaml


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def validate(root: Path, config_path: Path) -> dict[str, object]:
    cfg = read_yaml(config_path)
    source_cfg = cfg["source"]
    output_cfg = cfg["output"]
    validation_cfg = cfg["validation"]
    bootstrap_cfg = cfg["bootstrap"]

    data_dir = _resolve(root, source_cfg["data_dir"])
    demand = pd.read_parquet(data_dir / source_cfg["demand_file"])
    split_cfg = read_yaml(_resolve(root, source_cfg["split_config"]))
    features_cfg = read_yaml(_resolve(root, source_cfg["features_config"]))
    tz = demand.index.tz
    train_start = parse_timestamp(split_cfg["split"]["train_start"], tz)
    train_end = parse_timestamp(
        split_cfg["split"]["train_end_inclusive"],
        tz,
    )
    dma_columns = list(split_cfg["features"]["demand"]["dma_columns"])
    node_names = list(split_cfg["features"]["demand"]["dma_letters"])
    mapped = [features_cfg["dma_mapping"][name] for name in node_names]
    if mapped != dma_columns:
        raise ValueError("DMA mapping mismatch between configs.")

    demand_train = demand.loc[
        (demand.index >= train_start)
        & (demand.index <= train_end),
        dma_columns,
    ]
    artifact_path = _resolve(root, output_cfg["artifact_path"])
    diagnostics_dir = _resolve(root, output_cfg["diagnostics_dir"])
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    graph = load_graph(artifact_path)

    checks = validate_graph(
        graph,
        expected_nodes=int(validation_cfg["expected_nodes"]),
        expected_fit_rows=int(validation_cfg["expected_fit_rows"]),
        expected_node_names=node_names,
        expected_dma_columns=dma_columns,
        atol=float(validation_cfg["atol"]),
    )

    recomputed = build_pearson_graph(
        demand_train,
        dma_columns=dma_columns,
        node_names=node_names,
    )
    atol = float(validation_cfg["atol"])
    for matrix_name in ["static_corr", "static_adj", "random_walk"]:
        error = float(
            np.max(
                np.abs(
                    np.asarray(graph[matrix_name])
                    - np.asarray(recomputed[matrix_name])
                )
            )
        )
        checks.append(
            {
                "check": f"artifact_recomputed_{matrix_name}",
                "passed": error <= atol,
                "observed": error,
                "expected": f"<={atol}",
            }
        )
    checks.append(
        {
            "check": "artifact_demand_sha256",
            "passed": graph["demand_sha256"] == recomputed["demand_sha256"],
            "observed": graph["demand_sha256"],
            "expected": recomputed["demand_sha256"],
        }
    )
    checks.append(
        {
            "check": "artifact_fit_start",
            "passed": graph["fit_start"] == recomputed["fit_start"],
            "observed": graph["fit_start"],
            "expected": recomputed["fit_start"],
        }
    )
    checks.append(
        {
            "check": "artifact_fit_end",
            "passed": graph["fit_end"] == recomputed["fit_end"],
            "observed": graph["fit_end"],
            "expected": recomputed["fit_end"],
        }
    )

    metrics = compute_graph_metrics(graph)
    stability = compute_segment_stability(
        demand_train,
        dma_columns=dma_columns,
        n_segments=int(validation_cfg["stability_segments"]),
    )
    np.savetxt(
        diagnostics_dir / "segment_stability.csv",
        stability["similarity_matrix"],
        delimiter=",",
        fmt="%.12f",
    )
    (diagnostics_dir / "segment_stability.json").write_text(
        json.dumps(
            {
                "n_segments": stability["n_segments"],
                "segment_ranges": stability["segment_ranges"],
                "similarity_matrix": stability[
                    "similarity_matrix"
                ].tolist(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if bool(bootstrap_cfg["enabled"]):
        bootstrap = moving_block_bootstrap(
            demand_train,
            dma_columns=dma_columns,
            block_hours=int(bootstrap_cfg["block_hours"]),
            n_resamples=int(bootstrap_cfg["n_resamples"]),
            seed=int(bootstrap_cfg["seed"]),
        )
        bootstrap.to_csv(
            diagnostics_dir / "edge_bootstrap_intervals.csv",
            index=False,
        )

    all_passed = all(bool(row["passed"]) for row in checks)
    report = {
        "all_passed": all_passed,
        "artifact_path": str(artifact_path),
        "graph_method": graph["graph_method"],
        "fit_start": graph["fit_start"],
        "fit_end": graph["fit_end"],
        "fit_rows": graph["fit_rows"],
        "positive_undirected_edges": metrics["positive_undirected_edges"],
        "possible_undirected_edges": metrics["possible_undirected_edges"],
        "density": metrics["density"],
        "delta2": metrics["delta2"],
        "checks": checks,
    }
    (diagnostics_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


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
    report = validate(root, _resolve(root, args.config))
    if not bool(report["all_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
