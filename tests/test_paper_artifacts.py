"""论文表图生成器的端到端合成数据回归测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_MODELS = ("DCRNN", "State", "FA-DPR", "Full")


def _evaluation_dir(root: Path, model: str, task: str) -> Path:
    if model == "DCRNN":
        path = root / "models/star_gnn/Base" / task / "seed_0/evaluation"
    elif model in ABLATION_MODELS:
        path = root / "models/star_gnn" / model / task / "seed_0/evaluation"
    else:
        path = root / "models/baselines" / model.lower() / task / "seed_0/evaluation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_evaluation(root: Path, model: str, task: str, offset: float) -> None:
    output = _evaluation_dir(root, model, task)
    pd.DataFrame(
        [
            {
                "entity": "aggregate_total",
                "MAE": 1.0 + offset,
                "MAPE_percent": 2.0 + offset,
                "RMSE": 3.0 + offset,
                "NSE": 0.99 - 0.001 * offset,
            }
        ]
    ).to_csv(output / "metrics_aggregate_total_common_46.csv", index=False)
    dma_rows = [
        {
            "entity": letter,
            "MAE": 0.1 * (index + 1) + offset,
            "MAPE": 0.01 * (index + 1),
            "RMSE": 0.2 * (index + 1) + offset,
            "NSE": 0.98 - 0.001 * index,
        }
        for index, letter in enumerate("ABCDEFGHIJ")
    ]
    dma_rows.append(
        {"entity": "total", "MAE": 6.0, "MAPE": 0.02, "RMSE": 3.0, "NSE": 0.98}
    )
    pd.DataFrame(dma_rows).to_csv(output / "metrics_common_46.csv", index=False)

    horizon = 24 if task == "24h" else 168
    rng = np.random.default_rng(42 + horizon)
    truth = 20.0 + rng.normal(size=(46, horizon, 10))
    prediction = truth + offset + rng.normal(scale=0.1, size=truth.shape)
    if model in ABLATION_MODELS:
        np.savez_compressed(
            output / "predictions.npz",
            target=truth,
            prediction=prediction,
            operational_indices=np.arange(46, dtype=np.int64),
            strict_within_test_indices=np.arange(46, dtype=np.int64),
            common_46_indices=np.arange(46, dtype=np.int64),
        )
    else:
        np.savez_compressed(
            output / "predictions.npz",
            y_true=truth,
            y_pred=prediction,
            common_46_indices=np.arange(46, dtype=np.int64),
        )


def _write_graph(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = np.asarray(list("ABCDEFGHIJ"), dtype=str)
    corr = np.full((10, 10), 0.5, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    adjacency = corr.copy()
    np.fill_diagonal(adjacency, 0.0)
    walk = adjacency / adjacency.sum(axis=1, keepdims=True)
    np.savez_compressed(
        path,
        node_names=names,
        dma_columns=np.asarray([f"DMA {i}" for i in range(1, 11)], dtype=str),
        static_corr=corr,
        static_adj=adjacency,
        random_walk=walk,
        fit_start=np.asarray("2021-01-01"),
        fit_end=np.asarray("2022-12-15"),
        fit_rows=np.asarray(17136, dtype=np.int64),
        graph_method=np.asarray("pearson_full_positive"),
        corr_threshold=np.asarray(np.nan, dtype=np.float64),
        negative_policy=np.asarray("clip_zero"),
        self_loop_in_adjacency=np.asarray(False, dtype=np.bool_),
        static=np.asarray(True, dtype=np.bool_),
        normalization=np.asarray("random_walk"),
        demand_sha256=np.asarray("synthetic-test"),
    )


def test_build_detailed_test_artifacts(tmp_path: Path) -> None:
    release = tmp_path / "frozen"
    for model_index, model in enumerate(("STGCN", *ABLATION_MODELS)):
        # Full 设置为最小 offset，仅用于确认制图器能处理论文层级。
        offset = 0.0 if model == "Full" else 0.05 * (model_index + 1)
        for task in ("24h", "168h"):
            _write_evaluation(release, model, task, offset)
    graph = tmp_path / "graph.npz"
    _write_graph(graph)
    output = tmp_path / "paper"

    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce/build_detailed_test_artifacts.py",
            "--release",
            str(release),
            "--graph",
            str(graph),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    required = (
        "tables/test_all_models_common46.csv",
        "tables/test_dma_metrics_long.csv",
        "tables/test_day1_day7_metrics.csv",
        "tables/pearson_correlation.csv",
        "figures/test_day1_day7_ablation.png",
        "figures/test_day1_day7_models.pdf",
        "figures/pearson_correlation_heatmap.pdf",
        "reports/TEST_RESULTS_CN.md",
    )
    for relative in required:
        assert (output / relative).is_file(), relative
    daily = pd.read_csv(output / "tables/test_day1_day7_metrics.csv")
    assert len(daily) == 5 * 7
    dma = pd.read_csv(output / "tables/test_dma_metrics_long.csv")
    assert len(dma) == 2 * 5 * 10
