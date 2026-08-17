"""Tests for the DCRNN model-construction interface."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from dma_wdf.data.graph import build_pearson_graph, save_graph
from dma_wdf.evaluation.dcrnn_evaluator import (
    _validate_checkpoint_graph_identity,
    evaluate_aggregate_total_predictions,
    evaluate_predictions,
)
from dma_wdf.models.dcrnn import DCRNN, build_dcrnn_model
from dma_wdf.utils.config import read_yaml


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, dict]:
    graph_path = tmp_path / "artifacts" / "graphs" / "graph.npz"
    rng = np.random.default_rng(7)
    rows = 336
    base = np.sin(np.arange(rows) * 2.0 * np.pi / 24.0)
    values = {
        f"DMA {index}": (
            (1.0 + index * 0.05) * base
            + rng.normal(scale=0.1, size=rows)
        )
        for index in range(1, 11)
    }
    frame = pd.DataFrame(
        values,
        index=pd.date_range(
            "2021-01-01",
            periods=rows,
            freq="h",
            tz="Europe/Rome",
        ),
    )
    graph = build_pearson_graph(
        frame,
        dma_columns=list(frame.columns),
        node_names=list("ABCDEFGHIJ"),
    )
    save_graph(graph, graph_path)

    config = {
        "model": {
            "name": "dcrnn",
            "num_nodes": 10,
            "hidden_dim": 8,
            "num_rnn_layers": 1,
            "max_diffusion_step": 2,
            "num_graph_supports": 1,
            "dropout": 0.0,
        },
        "graph": {
            "artifact_path": str(graph_path.relative_to(tmp_path)),
            "matrix_key": "random_walk",
            "expected_method": "pearson_full_positive",
            "expected_corr_threshold": None,
            "expected_negative_policy": "clip_zero",
            "expected_self_loop_in_adjacency": False,
            "expected_nodes": 10,
            "node_names": list("ABCDEFGHIJ"),
            "dma_columns": [f"DMA {i}" for i in range(1, 11)],
        },
    }
    return tmp_path, config


def test_model_config_contains_only_model_concerns() -> None:
    root = Path(__file__).resolve().parents[1]
    config = read_yaml(root / "configs" / "model" / "dcrnn.yaml")
    assert "training" not in config
    assert "evaluation" not in config
    assert "scheduled_sampling" not in config
    assert config["model"]["max_diffusion_step"] == 2
    assert config["model"]["num_graph_supports"] == 1
    assert config["graph"]["expected_method"] == "pearson_full_positive"
    assert config["graph"]["expected_corr_threshold"] is None


@pytest.mark.parametrize("horizon", [24, 168])
def test_dcrnn_builds_both_tasks_from_same_interface(
    project: tuple[Path, dict],
    horizon: int,
) -> None:
    root, config = project
    model = build_dcrnn_model(
        config,
        project_root=root,
        input_dim=12,
        future_exog_dim=7,
        horizon=horizon,
        device="cpu",
    )
    assert isinstance(model, DCRNN)
    assert model.horizon == horizon
    assert model.max_diffusion_step == 2
    assert model.num_nodes == 10
    assert model.random_walk.device.type == "cpu"


def test_24h_and_168h_share_exact_graph(
    project: tuple[Path, dict],
) -> None:
    root, config = project
    model_24 = build_dcrnn_model(
        config,
        project_root=root,
        input_dim=12,
        future_exog_dim=7,
        horizon=24,
    )
    model_168 = build_dcrnn_model(
        config,
        project_root=root,
        input_dim=12,
        future_exog_dim=7,
        horizon=168,
    )
    assert torch.equal(model_24.random_walk, model_168.random_walk)
    meta_24 = model_24.model_metadata()["graph"]
    meta_168 = model_168.model_metadata()["graph"]
    assert meta_24["artifact_sha256"] == meta_168["artifact_sha256"]
    assert meta_24["demand_sha256"] == meta_168["demand_sha256"]


def test_dcrnn_records_graph_provenance(
    project: tuple[Path, dict],
) -> None:
    root, config = project
    model = build_dcrnn_model(
        config,
        project_root=root,
        input_dim=12,
        future_exog_dim=7,
        horizon=24,
    )
    metadata = model.model_metadata()["graph"]
    assert metadata["graph_method"] == "pearson_full_positive"
    assert metadata["corr_threshold"] is None
    assert metadata["fit_rows"] == 336
    assert len(metadata["artifact_sha256"]) == 64
    assert len(metadata["demand_sha256"]) == 64


def test_evaluator_accepts_exact_checkpoint_graph_identity(
    project: tuple[Path, dict],
) -> None:
    root, config = project
    model = build_dcrnn_model(
        config,
        project_root=root,
        input_dim=12,
        future_exog_dim=7,
        horizon=24,
    )
    checkpoint = {"model_metadata": model.model_metadata()}
    identity = _validate_checkpoint_graph_identity(
        checkpoint=checkpoint,
        model=model,
    )
    assert identity["verified"] is True
    assert identity["artifact_sha256"] == (
        model.model_metadata()["graph"]["artifact_sha256"]
    )


def test_evaluator_rejects_checkpoint_graph_hash_mismatch(
    project: tuple[Path, dict],
) -> None:
    root, config = project
    model = build_dcrnn_model(
        config,
        project_root=root,
        input_dim=12,
        future_exog_dim=7,
        horizon=24,
    )
    checkpoint = {
        "model_metadata": deepcopy(model.model_metadata())
    }
    checkpoint["model_metadata"]["graph"][
        "artifact_sha256"
    ] = "different-graph"
    with pytest.raises(
        ValueError,
        match="graph mismatch for artifact_sha256",
    ):
        _validate_checkpoint_graph_identity(
            checkpoint=checkpoint,
            model=model,
        )


def test_publisher_total_mae_is_sum_of_dma_mae() -> None:
    truth = np.full((2, 3, 2), 10.0, dtype=np.float32)
    prediction = truth.copy()
    prediction[:, :, 0] += 1.0
    prediction[:, :, 1] -= 1.0

    metrics = evaluate_predictions(
        y_true=truth,
        y_pred=prediction,
        dma_names=["A", "B"],
    ).set_index("entity")

    assert metrics.loc["A", "MAE"] == pytest.approx(1.0)
    assert metrics.loc["B", "MAE"] == pytest.approx(1.0)
    assert metrics.loc["total", "MAE"] == pytest.approx(2.0)
    # The aggregate demand is predicted perfectly; the other total metrics
    # therefore still use the summed hourly series rather than DMA sums.
    assert metrics.loc["total", "RMSE"] == pytest.approx(0.0)
    aggregate = evaluate_aggregate_total_predictions(
        y_true=truth,
        y_pred=prediction,
    ).set_index("entity")
    assert aggregate.loc["aggregate_total", "MAE"] == pytest.approx(0.0)


def test_dcrnn_rejects_k1(
    project: tuple[Path, dict],
) -> None:
    root, config = project
    config["model"]["max_diffusion_step"] = 1
    with pytest.raises(ValueError, match="max_diffusion_step=2"):
        build_dcrnn_model(
            config,
            project_root=root,
            input_dim=12,
            future_exog_dim=7,
            horizon=24,
        )


def test_dcrnn_rejects_duplicate_supports(
    project: tuple[Path, dict],
) -> None:
    root, config = project
    config["model"]["num_graph_supports"] = 2
    with pytest.raises(ValueError, match="one random-walk support"):
        build_dcrnn_model(
            config,
            project_root=root,
            input_dim=12,
            future_exog_dim=7,
            horizon=24,
        )


def test_dcrnn_rejects_graph_node_order_mismatch(
    project: tuple[Path, dict],
) -> None:
    root, config = project
    config["graph"]["node_names"] = list(reversed(list("ABCDEFGHIJ")))
    with pytest.raises(ValueError, match="node order mismatch"):
        build_dcrnn_model(
            config,
            project_root=root,
            input_dim=12,
            future_exog_dim=7,
            horizon=24,
        )


def test_dcrnn_rejects_wrong_model_name(
    project: tuple[Path, dict],
) -> None:
    root, config = project
    config["model"]["name"] = "gcn"
    with pytest.raises(ValueError, match="model.name='dcrnn'"):
        build_dcrnn_model(
            config,
            project_root=root,
            input_dim=12,
            future_exog_dim=7,
            horizon=24,
        )


def test_dcrnn_rejects_unsupported_horizon(
    project: tuple[Path, dict],
) -> None:
    root, config = project
    with pytest.raises(ValueError, match="must be 24 or 168"):
        build_dcrnn_model(
            config,
            project_root=root,
            input_dim=12,
            future_exog_dim=7,
            horizon=48,
        )
