"""STGCN architecture, graph-contract, and leakage-isolation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dma_wdf.data.graph import build_pearson_graph, save_graph
from dma_wdf.models.stgcn import (
    STGCN,
    build_chebyshev_supports,
    build_stgcn_model,
)


def _direct_model(*, horizon: int = 24) -> STGCN:
    adjacency = torch.ones(3, 3) - torch.eye(3)
    supports, _ = build_chebyshev_supports(
        adjacency,
        chebyshev_order=3,
    )
    return STGCN(
        chebyshev_supports=supports,
        input_dim=4,
        future_exog_dim=2,
        history_hours=12,
        horizon=horizon,
        num_nodes=3,
        temporal_kernel_size=2,
        block_channels=((8, 4, 8), (8, 4, 8)),
        head_channels=8,
        dropout=0.0,
        graph_metadata={"artifact_sha256": "synthetic"},
    )


def _project(tmp_path: Path) -> tuple[Path, dict]:
    graph_path = tmp_path / "artifacts" / "graphs" / "graph.npz"
    rows = 336
    rng = np.random.default_rng(11)
    base = np.sin(np.arange(rows) * 2.0 * np.pi / 24.0)
    frame = pd.DataFrame(
        {
            f"DMA {index}": (
                (1.0 + index * 0.03) * base
                + rng.normal(0.0, 0.1, rows)
            )
            for index in range(1, 11)
        },
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
            "name": "stgcn",
            "num_nodes": 10,
            "temporal_kernel_size": 2,
            "chebyshev_order": 3,
            "block_channels": [[8, 4, 8], [8, 4, 8]],
            "head_channels": 8,
            "dropout": 0.0,
        },
        "graph": {
            "artifact_path": str(
                graph_path.relative_to(tmp_path)
            ),
            "matrix_key": "static_adj",
            "expected_method": "pearson_full_positive",
            "expected_corr_threshold": None,
            "expected_negative_policy": "clip_zero",
            "expected_self_loop_in_adjacency": False,
            "expected_nodes": 10,
            "node_names": list("ABCDEFGHIJ"),
            "dma_columns": [
                f"DMA {index}" for index in range(1, 11)
            ],
        },
    }
    return tmp_path, config


def test_chebyshev_supports_include_identity_and_are_finite() -> None:
    adjacency = torch.ones(3, 3) - torch.eye(3)
    supports, lambda_max = build_chebyshev_supports(
        adjacency,
        chebyshev_order=3,
    )
    assert supports.shape == (3, 3, 3)
    assert torch.equal(supports[0], torch.eye(3))
    assert torch.isfinite(supports).all()
    assert lambda_max > 0.0


@pytest.mark.parametrize("horizon", [24, 168])
def test_direct_model_output_shape(horizon: int) -> None:
    model = _direct_model(horizon=horizon)
    x_past = torch.randn(2, 12, 3, 4)
    future = torch.randn(2, horizon, 3, 2)
    output = model(
        x_past,
        x_future_exog=future,
        teacher_forcing_ratio=0.0,
    )
    assert output.shape == (2, horizon, 3)
    assert torch.isfinite(output).all()


def test_direct_decoder_rejects_targets_and_teacher_forcing() -> None:
    model = _direct_model()
    x_past = torch.randn(1, 12, 3, 4)
    future = torch.randn(1, 24, 3, 2)
    target = torch.randn(1, 24, 3)
    with pytest.raises(ValueError, match="never accepts y_target"):
        model(
            x_past,
            y_target=target,
            x_future_exog=future,
        )
    with pytest.raises(ValueError, match="ratio must be 0"):
        model(
            x_past,
            x_future_exog=future,
            teacher_forcing_ratio=0.5,
        )


@pytest.mark.parametrize("horizon", [24, 168])
def test_builder_uses_static_adj_and_same_artifact(
    tmp_path: Path,
    horizon: int,
) -> None:
    root, config = _project(tmp_path)
    model = build_stgcn_model(
        config,
        project_root=root,
        input_dim=12,
        future_exog_dim=7,
        horizon=horizon,
        history_hours=12,
    )
    metadata = model.model_metadata()
    assert metadata["model_name"] == "stgcn"
    assert metadata["graph"]["matrix_key"] == "static_adj"
    assert metadata["graph"]["graph_method"] == (
        "pearson_full_positive"
    )
    assert metadata["graph"]["chebyshev_order"] == 3
    assert len(metadata["graph"]["artifact_sha256"]) == 64


def test_builder_rejects_dcrnn_random_walk_key(
    tmp_path: Path,
) -> None:
    root, config = _project(tmp_path)
    config["graph"]["matrix_key"] = "random_walk"
    with pytest.raises(ValueError, match="matrix_key='static_adj'"):
        build_stgcn_model(
            config,
            project_root=root,
            input_dim=12,
            future_exog_dim=7,
            horizon=24,
            history_hours=12,
        )
