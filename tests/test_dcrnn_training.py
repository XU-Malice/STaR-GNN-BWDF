"""Tests for the DCRNN training engine."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml

from dma_wdf.data.dcrnn_dataset import (
    DCRNNTrainingData,
    DCRNNWindowSubset,
    ZScoreScaler,
)
from dma_wdf.models.dcrnn import DCRNN
from dma_wdf.training.engine import (
    inverse_sigmoid_ratio,
    train_dcrnn,
)


def _load_finalize_module():
    root = Path(__file__).resolve().parent.parent
    path = root / "scripts" / "train" / "finalize_completed_run.py"
    spec = importlib.util.spec_from_file_location(
        "finalize_completed_run_script",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subset(
    *,
    rng: np.random.Generator,
    samples: int,
    history: int,
    horizon: int,
    nodes: int,
    input_dim: int,
    future_dim: int,
) -> DCRNNWindowSubset:
    y_scaled = rng.normal(
        size=(samples, horizon, nodes)
    ).astype(np.float32)
    return DCRNNWindowSubset(
        x_past=rng.normal(
            size=(samples, history, nodes, input_dim)
        ).astype(np.float32),
        y_scaled=y_scaled,
        y_raw=y_scaled.copy(),
        future_exog=rng.normal(
            size=(samples, horizon, nodes, future_dim)
        ).astype(np.float32),
        forecast_starts=tuple(
            f"2021-01-{index + 1:02d}"
            for index in range(samples)
        ),
    )


def _training_data() -> DCRNNTrainingData:
    rng = np.random.default_rng(7)
    nodes = 2
    scaler = ZScoreScaler(
        mean=np.zeros(nodes, dtype=np.float32),
        std=np.ones(nodes, dtype=np.float32),
        feature_names=("DMA 1", "DMA 2"),
        fit_start="2021-01-01 00:00:00+01:00",
        fit_end="2021-01-10 23:00:00+01:00",
        fit_rows=240,
    )
    weather_scaler = ZScoreScaler(
        mean=np.zeros(1, dtype=np.float32),
        std=np.ones(1, dtype=np.float32),
        feature_names=("weather",),
        fit_start=scaler.fit_start,
        fit_end=scaler.fit_end,
        fit_rows=scaler.fit_rows,
    )
    return DCRNNTrainingData(
        fit=_subset(
            rng=rng,
            samples=6,
            history=4,
            horizon=3,
            nodes=nodes,
            input_dim=2,
            future_dim=1,
        ),
        validation=_subset(
            rng=rng,
            samples=2,
            history=4,
            horizon=3,
            nodes=nodes,
            input_dim=2,
            future_dim=1,
        ),
        demand_scaler=scaler,
        weather_scaler=weather_scaler,
        input_feature_names=("demand", "weather"),
        future_exog_feature_names=("calendar",),
        dma_columns=("DMA 1", "DMA 2"),
        purged_forecast_starts=(),
        metadata={
            "development_samples": 8,
            "fit_samples": 6,
            "validation_samples": 2,
            "labels_overlap": False,
            "graph_fit_rows": 17136,
        },
    )


def _config() -> dict:
    return {
        "task": {"name": "synthetic", "horizon": 3},
        "training": {
            "loss": "mae",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 2,
            "max_epochs": 2,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0,
            "gradient_clip_norm": 5.0,
            "shuffle_train_samples": True,
            "scheduled_sampling": {
                "schedule": "inverse_sigmoid",
                "cl_decay_steps": 350,
            },
            "tensorboard": {"enabled": False},
        },
        # The resume test deliberately retains the normally-temporary last
        # checkpoint. Formal configs set this to false.
        "output": {"keep_last_checkpoint_after_training": True},
    }


def _model() -> DCRNN:
    return DCRNN(
        random_walk=torch.eye(2),
        input_dim=2,
        hidden_dim=4,
        horizon=3,
        num_nodes=2,
        num_rnn_layers=1,
        max_diffusion_step=1,
        future_exog_dim=1,
        graph_metadata={"artifact_sha256": "synthetic-graph"},
    )


def test_inverse_sigmoid_schedule_is_monotone_and_uses_350() -> None:
    values = [
        inverse_sigmoid_ratio(step, 350)
        for step in [0, 350, 1750, 3500]
    ]
    assert 0.99 < values[0] < 1.0
    assert values == sorted(values, reverse=True)
    assert values[-1] < 0.02


def test_train_checkpoint_and_resume(tmp_path: Path) -> None:
    data = _training_data()
    config = _config()
    result = train_dcrnn(
        config=config,
        model=_model(),
        training_data=data,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        seed=0,
        runtime_metadata={"test": True},
        max_epochs_override=1,
        max_train_batches=1,
    )
    assert result.epochs_completed == 1
    assert result.best_checkpoint.is_file()
    assert result.last_checkpoint is not None
    assert result.last_checkpoint.is_file()

    checkpoint = torch.load(
        result.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["kind"] == (
        "dma_wdf_dcrnn_training_checkpoint"
    )
    assert checkpoint["demand_scaler"] == (
        data.demand_scaler.state_dict()
    )
    assert checkpoint["weather_scaler"] == (
        data.weather_scaler.state_dict()
    )
    assert checkpoint["model_metadata"]["graph"][
        "artifact_sha256"
    ] == "synthetic-graph"
    assert "test" not in checkpoint["data_metadata"]

    resumed = train_dcrnn(
        config=config,
        model=_model(),
        training_data=data,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        seed=0,
        runtime_metadata={"test": True},
        resume_checkpoint=result.last_checkpoint,
        max_epochs_override=2,
        max_train_batches=1,
    )
    assert resumed.epochs_completed == 2
    assert resumed.global_step == 2
    summary = json.loads(
        (tmp_path / "training_summary.json").read_text()
    )
    assert summary["test_data_used"] is False
    history = json.loads(
        (tmp_path / "history.json").read_text()
    )
    assert history[-1]["train_seconds"] >= 0.0
    assert history[-1]["validation_seconds"] >= 0.0


def test_successful_formal_run_removes_last_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config()
    config["output"]["keep_last_checkpoint_after_training"] = False
    result = train_dcrnn(
        config=config,
        model=_model(),
        training_data=_training_data(),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        seed=0,
        max_epochs_override=1,
        max_train_batches=1,
    )
    assert result.best_checkpoint.is_file()
    assert result.last_checkpoint is None
    assert not (tmp_path / "checkpoint_last.pt").exists()
    summary = json.loads(
        (tmp_path / "training_summary.json").read_text()
    )
    assert summary["last_checkpoint"] is None
    assert summary["last_checkpoint_removed_after_success"] is True


def test_finalize_completed_legacy_run_removes_only_last(
    tmp_path: Path,
) -> None:
    config = _config()
    result = train_dcrnn(
        config=config,
        model=_model(),
        training_data=_training_data(),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        seed=0,
        max_epochs_override=1,
        max_train_batches=1,
    )
    assert result.last_checkpoint is not None
    unrelated = tmp_path / "keep_me.txt"
    unrelated.write_text("preserve", encoding="utf-8")

    module = _load_finalize_module()
    finalized = module.finalize_completed_run(tmp_path)
    assert finalized["last_checkpoint_removed"] is True
    assert result.best_checkpoint.is_file()
    assert not result.last_checkpoint.exists()
    assert unrelated.read_text(encoding="utf-8") == "preserve"


def test_real_configs_keep_schedule_nested_under_training() -> None:
    root = Path(__file__).resolve().parent.parent
    for task in ["24h", "168h"]:
        config = yaml.safe_load(
            (
                root
                / "configs"
                / "train"
                / f"dcrnn_{task}.yaml"
            ).read_text(encoding="utf-8")
        )
        schedule = config["training"]["scheduled_sampling"]
        assert schedule["schedule"] == "inverse_sigmoid"
        assert schedule["cl_decay_steps"] == 350
        assert config["training"]["early_stopping_patience"] == 15
        assert config["training"]["validation_batch_size"] > 0
        assert config["training"]["tensorboard"] == {
            "enabled": True,
            "subdir": "tensorboard",
            "flush_every_epochs": 1,
        }
        assert "scheduled_sampling" not in config
        assert config["runtime"]["device"] == "auto"
        assert config["runtime"]["allow_cpu_fallback"] is False
        assert config["runtime"]["minimum_free_memory_mib"] == 8192
        assert config["runtime"]["maximum_used_memory_mib"] == 2048
        assert (
            config["output"][
                "keep_last_checkpoint_after_training"
            ]
            is False
        )
