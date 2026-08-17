"""Training-contract tests for the isolated STGCN engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dma_wdf.data.forecast_dataset import (
    ForecastTrainingData,
    ForecastWindowSubset,
    ZScoreScaler,
)
from dma_wdf.models.stgcn import STGCN, build_chebyshev_supports
from dma_wdf.training.stgcn_engine import train_stgcn


def _subset(
    rng: np.random.Generator,
    *,
    samples: int,
) -> ForecastWindowSubset:
    y = rng.normal(size=(samples, 3, 2)).astype(np.float32)
    return ForecastWindowSubset(
        x_past=rng.normal(
            size=(samples, 8, 2, 3)
        ).astype(np.float32),
        y_scaled=y,
        y_raw=y.copy(),
        future_exog=rng.normal(
            size=(samples, 3, 2, 1)
        ).astype(np.float32),
        forecast_starts=tuple(
            f"2021-01-{index + 1:02d}"
            for index in range(samples)
        ),
    )


def _data() -> ForecastTrainingData:
    rng = np.random.default_rng(3)
    demand_scaler = ZScoreScaler(
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
        feature_names=("DMA 1", "DMA 2"),
        fit_start="2021-01-01 00:00:00+01:00",
        fit_end="2021-01-10 23:00:00+01:00",
        fit_rows=240,
    )
    weather_scaler = ZScoreScaler(
        mean=np.zeros(1, dtype=np.float32),
        std=np.ones(1, dtype=np.float32),
        feature_names=("weather",),
        fit_start=demand_scaler.fit_start,
        fit_end=demand_scaler.fit_end,
        fit_rows=demand_scaler.fit_rows,
    )
    return ForecastTrainingData(
        fit=_subset(rng, samples=4),
        validation=_subset(rng, samples=2),
        demand_scaler=demand_scaler,
        weather_scaler=weather_scaler,
        input_feature_names=("demand", "weather", "calendar"),
        future_exog_feature_names=("calendar",),
        dma_columns=("DMA 1", "DMA 2"),
        purged_forecast_starts=(),
        metadata={
            "development_samples": 6,
            "fit_samples": 4,
            "validation_samples": 2,
            "labels_overlap": False,
        },
    )


def _model() -> STGCN:
    supports, _ = build_chebyshev_supports(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        chebyshev_order=2,
    )
    return STGCN(
        chebyshev_supports=supports,
        input_dim=3,
        future_exog_dim=1,
        history_hours=8,
        horizon=3,
        num_nodes=2,
        temporal_kernel_size=2,
        block_channels=((4, 2, 4), (4, 2, 4)),
        head_channels=4,
        graph_metadata={"artifact_sha256": "synthetic"},
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
            "validation_batch_size": 2,
            "max_epochs": 1,
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 0.0,
            "gradient_clip_norm": 5.0,
            "shuffle_train_samples": True,
            "tensorboard": {"enabled": False},
        },
        "output": {"keep_last_checkpoint_after_training": False},
    }


def test_train_stgcn_checkpoint_is_model_specific(
    tmp_path: Path,
) -> None:
    result = train_stgcn(
        config=_config(),
        model=_model(),
        training_data=_data(),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        seed=0,
        max_epochs_override=1,
        max_train_batches=1,
    )
    checkpoint = torch.load(
        result.best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["kind"] == (
        "dma_wdf_stgcn_training_checkpoint"
    )
    assert checkpoint["model_name"] == "stgcn"
    assert result.last_checkpoint is None
    assert not (tmp_path / "checkpoint_last.pt").exists()
    summary = json.loads(
        (tmp_path / "training_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["model"] == "stgcn"
    assert summary["teacher_forcing_used"] is False
    assert summary["test_data_used"] is False


def test_stgcn_config_rejects_scheduled_sampling(
    tmp_path: Path,
) -> None:
    config = _config()
    config["training"]["scheduled_sampling"] = {
        "schedule": "inverse_sigmoid"
    }
    try:
        train_stgcn(
            config=config,
            model=_model(),
            training_data=_data(),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            seed=0,
        )
    except ValueError as error:
        assert "must not configure scheduled_sampling" in str(error)
    else:
        raise AssertionError(
            "STGCN accepted DCRNN scheduled sampling."
        )
