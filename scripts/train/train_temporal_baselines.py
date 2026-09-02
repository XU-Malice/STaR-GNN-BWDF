#!/usr/bin/env python
"""Train and evaluate all six Que et al. temporal baselines.

Formal runs train a one-day (24 h) model to the fixed paper-selected epoch,
freeze that single checkpoint, and use it for both direct 24 h and recursive
168 h common-46 evaluation.  Preprocessing provenance is checked before a
formal run and every scaler is fitted from training arrays only.

Examples
--------
Run all six models on physical GPU 6 (visible as logical cuda:0)::

    CUDA_VISIBLE_DEVICES=6 python scripts/train/train_temporal_baselines.py \
        --model all --device cuda:0

Short smoke run::

    python scripts/train/train_temporal_baselines.py --model mscmnet_w \
        --device cpu --allow-cpu --max-epochs 1 --max-train-batches 1
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.data.metrics import compute_metrics  # noqa: E402
from dma_wdf.data.mscmnet_dataset import (  # noqa: E402
    JointTemporalSamples,
    build_independent_temporal_samples,
    build_joint_temporal_samples,
    forecast_day_features,
    load_paper_data,
)
from dma_wdf.data.sliding_window import slice_hours  # noqa: E402
from dma_wdf.models.mscmnet import (  # noqa: E402
    GRUForecast,
    LSTMForecast,
    MSCMNetOutput,
    build_joint_model_from_config,
)
from dma_wdf.utils.config import read_yaml  # noqa: E402


CANONICAL_MODELS = (
    "gru",
    "lstm",
    "msnet",
    "mscmnet_m",
    "mscmnet_wm",
    "mscmnet_w",
)
ALIASES = {"mscmnet_mw": "mscmnet_wm"}


@dataclass(frozen=True)
class Standardizer:
    """NumPy z-score parameters fitted on training arrays only."""

    mean: np.ndarray
    std: np.ndarray
    fitted_from: str = "train_only"

    @classmethod
    def fit_features(cls, values: np.ndarray) -> "Standardizer":
        if values.ndim < 2:
            raise ValueError("Feature arrays must have at least two dimensions.")
        if values.shape[-1] == 0:
            return cls(
                mean=np.empty((0,), dtype=np.float32),
                std=np.empty((0,), dtype=np.float32),
            )
        flattened = values.reshape(-1, values.shape[-1]).astype(np.float64)
        mean = flattened.mean(axis=0).astype(np.float32)
        std = flattened.std(axis=0).astype(np.float32)
        std[~np.isfinite(std) | (std <= 1.0e-6)] = 1.0
        return cls(mean=mean, std=std)

    @classmethod
    def fit_scalar(cls, *values: np.ndarray) -> "Standardizer":
        flattened = np.concatenate(
            [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
        )
        mean = np.asarray([flattened.mean()], dtype=np.float32)
        std_value = float(flattened.std())
        if not np.isfinite(std_value) or std_value <= 1.0e-6:
            std_value = 1.0
        return cls(mean=mean, std=np.asarray([std_value], dtype=np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "normalization": "zscore",
            "mean": self.mean.copy(),
            "std": self.std.copy(),
            "fitted_from": self.fitted_from,
        }


@dataclass(frozen=True)
class MinMaxScaler:
    """NumPy [0,1] parameters fitted on training arrays only."""

    minimum: np.ndarray
    value_range: np.ndarray
    fitted_from: str = "train_only"

    @classmethod
    def fit_features(cls, values: np.ndarray) -> "MinMaxScaler":
        if values.ndim < 2:
            raise ValueError("Feature arrays must have at least two dimensions.")
        if values.shape[-1] == 0:
            return cls(
                minimum=np.empty((0,), dtype=np.float32),
                value_range=np.empty((0,), dtype=np.float32),
            )
        flattened = values.reshape(-1, values.shape[-1]).astype(np.float64)
        minimum = flattened.min(axis=0).astype(np.float32)
        maximum = flattened.max(axis=0).astype(np.float32)
        value_range = maximum - minimum
        value_range[~np.isfinite(value_range) | (value_range <= 1.0e-6)] = 1.0
        return cls(minimum=minimum, value_range=value_range)

    @classmethod
    def fit_scalar(cls, *values: np.ndarray) -> "MinMaxScaler":
        flattened = np.concatenate(
            [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
        )
        minimum = float(flattened.min())
        value_range = float(flattened.max() - minimum)
        if not np.isfinite(value_range) or value_range <= 1.0e-6:
            value_range = 1.0
        return cls(
            minimum=np.asarray([minimum], dtype=np.float32),
            value_range=np.asarray([value_range], dtype=np.float32),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.minimum) / self.value_range).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.value_range + self.minimum).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "normalization": "minmax",
            "minimum": self.minimum.copy(),
            "value_range": self.value_range.copy(),
            "fitted_from": self.fitted_from,
        }


TrainOnlyScaler = Standardizer | MinMaxScaler


def _normalization_name(value: str) -> str:
    name = str(value).lower()
    if name not in {"zscore", "minmax"}:
        raise ValueError("normalization must be 'zscore' or 'minmax'.")
    return name


def _fit_features(values: np.ndarray, normalization: str) -> TrainOnlyScaler:
    if _normalization_name(normalization) == "minmax":
        return MinMaxScaler.fit_features(values)
    return Standardizer.fit_features(values)


def _fit_scalar(normalization: str, *values: np.ndarray) -> TrainOnlyScaler:
    if _normalization_name(normalization) == "minmax":
        return MinMaxScaler.fit_scalar(*values)
    return Standardizer.fit_scalar(*values)


def _build_optimizer(
    parameters,
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build an explicit optimizer so Adam/AdamW semantics stay auditable."""
    name = str(optimizer_name).lower()
    classes = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if name not in classes:
        raise ValueError("optimizer must be 'adam' or 'adamw'.")
    if float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    return classes[name](
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def canonical_model_name(value: str) -> str:
    name = str(value).lower()
    name = ALIASES.get(name, name)
    if name not in CANONICAL_MODELS and name != "all":
        raise ValueError(f"Unknown temporal baseline: {value!r}.")
    return name


def set_reproducible_seed(seed: int, *, deterministic: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(bool(deterministic))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)


def resolve_device(requested: str, *, allow_cpu: bool) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} logical "
                "CUDA devices are visible."
            )
    elif not allow_cpu:
        raise RuntimeError("Formal runs require CUDA; pass --allow-cpu for a smoke run.")
    return device


def preflight_resources(
    *,
    device: torch.device,
    output_root: Path,
    minimum_free_gib: float,
    minimum_disk_gib: float,
) -> dict[str, Any]:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_root.parent)
    disk_free_gib = disk.free / 1024**3
    if disk_free_gib < float(minimum_disk_gib):
        raise RuntimeError(
            f"Only {disk_free_gib:.2f} GiB disk is free; "
            f"{minimum_disk_gib:.2f} GiB is required."
        )
    payload: dict[str, Any] = {
        "device": str(device),
        "disk_free_gib": disk_free_gib,
        "minimum_disk_gib": float(minimum_disk_gib),
    }
    if device.type == "cuda":
        logical_index = device.index or 0
        with torch.cuda.device(logical_index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / 1024**3
        total_gib = total_bytes / 1024**3
        if free_gib < float(minimum_free_gib):
            raise RuntimeError(
                f"{device} has {free_gib:.2f} GiB free; "
                f"{minimum_free_gib:.2f} GiB is required."
            )
        payload.update(
            {
                "cuda_name": torch.cuda.get_device_name(logical_index),
                "cuda_free_gib": free_gib,
                "cuda_total_gib": total_gib,
                "minimum_cuda_free_gib": float(minimum_free_gib),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
    return payload


def prepare_output_dir(path: Path, *, overwrite: bool) -> Path | None:
    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return None
    if not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --overwrite to "
            "archive it before a new run."
        )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.backup-{stamp}")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.backup-{stamp}-{counter}")
        counter += 1
    shutil.move(str(path), str(backup))
    path.mkdir(parents=True, exist_ok=True)
    return backup


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    family: str,
    share_weight: float,
    max_train_batches: int | None,
) -> float:
    model.train()
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_train_batches is not None and batch_index >= max_train_batches:
            break
        moved = [tensor.to(device, non_blocking=True) for tensor in batch]
        optimizer.zero_grad(set_to_none=True)
        if family == "independent_recurrent":
            prediction = model(moved[0])
            target = moved[1]
            share_prediction = None
        else:
            branches = moved[:10]
            cursor = 10
            future = moved[cursor]
            cursor += 1
            if family == "msnet":
                prediction = model(branches)
                share_prediction = None
            elif family == "mscmnet_m":
                output = model(branches, future)
                prediction = output.prediction
                share_prediction = None
            else:
                fc2_history = moved[cursor]
                cursor += 1
                output = model(branches, future, fc2_history)
                prediction = output.prediction
                share_prediction = output.predicted_daily_share
            target = moved[cursor]
            cursor += 1
        loss = F.mse_loss(prediction, target)
        if share_weight > 0.0:
            if share_prediction is None:
                raise RuntimeError("Share supervision requires an FC2 model.")
            share_target = moved[cursor]
            loss = loss + float(share_weight) * F.mse_loss(
                share_prediction, share_target
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise RuntimeError("No training batches were executed.")
    return float(np.mean(losses))


@torch.no_grad()
def predict_independent_24h(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(x), int(batch_size)):
        batch = torch.as_tensor(
            x[start : start + batch_size], dtype=torch.float32, device=device
        )
        output.append(model(batch).cpu().numpy())
    return np.concatenate(output, axis=0)


@torch.no_grad()
def predict_independent_168h(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    steps: int = 7,
) -> np.ndarray:
    model.eval()
    history = torch.as_tensor(x, dtype=torch.float32, device=device)
    predictions: list[np.ndarray] = []
    for _ in range(int(steps)):
        predicted_day = model(history)
        predictions.append(predicted_day.cpu().numpy())
        history = torch.cat(
            [history[:, 24:, :], predicted_day.unsqueeze(-1)], dim=1
        )
    return np.concatenate(predictions, axis=1)


def _scaled_arrays(
    samples: JointTemporalSamples,
    *,
    normalization: str,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[TrainOnlyScaler, ...],
    np.ndarray,
    TrainOnlyScaler,
    np.ndarray,
    np.ndarray,
    TrainOnlyScaler,
    np.ndarray | None,
    np.ndarray | None,
    TrainOnlyScaler | None,
]:
    branch_scalers = tuple(
        _fit_features(array, normalization) for array in samples.train_branches
    )
    train_branches = tuple(
        scaler.transform(array)
        for scaler, array in zip(branch_scalers, samples.train_branches)
    )
    test_branches = tuple(
        scaler.transform(array)
        for scaler, array in zip(branch_scalers, samples.test_branches)
    )
    target_scaler = _fit_features(samples.y_train_24h, normalization)
    target_train = target_scaler.transform(samples.y_train_24h)
    future_scaler = _fit_features(samples.future_train, normalization)
    future_train = future_scaler.transform(samples.future_train)
    future_test = future_scaler.transform(samples.future_test)
    fc2_scaler: TrainOnlyScaler | None = None
    fc2_train: np.ndarray | None = None
    fc2_test: np.ndarray | None = None
    if samples.fc2_train is not None:
        if samples.fc2_test is None:
            raise ValueError("FC2 test history is missing.")
        fc2_scaler = _fit_features(samples.fc2_train, normalization)
        fc2_train = fc2_scaler.transform(samples.fc2_train)
        fc2_test = fc2_scaler.transform(samples.fc2_test)
    return (
        train_branches,
        test_branches,
        branch_scalers,
        target_train,
        target_scaler,
        future_train,
        future_test,
        future_scaler,
        fc2_train,
        fc2_test,
        fc2_scaler,
    )


def _joint_loader(
    *,
    train_branches: Sequence[np.ndarray],
    future_train: np.ndarray,
    fc2_train: np.ndarray | None,
    target_train: np.ndarray,
    share_target: np.ndarray | None,
    batch_size: int,
    seed: int,
) -> DataLoader:
    arrays: list[np.ndarray] = list(train_branches) + [future_train]
    if fc2_train is not None:
        arrays.append(fc2_train)
    arrays.append(target_train)
    if share_target is not None:
        arrays.append(share_target)
    tensors = [torch.as_tensor(array, dtype=torch.float32) for array in arrays]
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
        pin_memory=torch.cuda.is_available(),
    )


def _joint_prediction(
    model: nn.Module,
    family: str,
    branches: Sequence[torch.Tensor],
    future: torch.Tensor,
    fc2_history: torch.Tensor | None,
) -> torch.Tensor:
    if family == "msnet":
        return model(branches)
    if family == "mscmnet_m":
        output: MSCMNetOutput = model(branches, future)
        return output.prediction
    if fc2_history is None:
        raise ValueError("An FC2 model requires fc2_history.")
    output = model(branches, future, fc2_history)
    return output.prediction


@torch.no_grad()
def predict_joint_24h(
    *,
    model: nn.Module,
    family: str,
    branches: Sequence[np.ndarray],
    future: np.ndarray,
    fc2_history: np.ndarray | None,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    count = len(branches[0])
    for start in range(0, count, int(batch_size)):
        end = start + int(batch_size)
        branch_tensors = [
            torch.as_tensor(array[start:end], dtype=torch.float32, device=device)
            for array in branches
        ]
        future_tensor = torch.as_tensor(
            future[start:end], dtype=torch.float32, device=device
        )
        fc2_tensor = (
            None
            if fc2_history is None
            else torch.as_tensor(
                fc2_history[start:end], dtype=torch.float32, device=device
            )
        )
        outputs.append(
            _joint_prediction(
                model, family, branch_tensors, future_tensor, fc2_tensor
            ).cpu().numpy()
        )
    return np.concatenate(outputs, axis=0)


def _raw_branch_day(
    *,
    prediction: np.ndarray,
    branch_index: int,
    starts: Sequence[pd.Timestamp],
    day_offset: int,
    feature_columns: Sequence[str],
    weather: pd.DataFrame,
    temporal: pd.DataFrame,
) -> np.ndarray:
    exogenous = pd.concat([weather, temporal], axis=1)
    rows: list[np.ndarray] = []
    for sample_index, forecast_start in enumerate(starts):
        day_start = pd.Timestamp(forecast_start) + pd.Timedelta(days=day_offset)
        day = np.empty((24, len(feature_columns)), dtype=np.float32)
        for feature_index, column in enumerate(feature_columns):
            if column == "own_dma_demand":
                day[:, feature_index] = prediction[sample_index, :, branch_index]
            else:
                day[:, feature_index] = slice_hours(
                    exogenous[column], day_start, 24
                )
        rows.append(day)
    return np.stack(rows).astype(np.float32)


def _temperature_extrema(
    weather: pd.DataFrame,
    starts: Sequence[pd.Timestamp],
    *,
    day_offset: int,
) -> np.ndarray:
    rows: list[list[float]] = []
    for forecast_start in starts:
        day_start = pd.Timestamp(forecast_start) + pd.Timedelta(days=day_offset)
        values = slice_hours(weather["air_temperature"], day_start, 24)
        rows.append([float(values.max()), float(values.min())])
    return np.asarray(rows, dtype=np.float32)


@torch.no_grad()
def predict_joint_168h(
    *,
    model: nn.Module,
    family: str,
    branches: Sequence[np.ndarray],
    branch_scalers: Sequence[TrainOnlyScaler],
    branch_feature_columns: Sequence[Sequence[str]],
    target_scaler: TrainOnlyScaler,
    future_scaler: TrainOnlyScaler,
    future_columns: Sequence[str],
    fc2_history_raw: np.ndarray | None,
    fc2_scaler: TrainOnlyScaler | None,
    include_fc2_temperature: bool,
    starts: Sequence[pd.Timestamp],
    weather: pd.DataFrame,
    temporal: pd.DataFrame,
    device: torch.device,
    steps: int = 7,
) -> np.ndarray:
    """Recursive daily rollout using predictions, never future target demand."""
    model.eval()
    history = [array.copy() for array in branches]
    fc2_raw = None if fc2_history_raw is None else fc2_history_raw.copy()
    output_days: list[np.ndarray] = []
    for step in range(int(steps)):
        branch_tensors = [
            torch.as_tensor(array, dtype=torch.float32, device=device)
            for array in history
        ]
        future_raw = forecast_day_features(
            weather=weather,
            temporal=temporal,
            starts=starts,
            columns=future_columns,
            day_offset=step,
        )
        future_tensor = torch.as_tensor(
            future_scaler.transform(future_raw),
            dtype=torch.float32,
            device=device,
        )
        fc2_tensor: torch.Tensor | None = None
        if fc2_raw is not None:
            if fc2_scaler is None:
                raise ValueError("FC2 scaler is missing.")
            fc2_tensor = torch.as_tensor(
                fc2_scaler.transform(fc2_raw),
                dtype=torch.float32,
                device=device,
            )
        prediction_normalized = _joint_prediction(
            model, family, branch_tensors, future_tensor, fc2_tensor
        )
        prediction = target_scaler.inverse(prediction_normalized.cpu().numpy())
        output_days.append(prediction)

        updated_histories: list[np.ndarray] = []
        for index, (branch, scaler, columns) in enumerate(
            zip(history, branch_scalers, branch_feature_columns)
        ):
            raw_day = _raw_branch_day(
                prediction=prediction,
                branch_index=index,
                starts=starts,
                day_offset=step,
                feature_columns=columns,
                weather=weather,
                temporal=temporal,
            )
            normalized_day = scaler.transform(raw_day)
            updated_histories.append(
                np.concatenate(
                    [branch[:, 1:, :, :], normalized_day[:, None, :, :]],
                    axis=1,
                ).astype(np.float32)
            )
        history = updated_histories

        if fc2_raw is not None:
            daily = prediction.sum(axis=1)
            shares = daily / np.maximum(daily.sum(axis=1, keepdims=True), 1.0e-6)
            next_row = shares.astype(np.float32)
            if include_fc2_temperature:
                next_row = np.concatenate(
                    [
                        next_row,
                        _temperature_extrema(weather, starts, day_offset=step),
                    ],
                    axis=1,
                ).astype(np.float32)
            fc2_raw = np.concatenate(
                [fc2_raw[:, 1:, :], next_row[:, None, :]], axis=1
            ).astype(np.float32)
    return np.concatenate(output_days, axis=1)


def metric_table(
    *,
    model_display_name: str,
    y_true_24h: np.ndarray,
    y_pred_24h: np.ndarray,
    y_true_168h: np.ndarray,
    y_pred_168h: np.ndarray,
    dma_letters: Sequence[str],
    literature_config: dict[str, Any] | None,
) -> pd.DataFrame:
    """Compute S1-compatible DMA and total metrics for both horizons."""
    rows: list[dict[str, Any]] = []
    for task, truth, prediction in (
        ("24h", y_true_24h, y_pred_24h),
        ("168h", y_true_168h, y_pred_168h),
    ):
        dma_metrics: list[dict[str, float]] = []
        for index, letter in enumerate(dma_letters):
            metrics = compute_metrics(truth[:, :, index], prediction[:, :, index])
            dma_metrics.append(metrics)
            for metric, value in metrics.items():
                rows.append(
                    {
                        "model": model_display_name,
                        "task": task,
                        "series": str(letter),
                        "metric": metric,
                        "value": float(value),
                        "paper_value": np.nan,
                    }
                )
        aggregate_metrics = compute_metrics(
            truth.sum(axis=2), prediction.sum(axis=2)
        )
        aggregate_metrics["MAE"] = float(
            sum(metrics["MAE"] for metrics in dma_metrics)
        )
        paper_total = None
        if literature_config is not None:
            paper_total = (
                literature_config.get("tasks", {})
                .get(task, {})
                .get(model_display_name)
            )
        for metric, value in aggregate_metrics.items():
            paper_value = (
                np.nan if paper_total is None else float(paper_total[metric])
            )
            rows.append(
                {
                    "model": model_display_name,
                    "task": task,
                    "series": "total",
                    "metric": metric,
                    "value": float(value),
                    "paper_value": paper_value,
                }
            )
    frame = pd.DataFrame(rows)
    frame["local_minus_paper"] = frame["value"] - frame["paper_value"]
    return frame


def _save_predictions(
    path: Path,
    *,
    samples_24h: np.ndarray,
    predictions_24h: np.ndarray,
    samples_168h: np.ndarray,
    predictions_168h: np.ndarray,
    starts: Sequence[pd.Timestamp],
    dma_letters: Sequence[str],
) -> None:
    np.savez_compressed(
        path,
        y_true_24h=samples_24h.astype(np.float32),
        y_pred_24h=predictions_24h.astype(np.float32),
        y_true_168h=samples_168h.astype(np.float32),
        y_pred_168h=predictions_168h.astype(np.float32),
        forecast_starts=np.asarray([str(value) for value in starts]),
        dma_letters=np.asarray(list(dma_letters)),
    )


def train_independent_family(
    *,
    canonical: str,
    model_config: dict[str, Any],
    protocol: dict[str, Any],
    training_config: dict[str, Any],
    demand: pd.DataFrame,
    bounds: dict[str, pd.Timestamp],
    device: torch.device,
    output_dir: Path,
    seed: int,
    max_epochs_override: int | None,
    max_train_batches: int | None,
    literature_config: dict[str, Any] | None,
) -> dict[str, Any]:
    dma_columns = list(protocol["dma_columns"])
    dma_letters = list(protocol["dma_letters"])
    batch_size = int(training_config["batch_size"])
    predictions_24: list[np.ndarray] = []
    predictions_168: list[np.ndarray] = []
    truth_24: list[np.ndarray] = []
    truth_168: list[np.ndarray] = []
    checkpoint_files: list[str] = []
    loss_rows: list[dict[str, Any]] = []
    parameter_counts: dict[str, int] = {}
    sample_count: int | None = None
    test_starts: np.ndarray | None = None

    for index, (letter, column) in enumerate(zip(dma_letters, dma_columns)):
        model_seed = int(seed) + index
        set_reproducible_seed(
            model_seed,
            deterministic=bool(training_config["deterministic_algorithms"]),
        )
        samples = build_independent_temporal_samples(
            demand=demand,
            bounds=bounds,
            dma_column=column,
            input_weeks=int(model_config["input_weeks"][index]),
        )
        if samples["x_test_eval"].shape[0] != int(
            protocol["expected_test_sequences"]
        ):
            raise ValueError("Independent baseline did not build common-46 test data.")
        scaler = _fit_scalar(
            training_config.get("normalization", "zscore"),
            samples["x_train"],
            samples["y_train_24h"],
        )
        x_train = scaler.transform(samples["x_train"])
        y_train = scaler.transform(samples["y_train_24h"])
        x_test = scaler.transform(samples["x_test_eval"])
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(x_train, dtype=torch.float32),
                torch.as_tensor(y_train, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(model_seed),
            pin_memory=torch.cuda.is_available(),
        )
        hidden_sizes = [int(value) for value in model_config["hidden_sizes"][index]]
        model: nn.Module
        if canonical == "gru":
            model = GRUForecast(hidden_sizes)
        else:
            model = LSTMForecast(hidden_sizes)
        model.to(device)
        parameter_counts[str(letter)] = _parameter_count(model)
        optimizer = _build_optimizer(
            model.parameters(),
            optimizer_name=training_config.get("optimizer", "adam"),
            learning_rate=float(model_config["learning_rates"][index]),
            weight_decay=float(model_config["weight_decays"][index]),
        )
        epochs = (
            int(max_epochs_override)
            if max_epochs_override is not None
            else int(model_config["best_epochs"][index])
        )
        for epoch in range(1, epochs + 1):
            loss = _run_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device=device,
                family="independent_recurrent",
                share_weight=0.0,
                max_train_batches=max_train_batches,
            )
            loss_rows.append(
                {"dma": letter, "epoch": epoch, "train_loss": loss}
            )
            print(
                f"model={canonical} dma={letter} epoch={epoch}/{epochs} "
                f"train_loss={loss:.6f}",
                flush=True,
            )
        prediction_24_norm = predict_independent_24h(
            model, x_test, device=device, batch_size=batch_size
        )
        prediction_168_norm = predict_independent_168h(
            model, x_test, device=device, steps=7
        )
        predictions_24.append(scaler.inverse(prediction_24_norm))
        predictions_168.append(scaler.inverse(prediction_168_norm))
        truth_24.append(samples["y_test_eval_24h"])
        truth_168.append(samples["y_test_eval_168h"])
        sample_count = int(samples["x_train"].shape[0])
        test_starts = samples.get("test_forecast_start")
        checkpoint = output_dir / f"checkpoint_{canonical}_dma_{letter}.pt"
        torch.save(
            {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "model": canonical,
                "dma_letter": letter,
                "dma_column": column,
                "model_seed": model_seed,
                "fixed_epoch": epochs,
                "hidden_sizes": hidden_sizes,
                "input_weeks": int(model_config["input_weeks"][index]),
                "scaler": scaler.state_dict(),
                "checkpoint_policy": "final_fixed_paper_epoch",
            },
            checkpoint,
        )
        checkpoint_files.append(checkpoint.name)
        del model, optimizer, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    y_pred_24 = np.stack(predictions_24, axis=2)
    y_pred_168 = np.stack(predictions_168, axis=2)
    y_true_24 = np.stack(truth_24, axis=2)
    y_true_168 = np.stack(truth_168, axis=2)
    # Stacking univariate arrays yields (S,H,N), exactly the joint convention.
    metrics = metric_table(
        model_display_name=str(model_config["display_name"]),
        y_true_24h=y_true_24,
        y_pred_24h=y_pred_24,
        y_true_168h=y_true_168,
        y_pred_168h=y_pred_168,
        dma_letters=dma_letters,
        literature_config=literature_config,
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(output_dir / "loss_curve.csv", index=False)
    starts = (
        np.arange(y_true_24.shape[0])
        if test_starts is None
        else test_starts
    )
    _save_predictions(
        output_dir / "predictions_common46.npz",
        samples_24h=y_true_24,
        predictions_24h=y_pred_24,
        samples_168h=y_true_168,
        predictions_168h=y_pred_168,
        starts=starts,
        dma_letters=dma_letters,
    )
    return {
        "checkpoint_files": checkpoint_files,
        "parameter_counts_by_dma": parameter_counts,
        "last_dma_train_samples": sample_count,
        "prediction_24h_shape": list(y_pred_24.shape),
        "prediction_168h_shape": list(y_pred_168.shape),
    }


def train_joint_family(
    *,
    canonical: str,
    model_config: dict[str, Any],
    cam_config: dict[str, Any],
    protocol: dict[str, Any],
    training_config: dict[str, Any],
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    temporal: pd.DataFrame,
    bounds: dict[str, pd.Timestamp],
    device: torch.device,
    output_dir: Path,
    seed: int,
    max_epochs_override: int | None,
    max_train_batches: int | None,
    literature_config: dict[str, Any] | None,
) -> dict[str, Any]:
    family = str(model_config["family"])
    future_columns = tuple(model_config.get("fc1", {}).get("future_features", []))
    fc2_config = model_config.get("fc2")
    fc2_days = None if fc2_config is None else int(fc2_config["history_days"])
    include_temperature = canonical == "mscmnet_wm"
    samples = build_joint_temporal_samples(
        demand=demand,
        weather=weather,
        temporal=temporal,
        bounds=bounds,
        dma_columns=protocol["dma_columns"],
        branch_features=model_config["branch_features"],
        input_weeks=model_config["input_weeks"],
        future_features=future_columns,
        fc2_history_days=fc2_days,
        fc2_include_temperature=include_temperature,
        max_history_weeks=int(protocol["max_history_weeks"]),
        expected_train_samples=int(protocol["expected_train_samples_joint"]),
        expected_test_sequences=int(protocol["expected_test_sequences"]),
    )
    (
        train_branches,
        test_branches,
        branch_scalers,
        target_train,
        target_scaler,
        future_train,
        future_test,
        future_scaler,
        fc2_train,
        fc2_test,
        fc2_scaler,
    ) = _scaled_arrays(
        samples,
        normalization=training_config.get("normalization", "zscore"),
    )
    share_weight = (
        0.0
        if fc2_config is None
        else float(fc2_config.get("share_supervision_weight", 0.0))
    )
    loader = _joint_loader(
        train_branches=train_branches,
        future_train=future_train,
        fc2_train=fc2_train,
        target_train=target_train,
        share_target=(
            samples.fc2_share_target_train if share_weight > 0.0 else None
        ),
        batch_size=int(training_config["batch_size"]),
        seed=seed,
    )
    set_reproducible_seed(
        seed,
        deterministic=bool(training_config["deterministic_algorithms"]),
    )
    model = build_joint_model_from_config(
        canonical, model_config, cam_config
    ).to(device)
    optimizer_name = str(training_config.get("optimizer", "adam")).lower()
    effective_weight_decay = float(
        training_config.get(
            "joint_weight_decay_override", model_config["weight_decay"]
        )
    )
    optimizer = _build_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        learning_rate=float(model_config["learning_rate"]),
        weight_decay=effective_weight_decay,
    )
    epochs = (
        int(max_epochs_override)
        if max_epochs_override is not None
        else int(model_config["best_epoch"])
    )
    losses: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        loss = _run_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            family=family,
            share_weight=share_weight,
            max_train_batches=max_train_batches,
        )
        losses.append({"epoch": epoch, "train_loss": loss})
        pd.DataFrame(losses).to_csv(output_dir / "loss_curve.csv", index=False)
        print(
            f"model={canonical} epoch={epoch}/{epochs} train_loss={loss:.6f}",
            flush=True,
        )
    prediction_24_norm = predict_joint_24h(
        model=model,
        family=family,
        branches=test_branches,
        future=future_test,
        fc2_history=fc2_test,
        device=device,
        batch_size=int(training_config["batch_size"]),
    )
    prediction_24 = target_scaler.inverse(prediction_24_norm)
    prediction_168 = predict_joint_168h(
        model=model,
        family=family,
        branches=test_branches,
        branch_scalers=branch_scalers,
        branch_feature_columns=samples.branch_feature_columns,
        target_scaler=target_scaler,
        future_scaler=future_scaler,
        future_columns=future_columns,
        fc2_history_raw=samples.fc2_test,
        fc2_scaler=fc2_scaler,
        include_fc2_temperature=include_temperature,
        starts=samples.test_forecast_starts,
        weather=weather,
        temporal=temporal,
        device=device,
        steps=7,
    )
    metrics = metric_table(
        model_display_name=str(model_config["display_name"]),
        y_true_24h=samples.y_test_24h,
        y_pred_24h=prediction_24,
        y_true_168h=samples.y_test_168h,
        y_pred_168h=prediction_168,
        dma_letters=protocol["dma_letters"],
        literature_config=literature_config,
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    _save_predictions(
        output_dir / "predictions_common46.npz",
        samples_24h=samples.y_test_24h,
        predictions_24h=prediction_24,
        samples_168h=samples.y_test_168h,
        predictions_168h=prediction_168,
        starts=samples.test_forecast_starts,
        dma_letters=protocol["dma_letters"],
    )
    checkpoint = output_dir / f"checkpoint_{canonical}.pt"
    torch.save(
        {
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "model": canonical,
            "fixed_epoch": epochs,
            "seed": int(seed),
            "optimizer": optimizer_name,
            "effective_weight_decay": effective_weight_decay,
            "model_config": model_config,
            "cam_config": cam_config,
            "branch_scalers": [value.state_dict() for value in branch_scalers],
            "target_scaler": target_scaler.state_dict(),
            "future_scaler": future_scaler.state_dict(),
            "fc2_scaler": None if fc2_scaler is None else fc2_scaler.state_dict(),
            "checkpoint_policy": "final_fixed_paper_epoch",
        },
        checkpoint,
    )
    return {
        "checkpoint_files": [checkpoint.name],
        "parameter_count": _parameter_count(model),
        "train_samples": int(samples.y_train_24h.shape[0]),
        "test_sequences": int(samples.y_test_24h.shape[0]),
        "prediction_24h_shape": list(prediction_24.shape),
        "prediction_168h_shape": list(prediction_168.shape),
        "optimizer": optimizer_name,
        "effective_weight_decay": effective_weight_decay,
        "fc2_share_supervision_weight": share_weight,
        "correction_mode": str(model_config.get("correction_mode", "direct")),
        "zero_init_correction": bool(
            model_config.get("zero_init_correction", False)
        ),
    }


def run_one_model(
    *,
    canonical: str,
    config: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    bounds: dict[str, pd.Timestamp],
    data_audit: dict[str, Any],
    device: torch.device,
    output_root: Path,
    seed: int,
    overwrite: bool,
    max_epochs_override: int | None,
    max_train_batches: int | None,
    literature_config: dict[str, Any] | None,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    model_config = config["models"][canonical]
    output_dir = output_root / canonical / f"seed_{seed}"
    archived = prepare_output_dir(output_dir, overwrite=overwrite)
    started = time.perf_counter()
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "protocol": config["protocol"],
                "cam": config["cam"],
                "model": model_config,
                "training": config["training"],
                "seed": seed,
                "max_epochs_override": max_epochs_override,
                "max_train_batches": max_train_batches,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if str(model_config["family"]) == "independent_recurrent":
        details = train_independent_family(
            canonical=canonical,
            model_config=model_config,
            protocol=config["protocol"],
            training_config=config["training"],
            demand=frames["demand"],
            bounds=bounds,
            device=device,
            output_dir=output_dir,
            seed=seed,
            max_epochs_override=max_epochs_override,
            max_train_batches=max_train_batches,
            literature_config=literature_config,
        )
    else:
        details = train_joint_family(
            canonical=canonical,
            model_config=model_config,
            cam_config=config["cam"],
            protocol=config["protocol"],
            training_config=config["training"],
            demand=frames["demand"],
            weather=frames["weather"],
            temporal=frames["temporal"],
            bounds=bounds,
            device=device,
            output_dir=output_dir,
            seed=seed,
            max_epochs_override=max_epochs_override,
            max_train_batches=max_train_batches,
            literature_config=literature_config,
        )
    status = {
        "status": "completed",
        "model": canonical,
        "display_name": model_config["display_name"],
        "seed": int(seed),
        "device": str(device),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "git_commit": _git_commit(),
        "data_audit": data_audit,
        "resource_preflight": preflight,
        "output_archived_before_run": None if archived is None else str(archived),
        "formal_protocol": max_epochs_override is None and max_train_batches is None,
        "single_frozen_checkpoint_for_24h_and_168h": True,
        **details,
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, indent=2, default=str), encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate Que et al. GRU, LSTM, MSNet and MSCMNet baselines."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[*CANONICAL_MODELS, *ALIASES, "all"],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/model/mscmnet_baselines.yaml",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/paper_split.yaml",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/data_build",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results/temporal_baselines",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--normalization",
        choices=["zscore", "minmax"],
        default=None,
        help=(
            "Override the train-only scaler. The paper does not disclose this "
            "choice, so zscore/minmax comparisons must use separate output roots."
        ),
    )
    parser.add_argument(
        "--cam-channel-sizes",
        type=int,
        nargs=3,
        metavar=("C1", "C2", "C3"),
        default=None,
        help=(
            "Override the three unpublished CAM convolution widths. C3 must "
            "remain 1 to match Table 3."
        ),
    )
    parser.add_argument(
        "--cam-attention-update",
        choices=["replace", "residual", "final_residual", "skip_final"],
        default=None,
        help=(
            "Diagnostic CAM attention update. 'replace' is the literal paper "
            "reconstruction; the other modes test whether attention "
            "over-smoothing causes the observed one-channel collapse."
        ),
    )
    parser.add_argument(
        "--cam-attention-scaling",
        choices=["sqrt_dim", "none"],
        default=None,
        help=(
            "Diagnostic QK score scaling. The article defines Q/K/V attention "
            "but does not disclose this low-level denominator."
        ),
    )
    parser.add_argument(
        "--cam-temporal-layout",
        choices=["full_history_flat", "per_day_flat", "per_day_vectors"],
        default=None,
        help=(
            "Diagnostic interpretation of the paper's d_i x 24 CAM input: "
            "flatten all hours, apply CAM within each day then flatten, or "
            "send one 24-hour CAM vector per day to LSTM."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default=None,
        help=(
            "Override optimizer semantics. Adam uses coupled L2 weight decay; "
            "AdamW uses decoupled weight decay."
        ),
    )
    parser.add_argument(
        "--joint-weight-decay",
        type=float,
        default=None,
        help=(
            "Diagnostic override for MSNet/MSCMNet weight decay. The paper "
            "value remains recorded in the model config."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the unpublished training batch size for diagnostics.",
    )
    parser.add_argument(
        "--correction-mode",
        choices=["direct", "residual"],
        default=None,
        help=(
            "Diagnostic FC1/FC2 composition. 'direct' is the formal literal "
            "reconstruction; 'residual' adds each correction to its input."
        ),
    )
    parser.add_argument(
        "--zero-init-correction",
        action="store_true",
        help=(
            "Zero-initialize the final FC1/FC2 layers. This is a diagnostic "
            "stabilizer intended for residual correction only."
        ),
    )
    parser.add_argument(
        "--fc2-share-supervision-weight",
        type=float,
        default=None,
        help=(
            "Diagnostic MSE weight for the FC2 daily DMA-share target. The "
            "paper does not disclose an auxiliary-loss coefficient."
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--minimum-free-gib", type=float, default=8.0)
    parser.add_argument("--minimum-disk-gib", type=float, default=5.0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = read_yaml(args.config.resolve())
    if args.normalization is not None:
        config["training"]["normalization"] = args.normalization
    if args.cam_channel_sizes is not None:
        if args.cam_channel_sizes[-1] != 1:
            raise ValueError("Table 3 requires the final CAM channel size to be 1.")
        config["cam"]["channel_sizes"] = list(args.cam_channel_sizes)
    if args.cam_attention_update is not None:
        config["cam"]["attention_update"] = args.cam_attention_update
    if args.cam_attention_scaling is not None:
        config["cam"]["attention_scaling"] = args.cam_attention_scaling
    if args.cam_temporal_layout is not None:
        config["cam"]["temporal_layout"] = args.cam_temporal_layout
    if args.optimizer is not None:
        config["training"]["optimizer"] = args.optimizer
    if args.joint_weight_decay is not None:
        if args.joint_weight_decay < 0.0:
            raise ValueError("--joint-weight-decay must be non-negative.")
        config["training"]["joint_weight_decay_override"] = float(
            args.joint_weight_decay
        )
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive.")
        config["training"]["batch_size"] = int(args.batch_size)
    correction_models = ("mscmnet_m", "mscmnet_wm", "mscmnet_w")
    if args.correction_mode is not None:
        for model_name in correction_models:
            config["models"][model_name]["correction_mode"] = args.correction_mode
    if args.zero_init_correction:
        if args.correction_mode != "residual":
            raise ValueError(
                "--zero-init-correction requires --correction-mode residual."
            )
        for model_name in correction_models:
            config["models"][model_name]["zero_init_correction"] = True
    if args.fc2_share_supervision_weight is not None:
        if args.fc2_share_supervision_weight < 0.0:
            raise ValueError(
                "--fc2-share-supervision-weight must be non-negative."
            )
        for model_name in ("mscmnet_wm", "mscmnet_w"):
            config["models"][model_name]["fc2"]["share_supervision_weight"] = (
                float(args.fc2_share_supervision_weight)
            )
    requested = canonical_model_name(args.model)
    selected = list(CANONICAL_MODELS) if requested == "all" else [requested]
    seed = (
        int(args.seed)
        if args.seed is not None
        else int(config["training"]["seed"])
    )
    device = resolve_device(args.device, allow_cpu=bool(args.allow_cpu))
    preflight = preflight_resources(
        device=device,
        output_root=args.output_root.resolve(),
        minimum_free_gib=float(args.minimum_free_gib),
        minimum_disk_gib=float(args.minimum_disk_gib),
    )
    print(json.dumps({"resource_preflight": preflight}, indent=2), flush=True)
    frames, _, bounds = load_paper_data(
        data_dir=args.data_dir.resolve(),
        split_config_path=args.split_config.resolve(),
        require_audit=True,
    )
    # load_paper_data already fails closed; record the exact provenance here.
    from dma_wdf.data.mscmnet_dataset import validate_leakage_safe_data_build

    data_audit = validate_leakage_safe_data_build(
        args.data_dir.resolve(), expected_train_end=bounds["train_end"]
    )
    literature_path = PROJECT_ROOT / "configs/evaluation/mscmnet_literature_totals.yaml"
    literature = read_yaml(literature_path) if literature_path.is_file() else None

    statuses: list[dict[str, Any]] = []
    for canonical in selected:
        statuses.append(
            run_one_model(
                canonical=canonical,
                config=config,
                frames=frames,
                bounds=bounds,
                data_audit=data_audit,
                device=device,
                output_root=args.output_root.resolve(),
                seed=seed,
                overwrite=bool(args.overwrite),
                max_epochs_override=args.max_epochs,
                max_train_batches=args.max_train_batches,
                literature_config=literature,
                preflight=preflight,
            )
        )
    summary = {
        "status": "completed",
        "models": selected,
        "runs": statuses,
    }
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    (args.output_root.resolve() / "last_run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
