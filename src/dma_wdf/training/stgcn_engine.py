"""Leakage-safe training loop for the direct-decoding STGCN baseline.

The DCRNN training loop remains unchanged.  This module shares the established
data tensors, seed policy, checkpoint conventions, early-stopping behavior,
and TensorBoard scalar schema, while keeping model-specific forward behavior
separate: STGCN never receives labels as decoder inputs and never applies
scheduled sampling.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from dma_wdf.data.forecast_dataset import ForecastTrainingData
from dma_wdf.data.metrics import compute_metrics
from dma_wdf.models.stgcn import STGCN
from dma_wdf.training.engine import (
    ProgressCallback,
    TrainingResult,
    _import_summary_writer,
    _log_tensorboard_epoch,
    set_reproducible_seed,
)


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _loader(
    *,
    x_past: np.ndarray,
    y_scaled: np.ndarray,
    future_exog: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(
            torch.from_numpy(x_past),
            torch.from_numpy(y_scaled),
            torch.from_numpy(future_exog),
        ),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=bool(pin_memory),
        drop_last=False,
        generator=generator,
    )


def _move(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return tuple(
        value.to(device, non_blocking=non_blocking)
        for value in batch
    )  # type: ignore[return-value]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate(
    *,
    model: STGCN,
    loader: DataLoader,
    training_data: ForecastTrainingData,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    error_sum = 0.0
    count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            x_past, y_scaled, future_exog = _move(batch, device)
            output = model(
                x_past,
                x_future_exog=future_exog,
                teacher_forcing_ratio=0.0,
            )
            error_sum += float(torch.abs(output - y_scaled).sum())
            count += int(y_scaled.numel())
            predictions.append(
                training_data.demand_scaler.inverse_transform(
                    output.cpu().numpy()
                )
            )
            targets.append(
                training_data.demand_scaler.inverse_transform(
                    y_scaled.cpu().numpy()
                )
            )
    if count == 0:
        raise RuntimeError("Validation loader produced no values.")
    return (
        error_sum / count,
        compute_metrics(
            np.concatenate(targets, axis=0),
            np.concatenate(predictions, axis=0),
        ),
    )


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_history(
    history: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(
        output_dir / "history.csv",
        index=False,
    )


def _scalers_equal(
    current: dict[str, Any],
    saved: dict[str, Any],
) -> bool:
    return (
        current["feature_names"] == saved["feature_names"]
        and current["fit_start"] == saved["fit_start"]
        and current["fit_end"] == saved["fit_end"]
        and current["fit_rows"] == saved["fit_rows"]
        and np.array_equal(current["mean"], saved["mean"])
        and np.array_equal(current["std"], saved["std"])
    )


def _checkpoint(
    *,
    config: dict[str, Any],
    model: STGCN,
    optimizer: torch.optim.Optimizer,
    training_data: ForecastTrainingData,
    seed: int,
    epoch: int,
    global_step: int,
    best_epoch: int,
    best_validation_loss: float,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
    runtime_metadata: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "dma_wdf_stgcn_training_checkpoint",
        "model_name": "stgcn",
        "task_name": str(config["task"]["name"]),
        "horizon": int(config["task"]["horizon"]),
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "epochs_without_improvement": int(
            epochs_without_improvement
        ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_metadata": model.model_metadata(),
        "demand_scaler": training_data.demand_scaler.state_dict(),
        "weather_scaler": training_data.weather_scaler.state_dict(),
        "input_feature_names": list(
            training_data.input_feature_names
        ),
        "future_exog_feature_names": list(
            training_data.future_exog_feature_names
        ),
        "dma_columns": list(training_data.dma_columns),
        "data_metadata": dict(training_data.metadata),
        "resolved_config": config,
        "resolved_config_sha256": _config_hash(config),
        "runtime_metadata": runtime_metadata,
        "history": history,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None
            ),
        },
    }


def _restore(
    *,
    path: Path,
    model: STGCN,
    optimizer: torch.optim.Optimizer,
    training_data: ForecastTrainingData,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    saved = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    expected = {
        "kind": "dma_wdf_stgcn_training_checkpoint",
        "model_name": "stgcn",
        "task_name": str(config["task"]["name"]),
        "horizon": int(config["task"]["horizon"]),
        "seed": int(seed),
        "resolved_config_sha256": _config_hash(config),
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(
                f"Resume checkpoint mismatch for {key}: "
                f"{saved.get(key)!r} != {value!r}."
            )
    current_graph = model.model_metadata()["graph"]
    saved_graph = saved["model_metadata"]["graph"]
    if (
        current_graph["artifact_sha256"]
        != saved_graph["artifact_sha256"]
    ):
        raise ValueError("Resume checkpoint graph hash mismatch.")
    if not _scalers_equal(
        training_data.demand_scaler.state_dict(),
        saved["demand_scaler"],
    ):
        raise ValueError("Resume checkpoint demand scaler mismatch.")
    if not _scalers_equal(
        training_data.weather_scaler.state_dict(),
        saved["weather_scaler"],
    ):
        raise ValueError("Resume checkpoint weather scaler mismatch.")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    state = saved["rng_state"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if state["torch_cuda"] is None:
            raise ValueError(
                "CUDA resume checkpoint has no CUDA RNG state."
            )
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    return saved


def train_stgcn(
    *,
    config: dict[str, Any],
    model: STGCN,
    training_data: ForecastTrainingData,
    device: torch.device,
    output_dir: Path,
    seed: int,
    runtime_metadata: dict[str, Any] | None = None,
    resume_checkpoint: Path | None = None,
    max_epochs_override: int | None = None,
    max_train_batches: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Train STGCN without ever constructing or reading test tensors."""
    training = config["training"]
    if str(training["loss"]).lower() != "mae":
        raise ValueError("Only MAE training loss is supported.")
    if str(training["optimizer"]).lower() != "adam":
        raise ValueError("Only Adam optimizer is supported.")
    if "scheduled_sampling" in training:
        raise ValueError(
            "STGCN is direct-decoding and must not configure "
            "scheduled_sampling."
        )

    set_reproducible_seed(seed)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    validation_loader = _loader(
        x_past=training_data.validation.x_past,
        y_scaled=training_data.validation.y_scaled,
        future_exog=training_data.validation.future_exog,
        batch_size=int(
            training.get(
                "validation_batch_size",
                training["batch_size"],
            )
        ),
        shuffle=False,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    max_epochs = int(
        training["max_epochs"]
        if max_epochs_override is None
        else max_epochs_override
    )
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    patience = int(training["early_stopping_patience"])
    if patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")
    min_delta = float(training.get("early_stopping_min_delta", 0.0))
    gradient_clip = float(training["gradient_clip_norm"])
    if min_delta < 0.0 or gradient_clip <= 0.0:
        raise ValueError("Invalid early-stopping or gradient-clip value.")

    start_epoch = 1
    global_step = 0
    best_epoch = 0
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        restored = _restore(
            path=resume_checkpoint.resolve(),
            model=model,
            optimizer=optimizer,
            training_data=training_data,
            config=config,
            seed=seed,
            device=device,
        )
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
        best_epoch = int(restored["best_epoch"])
        best_loss = float(restored["best_validation_loss"])
        stale_epochs = int(restored["epochs_without_improvement"])
        history = list(restored["history"])

    best_path = output_dir / "checkpoint_best.pt"
    last_path = output_dir / "checkpoint_last.pt"
    runtime_metadata = dict(runtime_metadata or {})
    tensorboard = dict(training.get("tensorboard", {}))
    writer: Any | None = None
    log_dir: Path | None = None
    flush_every = int(tensorboard.get("flush_every_epochs", 1))
    if bool(tensorboard.get("enabled", True)):
        log_dir = output_dir / str(
            tensorboard.get("subdir", "tensorboard")
        )
        SummaryWriter = _import_summary_writer()
        writer = SummaryWriter(
            log_dir=str(log_dir),
            purge_step=start_epoch if start_epoch > 1 else None,
        )
        if start_epoch == 1:
            writer.add_text(
                "run/resolved_config",
                "```json\n"
                + json.dumps(
                    config,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n```",
                0,
            )

    stopped_early = False
    for epoch in range(start_epoch, max_epochs + 1):
        _synchronize(device)
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_loader = _loader(
            x_past=training_data.fit.x_past,
            y_scaled=training_data.fit.y_scaled,
            future_exog=training_data.fit.future_exog,
            batch_size=int(training["batch_size"]),
            shuffle=bool(training["shuffle_train_samples"]),
            seed=int(seed) + int(epoch),
            pin_memory=device.type == "cuda",
        )
        model.train()
        loss_sum = 0.0
        value_count = 0
        gradient_sum = 0.0
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            if (
                max_train_batches is not None
                and batch_index >= int(max_train_batches)
            ):
                break
            x_past, y_scaled, future_exog = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                x_past,
                x_future_exog=future_exog,
                teacher_forcing_ratio=0.0,
            )
            loss = F.l1_loss(prediction, y_scaled)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, "
                    f"batch={batch_index}."
                )
            loss.backward()
            gradient = clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            )
            if not torch.isfinite(gradient):
                raise FloatingPointError("Non-finite gradient norm.")
            optimizer.step()
            loss_sum += (
                float(loss.detach().item())
                * int(y_scaled.numel())
            )
            value_count += int(y_scaled.numel())
            gradient_sum += float(
                gradient.detach().item()
            )
            batches += 1
            global_step += 1
        if batches == 0 or value_count == 0:
            raise RuntimeError("Training loader produced no batches.")
        _synchronize(device)
        train_seconds = time.perf_counter() - train_start

        validation_start = time.perf_counter()
        validation_loss, metrics = _validate(
            model=model,
            loader=validation_loader,
            training_data=training_data,
            device=device,
        )
        _synchronize(device)
        validation_seconds = (
            time.perf_counter() - validation_start
        )
        epoch_seconds = time.perf_counter() - epoch_start
        improved = validation_loss < best_loss - min_delta
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_normalized_mae": loss_sum / value_count,
            "validation_loss_normalized_mae": validation_loss,
            "validation_MAE": metrics["MAE"],
            "validation_MAPE": metrics["MAPE"],
            "validation_RMSE": metrics["RMSE"],
            "validation_NSE": metrics["NSE"],
            "teacher_forcing_ratio_mean": 0.0,
            "gradient_norm_mean_before_clip": gradient_sum / batches,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "epoch_compute_seconds": epoch_seconds,
            "validation_fraction_of_compute": (
                validation_seconds / epoch_seconds
                if epoch_seconds > 0.0
                else 0.0
            ),
            "improved": improved,
            "epochs_without_improvement": stale_epochs,
        }
        history.append(row)
        payload = _checkpoint(
            config=config,
            model=model,
            optimizer=optimizer,
            training_data=training_data,
            seed=seed,
            epoch=epoch,
            global_step=global_step,
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            epochs_without_improvement=stale_epochs,
            history=history,
            runtime_metadata=runtime_metadata,
            device=device,
        )
        _atomic_save(payload, last_path)
        if improved:
            _atomic_save(payload, best_path)
        _write_history(history, output_dir)
        if writer is not None:
            _log_tensorboard_epoch(writer, row)
            if epoch % flush_every == 0:
                writer.flush()
        if progress_callback is not None:
            progress_callback(dict(row))
        if stale_epochs >= patience:
            stopped_early = True
            break

    if writer is not None:
        writer.flush()
        writer.close()
    if not best_path.is_file():
        raise RuntimeError("Training completed without a best checkpoint.")
    keep_last = bool(
        config.get("output", {}).get(
            "keep_last_checkpoint_after_training",
            False,
        )
    )
    retained_last: Path | None
    if keep_last:
        retained_last = last_path
    else:
        last_path.unlink(missing_ok=True)
        retained_last = None
    summary = {
        "status": "completed",
        "model": "stgcn",
        "task": str(config["task"]["name"]),
        "horizon": int(config["task"]["horizon"]),
        "seed": int(seed),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "global_step": global_step,
        "best_checkpoint": str(best_path),
        "last_checkpoint": (
            None if retained_last is None else str(retained_last)
        ),
        "last_checkpoint_removed_after_success": (
            retained_last is None
        ),
        "tensorboard_log_dir": (
            None if log_dir is None else str(log_dir)
        ),
        "test_data_used": False,
        "teacher_forcing_used": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return TrainingResult(
        output_dir=output_dir,
        best_checkpoint=best_path,
        last_checkpoint=retained_last,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        stopped_early=stopped_early,
        epochs_completed=len(history),
        global_step=global_step,
        history=tuple(history),
    )
