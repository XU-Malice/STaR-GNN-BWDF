"""Leakage-safe DCRNN training engine with early stopping.

This module consumes :class:`DCRNNTrainingData`; it never constructs or reads
official test tensors.  Validation is always autoregressive
(``teacher_forcing_ratio=0``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from dma_wdf.data.dcrnn_dataset import DCRNNTrainingData
from dma_wdf.data.metrics import compute_metrics
from dma_wdf.models.dcrnn import DCRNN


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class TrainingResult:
    """Summary of one completed training run."""

    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path | None
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    epochs_completed: int
    global_step: int
    history: tuple[dict[str, Any], ...]

    def state_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "best_checkpoint": str(self.best_checkpoint),
            "last_checkpoint": (
                None
                if self.last_checkpoint is None
                else str(self.last_checkpoint)
            ),
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "stopped_early": self.stopped_early,
            "epochs_completed": self.epochs_completed,
            "global_step": self.global_step,
        }


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without assuming a CUDA device."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def inverse_sigmoid_ratio(global_step: int, decay_steps: int) -> float:
    """DCRNN inverse-sigmoid teacher-forcing schedule.

    ``epsilon_i = k / (k + exp(i / k))``.
    """
    global_step = int(global_step)
    decay_steps = int(decay_steps)
    if global_step < 0:
        raise ValueError("global_step must be non-negative.")
    if decay_steps <= 0:
        raise ValueError("decay_steps must be positive.")
    exponent = min(global_step / decay_steps, 60.0)
    return float(decay_steps / (decay_steps + math.exp(exponent)))


def _canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_loader(
    *,
    x_past: np.ndarray,
    y_scaled: np.ndarray,
    future_exog: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x_past),
        torch.from_numpy(y_scaled),
        torch.from_numpy(future_exog),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=bool(pin_memory),
        drop_last=False,
        generator=generator,
    )


def _move_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return tuple(
        tensor.to(device, non_blocking=non_blocking)
        for tensor in batch
    )  # type: ignore[return-value]


def _synchronize_device(device: torch.device) -> None:
    """Synchronize CUDA only when wall-clock timing requires it."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_epoch(
    *,
    model: DCRNN,
    loader: DataLoader,
    training_data: DCRNNTrainingData,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    absolute_error_sum = 0.0
    value_count = 0
    predictions_raw: list[np.ndarray] = []
    targets_raw: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            x_past, y_scaled, future_exog = _move_batch(
                batch,
                device=device,
            )
            prediction_scaled = model(
                x_past,
                x_future_exog=future_exog,
                teacher_forcing_ratio=0.0,
            )
            absolute_error_sum += float(
                torch.abs(prediction_scaled - y_scaled).sum().item()
            )
            value_count += int(y_scaled.numel())
            prediction_np = prediction_scaled.detach().cpu().numpy()
            target_np = y_scaled.detach().cpu().numpy()
            predictions_raw.append(
                training_data.demand_scaler.inverse_transform(
                    prediction_np
                )
            )
            targets_raw.append(
                training_data.demand_scaler.inverse_transform(target_np)
            )
    if value_count == 0:
        raise RuntimeError("Validation loader produced no values.")
    normalized_mae = absolute_error_sum / value_count
    raw_prediction = np.concatenate(predictions_raw, axis=0)
    raw_target = np.concatenate(targets_raw, axis=0)
    return float(normalized_mae), compute_metrics(
        raw_target,
        raw_prediction,
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


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


TENSORBOARD_SCALARS = {
    "loss/train_normalized_mae": "train_loss_normalized_mae",
    "loss/validation_normalized_mae": (
        "validation_loss_normalized_mae"
    ),
    "metrics/validation_MAE_Lps": "validation_MAE",
    "metrics/validation_MAPE_fraction": "validation_MAPE",
    "metrics/validation_RMSE_Lps": "validation_RMSE",
    "metrics/validation_NSE": "validation_NSE",
    "schedule/teacher_forcing_ratio": (
        "teacher_forcing_ratio_mean"
    ),
    "optimization/gradient_norm_before_clip": (
        "gradient_norm_mean_before_clip"
    ),
    "optimization/learning_rate": "learning_rate",
    "early_stopping/epochs_without_improvement": (
        "epochs_without_improvement"
    ),
    "timing/train_seconds": "train_seconds",
    "timing/validation_seconds": "validation_seconds",
    "timing/compute_seconds": "epoch_compute_seconds",
    "timing/validation_fraction": (
        "validation_fraction_of_compute"
    ),
}


def _import_summary_writer() -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "TensorBoard logging is enabled but tensorboard is not "
            "installed. Run: python -m pip install tensorboard"
        ) from exc
    return SummaryWriter


def _log_tensorboard_epoch(writer: Any, row: dict[str, Any]) -> None:
    epoch = int(row["epoch"])
    for tag, key in TENSORBOARD_SCALARS.items():
        writer.add_scalar(tag, float(row[key]), epoch)
    writer.add_scalar(
        "early_stopping/improved",
        int(bool(row["improved"])),
        epoch,
    )


def export_history_to_tensorboard(
    *,
    history: list[dict[str, Any]],
    log_dir: Path,
) -> Path:
    """Create TensorBoard events from an existing history list."""
    if not history:
        raise ValueError("Cannot export an empty training history.")
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    SummaryWriter = _import_summary_writer()
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for row in history:
            # Older runs predate timing instrumentation. Preserve every
            # available scalar rather than inventing timing values.
            epoch = int(row["epoch"])
            for tag, key in TENSORBOARD_SCALARS.items():
                if key in row:
                    writer.add_scalar(tag, float(row[key]), epoch)
            writer.add_scalar(
                "early_stopping/improved",
                int(bool(row["improved"])),
                epoch,
            )
        writer.flush()
    finally:
        writer.close()
    return log_dir


def _scaler_states_match(
    current: dict[str, Any],
    saved: dict[str, Any],
) -> bool:
    return (
        current["feature_names"] == saved["feature_names"]
        and current["fit_start"] == saved["fit_start"]
        and current["fit_end"] == saved["fit_end"]
        and current["fit_rows"] == saved["fit_rows"]
        and np.allclose(current["mean"], saved["mean"], atol=0.0, rtol=0.0)
        and np.allclose(current["std"], saved["std"], atol=0.0, rtol=0.0)
    )


def _checkpoint_payload(
    *,
    config: dict[str, Any],
    model: DCRNN,
    optimizer: torch.optim.Optimizer,
    training_data: DCRNNTrainingData,
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
        "kind": "dma_wdf_dcrnn_training_checkpoint",
        "model_name": "dcrnn",
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
        "resolved_config_sha256": _canonical_config_hash(config),
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


def _restore_checkpoint(
    *,
    path: Path,
    model: DCRNN,
    optimizer: torch.optim.Optimizer,
    training_data: DCRNNTrainingData,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    expected = {
        "kind": "dma_wdf_dcrnn_training_checkpoint",
        "model_name": "dcrnn",
        "task_name": str(config["task"]["name"]),
        "horizon": int(config["task"]["horizon"]),
        "seed": int(seed),
        "resolved_config_sha256": _canonical_config_hash(config),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Resume checkpoint mismatch for {key}: "
                f"{checkpoint.get(key)!r} != {value!r}."
            )
    if (
        checkpoint["model_metadata"]["graph"]["artifact_sha256"]
        != model.model_metadata()["graph"]["artifact_sha256"]
    ):
        raise ValueError("Resume checkpoint graph hash mismatch.")
    if not _scaler_states_match(
        training_data.demand_scaler.state_dict(),
        checkpoint["demand_scaler"],
    ):
        raise ValueError("Resume checkpoint demand scaler mismatch.")
    if not _scaler_states_match(
        training_data.weather_scaler.state_dict(),
        checkpoint["weather_scaler"],
    ):
        raise ValueError("Resume checkpoint weather scaler mismatch.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    rng_state = checkpoint["rng_state"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"].detach().cpu())
    if device.type == "cuda":
        if rng_state["torch_cuda"] is None:
            raise ValueError(
                "CUDA resume requested from a checkpoint without CUDA "
                "RNG state."
            )
        torch.cuda.set_rng_state(
            rng_state["torch_cuda"].detach().cpu(),
            device,
        )
    return checkpoint


def train_dcrnn(
    *,
    config: dict[str, Any],
    model: DCRNN,
    training_data: DCRNNTrainingData,
    device: torch.device,
    output_dir: Path,
    seed: int,
    runtime_metadata: dict[str, Any] | None = None,
    resume_checkpoint: Path | None = None,
    max_epochs_override: int | None = None,
    max_train_batches: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Train DCRNN, save best/last checkpoints, and return run metadata."""
    training_config = config["training"]
    if str(training_config["loss"]).lower() != "mae":
        raise ValueError("Only MAE training loss is supported.")
    if str(training_config["optimizer"]).lower() != "adam":
        raise ValueError("Only Adam optimizer is supported.")
    schedule = training_config["scheduled_sampling"]
    if str(schedule["schedule"]).lower() != "inverse_sigmoid":
        raise ValueError("Only inverse_sigmoid scheduled sampling is supported.")

    set_reproducible_seed(seed)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    pin_memory = device.type == "cuda"
    validation_loader = _make_loader(
        x_past=training_data.validation.x_past,
        y_scaled=training_data.validation.y_scaled,
        future_exog=training_data.validation.future_exog,
        batch_size=int(
            training_config.get(
                "validation_batch_size",
                training_config["batch_size"],
            )
        ),
        shuffle=False,
        seed=seed,
        pin_memory=pin_memory,
    )

    configured_epochs = int(training_config["max_epochs"])
    max_epochs = (
        configured_epochs
        if max_epochs_override is None
        else int(max_epochs_override)
    )
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    patience = int(training_config["early_stopping_patience"])
    if patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")
    min_delta = float(
        training_config.get("early_stopping_min_delta", 0.0)
    )
    if min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative.")
    gradient_clip = float(training_config["gradient_clip_norm"])
    if gradient_clip <= 0.0:
        raise ValueError("gradient_clip_norm must be positive.")
    validation_batch_size = int(
        training_config.get(
            "validation_batch_size",
            training_config["batch_size"],
        )
    )
    if validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be positive.")
    decay_steps = int(schedule["cl_decay_steps"])

    start_epoch = 1
    global_step = 0
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        restored = _restore_checkpoint(
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
        best_validation_loss = float(
            restored["best_validation_loss"]
        )
        epochs_without_improvement = int(
            restored["epochs_without_improvement"]
        )
        history = list(restored["history"])

    best_path = output_dir / "checkpoint_best.pt"
    last_path = output_dir / "checkpoint_last.pt"
    stopped_early = False
    runtime_metadata = dict(runtime_metadata or {})
    tensorboard_config = dict(
        training_config.get("tensorboard", {})
    )
    tensorboard_enabled = bool(
        tensorboard_config.get("enabled", True)
    )
    tensorboard_writer: Any | None = None
    tensorboard_log_dir: Path | None = None
    tensorboard_flush_every = int(
        tensorboard_config.get("flush_every_epochs", 1)
    )
    if tensorboard_flush_every <= 0:
        raise ValueError(
            "tensorboard.flush_every_epochs must be positive."
        )
    if tensorboard_enabled:
        tensorboard_log_dir = (
            output_dir
            / str(tensorboard_config.get("subdir", "tensorboard"))
        )
        SummaryWriter = _import_summary_writer()
        tensorboard_writer = SummaryWriter(
            log_dir=str(tensorboard_log_dir),
            purge_step=(
                start_epoch if start_epoch > 1 else None
            ),
        )
        if start_epoch == 1:
            tensorboard_writer.add_text(
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

    for epoch in range(start_epoch, max_epochs + 1):
        _synchronize_device(device)
        epoch_compute_start = time.perf_counter()
        train_start_time = time.perf_counter()
        # Epoch-specific shuffle seeds make sample order reproducible after
        # resuming without serialising DataLoader internals.
        train_loader = _make_loader(
            x_past=training_data.fit.x_past,
            y_scaled=training_data.fit.y_scaled,
            future_exog=training_data.fit.future_exog,
            batch_size=int(training_config["batch_size"]),
            shuffle=bool(training_config["shuffle_train_samples"]),
            seed=int(seed) + int(epoch),
            pin_memory=pin_memory,
        )
        model.train()
        loss_sum = 0.0
        value_count = 0
        gradient_norm_sum = 0.0
        teacher_ratio_sum = 0.0
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            if (
                max_train_batches is not None
                and batch_index >= int(max_train_batches)
            ):
                break
            x_past, y_scaled, future_exog = _move_batch(
                batch,
                device=device,
            )
            teacher_ratio = inverse_sigmoid_ratio(
                global_step,
                decay_steps,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction_scaled = model(
                x_past,
                y_target=y_scaled,
                x_future_exog=future_exog,
                teacher_forcing_ratio=teacher_ratio,
            )
            loss = F.l1_loss(
                prediction_scaled,
                y_scaled,
                reduction="mean",
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, "
                    f"batch={batch_index}."
                )
            loss.backward()
            gradient_norm = clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch={epoch}, "
                    f"batch={batch_index}."
                )
            optimizer.step()
            loss_sum += float(loss.item()) * int(y_scaled.numel())
            value_count += int(y_scaled.numel())
            gradient_norm_sum += float(gradient_norm.item())
            teacher_ratio_sum += teacher_ratio
            batches += 1
            global_step += 1
        if batches == 0 or value_count == 0:
            raise RuntimeError("Training loader produced no batches.")

        _synchronize_device(device)
        train_loss = loss_sum / value_count
        train_seconds = time.perf_counter() - train_start_time
        validation_start_time = time.perf_counter()
        validation_loss, validation_metrics = _validate_epoch(
            model=model,
            loader=validation_loader,
            training_data=training_data,
            device=device,
        )
        _synchronize_device(device)
        validation_seconds = (
            time.perf_counter() - validation_start_time
        )
        epoch_compute_seconds = (
            time.perf_counter() - epoch_compute_start
        )
        improved = (
            validation_loss
            < best_validation_loss - min_delta
        )
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_normalized_mae": train_loss,
            "validation_loss_normalized_mae": validation_loss,
            "validation_MAE": validation_metrics["MAE"],
            "validation_MAPE": validation_metrics["MAPE"],
            "validation_RMSE": validation_metrics["RMSE"],
            "validation_NSE": validation_metrics["NSE"],
            "teacher_forcing_ratio_mean": (
                teacher_ratio_sum / batches
            ),
            "gradient_norm_mean_before_clip": (
                gradient_norm_sum / batches
            ),
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "epoch_compute_seconds": epoch_compute_seconds,
            "validation_fraction_of_compute": (
                validation_seconds / epoch_compute_seconds
                if epoch_compute_seconds > 0.0
                else 0.0
            ),
            "improved": improved,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
        }
        history.append(row)
        payload = _checkpoint_payload(
            config=config,
            model=model,
            optimizer=optimizer,
            training_data=training_data,
            seed=seed,
            epoch=epoch,
            global_step=global_step,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            runtime_metadata=runtime_metadata,
            device=device,
        )
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, best_path)
        _write_history(history, output_dir)
        if tensorboard_writer is not None:
            _log_tensorboard_epoch(tensorboard_writer, row)
            if epoch % tensorboard_flush_every == 0:
                tensorboard_writer.flush()
        if progress_callback is not None:
            progress_callback(dict(row))

        if epochs_without_improvement >= patience:
            stopped_early = True
            break

    if tensorboard_writer is not None:
        tensorboard_writer.flush()
        tensorboard_writer.close()
    if not best_path.is_file():
        raise RuntimeError("Training completed without a best checkpoint.")
    keep_last_checkpoint = bool(
        config.get("output", {}).get(
            "keep_last_checkpoint_after_training",
            False,
        )
    )
    retained_last_path: Path | None
    if keep_last_checkpoint:
        retained_last_path = last_path
    else:
        last_path.unlink(missing_ok=True)
        retained_last_path = None
    summary = {
        "status": "completed",
        "task": str(config["task"]["name"]),
        "horizon": int(config["task"]["horizon"]),
        "seed": int(seed),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "global_step": global_step,
        "best_checkpoint": str(best_path),
        "last_checkpoint": (
            None
            if retained_last_path is None
            else str(retained_last_path)
        ),
        "last_checkpoint_removed_after_success": (
            retained_last_path is None
        ),
        "tensorboard_log_dir": (
            None
            if tensorboard_log_dir is None
            else str(tensorboard_log_dir)
        ),
        "test_data_used": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return TrainingResult(
        output_dir=output_dir,
        best_checkpoint=best_path,
        last_checkpoint=retained_last_path,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
        epochs_completed=len(history),
        global_step=global_step,
        history=tuple(history),
    )
