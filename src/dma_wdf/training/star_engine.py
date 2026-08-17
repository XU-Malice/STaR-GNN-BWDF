"""Leakage-safe training engine for the additive STaR-DCRNN model."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from dma_wdf.data.dcrnn_dataset import DCRNNTrainingData
from dma_wdf.models.star_dcrnn import STaRDCRNN, STaRForwardDetails
from dma_wdf.training.engine import (
    TrainingResult,
    _atomic_torch_save,
    _canonical_config_hash,
    _import_summary_writer,
    _log_tensorboard_epoch,
    _make_loader,
    _move_batch,
    _scaler_states_match,
    _synchronize_device,
    _validate_epoch,
    _write_history,
    inverse_sigmoid_ratio,
    set_reproducible_seed,
)


ProgressCallback = Callable[[dict[str, Any]], None]
CHECKPOINT_KIND = "dma_wdf_star_dcrnn_training_checkpoint"


def _build_optimizer(
    *,
    model: STaRDCRNN,
    learning_rate: float,
    weight_decay: float,
    alpha_weight_decay: float,
) -> torch.optim.Adam:
    """Keep convex-weight logits free from shrinkage toward alpha=0.5."""
    alpha_parameters = model.sasr_parameters()
    alpha_ids = {id(parameter) for parameter in alpha_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in alpha_ids
    ]
    groups: list[dict[str, Any]] = [
        {
            "params": backbone_parameters,
            "weight_decay": float(weight_decay),
            "name": "model",
        }
    ]
    if alpha_parameters:
        groups.append(
            {
                "params": alpha_parameters,
                "weight_decay": float(alpha_weight_decay),
                "name": "sasr_alpha",
            }
        )
    return torch.optim.Adam(groups, lr=float(learning_rate))


def _checkpoint_payload(
    *,
    config: dict[str, Any],
    model: STaRDCRNN,
    optimizer: torch.optim.Optimizer,
    training_data: DCRNNTrainingData,
    variant: str,
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
        "kind": CHECKPOINT_KIND,
        "model_name": "star_dcrnn",
        "variant": str(variant),
        "task_name": str(config["task"]["name"]),
        "horizon": int(config["task"]["horizon"]),
        "seed": int(seed),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "epochs_without_improvement": int(epochs_without_improvement),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_metadata": model.model_metadata(),
        "demand_scaler": training_data.demand_scaler.state_dict(),
        "weather_scaler": training_data.weather_scaler.state_dict(),
        "input_feature_names": list(training_data.input_feature_names),
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
    model: STaRDCRNN,
    optimizer: torch.optim.Optimizer,
    training_data: DCRNNTrainingData,
    config: dict[str, Any],
    variant: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected = {
        "kind": CHECKPOINT_KIND,
        "model_name": "star_dcrnn",
        "variant": str(variant),
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
    current_graph = model.model_metadata()["graph"]["artifact_sha256"]
    saved_graph = checkpoint["model_metadata"]["graph"]["artifact_sha256"]
    if current_graph != saved_graph:
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
    random.setstate(checkpoint["rng_state"]["python"])
    np.random.set_state(checkpoint["rng_state"]["numpy"])
    torch.set_rng_state(
        checkpoint["rng_state"]["torch_cpu"].detach().cpu()
    )
    if device.type == "cuda":
        cuda_state = checkpoint["rng_state"]["torch_cuda"]
        if cuda_state is None:
            raise ValueError("CUDA resume checkpoint has no CUDA RNG state.")
        torch.cuda.set_rng_state(cuda_state.detach().cpu(), device)
    return checkpoint


def _mechanism_scalars(
    details: STaRForwardDetails,
) -> dict[str, float]:
    values: dict[str, float] = {}
    state = details.dssn_sasr
    if state is not None:
        values["alpha_mean_average"] = float(
            state.alpha_mean.detach().mean().item()
        )
        values["alpha_std_average"] = float(
            state.alpha_std.detach().mean().item()
        )
        values["future_std_average"] = float(
            state.future_std.detach().mean().item()
        )
    readout = details.fa_dpr
    if readout is not None:
        weights = readout.attention_weights.detach().clamp_min(1.0e-12)
        entropy = -(weights * torch.log(weights)).sum(dim=-1)
        entropy = entropy / np.log(weights.shape[-1])
        values["attention_entropy_normalized"] = float(
            entropy.mean().item()
        )
        values["attention_last_day"] = float(weights[..., -1].mean().item())
        values["attention_one_week_lag"] = float(
            weights[..., -8].mean().item()
        )
        values["attention_two_week_lag"] = float(
            weights[..., -15].mean().item()
        )
        values["gate_average"] = float(readout.gate.detach().mean().item())
    return values


def train_star_dcrnn(
    *,
    config: dict[str, Any],
    model: STaRDCRNN,
    training_data: DCRNNTrainingData,
    variant: str,
    device: torch.device,
    output_dir: Path,
    seed: int,
    runtime_metadata: dict[str, Any] | None = None,
    resume_checkpoint: Path | None = None,
    max_epochs_override: int | None = None,
    max_train_batches: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Train one predeclared 2x2 STaR variant without reading test data."""
    training = config["training"]
    if str(training["loss"]).lower() != "mae":
        raise ValueError("Only MAE forecast loss is supported.")
    if str(training["optimizer"]).lower() != "adam":
        raise ValueError("Only Adam is supported.")
    schedule = training["scheduled_sampling"]
    if str(schedule["schedule"]).lower() != "inverse_sigmoid":
        raise ValueError("Only inverse-sigmoid sampling is supported.")
    if model.variant != variant:
        raise ValueError(
            f"Model variant {model.variant!r} != requested {variant!r}."
        )

    set_reproducible_seed(seed)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    state_config = config["innovation"]["dssn_sasr"]
    state_loss_weight = (
        float(state_config["state_loss_weight"])
        if model.use_dssn_sasr
        else 0.0
    )
    if state_loss_weight < 0.0:
        raise ValueError("state_loss_weight must be non-negative.")
    optimizer = _build_optimizer(
        model=model,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        alpha_weight_decay=float(state_config["alpha_weight_decay"]),
    )
    pin_memory = device.type == "cuda"
    validation_loader = _make_loader(
        x_past=training_data.validation.x_past,
        y_scaled=training_data.validation.y_scaled,
        future_exog=training_data.validation.future_exog,
        batch_size=int(
            training.get("validation_batch_size", training["batch_size"])
        ),
        shuffle=False,
        seed=seed,
        pin_memory=pin_memory,
    )
    max_epochs = int(
        training["max_epochs"]
        if max_epochs_override is None
        else max_epochs_override
    )
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    patience = int(training["early_stopping_patience"])
    min_delta = float(training.get("early_stopping_min_delta", 0.0))
    gradient_clip = float(training["gradient_clip_norm"])
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
            variant=variant,
            seed=seed,
            device=device,
        )
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
        best_epoch = int(restored["best_epoch"])
        best_validation_loss = float(restored["best_validation_loss"])
        epochs_without_improvement = int(
            restored["epochs_without_improvement"]
        )
        history = list(restored["history"])

    best_path = output_dir / "checkpoint_best.pt"
    last_path = output_dir / "checkpoint_last.pt"
    runtime_metadata = dict(runtime_metadata or {})
    tensorboard_config = dict(training.get("tensorboard", {}))
    tensorboard_writer: Any | None = None
    tensorboard_log_dir: Path | None = None
    if bool(tensorboard_config.get("enabled", True)):
        tensorboard_log_dir = output_dir / str(
            tensorboard_config.get("subdir", "tensorboard")
        )
        SummaryWriter = _import_summary_writer()
        tensorboard_writer = SummaryWriter(
            log_dir=str(tensorboard_log_dir),
            purge_step=start_epoch if start_epoch > 1 else None,
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
            tensorboard_writer.add_text("run/variant", variant, 0)

    stopped_early = False
    for epoch in range(start_epoch, max_epochs + 1):
        _synchronize_device(device)
        epoch_start = time.perf_counter()
        train_loader = _make_loader(
            x_past=training_data.fit.x_past,
            y_scaled=training_data.fit.y_scaled,
            future_exog=training_data.fit.future_exog,
            batch_size=int(training["batch_size"]),
            shuffle=bool(training["shuffle_train_samples"]),
            seed=int(seed) + int(epoch),
            pin_memory=pin_memory,
        )
        model.train()
        forecast_sum = 0.0
        state_sum = 0.0
        state_mean_sum = 0.0
        state_log_std_sum = 0.0
        total_sum = 0.0
        value_count = 0
        gradient_sum = 0.0
        teacher_sum = 0.0
        batches = 0
        mechanism_sums: dict[str, float] = {}

        train_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            if (
                max_train_batches is not None
                and batch_index >= int(max_train_batches)
            ):
                break
            x_past, y_scaled, future_exog = _move_batch(batch, device=device)
            teacher_ratio = inverse_sigmoid_ratio(global_step, decay_steps)
            optimizer.zero_grad(set_to_none=True)
            details = model(
                x_past,
                y_target=y_scaled,
                x_future_exog=future_exog,
                teacher_forcing_ratio=teacher_ratio,
                return_details=True,
            )
            if not isinstance(details, STaRForwardDetails):
                raise TypeError("STaR model did not return forward details.")
            forecast_loss = F.l1_loss(details.prediction, y_scaled)
            state_loss = forecast_loss.new_zeros(())
            state_components = {
                "state_mean_mae": forecast_loss.new_zeros(()),
                "state_log_std_mae": forecast_loss.new_zeros(()),
            }
            if details.dssn_sasr is not None:
                state_loss, state_components = model.state_supervision_loss(
                    target=y_scaled,
                    state=details.dssn_sasr,
                )
            total_loss = forecast_loss + state_loss_weight * state_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_index}."
                )
            total_loss.backward()
            gradient = clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            )
            if not torch.isfinite(gradient):
                raise FloatingPointError("Non-finite gradient norm.")
            optimizer.step()

            count = int(y_scaled.numel())
            forecast_sum += float(forecast_loss.detach().item()) * count
            state_sum += float(state_loss.detach().item()) * count
            state_mean_sum += float(
                state_components["state_mean_mae"].detach().item()
            ) * count
            state_log_std_sum += float(
                state_components["state_log_std_mae"].detach().item()
            ) * count
            total_sum += float(total_loss.detach().item()) * count
            value_count += count
            gradient_sum += float(gradient.detach().item())
            teacher_sum += teacher_ratio
            batches += 1
            global_step += 1
            for name, value in _mechanism_scalars(details).items():
                mechanism_sums[name] = mechanism_sums.get(name, 0.0) + value
        if batches == 0 or value_count == 0:
            raise RuntimeError("Training loader produced no batches.")

        _synchronize_device(device)
        train_seconds = time.perf_counter() - train_start
        validation_start = time.perf_counter()
        validation_loss, validation_metrics = _validate_epoch(
            model=model,
            loader=validation_loader,
            training_data=training_data,
            device=device,
        )
        _synchronize_device(device)
        validation_seconds = time.perf_counter() - validation_start
        epoch_seconds = time.perf_counter() - epoch_start
        improved = validation_loss < best_validation_loss - min_delta
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "variant": variant,
            "train_loss_normalized_mae": forecast_sum / value_count,
            "train_state_loss": state_sum / value_count,
            "train_state_mean_mae": state_mean_sum / value_count,
            "train_state_log_std_mae": state_log_std_sum / value_count,
            "train_total_loss": total_sum / value_count,
            "state_loss_weight": state_loss_weight,
            "validation_loss_normalized_mae": validation_loss,
            "validation_MAE": validation_metrics["MAE"],
            "validation_MAPE": validation_metrics["MAPE"],
            "validation_RMSE": validation_metrics["RMSE"],
            "validation_NSE": validation_metrics["NSE"],
            "teacher_forcing_ratio_mean": teacher_sum / batches,
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
            "epochs_without_improvement": epochs_without_improvement,
        }
        for name, value in mechanism_sums.items():
            row[f"mechanism_{name}"] = value / batches
        history.append(row)
        payload = _checkpoint_payload(
            config=config,
            model=model,
            optimizer=optimizer,
            training_data=training_data,
            variant=variant,
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
            tensorboard_writer.add_scalar(
                "loss/train_total", row["train_total_loss"], epoch
            )
            tensorboard_writer.add_scalar(
                "loss/train_state", row["train_state_loss"], epoch
            )
            for key, value in row.items():
                if key.startswith("mechanism_"):
                    tensorboard_writer.add_scalar(
                        f"mechanism/{key.removeprefix('mechanism_')}",
                        value,
                        epoch,
                    )
            tensorboard_writer.flush()
        if progress_callback is not None:
            progress_callback(dict(row))
        if epochs_without_improvement >= patience:
            stopped_early = True
            break

    if tensorboard_writer is not None:
        tensorboard_writer.close()
    if not best_path.is_file():
        raise RuntimeError("Training completed without a best checkpoint.")
    keep_last = bool(
        config["output"].get("keep_last_checkpoint_after_training", False)
    )
    retained_last: Path | None = last_path if keep_last else None
    if not keep_last:
        last_path.unlink(missing_ok=True)

    summary = {
        "status": "completed",
        "model": "star_dcrnn",
        "variant": variant,
        "task": str(config["task"]["name"]),
        "horizon": int(config["task"]["horizon"]),
        "seed": int(seed),
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "stopped_early": bool(stopped_early),
        "epochs_completed": int(history[-1]["epoch"]),
        "global_step": int(global_step),
        "parameters_total": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "parameters_trainable": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "training_compute_seconds": float(
            sum(float(row["epoch_compute_seconds"]) for row in history)
        ),
        "average_epoch_seconds": float(
            np.mean([float(row["epoch_compute_seconds"]) for row in history])
        ),
        "peak_cuda_memory_mib": (
            None
            if device.type != "cuda"
            else float(torch.cuda.max_memory_allocated(device) / 2**20)
        ),
        "best_checkpoint": str(best_path),
        "last_checkpoint": (
            None if retained_last is None else str(retained_last)
        ),
        "last_checkpoint_removed_after_success": not keep_last,
        "tensorboard_log_dir": (
            None if tensorboard_log_dir is None else str(tensorboard_log_dir)
        ),
        "test_data_used": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    return TrainingResult(
        output_dir=output_dir,
        best_checkpoint=best_path,
        last_checkpoint=retained_last,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
        epochs_completed=int(history[-1]["epoch"]),
        global_step=global_step,
        history=tuple(history),
    )
