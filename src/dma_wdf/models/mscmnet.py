"""Que et al. (2024) temporal baselines for multi-DMA forecasting.

This module implements the six neural-network families compared in
"Water demand forecasting in multiple district metered areas based on a
multi-scale correction module neural network architecture":

* independent GRU and LSTM models (one model per DMA);
* MSNet with ten jointly-trained CNN-Attention-LSTM branches;
* MSCMNet_M with the forecast-day meteorological/temporal FC1 correction;
* MSCMNet_WM (called MSCMNet_MW in parts of the article) with FC1 and FC2;
* MSCMNet_W, which removes meteorological inputs but retains temporal FC1
  inputs and the demand-share FC2 correction.

The article specifies the data flow and the Hyperopt-selected recurrent and
fully-connected dimensions, but does not publish every low-level framework
choice.  Figure 7 does, however, fix the CAM ordering: Conv1d and Attention
alternate three times.  Table 3 fixes a one-channel CAM output before LSTM but
does not disclose the two intermediate convolution widths.  The formal
reconstruction treats FC1/FC2 as direct corrections; an explicitly labelled
residual mode is retained only for diagnostics because the article does not
publish the low-level composition operator.  No target or future demand is
accepted by a forward method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


PAPER_DAY_HOURS = 24
PAPER_NUM_DMAS = 10


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}.")
    return value


def _attention_update(value: str) -> str:
    value = str(value).lower()
    valid = {"replace", "residual", "final_residual", "skip_final"}
    if value not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"attention_update must be one of {choices}, got {value!r}.")
    return value


def _attention_scaling(value: str) -> str:
    value = str(value).lower()
    valid = {"sqrt_dim", "none"}
    if value not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"attention_scaling must be one of {choices}, got {value!r}.")
    return value


def _temporal_layout(value: str) -> str:
    value = str(value).lower()
    valid = {"full_history_flat", "per_day_flat", "per_day_vectors"}
    if value not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"temporal_layout must be one of {choices}, got {value!r}.")
    return value


def _correction_mode(value: str) -> str:
    value = str(value).lower()
    valid = {"direct", "residual"}
    if value not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"correction_mode must be one of {choices}, got {value!r}.")
    return value


@dataclass(frozen=True)
class ForecastBranchConfig:
    """Paper-selected settings for one DMA forecast branch."""

    input_features: int
    input_weeks: int
    lstm_layers: int
    hidden_size: int

    def __post_init__(self) -> None:
        _positive_int("input_features", self.input_features)
        _positive_int("input_weeks", self.input_weeks)
        _positive_int("lstm_layers", self.lstm_layers)
        _positive_int("hidden_size", self.hidden_size)

    @property
    def history_days(self) -> int:
        return int(self.input_weeks) * 7


@dataclass(frozen=True)
class MSCMNetOutput:
    """Forecast and auditable intermediate correction outputs."""

    prediction: torch.Tensor
    msnet_prediction: torch.Tensor
    fc1_prediction: torch.Tensor | None = None
    predicted_daily_share: torch.Tensor | None = None


class StackedRecurrentForecaster(nn.Module):
    """Demand-only GRU/LSTM forecaster used independently for one DMA."""

    def __init__(
        self,
        *,
        cell_type: str,
        hidden_sizes: Sequence[int],
        horizon: int = PAPER_DAY_HOURS,
        input_features: int = 1,
    ) -> None:
        super().__init__()
        cell_type = str(cell_type).upper()
        if cell_type not in {"GRU", "LSTM"}:
            raise ValueError("cell_type must be 'GRU' or 'LSTM'.")
        sizes = tuple(_positive_int("hidden_size", value) for value in hidden_sizes)
        if not sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")
        self.cell_type = cell_type
        self.input_features = _positive_int("input_features", input_features)
        self.horizon = _positive_int("horizon", horizon)

        recurrent = nn.GRU if cell_type == "GRU" else nn.LSTM
        layers: list[nn.Module] = []
        layer_input = self.input_features
        for hidden_size in sizes:
            layers.append(
                recurrent(
                    input_size=layer_input,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
            )
            layer_input = hidden_size
        self.recurrent_layers = nn.ModuleList(layers)
        self.output = nn.Linear(layer_input, self.horizon)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self.input_features:
            raise ValueError(
                "history must have shape (B,T,"
                f"{self.input_features}), got {tuple(history.shape)}."
            )
        if not torch.is_floating_point(history):
            raise TypeError("history must be a floating-point tensor.")
        encoded = history
        for recurrent in self.recurrent_layers:
            encoded, _ = recurrent(encoded)
        return self.output(encoded[:, -1, :])


class GRUForecast(StackedRecurrentForecaster):
    """Independent GRU baseline for one DMA."""

    def __init__(self, hidden_sizes: Sequence[int], horizon: int = PAPER_DAY_HOURS) -> None:
        super().__init__(cell_type="GRU", hidden_sizes=hidden_sizes, horizon=horizon)


class LSTMForecast(StackedRecurrentForecaster):
    """Independent LSTM baseline for one DMA."""

    def __init__(self, hidden_sizes: Sequence[int], horizon: int = PAPER_DAY_HOURS) -> None:
        super().__init__(cell_type="LSTM", hidden_sizes=hidden_sizes, horizon=horizon)


class ScaledDotProductSelfAttention(nn.Module):
    """Single-head Q/K/V attention with an auditable score scaling choice."""

    def __init__(
        self,
        features: int,
        dropout: float = 0.0,
        scaling: str = "sqrt_dim",
    ) -> None:
        super().__init__()
        self.features = _positive_int("features", features)
        self.scaling = _attention_scaling(scaling)
        self.query = nn.Linear(self.features, self.features, bias=False)
        self.key = nn.Linear(self.features, self.features, bias=False)
        self.value = nn.Linear(self.features, self.features, bias=False)
        self.dropout = nn.Dropout(_probability("dropout", dropout))
        self.scale = (
            float(self.features) ** -0.5 if self.scaling == "sqrt_dim" else 1.0
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.features:
            raise ValueError(
                "attention input must have shape (B,T,"
                f"{self.features}), got {tuple(sequence.shape)}."
            )
        query = self.query(sequence)
        key = self.key(sequence)
        value = self.value(sequence)
        weights = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) * self.scale,
            dim=-1,
        )
        return torch.matmul(self.dropout(weights), value)


class ConvAttentionBlock(nn.Module):
    """Paper CAM: three alternating Conv1d/ReLU and Attention stages."""

    def __init__(
        self,
        *,
        input_features: int,
        channel_sizes: Sequence[int] = (16, 16, 1),
        cnn_layers: int = 3,
        attention_layers: int = 3,
        kernel_size: int = 3,
        attention_heads: int = 1,
        dropout: float = 0.0,
        attention_update: str = "replace",
        attention_scaling: str = "sqrt_dim",
    ) -> None:
        super().__init__()
        input_features = _positive_int("input_features", input_features)
        cnn_layers = _positive_int("cnn_layers", cnn_layers)
        attention_layers = _positive_int("attention_layers", attention_layers)
        kernel_size = _positive_int("kernel_size", kernel_size)
        attention_heads = _positive_int("attention_heads", attention_heads)
        dropout = _probability("dropout", dropout)
        self.attention_update = _attention_update(attention_update)
        self.attention_scaling = _attention_scaling(attention_scaling)
        channels = tuple(
            _positive_int("channel_size", value) for value in channel_sizes
        )
        if len(channels) != cnn_layers:
            raise ValueError(
                "channel_sizes must provide one output size per CNN layer."
            )
        if attention_layers != cnn_layers:
            raise ValueError(
                "Figure 7 requires one Attention stage after every Conv1d stage."
            )
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")
        if attention_heads != 1:
            raise ValueError(
                "The paper defines single scaled dot-product attention, not "
                "multi-head attention."
            )

        self.output_features = channels[-1]
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        in_channels = input_features
        for out_channels in channels:
            self.convolutions.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.attention.append(
                ScaledDotProductSelfAttention(
                    out_channels,
                    dropout=dropout,
                    scaling=self.attention_scaling,
                )
            )
            in_channels = out_channels
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError(
                "CAM input must have shape (B,T,F), got "
                f"{tuple(sequence.shape)}."
            )
        encoded = sequence
        final_index = len(self.convolutions) - 1
        for index, (convolution, attention) in enumerate(
            zip(self.convolutions, self.attention)
        ):
            encoded = convolution(encoded.transpose(1, 2)).transpose(1, 2)
            encoded = self.dropout(self.activation(encoded))
            if self.attention_update == "skip_final" and index == final_index:
                continue
            attended = attention(encoded)
            if self.attention_update == "residual" or (
                self.attention_update == "final_residual" and index == final_index
            ):
                encoded = encoded + attended
            else:
                encoded = attended
        return encoded


class CAMLSTMForecastBranch(nn.Module):
    """One paper Forecast module: 1D CNN + Attention + LSTM."""

    def __init__(
        self,
        config: ForecastBranchConfig,
        *,
        channel_sizes: Sequence[int] = (16, 16, 1),
        cnn_layers: int = 3,
        attention_layers: int = 3,
        kernel_size: int = 3,
        attention_heads: int = 1,
        dropout: float = 0.0,
        attention_update: str = "replace",
        attention_scaling: str = "sqrt_dim",
        temporal_layout: str = "full_history_flat",
        horizon: int = PAPER_DAY_HOURS,
    ) -> None:
        super().__init__()
        self.config = config
        self.horizon = _positive_int("horizon", horizon)
        self.temporal_layout = _temporal_layout(temporal_layout)
        dropout = _probability("dropout", dropout)
        self.cam = ConvAttentionBlock(
            input_features=config.input_features,
            channel_sizes=channel_sizes,
            cnn_layers=cnn_layers,
            attention_layers=attention_layers,
            kernel_size=kernel_size,
            attention_heads=attention_heads,
            dropout=dropout,
            attention_update=attention_update,
            attention_scaling=attention_scaling,
        )
        lstm_input_size = self.cam.output_features
        if self.temporal_layout == "per_day_vectors":
            lstm_input_size *= PAPER_DAY_HOURS
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=dropout if config.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_size, self.horizon)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        expected = (self.config.history_days, PAPER_DAY_HOURS, self.config.input_features)
        if history.ndim != 4 or tuple(history.shape[1:]) != expected:
            raise ValueError(
                f"branch history must have shape (B,{expected[0]},"
                f"{expected[1]},{expected[2]}), got {tuple(history.shape)}."
            )
        batch_size = history.shape[0]
        if self.temporal_layout == "full_history_flat":
            sequence = history.reshape(
                batch_size,
                self.config.history_days * PAPER_DAY_HOURS,
                self.config.input_features,
            )
            encoded = self.cam(sequence)
        else:
            daily = history.reshape(
                batch_size * self.config.history_days,
                PAPER_DAY_HOURS,
                self.config.input_features,
            )
            daily = self.cam(daily)
            if self.temporal_layout == "per_day_flat":
                encoded = daily.reshape(
                    batch_size,
                    self.config.history_days * PAPER_DAY_HOURS,
                    self.cam.output_features,
                )
            else:
                encoded = daily.reshape(
                    batch_size,
                    self.config.history_days,
                    PAPER_DAY_HOURS * self.cam.output_features,
                )
        recurrent, _ = self.lstm(encoded)
        return self.output(recurrent[:, -1, :])


class MSNet(nn.Module):
    """Joint ten-DMA MSNet forecast trunk."""

    def __init__(
        self,
        branch_configs: Sequence[ForecastBranchConfig],
        *,
        channel_sizes: Sequence[int] = (16, 16, 1),
        cnn_layers: int = 3,
        attention_layers: int = 3,
        kernel_size: int = 3,
        attention_heads: int = 1,
        dropout: float = 0.0,
        attention_update: str = "replace",
        attention_scaling: str = "sqrt_dim",
        temporal_layout: str = "full_history_flat",
        horizon: int = PAPER_DAY_HOURS,
    ) -> None:
        super().__init__()
        configs = tuple(branch_configs)
        if len(configs) != PAPER_NUM_DMAS:
            raise ValueError(
                f"MSNet requires {PAPER_NUM_DMAS} DMA branches, got {len(configs)}."
            )
        self.horizon = _positive_int("horizon", horizon)
        self.num_dmas = len(configs)
        self.branches = nn.ModuleList(
            [
                CAMLSTMForecastBranch(
                    config,
                    channel_sizes=channel_sizes,
                    cnn_layers=cnn_layers,
                    attention_layers=attention_layers,
                    kernel_size=kernel_size,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_update=attention_update,
                    attention_scaling=attention_scaling,
                    temporal_layout=temporal_layout,
                    horizon=self.horizon,
                )
                for config in configs
            ]
        )
        joined = self.horizon * self.num_dmas
        self.joint_fully_connected = nn.Linear(joined, joined)

    def forward(self, histories: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(histories) != self.num_dmas:
            raise ValueError(
                f"Expected {self.num_dmas} branch histories, got {len(histories)}."
            )
        branch_predictions = [
            branch(history) for branch, history in zip(self.branches, histories)
        ]
        concatenated = torch.stack(branch_predictions, dim=-1)
        corrected = self.joint_fully_connected(
            concatenated.reshape(concatenated.shape[0], -1)
        )
        return corrected.reshape(
            concatenated.shape[0], self.horizon, self.num_dmas
        )


class FullyConnectedCorrection(nn.Module):
    """Direct fully-connected correction used by FC1 and FC2."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        input_size = _positive_int("input_size", input_size)
        hidden_size = _positive_int("hidden_size", hidden_size)
        output_size = _positive_int("output_size", output_size)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(_probability("dropout", dropout)),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("correction features must have shape (B,F).")
        return self.network(features)


class DailyShareForecaster(nn.Module):
    """FC2 CAM-LSTM that predicts the next-day share of every DMA."""

    def __init__(
        self,
        *,
        input_features: int,
        cam_channel_sizes: Sequence[int],
        cam_kernel_size: int,
        cam_dropout: float,
        hidden_size: int,
        lstm_layers: int,
        fully_connected_nodes: int,
        dropout: float,
        num_dmas: int = PAPER_NUM_DMAS,
    ) -> None:
        super().__init__()
        input_features = _positive_int("input_features", input_features)
        hidden_size = _positive_int("hidden_size", hidden_size)
        lstm_layers = _positive_int("lstm_layers", lstm_layers)
        fully_connected_nodes = _positive_int(
            "fully_connected_nodes", fully_connected_nodes
        )
        dropout = _probability("dropout", dropout)
        self.input_features = input_features
        self.num_dmas = _positive_int("num_dmas", num_dmas)
        self.cam = ConvAttentionBlock(
            input_features=input_features,
            channel_sizes=cam_channel_sizes,
            cnn_layers=len(tuple(cam_channel_sizes)),
            attention_layers=len(tuple(cam_channel_sizes)),
            kernel_size=cam_kernel_size,
            attention_heads=1,
            dropout=cam_dropout,
        )
        self.lstm = nn.LSTM(
            input_size=self.cam.output_features,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size, fully_connected_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fully_connected_nodes, self.num_dmas),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self.input_features:
            raise ValueError(
                "FC2 history must have shape (B,D,"
                f"{self.input_features}), got {tuple(history.shape)}."
            )
        encoded = self.cam(history)
        _, (hidden, _) = self.lstm(encoded)
        return torch.softmax(self.output(hidden[-1]), dim=-1)


class MSCMNetM(nn.Module):
    """MSCMNet_M: MSNet plus forecast-day weather/time FC1."""

    def __init__(
        self,
        msnet: MSNet,
        *,
        future_features: int,
        fc1_nodes: int,
        fc1_dropout: float,
        correction_mode: str = "direct",
        zero_init_correction: bool = False,
    ) -> None:
        super().__init__()
        self.msnet = msnet
        self.future_features = _positive_int("future_features", future_features)
        self.correction_mode = _correction_mode(correction_mode)
        output_size = msnet.horizon * msnet.num_dmas
        self.fc1 = FullyConnectedCorrection(
            input_size=output_size + msnet.horizon * self.future_features,
            hidden_size=fc1_nodes,
            output_size=output_size,
            dropout=fc1_dropout,
        )
        if bool(zero_init_correction):
            output = self.fc1.network[-1]
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def _validate_future(self, future: torch.Tensor) -> None:
        expected = (self.msnet.horizon, self.future_features)
        if future.ndim != 3 or tuple(future.shape[1:]) != expected:
            raise ValueError(
                f"future_features must have shape (B,{expected[0]},"
                f"{expected[1]}), got {tuple(future.shape)}."
            )

    def forward(
        self,
        histories: Sequence[torch.Tensor],
        future_features: torch.Tensor,
    ) -> MSCMNetOutput:
        self._validate_future(future_features)
        msnet_prediction = self.msnet(histories)
        correction_features = torch.cat(
            [
                msnet_prediction.reshape(msnet_prediction.shape[0], -1),
                future_features.reshape(future_features.shape[0], -1),
            ],
            dim=1,
        )
        corrected = self.fc1(correction_features).reshape_as(msnet_prediction)
        if self.correction_mode == "residual":
            corrected = msnet_prediction + corrected
        return MSCMNetOutput(
            prediction=corrected,
            msnet_prediction=msnet_prediction,
            fc1_prediction=corrected,
        )


class MSCMNetWM(MSCMNetM):
    """MSCMNet_WM/MW: FC1 followed by daily-share/temperature FC2."""

    def __init__(
        self,
        msnet: MSNet,
        *,
        future_features: int,
        fc1_nodes: int,
        fc1_dropout: float,
        fc2_input_features: int,
        fc2_cam_channel_sizes: Sequence[int],
        fc2_cam_kernel_size: int,
        fc2_cam_dropout: float,
        fc2_hidden_size: int,
        fc2_lstm_layers: int,
        fc2_nodes: int,
        fc2_dropout: float,
        correction_mode: str = "direct",
        zero_init_correction: bool = False,
    ) -> None:
        super().__init__(
            msnet,
            future_features=future_features,
            fc1_nodes=fc1_nodes,
            fc1_dropout=fc1_dropout,
            correction_mode=correction_mode,
            zero_init_correction=zero_init_correction,
        )
        self.share_forecaster = DailyShareForecaster(
            input_features=fc2_input_features,
            cam_channel_sizes=fc2_cam_channel_sizes,
            cam_kernel_size=fc2_cam_kernel_size,
            cam_dropout=fc2_cam_dropout,
            hidden_size=fc2_hidden_size,
            lstm_layers=fc2_lstm_layers,
            fully_connected_nodes=fc2_nodes,
            dropout=fc2_dropout,
            num_dmas=msnet.num_dmas,
        )
        output_size = msnet.horizon * msnet.num_dmas
        self.fc2 = FullyConnectedCorrection(
            input_size=output_size + msnet.num_dmas,
            hidden_size=fc2_nodes,
            output_size=output_size,
            dropout=fc2_dropout,
        )
        if bool(zero_init_correction):
            output = self.fc2.network[-1]
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(
        self,
        histories: Sequence[torch.Tensor],
        future_features: torch.Tensor,
        fc2_history: torch.Tensor,
    ) -> MSCMNetOutput:
        fc1_output = super().forward(histories, future_features)
        predicted_share = self.share_forecaster(fc2_history)
        correction_features = torch.cat(
            [
                fc1_output.prediction.reshape(fc1_output.prediction.shape[0], -1),
                predicted_share,
            ],
            dim=1,
        )
        corrected = self.fc2(correction_features).reshape_as(fc1_output.prediction)
        if self.correction_mode == "residual":
            corrected = fc1_output.prediction + corrected
        return MSCMNetOutput(
            prediction=corrected,
            msnet_prediction=fc1_output.msnet_prediction,
            fc1_prediction=fc1_output.prediction,
            predicted_daily_share=predicted_share,
        )


class MSCMNetW(MSCMNetWM):
    """Meteorology-removed MSCMNet_W (temporal FC1 + share-only FC2)."""


# The article body commonly says MW; its supplementary metric tables say WM.
MSCMNetMW = MSCMNetWM


def build_msnet_from_config(
    model_config: Mapping[str, Any],
    cam_config: Mapping[str, Any],
) -> MSNet:
    """Construct an MSNet trunk from the paper-parameter YAML mappings."""
    features = tuple(model_config["branch_features"])
    weeks = tuple(model_config["input_weeks"])
    layers = tuple(model_config["lstm_layers"])
    hidden = tuple(model_config["hidden_sizes"])
    lengths = {len(weeks), len(layers), len(hidden)}
    if lengths != {PAPER_NUM_DMAS}:
        raise ValueError(
            "input_weeks, lstm_layers and hidden_sizes must each contain "
            f"{PAPER_NUM_DMAS} entries."
        )
    branches = [
        ForecastBranchConfig(
            input_features=len(features),
            input_weeks=weeks[index],
            lstm_layers=layers[index],
            hidden_size=hidden[index],
        )
        for index in range(PAPER_NUM_DMAS)
    ]
    if bool(cam_config.get("pooling", False)):
        raise ValueError("The paper CAM explicitly does not use pooling.")
    convolution = str(cam_config.get("convolution", "conv1d")).lower()
    if convolution != "conv1d":
        raise ValueError("The paper CAM requires temporal Conv1d.")
    return MSNet(
        branches,
        channel_sizes=tuple(cam_config.get("channel_sizes", (16, 16, 1))),
        cnn_layers=int(cam_config.get("cnn_layers", 3)),
        attention_layers=int(cam_config.get("attention_layers", 3)),
        kernel_size=int(cam_config.get("kernel_size", 3)),
        attention_heads=int(cam_config.get("attention_heads", 1)),
        dropout=float(cam_config.get("dropout", 0.0)),
        attention_update=str(cam_config.get("attention_update", "replace")),
        attention_scaling=str(cam_config.get("attention_scaling", "sqrt_dim")),
        temporal_layout=str(
            cam_config.get("temporal_layout", "full_history_flat")
        ),
    )


def build_joint_model_from_config(
    model_name: str,
    model_config: Mapping[str, Any],
    cam_config: Mapping[str, Any],
) -> nn.Module:
    """Build MSNet or an MSCMNet variant from a validated config mapping."""
    canonical = str(model_name).lower()
    if canonical == "mscmnet_mw":
        canonical = "mscmnet_wm"
    trunk = build_msnet_from_config(model_config, cam_config)
    if canonical == "msnet":
        return trunk
    fc1 = model_config.get("fc1")
    if not isinstance(fc1, Mapping):
        raise ValueError(f"{canonical} requires an fc1 mapping.")
    common: dict[str, Any] = {
        "msnet": trunk,
        "future_features": len(tuple(fc1["future_features"])),
        "fc1_nodes": int(fc1["nodes"]),
        "fc1_dropout": float(fc1["dropout"]),
        "correction_mode": str(model_config.get("correction_mode", "direct")),
        "zero_init_correction": bool(
            model_config.get("zero_init_correction", False)
        ),
    }
    if canonical == "mscmnet_m":
        return MSCMNetM(**common)
    if canonical not in {"mscmnet_wm", "mscmnet_w"}:
        raise ValueError(f"Unsupported joint temporal model: {model_name!r}.")
    fc2 = model_config.get("fc2")
    if not isinstance(fc2, Mapping):
        raise ValueError(f"{canonical} requires an fc2 mapping.")
    full = {
        **common,
        "fc2_input_features": int(fc2["input_size"]),
        "fc2_cam_channel_sizes": tuple(
            fc2.get("cam_channel_sizes", cam_config.get("channel_sizes", (16, 16, 1)))
        ),
        "fc2_cam_kernel_size": int(cam_config.get("kernel_size", 3)),
        "fc2_cam_dropout": float(cam_config.get("dropout", 0.0)),
        "fc2_hidden_size": int(fc2["hidden_size"]),
        "fc2_lstm_layers": int(fc2["lstm_layers"]),
        "fc2_nodes": int(fc2["nodes"]),
        "fc2_dropout": float(fc2["dropout"]),
    }
    if canonical == "mscmnet_w":
        return MSCMNetW(**full)
    return MSCMNetWM(**full)


__all__ = [
    "CAMLSTMForecastBranch",
    "ConvAttentionBlock",
    "DailyShareForecaster",
    "ForecastBranchConfig",
    "GRUForecast",
    "LSTMForecast",
    "MSCMNetM",
    "MSCMNetMW",
    "MSCMNetOutput",
    "MSCMNetW",
    "MSCMNetWM",
    "MSNet",
    "PAPER_DAY_HOURS",
    "PAPER_NUM_DMAS",
    "ScaledDotProductSelfAttention",
    "StackedRecurrentForecaster",
    "build_joint_model_from_config",
    "build_msnet_from_config",
]
