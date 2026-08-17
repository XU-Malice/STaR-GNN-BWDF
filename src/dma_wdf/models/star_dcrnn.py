"""STaR-DCRNN: slice normalization, restoration, and daily retrieval.

This file is additive: the sealed DCRNN implementation is imported and left
unchanged.  With both innovations disabled, :class:`STaRDCRNN` calls the
original :class:`~dma_wdf.models.dcrnn.DCRNN.forward` method directly.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dma_wdf.models.dcrnn import DCRNN, build_dcrnn_model
from dma_wdf.models.star_components import (
    DSSNSASR,
    DSSNSASROutput,
    FADPROutput,
    ForecastAlignedDailyPatternRetrieval,
)


VARIANT_FEATURES: dict[str, tuple[bool, bool]] = {
    "backbone": (False, False),
    "dssn_sasr": (True, False),
    "fa_dpr": (False, True),
    "full": (True, True),
}
FORMAL_VARIANTS = ("backbone", "dssn_sasr", "fa_dpr", "full")


@dataclass(frozen=True)
class STaRForwardDetails:
    """Prediction plus mechanism outputs used by training and diagnostics."""

    prediction: torch.Tensor
    decoder_prediction: torch.Tensor
    dssn_sasr: DSSNSASROutput | None
    fa_dpr: FADPROutput | None


class STaRDCRNN(DCRNN):
    """DCRNN with optional DSSN-SASR and FA-DPR innovations."""

    model_name = "star_dcrnn"

    def __init__(
        self,
        *,
        random_walk: torch.Tensor,
        input_dim: int,
        hidden_dim: int = 32,
        horizon: int = 24,
        history_hours: int = 672,
        num_nodes: int = 10,
        num_rnn_layers: int = 1,
        max_diffusion_step: int = 2,
        future_exog_dim: int = 0,
        dropout: float = 0.0,
        graph_metadata: dict[str, Any] | None = None,
        use_dssn_sasr: bool = True,
        use_fa_dpr: bool = True,
        slice_hours: int = 24,
        state_epsilon: float = 1.0e-5,
        initial_alpha: float = 0.5,
        attention_dim: int = 16,
        attention_dropout: float = 0.0,
        gate_bias: float = -2.0,
        condition_on_future_calendar: bool = True,
    ) -> None:
        super().__init__(
            random_walk=random_walk,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
            max_diffusion_step=max_diffusion_step,
            future_exog_dim=future_exog_dim,
            dropout=dropout,
            graph_metadata=graph_metadata,
        )
        self.history_hours = int(history_hours)
        self.use_dssn_sasr = bool(use_dssn_sasr)
        self.use_fa_dpr = bool(use_fa_dpr)
        self.slice_hours = int(slice_hours)
        self.condition_on_future_calendar = bool(
            condition_on_future_calendar
        )

        self.dssn_sasr: DSSNSASR | None = None
        if self.use_dssn_sasr:
            self.dssn_sasr = DSSNSASR(
                num_nodes=self.num_nodes,
                history_hours=self.history_hours,
                horizon=self.horizon,
                slice_hours=self.slice_hours,
                epsilon=float(state_epsilon),
                initial_alpha=float(initial_alpha),
            )

        self.fa_dpr: ForecastAlignedDailyPatternRetrieval | None = None
        if self.use_fa_dpr:
            future_context_dim = (
                self.future_exog_dim
                if self.condition_on_future_calendar
                else 0
            )
            self.fa_dpr = ForecastAlignedDailyPatternRetrieval(
                hidden_dim=self.hidden_dim,
                history_hours=self.history_hours,
                patch_length=self.slice_hours,
                attention_dim=int(attention_dim),
                future_context_dim=future_context_dim,
                dropout=float(attention_dropout),
                gate_bias=float(gate_bias),
            )

    @property
    def variant(self) -> str:
        flags = (self.use_dssn_sasr, self.use_fa_dpr)
        for name, value in VARIANT_FEATURES.items():
            if value == flags:
                return name
        raise RuntimeError(f"Unsupported STaR feature flags: {flags}.")

    def _validate_star_input(self, x_past: torch.Tensor) -> None:
        expected = (self.history_hours, self.num_nodes, self.input_dim)
        if x_past.ndim != 4 or tuple(x_past.shape[1:]) != expected:
            raise ValueError(
                f"x_past must have shape (B,{expected[0]},"
                f"{expected[1]},{expected[2]}), got "
                f"{tuple(x_past.shape)}."
            )
        if not torch.is_floating_point(x_past):
            raise TypeError("x_past must be a floating-point tensor.")
        if not torch.isfinite(x_past).all():
            raise ValueError("x_past contains NaN/Inf.")

    def _encode_with_sequence(
        self,
        x_past: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the unchanged DCGRU cells and retain top-layer states."""
        batch_size, sequence_length, num_nodes, _ = x_past.shape
        states = [
            x_past.new_zeros(batch_size, num_nodes, self.hidden_dim)
            for _ in range(self.num_rnn_layers)
        ]
        top_sequence: list[torch.Tensor] = []
        for step in range(sequence_length):
            layer_input = x_past[:, step]
            next_states: list[torch.Tensor] = []
            for layer_index, cell in enumerate(self.encoder.layers):
                if layer_index > 0:
                    layer_input = self.encoder.dropout(layer_input)
                state = cell(
                    layer_input,
                    states[layer_index],
                    self.random_walk,
                )
                next_states.append(state)
                layer_input = state
            states = next_states
            top_sequence.append(states[-1])
        return torch.stack(states, dim=0), torch.stack(top_sequence, dim=1)

    def load_dcrnn_backbone(self, dcrnn: DCRNN) -> None:
        """Copy only sealed DCRNN parameters for identity/warm-start tests."""
        if (
            dcrnn.input_dim != self.input_dim
            or dcrnn.hidden_dim != self.hidden_dim
            or dcrnn.horizon != self.horizon
            or dcrnn.num_nodes != self.num_nodes
            or dcrnn.num_rnn_layers != self.num_rnn_layers
        ):
            raise ValueError("DCRNN backbone architecture mismatch.")
        if not torch.equal(dcrnn.random_walk, self.random_walk):
            raise ValueError("DCRNN backbone graph mismatch.")
        self.encoder.load_state_dict(dcrnn.encoder.state_dict(), strict=True)
        self.decoder.load_state_dict(dcrnn.decoder.state_dict(), strict=True)

    def sasr_parameters(self) -> list[nn.Parameter]:
        if self.dssn_sasr is None:
            return []
        return list(self.dssn_sasr.restorer.parameters())

    def _decode_with_fa_dpr(
        self,
        encoder_hidden: torch.Tensor,
        hidden_sequence: torch.Tensor,
        *,
        target: torch.Tensor | None,
        future_exog: torch.Tensor | None,
        teacher_forcing_ratio: float,
    ) -> tuple[torch.Tensor, FADPROutput]:
        """Run the original decoder recurrence with step-wise FA-DPR.

        Only the top-layer state is retrieval-conditioned.  DCGRU cells,
        output projection, autoregressive inputs, and the official scheduled
        sampling rule are the original decoder modules and semantics.
        """
        if self.fa_dpr is None:
            raise RuntimeError("FA-DPR decoding requires an FA-DPR module.")
        ratio = float(teacher_forcing_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1].")
        if encoder_hidden.ndim != 4:
            raise ValueError(
                "encoder_hidden must have shape (layers,B,N,H)."
            )
        if encoder_hidden.shape[0] != self.num_rnn_layers:
            raise ValueError("encoder_hidden layer count mismatch.")

        batch_size = encoder_hidden.shape[1]
        num_nodes = encoder_hidden.shape[2]
        if target is not None:
            expected_target = (batch_size, self.horizon, num_nodes)
            if tuple(target.shape) != expected_target:
                raise ValueError(
                    f"target must have shape {expected_target}, got "
                    f"{tuple(target.shape)}."
                )
        if self.training and ratio > 0.0 and target is None:
            raise ValueError(
                "target is required during training when "
                "teacher_forcing_ratio > 0."
            )
        if self.future_exog_dim > 0:
            expected_exog = (
                batch_size,
                self.horizon,
                num_nodes,
                self.future_exog_dim,
            )
            if future_exog is None:
                raise ValueError(
                    "future_exog is required because future_exog_dim="
                    f"{self.future_exog_dim}."
                )
            if tuple(future_exog.shape) != expected_exog:
                raise ValueError(
                    f"future_exog must have shape {expected_exog}, got "
                    f"{tuple(future_exog.shape)}."
                )
        elif future_exog is not None and future_exog.shape[-1] != 0:
            raise ValueError(
                "future_exog was supplied although future_exog_dim=0."
            )

        memory = self.fa_dpr.build_memory(hidden_sequence)
        states = [
            encoder_hidden[layer]
            for layer in range(self.num_rnn_layers)
        ]
        previous_demand = encoder_hidden.new_zeros(batch_size, num_nodes)
        predictions: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        fused_hidden: list[torch.Tensor] = []

        for step in range(self.horizon):
            future_context = None
            if self.fa_dpr.future_context_dim > 0:
                assert future_exog is not None
                future_context = future_exog[:, step]
            retrieval = self.fa_dpr.attend(
                memory,
                states[-1],
                future_context=future_context,
            )
            states[-1] = retrieval.fused_hidden
            weights.append(retrieval.attention_weights)
            gates.append(retrieval.gate)
            fused_hidden.append(retrieval.fused_hidden)

            layer_input = previous_demand.unsqueeze(-1)
            if future_exog is not None:
                layer_input = torch.cat(
                    [layer_input, future_exog[:, step]],
                    dim=-1,
                )
            next_states: list[torch.Tensor] = []
            for layer_index, cell in enumerate(self.decoder.layers):
                if layer_index > 0:
                    layer_input = self.decoder.dropout(layer_input)
                state = cell(
                    layer_input,
                    states[layer_index],
                    self.random_walk,
                )
                next_states.append(state)
                layer_input = state
            states = next_states

            prediction = self.decoder.output_projection(
                layer_input
            ).squeeze(-1)
            predictions.append(prediction)
            if self.training and target is not None and ratio > 0.0:
                if ratio >= 1.0:
                    previous_demand = target[:, step]
                else:
                    teacher_mask = (
                        torch.rand((), device=prediction.device) < ratio
                    )
                    previous_demand = torch.where(
                        teacher_mask,
                        target[:, step],
                        prediction,
                    )
            else:
                previous_demand = prediction

        return torch.stack(predictions, dim=1), FADPROutput(
            fused_hidden=torch.stack(fused_hidden, dim=1),
            attention_weights=torch.stack(weights, dim=1),
            gate=torch.stack(gates, dim=1),
            daily_tokens=memory.daily_tokens,
        )

    def state_supervision_loss(
        self,
        *,
        target: torch.Tensor,
        state: DSSNSASROutput,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute target-only training supervision outside model forward."""
        if self.dssn_sasr is None:
            raise RuntimeError("State supervision requires DSSN-SASR.")
        true_mean, true_log_std = self.dssn_sasr.target_daily_statistics(
            target
        )
        mean_loss = F.l1_loss(state.future_mean_daily, true_mean)
        log_std_loss = F.l1_loss(
            state.future_log_std_daily,
            true_log_std,
        )
        return mean_loss + log_std_loss, {
            "state_mean_mae": mean_loss,
            "state_log_std_mae": log_std_loss,
        }

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        y_target: torch.Tensor | None = None,
        x_future_exog: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        return_details: bool = False,
    ) -> torch.Tensor | STaRForwardDetails:
        # This exact direct path is the strict DCRNN degeneration contract.
        if not self.use_dssn_sasr and not self.use_fa_dpr:
            prediction = super().forward(
                x_past,
                y_target=y_target,
                x_future_exog=x_future_exog,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            if return_details:
                return STaRForwardDetails(
                    prediction=prediction,
                    decoder_prediction=prediction,
                    dssn_sasr=None,
                    fa_dpr=None,
                )
            return prediction

        self._validate_star_input(x_past)
        ratio = float(teacher_forcing_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1].")
        if not self.training:
            y_target = None
            ratio = 0.0

        state_output: DSSNSASROutput | None = None
        model_input = x_past
        decoder_target = y_target
        if self.dssn_sasr is not None:
            state_output = self.dssn_sasr(x_past[..., 0])
            model_input = torch.cat(
                [
                    state_output.normalized_history.unsqueeze(-1),
                    x_past[..., 1:],
                ],
                dim=-1,
            )
            if y_target is not None:
                # Scheduled-sampling targets must not provide a route by
                # which alpha parameters can manipulate decoder inputs.
                decoder_target = (
                    (y_target - state_output.future_mean)
                    / state_output.future_std
                ).detach()

        readout_output: FADPROutput | None = None
        hidden_sequence: torch.Tensor | None = None
        if self.fa_dpr is None:
            encoder_hidden = self.encoder(model_input, self.random_walk)
        else:
            encoder_hidden, hidden_sequence = self._encode_with_sequence(
                model_input
            )
        if self.fa_dpr is None:
            decoder_prediction = self.decoder(
                encoder_hidden,
                self.random_walk,
                target=decoder_target,
                future_exog=x_future_exog,
                teacher_forcing_ratio=ratio,
            )
        else:
            assert hidden_sequence is not None
            decoder_prediction, readout_output = self._decode_with_fa_dpr(
                encoder_hidden,
                hidden_sequence,
                target=decoder_target,
                future_exog=x_future_exog,
                teacher_forcing_ratio=ratio,
            )
        prediction = decoder_prediction
        if state_output is not None:
            prediction = (
                decoder_prediction * state_output.future_std
                + state_output.future_mean
            )

        if return_details:
            return STaRForwardDetails(
                prediction=prediction,
                decoder_prediction=decoder_prediction,
                dssn_sasr=state_output,
                fa_dpr=readout_output,
            )
        return prediction

    def model_metadata(self) -> dict[str, Any]:
        metadata = super().model_metadata()
        metadata.update(
            {
                "model_name": self.model_name,
                "variant": self.variant,
                "history_hours": self.history_hours,
                "dssn_sasr": {
                    "enabled": self.use_dssn_sasr,
                    "slice_hours": self.slice_hours,
                    "future_state_inputs": "history_only",
                    "parameters_per_dma": 2,
                },
                "fa_dpr": {
                    "enabled": self.use_fa_dpr,
                    "patch_length": self.slice_hours,
                    "num_patches": self.history_hours // self.slice_hours,
                    "attention_dim": (
                        None
                        if self.fa_dpr is None
                        else self.fa_dpr.attention_dim
                    ),
                    "num_heads": 1,
                    "query_source": "previous_decoder_state",
                    "forecast_conditioning": (
                        "future_calendar"
                        if self.condition_on_future_calendar
                        else "none"
                    ),
                    "retrieval_frequency": "every_decoder_step",
                },
            }
        )
        return metadata


def build_star_dcrnn_model(
    config: dict[str, Any],
    *,
    project_root: Path,
    input_dim: int,
    future_exog_dim: int,
    horizon: int,
    history_hours: int,
    variant: str,
    device: torch.device | str | None = None,
) -> STaRDCRNN:
    """Build STaR-DCRNN from the same graph contract as sealed DCRNN."""
    if variant not in VARIANT_FEATURES:
        raise ValueError(
            f"Unknown variant {variant!r}; expected "
            f"{sorted(VARIANT_FEATURES)}."
        )
    if str(config["model"]["name"]).lower() != "star_dcrnn":
        raise ValueError("STaR config must use model.name='star_dcrnn'.")
    if int(history_hours) != int(config["task"]["history_hours"]):
        raise ValueError("history_hours does not match task config.")

    # The sealed builder remains the single graph-contract authority.
    dcrnn_config = deepcopy(config)
    dcrnn_config["model"]["name"] = "dcrnn"
    base = build_dcrnn_model(
        dcrnn_config,
        project_root=project_root,
        input_dim=input_dim,
        future_exog_dim=future_exog_dim,
        horizon=horizon,
        device=None,
    )
    innovation = config["innovation"]
    state_config = innovation["dssn_sasr"]
    readout_config = innovation["fa_dpr"]
    if int(state_config["slice_hours"]) != int(
        readout_config["patch_length"]
    ):
        raise ValueError(
            "State slices and FA-DPR patches must have equal length."
        )
    if int(readout_config["num_heads"]) != 1:
        raise ValueError("The registered STaR protocol uses one attention head.")
    use_dssn_sasr, use_fa_dpr = VARIANT_FEATURES[variant]
    model = STaRDCRNN(
        random_walk=base.random_walk,
        input_dim=input_dim,
        hidden_dim=base.hidden_dim,
        horizon=horizon,
        history_hours=history_hours,
        num_nodes=base.num_nodes,
        num_rnn_layers=base.num_rnn_layers,
        max_diffusion_step=base.max_diffusion_step,
        future_exog_dim=future_exog_dim,
        dropout=float(config["model"].get("dropout", 0.0)),
        graph_metadata=base.graph_metadata,
        use_dssn_sasr=use_dssn_sasr,
        use_fa_dpr=use_fa_dpr,
        slice_hours=int(state_config["slice_hours"]),
        state_epsilon=float(state_config["epsilon"]),
        initial_alpha=float(state_config["initial_alpha"]),
        attention_dim=int(readout_config["attention_dim"]),
        attention_dropout=float(readout_config["dropout"]),
        gate_bias=float(readout_config["gate_bias"]),
        condition_on_future_calendar=bool(
            readout_config["condition_on_future_calendar"]
        ),
    )
    model.load_dcrnn_backbone(base)
    if device is not None:
        model = model.to(device)
    return model
