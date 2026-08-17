"""DSSN-SASR and forecast-aligned daily retrieval for STaR-DCRNN.

The implementation deliberately separates three roles:

* :class:`DMADailySliceNormalizer` performs parameter-free daily slice
  normalization (DSSN);
* :class:`SeasonallyAnchoredStateRestorer` estimates future daily state from
  history only (SASR);
* :class:`ForecastAlignedDailyPatternRetrieval` builds a compact daily memory
  from encoder states and retrieves it dynamically at every decoder step.

Future demand is never accepted by any forward method in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SliceNormalizationOutput:
    """DSSN-normalized history and per-DMA daily statistics."""

    normalized_history: torch.Tensor
    history_mean: torch.Tensor
    history_std: torch.Tensor


@dataclass(frozen=True)
class StateRestorationOutput:
    """SASR history-only future state estimates."""

    future_mean_daily: torch.Tensor
    future_log_std_daily: torch.Tensor
    future_mean: torch.Tensor
    future_std: torch.Tensor
    alpha_mean: torch.Tensor
    alpha_std: torch.Tensor


@dataclass(frozen=True)
class DSSNSASROutput:
    """Combined DSSN input transformation and SASR restoration state."""

    normalized_history: torch.Tensor
    history_mean: torch.Tensor
    history_std: torch.Tensor
    future_mean_daily: torch.Tensor
    future_log_std_daily: torch.Tensor
    future_mean: torch.Tensor
    future_std: torch.Tensor
    alpha_mean: torch.Tensor
    alpha_std: torch.Tensor


@dataclass(frozen=True)
class FADPRMemory:
    """Precomputed 28-day encoder memory used throughout decoding."""

    daily_tokens: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class FADPRStepOutput:
    """One forecast-aligned retrieval step."""

    fused_hidden: torch.Tensor
    attention_weights: torch.Tensor
    gate: torch.Tensor
    context: torch.Tensor


@dataclass(frozen=True)
class FADPROutput:
    """Horizon-wide FA-DPR diagnostics."""

    fused_hidden: torch.Tensor
    attention_weights: torch.Tensor
    gate: torch.Tensor
    daily_tokens: torch.Tensor


class DMADailySliceNormalizer(nn.Module):
    """Parameter-free daily slice normalization (DSSN)."""

    def __init__(
        self,
        *,
        num_nodes: int,
        history_hours: int,
        slice_hours: int = 24,
        epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.history_hours = int(history_hours)
        self.slice_hours = int(slice_hours)
        self.epsilon = float(epsilon)
        if min(self.num_nodes, self.history_hours, self.slice_hours) <= 0:
            raise ValueError("DSSN dimensions must be positive.")
        if self.history_hours % self.slice_hours != 0:
            raise ValueError(
                "history_hours must be divisible by slice_hours."
            )
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        self.num_history_slices = self.history_hours // self.slice_hours
        if self.num_history_slices != 28:
            raise ValueError(
                "The registered STaR protocol requires exactly 28 "
                "forecast-aligned history slices."
            )

    def _validate_history(self, history: torch.Tensor) -> None:
        expected = (self.history_hours, self.num_nodes)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(
                "history must have shape "
                f"(B,{self.history_hours},{self.num_nodes}), got "
                f"{tuple(history.shape)}."
            )
        if not torch.is_floating_point(history):
            raise TypeError("history must be a floating-point tensor.")
        if not torch.isfinite(history).all():
            raise ValueError("history contains NaN/Inf.")

    def forward(self, history: torch.Tensor) -> SliceNormalizationOutput:
        self._validate_history(history)
        slices = history.reshape(
            history.shape[0],
            self.num_history_slices,
            self.slice_hours,
            self.num_nodes,
        )
        mean = slices.mean(dim=2)
        std = slices.std(dim=2, unbiased=False).clamp_min(self.epsilon)
        normalized = (slices - mean.unsqueeze(2)) / std.unsqueeze(2)
        return SliceNormalizationOutput(
            normalized_history=normalized.reshape_as(history),
            history_mean=mean,
            history_std=std,
        )

    def target_daily_statistics(
        self,
        target: torch.Tensor,
        *,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return target daily mean/log-std for training supervision only."""
        horizon = int(horizon)
        if target.ndim != 3:
            raise ValueError("target must have shape (B,horizon,N).")
        expected = (target.shape[0], horizon, self.num_nodes)
        if tuple(target.shape) != expected:
            raise ValueError(
                f"target must have shape {expected}, got "
                f"{tuple(target.shape)}."
            )
        if horizon % self.slice_hours != 0:
            raise ValueError("horizon must be divisible by slice_hours.")
        future_slices = horizon // self.slice_hours
        if future_slices not in {1, 7}:
            raise ValueError("STaR supports 1 or 7 future slices.")
        slices = target.reshape(
            target.shape[0],
            future_slices,
            self.slice_hours,
            self.num_nodes,
        )
        mean = slices.mean(dim=2)
        std = slices.std(dim=2, unbiased=False).clamp_min(self.epsilon)
        return mean, torch.log(std)


class SeasonallyAnchoredStateRestorer(nn.Module):
    """SASR convexly combines recent and matching-weekday DMA states."""

    def __init__(
        self,
        *,
        num_nodes: int,
        horizon: int,
        slice_hours: int = 24,
        epsilon: float = 1.0e-5,
        initial_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.horizon = int(horizon)
        self.slice_hours = int(slice_hours)
        self.epsilon = float(epsilon)
        if self.horizon % self.slice_hours != 0:
            raise ValueError("horizon must be divisible by slice_hours.")
        self.num_future_slices = self.horizon // self.slice_hours
        if self.num_future_slices not in {1, 7}:
            raise ValueError("STaR supports 1 or 7 future slices.")
        if not 0.0 < float(initial_alpha) < 1.0:
            raise ValueError("initial_alpha must lie strictly in (0, 1).")
        initial_logit = torch.logit(
            torch.tensor(float(initial_alpha), dtype=torch.float32)
        )
        self.alpha_mean_logits = nn.Parameter(
            initial_logit.repeat(self.num_nodes)
        )
        self.alpha_std_logits = nn.Parameter(
            initial_logit.repeat(self.num_nodes)
        )

    @property
    def alpha_mean(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_mean_logits)

    @property
    def alpha_std(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_std_logits)

    def forward(
        self,
        history_mean: torch.Tensor,
        history_std: torch.Tensor,
    ) -> StateRestorationOutput:
        expected = (history_mean.shape[0], 28, self.num_nodes)
        if tuple(history_mean.shape) != expected:
            raise ValueError(
                f"history_mean must have shape {expected}, got "
                f"{tuple(history_mean.shape)}."
            )
        if tuple(history_std.shape) != expected:
            raise ValueError(
                f"history_std must have shape {expected}, got "
                f"{tuple(history_std.shape)}."
            )
        if torch.any(history_std <= 0.0):
            raise ValueError("history_std must be strictly positive.")
        if not torch.is_floating_point(history_mean) or not torch.is_floating_point(
            history_std
        ):
            raise TypeError("history statistics must be floating-point tensors.")
        if not torch.isfinite(history_mean).all() or not torch.isfinite(
            history_std
        ).all():
            raise ValueError("history statistics contain NaN/Inf.")

        future_day = torch.arange(
            self.num_future_slices,
            device=history_mean.device,
        )
        one_week_indices = 21 + future_day
        two_week_indices = 14 + future_day
        seasonal_mean = 0.5 * (
            history_mean[:, one_week_indices]
            + history_mean[:, two_week_indices]
        )
        history_log_std = torch.log(history_std.clamp_min(self.epsilon))
        seasonal_log_std = 0.5 * (
            history_log_std[:, one_week_indices]
            + history_log_std[:, two_week_indices]
        )
        last_mean = history_mean[:, -1:].expand(
            -1, self.num_future_slices, -1
        )
        last_log_std = history_log_std[:, -1:].expand(
            -1, self.num_future_slices, -1
        )
        alpha_mean = self.alpha_mean.view(1, 1, self.num_nodes)
        alpha_std = self.alpha_std.view(1, 1, self.num_nodes)
        future_mean_daily = (
            alpha_mean * last_mean
            + (1.0 - alpha_mean) * seasonal_mean
        )
        future_log_std_daily = (
            alpha_std * last_log_std
            + (1.0 - alpha_std) * seasonal_log_std
        )
        future_std_daily = torch.exp(future_log_std_daily).clamp_min(
            self.epsilon
        )
        return StateRestorationOutput(
            future_mean_daily=future_mean_daily,
            future_log_std_daily=future_log_std_daily,
            future_mean=future_mean_daily.repeat_interleave(
                self.slice_hours,
                dim=1,
            ),
            future_std=future_std_daily.repeat_interleave(
                self.slice_hours,
                dim=1,
            ),
            alpha_mean=self.alpha_mean,
            alpha_std=self.alpha_std,
        )


class DSSNSASR(nn.Module):
    """Compose parameter-free DSSN with history-only SASR."""

    def __init__(
        self,
        *,
        num_nodes: int,
        history_hours: int,
        horizon: int,
        slice_hours: int = 24,
        epsilon: float = 1.0e-5,
        initial_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.normalizer = DMADailySliceNormalizer(
            num_nodes=num_nodes,
            history_hours=history_hours,
            slice_hours=slice_hours,
            epsilon=epsilon,
        )
        self.restorer = SeasonallyAnchoredStateRestorer(
            num_nodes=num_nodes,
            horizon=horizon,
            slice_hours=slice_hours,
            epsilon=epsilon,
            initial_alpha=initial_alpha,
        )

    def forward(self, history: torch.Tensor) -> DSSNSASROutput:
        normalized = self.normalizer(history)
        restored = self.restorer(
            normalized.history_mean,
            normalized.history_std,
        )
        return DSSNSASROutput(
            normalized_history=normalized.normalized_history,
            history_mean=normalized.history_mean,
            history_std=normalized.history_std,
            future_mean_daily=restored.future_mean_daily,
            future_log_std_daily=restored.future_log_std_daily,
            future_mean=restored.future_mean,
            future_std=restored.future_std,
            alpha_mean=restored.alpha_mean,
            alpha_std=restored.alpha_std,
        )

    def target_daily_statistics(
        self,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.normalizer.target_daily_statistics(
            target,
            horizon=self.horizon,
        )


class ForecastAlignedDailyPatternRetrieval(nn.Module):
    """FA-DPR: forecast-conditioned retrieval from 28 daily tokens.

    Encoder states are pooled into a compact 28-day memory once.  During
    decoding, the previous top-layer decoder state is the primary query and
    the known exogenous features of the current forecast step optionally
    condition that query.  This produces a distinct node-wise retrieval for
    every forecast hour rather than one context reused over the full horizon.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        history_hours: int,
        patch_length: int = 24,
        attention_dim: int = 16,
        future_context_dim: int = 0,
        dropout: float = 0.0,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.history_hours = int(history_hours)
        self.patch_length = int(patch_length)
        self.attention_dim = int(attention_dim)
        self.future_context_dim = int(future_context_dim)
        if min(
            self.hidden_dim,
            self.history_hours,
            self.patch_length,
            self.attention_dim,
        ) <= 0:
            raise ValueError("FA-DPR dimensions must be positive.")
        if self.future_context_dim < 0:
            raise ValueError("future_context_dim must be non-negative.")
        if self.history_hours % self.patch_length != 0:
            raise ValueError(
                "history_hours must be divisible by patch_length."
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.num_patches = self.history_hours // self.patch_length
        if self.num_patches != 28:
            raise ValueError(
                "The registered FA-DPR protocol requires 28 days."
            )

        # A token contains only the mean of 24 hidden states.  It does not
        # concatenate the final hidden state of the day, avoiding a shortcut
        # between h_last (query) and the most recent daily key.
        self.token_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.query_projection = nn.Linear(
            self.hidden_dim,
            self.attention_dim,
            bias=False,
        )
        self.future_projection: nn.Linear | None = None
        if self.future_context_dim > 0:
            self.future_projection = nn.Linear(
                self.future_context_dim,
                self.attention_dim,
                bias=False,
            )
        self.key_projection = nn.Linear(
            self.hidden_dim,
            self.attention_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.attention_dropout = nn.Dropout(float(dropout))
        self.fusion_gate = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, float(gate_bias))
        self.scale = self.attention_dim ** -0.5

    def build_memory(
        self,
        hidden_sequence: torch.Tensor,
    ) -> FADPRMemory:
        """Pool hourly encoder states and precompute decoder-independent K/V."""
        if hidden_sequence.ndim != 4:
            raise ValueError(
                "hidden_sequence must have shape (B,L,N,H)."
            )
        batch_size, length, num_nodes, hidden_dim = hidden_sequence.shape
        if length != self.history_hours or hidden_dim != self.hidden_dim:
            raise ValueError(
                "hidden_sequence shape mismatch: expected L/H="
                f"{self.history_hours}/{self.hidden_dim}, got "
                f"{length}/{hidden_dim}."
            )
        patches = hidden_sequence.reshape(
            batch_size,
            self.num_patches,
            self.patch_length,
            num_nodes,
            self.hidden_dim,
        )
        daily_tokens = self.token_projection(patches.mean(dim=2))
        key = self.key_projection(daily_tokens)
        value = self.value_projection(daily_tokens)
        return FADPRMemory(
            daily_tokens=daily_tokens,
            key=key,
            value=value,
        )

    def attend(
        self,
        memory: FADPRMemory,
        decoder_hidden: torch.Tensor,
        *,
        future_context: torch.Tensor | None = None,
    ) -> FADPRStepOutput:
        """Retrieve one node-wise daily context for one forecast step."""
        if decoder_hidden.ndim != 3:
            raise ValueError(
                "decoder_hidden must have shape (B,N,H)."
            )
        batch_size, num_nodes, hidden_dim = decoder_hidden.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                "decoder_hidden hidden dimension mismatch: expected "
                f"{self.hidden_dim}, got {hidden_dim}."
            )
        expected_tokens = (
            batch_size,
            self.num_patches,
            num_nodes,
            self.hidden_dim,
        )
        if tuple(memory.daily_tokens.shape) != expected_tokens:
            raise ValueError(
                "FA-DPR memory shape does not match decoder state."
            )

        query = self.query_projection(decoder_hidden)
        if self.future_projection is not None:
            expected_future = (
                batch_size,
                num_nodes,
                self.future_context_dim,
            )
            if future_context is None:
                raise ValueError(
                    "future_context is required when future_context_dim > 0."
                )
            if tuple(future_context.shape) != expected_future:
                raise ValueError(
                    f"future_context must have shape {expected_future}, got "
                    f"{tuple(future_context.shape)}."
                )
            query = query + self.future_projection(future_context)
        elif future_context is not None and future_context.shape[-1] != 0:
            raise ValueError(
                "future_context was supplied although future_context_dim=0."
            )

        scores = (
            torch.einsum("bna,bpna->bnp", query, memory.key)
            * self.scale
        )
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum(
            "bnp,bpnh->bnh",
            self.attention_dropout(weights),
            memory.value,
        )
        gate = torch.sigmoid(
            self.fusion_gate(
                torch.cat([decoder_hidden, context], dim=-1)
            )
        )
        return FADPRStepOutput(
            fused_hidden=decoder_hidden + gate * context,
            attention_weights=weights,
            gate=gate,
            context=context,
        )
