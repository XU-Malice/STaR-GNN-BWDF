from __future__ import annotations

import torch

from dma_wdf.models.star_components import (
    DSSNSASR,
    DMADailySliceNormalizer,
    ForecastAlignedDailyPatternRetrieval,
    SeasonallyAnchoredStateRestorer,
)


def test_dssn_slice_shapes_and_normalization() -> None:
    module = DMADailySliceNormalizer(
        num_nodes=3,
        history_hours=672,
    )
    history = torch.randn(2, 672, 3)
    output = module(history)
    assert output.normalized_history.shape == (2, 672, 3)
    assert output.history_mean.shape == (2, 28, 3)
    assert output.history_std.shape == (2, 28, 3)
    slices = output.normalized_history.reshape(2, 28, 24, 3)
    torch.testing.assert_close(
        slices.mean(dim=2),
        torch.zeros(2, 28, 3),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        slices.std(dim=2, unbiased=False),
        torch.ones(2, 28, 3),
        atol=1.0e-5,
        rtol=0.0,
    )


def test_sasr_uses_declared_matching_weekday_indices() -> None:
    module = SeasonallyAnchoredStateRestorer(
        num_nodes=1,
        horizon=168,
        initial_alpha=0.5,
    )
    history_mean = torch.arange(28.0).reshape(1, 28, 1)
    history_std = torch.exp(history_mean / 10.0)
    output = module(history_mean, history_std)
    for q in range(7):
        seasonal_mean = 0.5 * ((21 + q) + (14 + q))
        expected_mean = 0.5 * 27.0 + 0.5 * seasonal_mean
        seasonal_log_std = 0.5 * (
            (21 + q) / 10.0 + (14 + q) / 10.0
        )
        expected_log_std = 0.5 * 2.7 + 0.5 * seasonal_log_std
        torch.testing.assert_close(
            output.future_mean_daily[0, q, 0],
            torch.tensor(expected_mean),
        )
        torch.testing.assert_close(
            output.future_log_std_daily[0, q, 0],
            torch.tensor(expected_log_std),
        )


def test_sasr_endpoint_logits_recover_recent_or_seasonal_state() -> None:
    history_mean = torch.arange(28.0).reshape(1, 28, 1)
    history_std = torch.exp(history_mean / 10.0)
    module = SeasonallyAnchoredStateRestorer(
        num_nodes=1,
        horizon=24,
    )
    with torch.no_grad():
        module.alpha_mean_logits.fill_(30.0)
        module.alpha_std_logits.fill_(30.0)
    recent = module(history_mean, history_std)
    torch.testing.assert_close(
        recent.future_mean_daily[0, 0, 0],
        history_mean[0, -1, 0],
        atol=1.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        recent.future_log_std_daily[0, 0, 0],
        torch.log(history_std[0, -1, 0]),
        atol=1.0e-5,
        rtol=0.0,
    )
    with torch.no_grad():
        module.alpha_mean_logits.fill_(-30.0)
        module.alpha_std_logits.fill_(-30.0)
    seasonal = module(history_mean, history_std)
    torch.testing.assert_close(
        seasonal.future_mean_daily[0, 0, 0],
        0.5 * (history_mean[0, 21, 0] + history_mean[0, 14, 0]),
        atol=1.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        seasonal.future_log_std_daily[0, 0, 0],
        0.5
        * (
            torch.log(history_std[0, 21, 0])
            + torch.log(history_std[0, 14, 0])
        ),
        atol=1.0e-5,
        rtol=0.0,
    )


def test_dssn_sasr_alpha_bounds_positive_std_and_24h_expansion() -> None:
    module = DSSNSASR(
        num_nodes=2,
        history_hours=672,
        horizon=24,
    )
    with torch.no_grad():
        module.restorer.alpha_mean_logits.copy_(
            torch.tensor([-30.0, 30.0])
        )
        module.restorer.alpha_std_logits.copy_(
            torch.tensor([30.0, -30.0])
        )
    output = module(torch.randn(1, 672, 2))
    assert torch.all((output.alpha_mean >= 0) & (output.alpha_mean <= 1))
    assert torch.all((output.alpha_std >= 0) & (output.alpha_std <= 1))
    assert torch.all(output.future_std > 0.0)
    assert output.future_mean_daily.shape == (1, 1, 2)
    for hour in range(24):
        torch.testing.assert_close(
            output.future_mean[:, hour],
            output.future_mean_daily[:, 0],
        )


def test_fa_dpr_mean_pool_memory_softmax_and_gate_initialization() -> None:
    module = ForecastAlignedDailyPatternRetrieval(
        hidden_dim=8,
        history_hours=672,
        patch_length=24,
        attention_dim=4,
        future_context_dim=2,
        gate_bias=-2.0,
    )
    sequence = torch.randn(2, 672, 3, 8)
    memory = module.build_memory(sequence)
    output = module.attend(
        memory,
        sequence[:, -1],
        future_context=torch.randn(2, 3, 2),
    )
    expected_tokens = module.token_projection(
        sequence.reshape(2, 28, 24, 3, 8).mean(dim=2)
    )
    torch.testing.assert_close(
        memory.daily_tokens,
        expected_tokens,
        atol=0.0,
        rtol=0.0,
    )
    assert module.token_projection.in_features == 8
    assert memory.daily_tokens.shape == (2, 28, 3, 8)
    assert output.attention_weights.shape == (2, 3, 28)
    assert output.gate.shape == (2, 3, 8)
    assert output.fused_hidden.shape == (2, 3, 8)
    torch.testing.assert_close(
        output.attention_weights.sum(dim=-1),
        torch.ones(2, 3),
        atol=1.0e-6,
        rtol=0.0,
    )
    expected_gate = torch.sigmoid(torch.tensor(-2.0))
    torch.testing.assert_close(
        output.gate,
        torch.full_like(output.gate, expected_gate),
    )


def test_fa_dpr_tokens_do_not_concatenate_patch_last_state() -> None:
    module = ForecastAlignedDailyPatternRetrieval(
        hidden_dim=4,
        history_hours=672,
        patch_length=24,
        attention_dim=2,
    )
    sequence = torch.zeros(1, 672, 1, 4)
    sequence[:, 671] = 24.0
    memory = module.build_memory(sequence)
    # The last daily raw token is the 24-hour mean (1.0), not h_672 (24.0)
    # and not [mean; h_672].  The Linear(H,H) shape enforces this contract.
    expected_last = module.token_projection(torch.ones(1, 1, 4))
    torch.testing.assert_close(
        memory.daily_tokens[:, -1],
        expected_last,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_fa_dpr_query_changes_with_decoder_state_and_future_calendar() -> None:
    module = ForecastAlignedDailyPatternRetrieval(
        hidden_dim=2,
        history_hours=672,
        patch_length=24,
        attention_dim=2,
        future_context_dim=2,
    )
    with torch.no_grad():
        module.token_projection.weight.copy_(torch.eye(2))
        module.token_projection.bias.zero_()
        module.query_projection.weight.copy_(torch.eye(2))
        module.key_projection.weight.copy_(torch.eye(2))
        assert module.future_projection is not None
        module.future_projection.weight.copy_(torch.eye(2))

    sequence = torch.zeros(1, 672, 1, 2)
    sequence[:, :24, :, 0] = 2.0
    sequence[:, 24:48, :, 1] = 2.0
    memory = module.build_memory(sequence)
    state_x = torch.tensor([[[1.0, 0.0]]])
    state_y = torch.tensor([[[0.0, 1.0]]])
    zero_calendar = torch.zeros(1, 1, 2)
    weights_x = module.attend(
        memory,
        state_x,
        future_context=zero_calendar,
    ).attention_weights
    weights_y = module.attend(
        memory,
        state_y,
        future_context=zero_calendar,
    ).attention_weights
    assert not torch.allclose(weights_x, weights_y)

    calendar_y = torch.tensor([[[0.0, 2.0]]])
    weights_calendar = module.attend(
        memory,
        state_x,
        future_context=calendar_y,
    ).attention_weights
    assert not torch.allclose(weights_x, weights_calendar)
