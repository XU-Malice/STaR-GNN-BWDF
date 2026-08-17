from __future__ import annotations

import torch
import torch.nn.functional as F

from dma_wdf.models.dcrnn import DCRNN
from dma_wdf.models.star_dcrnn import STaRDCRNN


def _support(nodes: int) -> torch.Tensor:
    return torch.eye(nodes)


def _star(
    *,
    horizon: int = 24,
    dssn_sasr: bool = True,
    fa_dpr: bool = True,
) -> STaRDCRNN:
    return STaRDCRNN(
        random_walk=_support(2),
        input_dim=3,
        hidden_dim=4,
        horizon=horizon,
        history_hours=672,
        num_nodes=2,
        num_rnn_layers=1,
        max_diffusion_step=2,
        future_exog_dim=1,
        use_dssn_sasr=dssn_sasr,
        use_fa_dpr=fa_dpr,
        attention_dim=3,
        condition_on_future_calendar=True,
    )


def _inputs(horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(1, 672, 2, 3),
        torch.randn(1, horizon, 2),
        torch.randn(1, horizon, 2, 1),
    )


def test_all_factorial_variants_have_expected_shape() -> None:
    for horizon in (24, 168):
        x, y, future = _inputs(horizon)
        for dssn_sasr, fa_dpr in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            model = _star(
                horizon=horizon,
                dssn_sasr=dssn_sasr,
                fa_dpr=fa_dpr,
            )
            model.eval()
            output = model(x, y_target=y, x_future_exog=future)
            assert output.shape == (1, horizon, 2)
            assert torch.isfinite(output).all()


def test_disabled_model_is_exact_sealed_dcrnn() -> None:
    baseline = DCRNN(
        random_walk=_support(2),
        input_dim=3,
        hidden_dim=4,
        horizon=24,
        num_nodes=2,
        num_rnn_layers=1,
        max_diffusion_step=2,
        future_exog_dim=1,
    )
    model = _star(dssn_sasr=False, fa_dpr=False)
    model.load_dcrnn_backbone(baseline)
    baseline.eval()
    model.eval()
    x, _, future = _inputs(24)
    expected = baseline(x, x_future_exog=future)
    observed = model(x, x_future_exog=future)
    torch.testing.assert_close(observed, expected, atol=0.0, rtol=0.0)


def test_evaluation_target_is_structurally_isolated() -> None:
    model = _star(dssn_sasr=True, fa_dpr=True)
    model.eval()
    x, y, future = _inputs(24)
    first = model(x, y_target=y, x_future_exog=future)
    second = model(x, y_target=y + 1000.0, x_future_exog=future)
    third = model(x, y_target=None, x_future_exog=future)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first, third, atol=0.0, rtol=0.0)


def test_joint_loss_reaches_dcrnn_fa_dpr_and_sasr() -> None:
    model = _star(dssn_sasr=True, fa_dpr=True)
    model.train()
    x, y, future = _inputs(24)
    details = model(
        x,
        y_target=y,
        x_future_exog=future,
        teacher_forcing_ratio=0.0,
        return_details=True,
    )
    assert details.dssn_sasr is not None
    assert details.fa_dpr is not None
    state_loss, _ = model.state_supervision_loss(
        target=y,
        state=details.dssn_sasr,
    )
    loss = F.l1_loss(details.prediction, y) + 0.1 * state_loss
    loss.backward()
    assert model.fa_dpr is not None
    assert model.dssn_sasr is not None
    parameters = {
        "encoder": model.encoder.layers[0].gate_conv.weight,
        "decoder": model.decoder.output_projection.weight,
        "query": model.fa_dpr.query_projection.weight,
        "future_calendar": model.fa_dpr.future_projection.weight,
        "gate": model.fa_dpr.fusion_gate.weight,
        "alpha_mean": model.dssn_sasr.restorer.alpha_mean_logits,
        "alpha_std": model.dssn_sasr.restorer.alpha_std_logits,
    }
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0.0, name


def test_teacher_forcing_input_is_detached_from_sasr_alpha() -> None:
    model = _star(dssn_sasr=True, fa_dpr=False)
    model.train()
    x, y, future = _inputs(24)
    details = model(
        x,
        y_target=y,
        x_future_exog=future,
        teacher_forcing_ratio=1.0,
        return_details=True,
    )
    loss = details.decoder_prediction.square().mean()
    loss.backward()
    assert model.dssn_sasr is not None
    assert model.dssn_sasr.restorer.alpha_mean_logits.grad is None
    assert model.dssn_sasr.restorer.alpha_std_logits.grad is None


def test_checkpoint_round_trip_preserves_output() -> None:
    model = _star(dssn_sasr=True, fa_dpr=True)
    clone = _star(dssn_sasr=True, fa_dpr=True)
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    clone.eval()
    x, _, future = _inputs(24)
    torch.testing.assert_close(
        model(x, x_future_exog=future),
        clone(x, x_future_exog=future),
        atol=0.0,
        rtol=0.0,
    )


def test_fa_dpr_produces_forecast_step_specific_diagnostics() -> None:
    model = _star(horizon=168, dssn_sasr=False, fa_dpr=True)
    model.eval()
    x, _, future = _inputs(168)
    details = model(
        x,
        x_future_exog=future,
        return_details=True,
    )
    assert details.fa_dpr is not None
    assert details.fa_dpr.attention_weights.shape == (1, 168, 2, 28)
    assert details.fa_dpr.gate.shape == (1, 168, 2, 4)
    assert details.fa_dpr.fused_hidden.shape == (1, 168, 2, 4)
    torch.testing.assert_close(
        details.fa_dpr.attention_weights.sum(dim=-1),
        torch.ones(1, 168, 2),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_fa_dpr_decoder_loop_matches_original_when_context_is_zero() -> None:
    model = _star(dssn_sasr=False, fa_dpr=True)
    assert model.fa_dpr is not None
    with torch.no_grad():
        model.fa_dpr.value_projection.weight.zero_()
    model.eval()
    x, _, future = _inputs(24)
    encoder_hidden, hidden_sequence = model._encode_with_sequence(x)
    expected = model.decoder(
        encoder_hidden,
        model.random_walk,
        future_exog=future,
        teacher_forcing_ratio=0.0,
    )
    observed, _ = model._decode_with_fa_dpr(
        encoder_hidden,
        hidden_sequence,
        target=None,
        future_exog=future,
        teacher_forcing_ratio=0.0,
    )
    torch.testing.assert_close(observed, expected, atol=0.0, rtol=0.0)
