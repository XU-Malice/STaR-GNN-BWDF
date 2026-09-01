"""Shape and interface tests for the Que et al. temporal baselines."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from dma_wdf.models.mscmnet import (  # noqa: E402
    ConvAttentionBlock,
    ForecastBranchConfig,
    GRUForecast,
    LSTMForecast,
    MSCMNetM,
    MSCMNetW,
    MSCMNetWM,
    MSNet,
)


def _small_msnet(*, input_features: int) -> MSNet:
    configs = [
        ForecastBranchConfig(
            input_features=input_features,
            input_weeks=1,
            lstm_layers=1,
            hidden_size=8,
        )
        for _ in range(10)
    ]
    return MSNet(
        configs,
        channel_sizes=(4,),
        cnn_layers=1,
        attention_layers=1,
        kernel_size=3,
        attention_heads=1,
    )


def _histories(*, input_features: int) -> list[torch.Tensor]:
    return [torch.randn(2, 7, 24, input_features) for _ in range(10)]


@pytest.mark.parametrize("model_class", [GRUForecast, LSTMForecast])
def test_independent_recurrent_baseline_shape(model_class) -> None:
    model = model_class([8, 4])
    output = model(torch.randn(3, 168, 1))
    assert output.shape == (3, 24)


def test_msnet_joint_output_shape() -> None:
    model = _small_msnet(input_features=10)
    output = model(_histories(input_features=10))
    assert output.shape == (2, 24, 10)


def test_paper_cam_interleaves_and_compresses_channels() -> None:
    cam = ConvAttentionBlock(
        input_features=10,
        channel_sizes=(16, 16, 1),
        cnn_layers=3,
        attention_layers=3,
        kernel_size=3,
        attention_heads=1,
    )
    output = cam(torch.randn(2, 168, 10))
    assert output.shape == (2, 168, 1)
    assert [layer.in_channels for layer in cam.convolutions] == [10, 16, 16]
    assert [layer.out_channels for layer in cam.convolutions] == [16, 16, 1]
    assert not any(isinstance(layer, torch.nn.LayerNorm) for layer in cam.modules())
    assert not any(
        isinstance(layer, torch.nn.MultiheadAttention) for layer in cam.modules()
    )


@pytest.mark.parametrize(
    ("attention_update", "expected_multiplier"),
    [
        ("replace", 0.0),
        ("residual", 1.0),
        ("final_residual", 1.0),
        ("skip_final", 1.0),
    ],
)
def test_cam_attention_update_diagnostics(
    attention_update: str,
    expected_multiplier: float,
) -> None:
    class ZeroAttention(torch.nn.Module):
        def forward(self, sequence: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(sequence)

    cam = ConvAttentionBlock(
        input_features=1,
        channel_sizes=(1,),
        cnn_layers=1,
        attention_layers=1,
        kernel_size=3,
        attention_heads=1,
        attention_update=attention_update,
    )
    with torch.no_grad():
        cam.convolutions[0].weight.zero_()
        cam.convolutions[0].weight[0, 0, 1] = 1.0
        cam.convolutions[0].bias.zero_()
    cam.attention[0] = ZeroAttention()
    sequence = torch.tensor([[[1.0], [2.0], [4.0]]])
    torch.testing.assert_close(cam(sequence), sequence * expected_multiplier)


def test_cam_rejects_unknown_attention_update() -> None:
    with pytest.raises(ValueError, match="attention_update"):
        ConvAttentionBlock(
            input_features=1,
            channel_sizes=(1,),
            cnn_layers=1,
            attention_layers=1,
            attention_update="unknown",
        )


def test_mscmnet_m_exposes_trunk_and_fc1_outputs() -> None:
    model = MSCMNetM(
        _small_msnet(input_features=10),
        future_features=9,
        fc1_nodes=24,
        fc1_dropout=0.0,
    )
    output = model(
        _histories(input_features=10),
        torch.randn(2, 24, 9),
    )
    assert output.prediction.shape == (2, 24, 10)
    assert output.msnet_prediction.shape == (2, 24, 10)
    assert output.fc1_prediction is not None
    assert output.predicted_daily_share is None


@pytest.mark.parametrize(
    ("model_class", "branch_features", "future_features", "fc2_features"),
    [
        (MSCMNetWM, 10, 9, 12),
        (MSCMNetW, 6, 5, 10),
    ],
)
def test_fc2_variants_predict_normalized_daily_shares(
    model_class,
    branch_features: int,
    future_features: int,
    fc2_features: int,
) -> None:
    model = model_class(
        _small_msnet(input_features=branch_features),
        future_features=future_features,
        fc1_nodes=12,
        fc1_dropout=0.0,
        fc2_input_features=fc2_features,
        fc2_cam_channel_sizes=(4,),
        fc2_cam_kernel_size=3,
        fc2_cam_dropout=0.0,
        fc2_hidden_size=8,
        fc2_lstm_layers=1,
        fc2_nodes=12,
        fc2_dropout=0.0,
    )
    output = model(
        _histories(input_features=branch_features),
        torch.randn(2, 24, future_features),
        torch.randn(2, 7, fc2_features),
    )
    assert output.prediction.shape == (2, 24, 10)
    assert output.predicted_daily_share is not None
    assert output.predicted_daily_share.shape == (2, 10)
    torch.testing.assert_close(
        output.predicted_daily_share.sum(dim=1), torch.ones(2)
    )
    assert output.predicted_daily_share.grad_fn is not None


def test_branch_rejects_wrong_paper_history_shape() -> None:
    model = _small_msnet(input_features=10)
    histories = _histories(input_features=10)
    histories[0] = torch.randn(2, 6, 24, 10)
    with pytest.raises(ValueError, match="branch history"):
        model(histories)
