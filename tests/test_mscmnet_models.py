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
    ScaledDotProductSelfAttention,
    build_joint_model_from_config,
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


def test_attention_score_scaling_is_explicit() -> None:
    sequence = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    scaled = ScaledDotProductSelfAttention(2, scaling="sqrt_dim")
    unscaled = ScaledDotProductSelfAttention(2, scaling="none")
    with torch.no_grad():
        identity = torch.eye(2)
        for module in (scaled, unscaled):
            module.query.weight.copy_(identity)
            module.key.weight.copy_(identity)
            module.value.weight.copy_(identity)
    assert scaled.scale == pytest.approx(2.0 ** -0.5)
    assert unscaled.scale == pytest.approx(1.0)
    assert not torch.allclose(scaled(sequence), unscaled(sequence))


@pytest.mark.parametrize("model_name", ["mscmnet_wm", "mscmnet_w"])
@pytest.mark.parametrize("attention_scaling", ["sqrt_dim", "none"])
@pytest.mark.parametrize(
    "attention_update", ["replace", "residual", "final_residual", "skip_final"]
)
def test_joint_builder_propagates_attention_settings_to_every_cam(
    model_name: str,
    attention_scaling: str,
    attention_update: str,
) -> None:
    model_config = {
        "branch_features": ["demand", "hour"],
        "input_weeks": [1] * 10,
        "lstm_layers": [1] * 10,
        "hidden_sizes": [4] * 10,
        "fc1": {"future_features": ["hour"], "nodes": 4, "dropout": 0.0},
        "fc2": {
            "input_size": 12 if model_name == "mscmnet_wm" else 10,
            "hidden_size": 4,
            "lstm_layers": 1,
            "nodes": 4,
            "dropout": 0.0,
        },
    }
    cam_config = {
        "channel_sizes": [4, 1],
        "cnn_layers": 2,
        "attention_layers": 2,
        "attention_scaling": attention_scaling,
        "attention_update": attention_update,
        "temporal_layout": "per_day_vectors",
    }
    model = build_joint_model_from_config(model_name, model_config, cam_config)
    cams = [
        module for module in model.modules() if isinstance(module, ConvAttentionBlock)
    ]
    assert len(cams) == 11  # Ten DMA branches plus the daily-share branch.
    for cam in cams:
        assert cam.attention_scaling == attention_scaling
        assert cam.attention_update == attention_update
        for attention in cam.attention:
            assert attention.scaling == attention_scaling
            expected_scale = (
                attention.features ** -0.5
                if attention_scaling == "sqrt_dim"
                else 1.0
            )
            assert attention.scale == pytest.approx(expected_scale)

    # FC2 already has a daily time axis: do not apply the trunk's 24-hour reshape.
    assert model.msnet.branches[0].lstm.input_size == 24
    assert model.share_forecaster.lstm.input_size == 1
    output = model.share_forecaster(
        torch.randn(2, 7, model_config["fc2"]["input_size"])
    )
    assert output.shape == (2, 10)


@pytest.mark.parametrize(
    ("temporal_layout", "expected_steps", "expected_features"),
    [
        ("full_history_flat", 168, 1),
        ("per_day_flat", 168, 1),
        ("per_day_vectors", 7, 24),
    ],
)
def test_cam_lstm_temporal_layouts(
    temporal_layout: str,
    expected_steps: int,
    expected_features: int,
) -> None:
    config = ForecastBranchConfig(
        input_features=2,
        input_weeks=1,
        lstm_layers=1,
        hidden_size=4,
    )
    model = MSNet(
        [config for _ in range(10)],
        channel_sizes=(1,),
        cnn_layers=1,
        attention_layers=1,
        temporal_layout=temporal_layout,
    )
    captured: list[tuple[int, ...]] = []

    def record_shape(_module, args) -> None:
        captured.append(tuple(args[0].shape))

    hook = model.branches[0].lstm.register_forward_pre_hook(record_shape)
    try:
        output = model([torch.randn(2, 7, 24, 2) for _ in range(10)])
    finally:
        hook.remove()
    assert output.shape == (2, 24, 10)
    assert captured == [(2, expected_steps, expected_features)]
    assert model.branches[0].lstm.input_size == expected_features


@pytest.mark.parametrize("setting", ["bad_scaling", "bad_layout"])
def test_cam_diagnostic_choices_reject_unknown_values(setting: str) -> None:
    kwargs = (
        {"attention_scaling": "unknown"}
        if setting == "bad_scaling"
        else {"temporal_layout": "unknown"}
    )
    with pytest.raises(ValueError, match="attention_scaling|temporal_layout"):
        MSNet(
            [
                ForecastBranchConfig(
                    input_features=1,
                    input_weeks=1,
                    lstm_layers=1,
                    hidden_size=2,
                )
                for _ in range(10)
            ],
            channel_sizes=(1,),
            cnn_layers=1,
            attention_layers=1,
            **kwargs,
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


@pytest.mark.parametrize("correction_mode", ["direct", "residual"])
def test_zero_initialized_correction_composition(correction_mode: str) -> None:
    model = MSCMNetM(
        _small_msnet(input_features=10),
        future_features=9,
        fc1_nodes=24,
        fc1_dropout=0.0,
        correction_mode=correction_mode,
        zero_init_correction=True,
    )
    output = model(
        _histories(input_features=10),
        torch.randn(2, 24, 9),
    )
    expected = (
        torch.zeros_like(output.prediction)
        if correction_mode == "direct"
        else output.msnet_prediction
    )
    torch.testing.assert_close(output.prediction, expected)


def test_unknown_correction_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="correction_mode"):
        MSCMNetM(
            _small_msnet(input_features=10),
            future_features=9,
            fc1_nodes=24,
            fc1_dropout=0.0,
            correction_mode="unknown",
        )


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
