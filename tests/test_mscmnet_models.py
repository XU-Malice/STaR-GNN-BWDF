"""Shape and interface tests for the Que et al. temporal baselines."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from dma_wdf.models.mscmnet import (  # noqa: E402
    CAMLSTMForecastBranch,
    ConvAttentionBlock,
    Conv2dDayHourAttentionBlock,
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


def _small_msnet(*, input_features: int, correction_layout: str = "global_flat") -> MSNet:
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
        correction_layout=correction_layout,
    )


def _histories(*, input_features: int) -> list[torch.Tensor]:
    return [torch.randn(2, 7, 24, input_features) for _ in range(10)]


@pytest.mark.parametrize("model_class", [GRUForecast, LSTMForecast])
def test_independent_recurrent_baseline_shape(model_class) -> None:
    model = model_class([8, 4])
    output = model(torch.randn(3, 168, 1))
    assert output.shape == (3, 24)


@pytest.mark.parametrize("model_class", [GRUForecast, LSTMForecast])
def test_recurrent_daily_vector_hypothesis_is_trainable(model_class) -> None:
    torch.manual_seed(71)
    model = model_class([8, 4], input_features=24)
    history = torch.randn(3, 7, 24, requires_grad=True)
    output = model(history)
    assert output.shape == (3, 24)
    assert model.recurrent_layers[0].input_size == 24
    output.square().mean().backward()
    assert history.grad is not None
    assert torch.isfinite(history.grad).all()
    assert history.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match="history must have shape"):
        model(torch.randn(3, 168, 1))


@pytest.mark.parametrize("model_class", [GRUForecast, LSTMForecast])
def test_recurrent_input_feature_default_preserves_numerical_path(model_class) -> None:
    torch.manual_seed(9)
    implicit = model_class([8, 4])
    torch.manual_seed(9)
    explicit = model_class([8, 4], input_features=1)
    history = torch.randn(2, 48, 1)
    torch.testing.assert_close(implicit(history), explicit(history), rtol=0, atol=0)


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
        ("conv2d_day_hour", 7, 24),
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


@pytest.mark.parametrize("attention_scaling", ["sqrt_dim", "none"])
def test_conv2d_day_hour_grid_trainable_without_batch_mixing(attention_scaling: str) -> None:
    torch.manual_seed(27)
    cam = Conv2dDayHourAttentionBlock(
        input_features=2,
        channel_sizes=(4, 4, 1),
        attention_scaling=attention_scaling,
    ).eval()
    with torch.no_grad():
        # Test connectivity rather than an accidental all-negative ReLU stage.
        for convolution in cam.convolutions:
            convolution.bias.fill_(1.0)
    history = torch.randn(2, 7, 24, 2, requires_grad=True)
    shapes = []
    handles = [
        layer.register_forward_pre_hook(
            lambda _module, args: shapes.append(tuple(args[0].shape))
        )
        for layer in cam.attention
    ]
    try:
        output = cam(history)
    finally:
        for handle in handles:
            handle.remove()
    assert output.shape == (2, 7, 24, 1)
    assert shapes == [(2, 7, 96), (2, 7, 96), (2, 7, 24)]
    assert all(layer.kernel_size == (3, 3) for layer in cam.convolutions)

    # A forecast origin must not depend on other origins in its evaluation batch.
    changed = history.detach().clone()
    changed[1] += 100
    torch.testing.assert_close(cam(changed)[0], output[0], rtol=1e-5, atol=1e-6)
    output[0].square().mean().backward()
    assert history.grad is not None
    assert torch.isfinite(history.grad).all()
    assert history.grad[0].abs().sum() > 0
    assert torch.count_nonzero(history.grad[1]) == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cam.parameters())


def test_conv2d_branch_rejects_future_appended_to_fixed_history() -> None:
    branch = CAMLSTMForecastBranch(
        ForecastBranchConfig(
            input_features=1, input_weeks=1, lstm_layers=1, hidden_size=4
        ),
        channel_sizes=(1,), cnn_layers=1, attention_layers=1,
        temporal_layout="conv2d_day_hour",
    )
    with pytest.raises(ValueError, match="branch history must have shape"):
        branch(torch.randn(2, 8, 24, 1))
    # There is no future demand/target argument in this forecast-only interface.
    with pytest.raises(TypeError):
        branch(torch.randn(2, 7, 24, 1), torch.randn(2, 24))


def test_conv2d_builder_preserves_fc2_daily_conv1d() -> None:
    model_config = {
        "branch_features": ["demand", "hour"],
        "input_weeks": [1] * 10,
        "lstm_layers": [1] * 10,
        "hidden_sizes": [4] * 10,
        "fc1": {"future_features": ["hour"], "nodes": 4, "dropout": 0.0},
        "fc2": {
            "input_size": 10, "hidden_size": 4, "lstm_layers": 1,
            "nodes": 4, "dropout": 0.0,
        },
    }
    cam_config = {
        "convolution": "conv2d", "temporal_layout": "conv2d_day_hour",
        "channel_sizes": [2, 1], "cnn_layers": 2, "attention_layers": 2,
        "attention_scaling": "none", "attention_update": "replace",
    }
    model = build_joint_model_from_config("mscmnet_w", model_config, cam_config)
    assert all(isinstance(b.cam, Conv2dDayHourAttentionBlock) for b in model.msnet.branches)
    assert isinstance(model.share_forecaster.cam, ConvAttentionBlock)
    assert model.share_forecaster.lstm.input_size == 1
    assert model.msnet.branches[0].lstm.input_size == 24
    output = model(_histories(input_features=2), torch.randn(2, 24, 1), torch.randn(2, 7, 10))
    assert output.prediction.shape == (2, 24, 10)
    torch.testing.assert_close(output.predicted_daily_share.sum(-1), torch.ones(2))
    with pytest.raises(ValueError, match="explicit conv2d_day_hour"):
        build_joint_model_from_config(
            "mscmnet_w", model_config,
            {**cam_config, "temporal_layout": "per_day_vectors"},
        )


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


@pytest.mark.parametrize("correction_layout", ["global_flat", "hourwise_shared"])
def test_mscmnet_m_exposes_trunk_and_fc1_outputs(correction_layout: str) -> None:
    model = MSCMNetM(
        _small_msnet(input_features=10, correction_layout=correction_layout),
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
@pytest.mark.parametrize("correction_layout", ["global_flat", "hourwise_shared"])
def test_zero_initialized_correction_composition(correction_mode: str, correction_layout: str) -> None:
    model = MSCMNetM(
        _small_msnet(input_features=10, correction_layout=correction_layout),
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
@pytest.mark.parametrize("correction_layout", ["global_flat", "hourwise_shared"])
def test_fc2_variants_predict_normalized_daily_shares(
    model_class,
    branch_features: int,
    future_features: int,
    fc2_features: int,
    correction_layout: str,
) -> None:
    model = model_class(
        _small_msnet(input_features=branch_features, correction_layout=correction_layout),
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


def test_hourwise_joint_fc_does_not_mix_forecast_hours() -> None:
    class ProvidedForecast(torch.nn.Module):
        def forward(self, history):
            return history

    model = _small_msnet(input_features=1, correction_layout="hourwise_shared").eval()
    model.branches = torch.nn.ModuleList([ProvidedForecast() for _ in range(10)])
    assert model.joint_fully_connected.in_features == 10
    assert model.joint_fully_connected.out_features == 10
    forecasts = [torch.randn(2, 24) for _ in range(10)]
    original = model(forecasts)
    forecasts[3] = forecasts[3].clone()
    forecasts[3][:, 5] += 3
    changed = model(forecasts)
    other_hours = [i for i in range(24) if i != 5]
    torch.testing.assert_close(original[:, other_hours], changed[:, other_hours], rtol=0, atol=0)
    assert not torch.allclose(original[:, 5], changed[:, 5])


@pytest.mark.parametrize("with_fc2", [False, True])
def test_hourwise_corrections_share_weights_preserve_hours_and_receive_gradients(with_fc2: bool) -> None:
    torch.manual_seed(68)
    model_config = {
        "branch_features": ["demand", "hour"],
        "input_weeks": [1] * 10,
        "lstm_layers": [1] * 10,
        "hidden_sizes": [4] * 10,
        "correction_layout": "hourwise_shared",
        "fc1": {"future_features": ["hour", "temperature"], "nodes": 24, "dropout": 0.0},
        "fc2": {
            "input_size": 12, "hidden_size": 4, "lstm_layers": 1,
            "nodes": 48, "dropout": 0.0,
        },
    }
    cam_config = {
        "channel_sizes": [2, 1], "cnn_layers": 2, "attention_layers": 2,
        "temporal_layout": "per_day_vectors", "attention_scaling": "none",
    }
    model = build_joint_model_from_config(
        "mscmnet_wm" if with_fc2 else "mscmnet_m", model_config, cam_config,
    ).eval()
    if with_fc2:
        with torch.no_grad():
            for convolution in model.share_forecaster.cam.convolutions:
                convolution.bias.fill_(1.0)
    assert model.fc1.network[0].in_features == 12
    assert model.fc1.network[0].out_features == 24
    assert model.fc1.network[-1].out_features == 10
    if with_fc2:
        assert model.fc2.network[0].in_features == 20
        assert model.fc2.network[0].out_features == 48
        assert model.fc2.network[-1].out_features == 10
    histories = _histories(input_features=2)
    future = torch.randn(2, 24, 2, requires_grad=True)
    share_history = torch.randn(2, 7, 12, requires_grad=True)
    args = [histories, future] + ([share_history] if with_fc2 else [])
    output = model(*args)
    output.prediction.square().mean().backward()
    assert future.grad is not None and torch.isfinite(future.grad).all()
    assert future.grad.abs().sum() > 0
    if with_fc2:
        assert share_history.grad is not None and torch.isfinite(share_history.grad).all()
        assert share_history.grad.abs().sum() > 0
    changed_future = future.detach().clone()
    changed_future[:, 5] += 5
    changed_args = [histories, changed_future] + ([share_history.detach()] if with_fc2 else [])
    changed = model(*changed_args).prediction
    other_hours = [i for i in range(24) if i != 5]
    torch.testing.assert_close(output.prediction[:, other_hours], changed[:, other_hours], rtol=0, atol=0)
    assert not torch.allclose(output.prediction[:, 5], changed[:, 5])


def test_joint_builder_default_global_correction_path_is_numerically_unchanged() -> None:
    config = {
        "branch_features": ["demand"], "input_weeks": [1] * 10,
        "lstm_layers": [1] * 10, "hidden_sizes": [4] * 10,
        "fc1": {"future_features": ["hour"], "nodes": 4, "dropout": 0.0},
    }
    cam = {"channel_sizes": [1], "cnn_layers": 1, "attention_layers": 1,
           "temporal_layout": "per_day_vectors"}
    torch.manual_seed(23)
    implicit = build_joint_model_from_config("mscmnet_m", config, cam).eval()
    torch.manual_seed(23)
    explicit = build_joint_model_from_config(
        "mscmnet_m", {**config, "correction_layout": "global_flat"}, cam,
    ).eval()
    histories, future = _histories(input_features=1), torch.randn(2, 24, 1)
    torch.testing.assert_close(
        implicit(histories, future).prediction, explicit(histories, future).prediction,
        rtol=0, atol=0,
    )
    assert implicit.msnet.joint_fully_connected.in_features == 240
    assert implicit.fc1.network[0].in_features == 264
    with pytest.raises(ValueError, match="correction_layout"):
        build_joint_model_from_config("mscmnet_m", {**config, "correction_layout": "unknown"}, cam)


def test_branch_rejects_wrong_paper_history_shape() -> None:
    model = _small_msnet(input_features=10)
    histories = _histories(input_features=10)
    histories[0] = torch.randn(2, 6, 24, 10)
    with pytest.raises(ValueError, match="branch history"):
        model(histories)
