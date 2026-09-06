"""Training-pipeline guards that require the CI PyTorch environment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]


def _load_training_script():
    path = ROOT / "scripts/train/train_temporal_baselines.py"
    spec = importlib.util.spec_from_file_location("temporal_baseline_training", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mw_alias_resolves_to_supplementary_wm_name() -> None:
    module = _load_training_script()
    assert module.canonical_model_name("mscmnet_mw") == "mscmnet_wm"


def test_standardizer_is_fitted_without_test_values() -> None:
    module = _load_training_script()
    train = np.asarray([[[1.0], [3.0]]], dtype=np.float32)
    test = np.asarray([[[10_000.0]]], dtype=np.float32)
    scaler = module.Standardizer.fit_features(train)
    assert float(scaler.mean[0]) == pytest.approx(2.0)
    _ = scaler.transform(test)
    assert float(scaler.mean[0]) == pytest.approx(2.0)


def test_minmax_scaler_is_fitted_without_test_values() -> None:
    module = _load_training_script()
    train = np.asarray([[[1.0], [3.0]]], dtype=np.float32)
    test = np.asarray([[[10_000.0]]], dtype=np.float32)
    scaler = module.MinMaxScaler.fit_features(train)
    assert float(scaler.minimum[0]) == pytest.approx(1.0)
    assert float(scaler.value_range[0]) == pytest.approx(2.0)
    transformed = scaler.transform(test)
    assert float(transformed[0, 0, 0]) > 1.0
    assert float(scaler.minimum[0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("name", "expected_class"),
    [("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)],
)
def test_optimizer_semantics_are_explicit(name, expected_class) -> None:
    module = _load_training_script()
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = module._build_optimizer(
        [parameter],
        optimizer_name=name,
        learning_rate=0.001,
        weight_decay=0.1,
    )
    assert isinstance(optimizer, expected_class)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)


def test_metric_table_uses_supplementary_total_mae_convention() -> None:
    module = _load_training_script()
    truth_24 = np.zeros((1, 24, 10), dtype=np.float32)
    pred_24 = np.ones((1, 24, 10), dtype=np.float32)
    truth_168 = np.zeros((1, 168, 10), dtype=np.float32)
    pred_168 = np.ones((1, 168, 10), dtype=np.float32)
    table = module.metric_table(
        model_display_name="MSNet",
        y_true_24h=truth_24,
        y_pred_24h=pred_24,
        y_true_168h=truth_168,
        y_pred_168h=pred_168,
        dma_letters=list("ABCDEFGHIJ"),
        literature_config=None,
    )
    total_mae = table.loc[
        (table["task"] == "24h")
        & (table["series"] == "total")
        & (table["metric"] == "MAE"),
        "value",
    ].item()
    total_rmse = table.loc[
        (table["task"] == "24h")
        & (table["series"] == "total")
        & (table["metric"] == "RMSE"),
        "value",
    ].item()
    assert total_mae == pytest.approx(10.0)
    assert total_rmse == pytest.approx(10.0)
    assert len(table) == 88


def test_joint_stage_prediction_retains_intermediate_outputs() -> None:
    module = _load_training_script()

    class FakeCorrection(torch.nn.Module):
        def forward(self, branches, future, fc2_history):
            base = branches[0][..., :1].squeeze(1)
            prediction = base.repeat(1, 1, 10)
            return module.MSCMNetOutput(
                prediction=prediction + 2.0,
                msnet_prediction=prediction,
                fc1_prediction=prediction + 1.0,
                predicted_daily_share=torch.softmax(fc2_history[:, -1, :10], dim=1),
            )

    branches = [np.ones((3, 1, 24, 1), dtype=np.float32)]
    stages = module.predict_joint_24h_stages(
        model=FakeCorrection(),
        family="mscmnet_w",
        branches=branches,
        future=np.zeros((3, 24, 1), dtype=np.float32),
        fc2_history=np.zeros((3, 7, 10), dtype=np.float32),
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert set(stages) == {
        "prediction",
        "msnet_prediction",
        "fc1_prediction",
        "predicted_daily_share",
    }
    assert stages["prediction"].shape == (3, 24, 10)
    assert stages["predicted_daily_share"].shape == (3, 10)
    assert np.allclose(stages["prediction"], 3.0)
    assert np.allclose(stages["fc1_prediction"], 2.0)


def test_metric_table_rejects_nonfinite_predictions() -> None:
    module = _load_training_script()
    truth = np.ones((2, 24, 10), dtype=np.float32)
    pred = truth.copy()
    pred[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        module.metric_table(
            model_display_name="MSNet",
            y_true_24h=truth, y_pred_24h=pred,
            y_true_168h=np.ones((2, 168, 10)),
            y_pred_168h=np.ones((2, 168, 10)),
            dma_letters=list("ABCDEFGHIJ"), literature_config=None,
        )


def test_first_day_is_read_from_one_rollout_for_every_stage() -> None:
    module = _load_training_script()
    rng = np.random.default_rng(1)
    stages = {
        key: rng.normal(size=(3, 168, 10)).astype(np.float32)
        for key in ("prediction", "msnet_prediction", "fc1_prediction")
    }
    stages["predicted_daily_share"] = rng.random((3, 7, 10)).astype(np.float32)
    first = module.first_day_stages(stages)
    for key in ("prediction", "msnet_prediction", "fc1_prediction"):
        np.testing.assert_array_equal(first[key], stages[key][:, :24])
    np.testing.assert_array_equal(first["predicted_daily_share"], stages["predicted_daily_share"][:, 0])


def _joint_scaler_fixture(module, *, empty_future=False):
    index = pd.date_range("2021-01-01", periods=5 * 24, freq="h", tz="Etc/GMT-1")
    t = np.arange(len(index), dtype=np.float32)
    demand = pd.DataFrame({f"dma_{i}": 10.0 * i + 2.0 + t for i in range(10)}, index=index)
    weather = pd.DataFrame({"air_temperature": 5.0 + t / 24}, index=index)
    temporal = pd.DataFrame({"hour": index.hour.astype(float)}, index=index)
    bounds = {"train_start": index[0], "train_end": index[71]}
    columns = tuple(("own_dma_demand", "air_temperature", "hour") for _ in range(10))
    branches = tuple(np.column_stack([
        demand[dma].to_numpy()[:48], weather.air_temperature.to_numpy()[:48],
        temporal.hour.to_numpy()[:48],
    ]).reshape(2, 1, 24, 3).astype(np.float32) for dma in demand.columns)
    y = demand.iloc[24:72].to_numpy(dtype=np.float32).reshape(2, 24, 10)
    future = np.empty((2, 24, 0), dtype=np.float32) if empty_future else weather.iloc[24:72].to_numpy(dtype=np.float32).reshape(2, 24, 1)
    fc2 = module.daily_share_history(
        demand=demand, weather=weather, starts=index[[24, 48]],
        dma_columns=list(demand.columns), history_days=1, include_temperature=True,
    )
    samples = module.JointTemporalSamples(
        train_branches=branches, test_branches=tuple(x + 1_000_000 for x in branches),
        branch_feature_columns=columns, y_train_24h=y,
        y_test_24h=y + 1_000_000, y_test_168h=np.zeros((2, 168, 10), dtype=np.float32),
        future_train=future, future_test=future + 1_000_000,
        fc2_train=fc2, fc2_test=fc2 + 1_000_000, fc2_share_target_train=None,
        train_forecast_starts=index[[24, 48]], test_forecast_starts=index[[72, 96]],
    )
    return samples, demand, weather, temporal, bounds


@pytest.mark.parametrize("normalization", ["zscore", "minmax"])
@pytest.mark.parametrize("empty_future", [False, True])
def test_default_scaling_matches_original_window_fitting(normalization, empty_future):
    module = _load_training_script()
    samples, *_ = _joint_scaler_fixture(module, empty_future=empty_future)
    result = module._scaled_arrays(samples, normalization=normalization)
    explicit = module._scaled_arrays(samples, normalization=normalization, fit_rows=None, demand_scaling="per_dma")
    for original, actual, scaler in zip(samples.train_branches, result[0], result[2]):
        expected_scaler = module._fit_features(original, normalization)
        assert module._scaler_json(scaler) == module._scaler_json(expected_scaler)
        np.testing.assert_array_equal(actual, expected_scaler.transform(original))
    for pos in (3, 5, 6, 8, 9):
        np.testing.assert_array_equal(result[pos], explicit[pos])
    np.testing.assert_array_equal(result[3], module._fit_features(samples.y_train_24h, normalization).transform(samples.y_train_24h))
    np.testing.assert_array_equal(result[5], module._fit_features(samples.future_train, normalization).transform(samples.future_train))


@pytest.mark.parametrize("normalization", ["zscore", "minmax"])
def test_unique_train_scalers_exclude_test_and_preserve_shared_units(normalization):
    module = _load_training_script()
    samples, demand, weather, temporal, bounds = _joint_scaler_fixture(module)

    def fit(demand_frame, weather_frame, temporal_frame):
        return module._joint_train_fit_rows(
            demand=demand_frame, weather=weather_frame, temporal=temporal_frame,
            bounds=bounds, dma_columns=list(demand.columns),
            branch_feature_columns=samples.branch_feature_columns,
            future_columns=["air_temperature"], include_fc2=True, include_temperature=True,
        )

    rows = fit(demand, weather, temporal)
    changed = []
    for frame in (demand, weather, temporal):
        copy = frame.copy()
        copy.iloc[72:] = -1_000_000
        changed.append(copy)
    other = fit(*changed)
    for key in ("target", "future", "fc2"):
        np.testing.assert_array_equal(rows[key], other[key])
    for first, second in zip(rows["branches"], other["branches"]):
        np.testing.assert_array_equal(first, second)
    assert rows["target"].shape == (72, 10)
    assert rows["fc2"].shape == (3, 12)
    # Last FC2 row is exactly the last TRAIN day's shares and extrema.
    daily = demand.iloc[48:72].sum().to_numpy()
    np.testing.assert_allclose(rows["fc2"][-1, :10], daily / daily.sum())
    np.testing.assert_allclose(rows["fc2"][-1, -2:], [weather.iloc[48:72].air_temperature.max(), weather.iloc[48:72].air_temperature.min()])
    result = module._scaled_arrays(samples, normalization=normalization, fit_rows=rows, demand_scaling="shared")
    expected = module._fit_scalar(normalization, demand.iloc[:72].to_numpy())
    target = result[4]
    np.testing.assert_allclose(target.transform(samples.y_train_24h), expected.transform(samples.y_train_24h))
    np.testing.assert_allclose(target.inverse(result[3]), samples.y_train_24h, rtol=2e-6, atol=2e-5)
    for branch_scaler in result[2]:
        raw = np.ones((4, 3), dtype=np.float32) * 42.0
        np.testing.assert_allclose(branch_scaler.transform(raw)[:, 0], expected.transform(raw[:, 0]))
    assert module._scaler_json(result[7]) == module._scaler_json(module._fit_features(weather.iloc[:72].to_numpy(), normalization))
    # Window fitting assigns repeated/window coverage weights, demonstrably distinct.
    legacy = module._scaled_arrays(samples, normalization=normalization)
    assert module._scaler_json(result[2][0]) != module._scaler_json(legacy[2][0])


@pytest.mark.parametrize("layout", ["hourly", "daily_vectors"])
def test_recurrent_seven_day_rollout_uses_only_predicted_days(layout):
    module = _load_training_script()
    raw = np.arange(2 * 48, dtype=np.float32).reshape(2, 48, 1)
    history = module._recurrent_history(raw, layout)
    seen = []

    class NextDay(torch.nn.Module):
        def forward(self, x):
            seen.append(x.clone())
            last_day = x[:, -1, :] if layout == "daily_vectors" else x[:, -24:, 0]
            return last_day + 1.0

    output = module.predict_independent_168h(
        NextDay(), history, device=torch.device("cpu"), steps=7, recurrent_layout=layout,
    )
    assert output.shape == (2, 168)
    for day in range(7):
        np.testing.assert_array_equal(output[:, day * 24:(day + 1) * 24], raw[:, -24:, 0] + day + 1)
    assert all(x.shape == history.shape for x in seen)
    if layout == "daily_vectors":
        torch.testing.assert_close(seen[1][:, 0], seen[0][:, 1])
        np.testing.assert_array_equal(history[:, 0, :], raw[:, :24, 0])


def test_scaler_train_rows_reject_gaps_and_do_not_repair_training_values():
    module = _load_training_script()
    _, demand, _, _, bounds = _joint_scaler_fixture(module)
    with pytest.raises(ValueError, match="complete unique"):
        module._train_rows(demand.drop(demand.index[2]), bounds)
    demand.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        module._train_rows(demand, bounds)


def test_scaler_audit_contains_numeric_parameters_and_scope():
    import json
    module = _load_training_script()
    _, demand, _, _, bounds = _joint_scaler_fixture(module)
    scaler = module.Standardizer.fit_features(demand.iloc[:72].to_numpy())
    audit = module._scaler_audit_base({"scaler_fit_scope": "train_rows"}, bounds)
    audit["target"] = module._scaler_audit_entry(scaler, demand.iloc[:72].to_numpy())
    restored = json.loads(json.dumps(audit, allow_nan=False))
    assert restored["test_values_used_for_fit"] is False
    assert restored["target"]["fit_rows"] == 72
    assert len(restored["target"]["parameters"]["mean"]) == 10
    assert isinstance(restored["target"]["parameters"]["mean"][0], float)


def test_cli_propagates_layout_scaling_and_diagnostic_overrides(monkeypatch, tmp_path):
    import dma_wdf.data.mscmnet_dataset as dataset
    module = _load_training_script()
    seen = []
    monkeypatch.setattr(sys, "argv", [
        "train_temporal_baselines.py", "--model", "all", "--device", "cpu", "--allow-cpu",
        "--output-root", str(tmp_path), "--normalization", "minmax",
        "--recurrent-layout", "daily_vectors", "--scaler-fit-scope", "train_rows",
        "--correction-layout", "hourwise_shared", "--cam-temporal-layout", "conv2d_day_hour",
        "--loss", "huber", "--learning-rate-scale", "0.5", "--best-epoch-scale", "2",
        "--joint-weight-decay", "0.002", "--fc2-share-supervision-weight", "0.05",
        "--correction-mode", "residual", "--zero-init-correction",
    ])
    monkeypatch.setattr(module, "preflight_resources", lambda **_: {})
    monkeypatch.setattr(module, "load_paper_data", lambda **_: ({}, {}, {"train_end": pd.Timestamp("2022-12-15")}))
    monkeypatch.setattr(dataset, "validate_leakage_safe_data_build", lambda *_, **__: {})

    def record(**kwargs):
        seen.append(kwargs)
        return {"status": "completed", "model": kwargs["canonical"]}

    monkeypatch.setattr(module, "run_one_model", record)
    module.main()
    assert len(seen) == 6
    config = seen[0]["config"]
    assert config["training"]["recurrent_layout"] == "daily_vectors"
    assert config["training"]["scaler_fit_scope"] == "train_rows"
    assert config["training"]["demand_scaling"] == "per_dma"
    assert config["training"]["loss"] == "huber"
    assert config["training"]["learning_rate_scale"] == 0.5
    assert config["training"]["best_epoch_scale"] == 2
    assert config["training"]["joint_weight_decay_override"] == 0.002
    assert config["cam"]["convolution"] == "conv2d"
    for model in ("msnet", "mscmnet_m", "mscmnet_wm", "mscmnet_w"):
        assert config["models"][model]["correction_layout"] == "hourwise_shared"
    assert config["models"]["mscmnet_wm"]["fc2"]["share_supervision_weight"] == 0.05
    assert config["models"]["mscmnet_w"]["zero_init_correction"] is True


@pytest.mark.parametrize("model", ["gru", "lstm", "all"])
def test_cli_rejects_shared_demand_for_independent_models_before_data_reads(monkeypatch, model):
    module = _load_training_script()
    monkeypatch.setattr(sys, "argv", ["train_temporal_baselines.py", "--model", model, "--demand-scaling", "shared"])
    with pytest.raises(ValueError, match="joint models only"):
        module.main()
