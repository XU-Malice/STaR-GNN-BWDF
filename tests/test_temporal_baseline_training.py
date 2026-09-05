"""Training-pipeline guards that require the CI PyTorch environment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
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
