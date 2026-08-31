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
