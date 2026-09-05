"""Source-only tests for final Que reproduction diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_summary_script():
    path = ROOT / "scripts/reproduce/summarize_que_final_reproduction.py"
    spec = importlib.util.spec_from_file_location("que_final_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_total_mae_uses_sum_of_dma_mae() -> None:
    module = _load_summary_script()
    truth = np.zeros((2, 24, 10), dtype=np.float32)
    prediction = np.ones_like(truth)
    metrics = module.total_metrics(truth, prediction)
    assert metrics["MAE"] == pytest.approx(10.0)
    assert metrics["RMSE"] == pytest.approx(10.0)


def test_train_only_affine_calibration_recovers_known_mapping() -> None:
    module = _load_summary_script()
    prediction = np.arange(480, dtype=np.float32).reshape(2, 24, 10)
    truth = 2.0 * prediction + 3.0
    slope, intercept = module.fit_calibration(truth, prediction, "affine")
    assert np.allclose(slope, 2.0)
    assert np.allclose(intercept, 3.0)


def test_intercept_calibration_does_not_change_slope() -> None:
    module = _load_summary_script()
    prediction = np.zeros((2, 24, 10), dtype=np.float32)
    truth = prediction + np.arange(10, dtype=np.float32)
    slope, intercept = module.fit_calibration(truth, prediction, "intercept")
    assert np.allclose(slope, 1.0)
    assert np.allclose(intercept, np.arange(10))
