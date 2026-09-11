"""CPU-only checks for immutable paper data protocol audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "que_data_audit", ROOT / "scripts/reproduce/audit_que_data_protocol.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def paper_frames():
    bounds = {key: pd.Timestamp(value) for key, value in MODULE.EXPECTED_BOUNDS.items()}
    index = pd.date_range(bounds["train_start"], bounds["test_end"], freq="h")
    values = 5 + np.sin(np.arange(len(index)) * np.pi / 12)
    return {
        "demand": pd.DataFrame({column: values + i for i, column in enumerate(MODULE.DMA_COLUMNS)}, index=index),
        "weather": pd.DataFrame({column: values for column in MODULE.WEATHER_REFERENCE}, index=index),
        "temporal": pd.DataFrame({column: index.hour for column in MODULE.TEMPORAL_COLUMNS}, index=index),
    }, bounds


def test_complete_data_audit_keeps_discrepant_values_and_common_truths(paper_frames):
    frames, bounds = paper_frames
    original = frames["demand"].copy(deep=True)
    report, origins = MODULE.audit_frames(frames, bounds)
    assert report["row_counts"] == {"train": 17136, "test": 1920}
    assert report["data_modified"] is False
    assert len(origins) == 46
    assert origins.iloc[0]["forecast_start"] == pd.Timestamp("2023-01-13T00:00:00+01:00")
    assert origins.iloc[-1]["forecast_start"] == pd.Timestamp("2023-02-27T00:00:00+01:00")
    truths = report["common_evaluation"]["truths"]
    assert truths["24h"]["shape"] == [46, 24, 10]
    assert truths["168h"]["shape"] == [46, 168, 10]
    assert truths["24h"]["total_truth_population_variance"] == pytest.approx(50, abs=1e-5)
    assert report["statistics"][0]["actual_minus_paper"] != 0
    pd.testing.assert_frame_equal(frames["demand"], original)


@pytest.mark.parametrize("problem", ["missing_hour", "duplicate", "nan", "temporal_nan", "wrong_bounds"])
def test_data_audit_fails_closed_on_invalid_inputs(paper_frames, problem):
    frames, bounds = paper_frames
    if problem == "missing_hour":
        frames["demand"] = frames["demand"].iloc[1:]
    elif problem == "duplicate":
        frame = frames["demand"]
        frames["demand"] = pd.concat([frame.iloc[:1], frame.iloc[:-1]])
    elif problem == "nan":
        frames["demand"].iloc[0, 0] = np.nan
    elif problem == "temporal_nan":
        frames["temporal"] = frames["temporal"].astype(float)
        frames["temporal"].iloc[0, 0] = np.nan
    else:
        bounds["test_start"] -= pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        MODULE.audit_frames(frames, bounds)


def test_pre_outlier_means_are_separate_diagnostics(paper_frames):
    frames, bounds = paper_frames
    before = frames["demand"] * 2
    report, _ = MODULE.audit_frames(frames, bounds, pre_outlier_demand=before)
    rows = pd.DataFrame(report["statistics"])
    means = rows[(rows.feature == "DMA A") & (rows.split == "train")].set_index("stage")
    assert means.loc["interpolated_before_outliers", "actual"] == pytest.approx(2 * means.loc["processed", "actual"])
    assert means.loc["interpolated_before_outliers", "paper_reference"] == means.loc["processed", "paper_reference"]


def test_transcription_matches_main_article_tables():
    assert MODULE.DEMAND_MEANS["train"] == (8.25, 9.60, 4.35, 33.05, 78.13, 8.02, 24.85, 20.59, 20.36, 26.34)
    assert MODULE.DEMAND_MEANS["test"] == (6.57, 9.56, 2.94, 32.18, 81.66, 10.46, 26.37, 23.92, 23.89, 24.15)
    assert MODULE.WEATHER_REFERENCE["windspeed"]["test"] == (16.42, 17.90, 1.0, 77.0)


def test_cli_missing_data_writes_failure_and_nonzero_status(tmp_path, monkeypatch):
    out = tmp_path / "audit"
    monkeypatch.setattr(MODULE.sys, "argv", [
        "audit_que_data_protocol.py", "--data-dir", str(tmp_path / "missing"),
        "--output-root", str(out),
    ])
    assert MODULE.main() == 1
    failure = json.loads((out / "paper_data_statistics_failure.json").read_text())
    assert failure["status"] == "INVALID_INPUTS"
    assert failure["data_modified"] is False
