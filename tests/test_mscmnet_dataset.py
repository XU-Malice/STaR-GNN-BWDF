"""Leakage and FC2 data guards for the MSCMNet baseline pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dma_wdf.data.mscmnet_dataset import (
    daily_share_history,
    validate_leakage_safe_data_build,
)


def _write_audit_files(tmp_path, *, threshold_source: str) -> None:
    (tmp_path / "quality_checks.json").write_text(
        json.dumps({"all_passed": True}), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "dma_column": ["DMA 1"],
            "threshold_source": [threshold_source],
            "fit_start": ["2021-01-01 00:00:00+01:00"],
            "fit_end": ["2022-12-15 23:00:00+01:00"],
            "fit_rows": [17136],
            "lower": [0.0],
            "upper": [100.0],
        }
    ).to_csv(tmp_path / "demand_iqr_thresholds.csv", index=False)
    pd.DataFrame(
        {
            "split": ["train", "test"],
            "column": ["DMA 1", "DMA 1"],
        }
    ).to_csv(tmp_path / "interpolation_split_profile.csv", index=False)


def test_formal_data_audit_accepts_train_only_thresholds(tmp_path) -> None:
    _write_audit_files(tmp_path, threshold_source="train")
    audit = validate_leakage_safe_data_build(
        tmp_path,
        expected_train_end=pd.Timestamp("2022-12-15 23:00:00+01:00"),
    )
    assert audit["iqr_threshold_source"] == "train"
    assert audit["interpolation_splits"] == ["test", "train"]


def test_formal_data_audit_rejects_full_period_thresholds(tmp_path) -> None:
    _write_audit_files(tmp_path, threshold_source="full_paper_period")
    with pytest.raises(ValueError, match="training rows"):
        validate_leakage_safe_data_build(tmp_path)


def test_fc2_history_uses_only_pre_forecast_demand() -> None:
    index = pd.date_range(
        "2023-01-01", periods=10 * 24, freq="h", tz="Europe/Rome"
    )
    demand = pd.DataFrame(
        {
            "DMA 1": np.ones(len(index), dtype=float),
            "DMA 2": np.full(len(index), 3.0),
        },
        index=index,
    )
    weather = pd.DataFrame(
        {"air_temperature": np.arange(len(index), dtype=float)}, index=index
    )
    forecast_start = index[7 * 24]
    first = daily_share_history(
        demand=demand,
        weather=weather,
        starts=[forecast_start],
        dma_columns=["DMA 1", "DMA 2"],
        history_days=7,
        include_temperature=False,
    )
    demand.loc[demand.index >= forecast_start, :] = 9999.0
    second = daily_share_history(
        demand=demand,
        weather=weather,
        starts=[forecast_start],
        dma_columns=["DMA 1", "DMA 2"],
        history_days=7,
        include_temperature=False,
    )
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[0, :, 0], 0.25)
    np.testing.assert_allclose(first[0, :, 1], 0.75)
