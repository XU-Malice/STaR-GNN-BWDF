"""Tests for the leakage-safe preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


sys.path.insert(0, str(Path(__file__).resolve().parent))

from dma_wdf.data.interpolation import interpolate_by_splits
from dma_wdf.data.outlier_detection import (
    apply_iqr_thresholds,
    fit_iqr_thresholds,
    preprocess_demand,
)


def _hourly(periods: int) -> pd.DatetimeIndex:
    return pd.date_range(
        "2022-12-15 20:00:00",
        periods=periods,
        freq="h",
        tz="Europe/Rome",
    )


def test_split_interpolation_cannot_use_test_value_for_train() -> None:
    index = _hourly(8)
    original = pd.DataFrame(
        {"DMA 1": [1.0, 1.0, 1.0, np.nan, 100.0, 100.0, 100.0, 100.0]},
        index=index,
    )
    changed_test = original.copy()
    changed_test.loc[index[4]:, "DMA 1"] = 10_000.0
    splits = {
        "train": (index[0], index[3]),
        "test": (index[4], index[7]),
    }

    result_a, _ = interpolate_by_splits(original, splits)
    result_b, _ = interpolate_by_splits(changed_test, splits)

    assert_frame_equal(result_a.loc[index[:4]], result_b.loc[index[:4]])
    assert result_a.loc[index[3], "DMA 1"] == pytest.approx(1.0)


def test_iqr_thresholds_are_fitted_only_on_training_rows() -> None:
    index = _hourly(12)
    train = pd.DataFrame(
        {
            "DMA 1": [8.0, 9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0],
            "DMA 2": [18.0, 19.0, 19.0, 20.0, 20.0, 21.0, 21.0, 22.0],
        },
        index=index[:8],
    )
    thresholds_a = fit_iqr_thresholds(train, threshold_source="train")

    altered_test = pd.DataFrame(
        {
            "DMA 1": [1.0e3, 2.0e3, 3.0e3, 4.0e3],
            "DMA 2": [5.0e3, 6.0e3, 7.0e3, 8.0e3],
        },
        index=index[8:],
    )
    _ = altered_test
    thresholds_b = fit_iqr_thresholds(train, threshold_source="train")

    assert_frame_equal(thresholds_a, thresholds_b)
    assert set(thresholds_a["threshold_source"]) == {"train"}
    assert set(thresholds_a["fit_rows"]) == {8}


def test_fixed_train_thresholds_are_applied_to_test_outlier() -> None:
    index = _hourly(12)
    train = pd.DataFrame(
        {"DMA 1": [8.0, 9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0]},
        index=index[:8],
    )
    test = pd.DataFrame(
        {"DMA 1": [10.0, 1000.0, 10.0, 10.0]},
        index=index[8:],
    )
    full = pd.concat([train, test])
    thresholds = fit_iqr_thresholds(train, threshold_source="train")
    splits = {
        "train": (index[0], index[7]),
        "test": (index[8], index[11]),
    }

    processed, profile, mask = apply_iqr_thresholds(
        full,
        thresholds,
        method="interpolate",
        split_ranges=splits,
    )

    assert bool(mask.loc[index[9], "DMA 1"])
    assert processed.loc[index[9], "DMA 1"] == pytest.approx(10.0)
    assert int(profile.loc[0, "outlier_count"]) == 1
    assert profile.loc[0, "threshold_source"] == "train"


def test_train_source_requires_explicit_training_frame() -> None:
    frame = pd.DataFrame({"DMA 1": [1.0, 2.0]}, index=_hourly(2))
    with pytest.raises(ValueError, match="requires fit_demand"):
        preprocess_demand(frame, threshold_source="train")


def test_split_ranges_must_cover_frame_without_overlap() -> None:
    index = _hourly(4)
    frame = pd.DataFrame({"DMA 1": [1.0, np.nan, 2.0, 3.0]}, index=index)
    incomplete = {"train": (index[0], index[2])}
    with pytest.raises(ValueError, match="do not cover"):
        interpolate_by_splits(frame, incomplete)
