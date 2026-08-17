"""Tests for publisher-table metrics and direction-aware comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dma_wdf.evaluation.dcrnn_evaluator import (
    evaluate_aggregate_total_predictions,
    evaluate_predictions,
)
from dma_wdf.evaluation.paper_comparison import (
    compare_metrics_to_paper,
    load_paper_reference,
)


ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = (
    ROOT / "configs" / "evaluation" / "mscmnet_paper_metrics.yaml"
)


def test_reference_locks_publisher_total_values() -> None:
    reference = load_paper_reference(REFERENCE_PATH)
    assert reference["paper"]["protocol"] == "common_46"
    assert reference["paper"]["expected_sequences"] == 46
    assert reference["paper"]["metric_conventions"] == {
        "dma_rows": "metric_per_dma",
        "total_MAE": "sum_of_A_to_J_dma_mae",
        "total_MAPE_RMSE_NSE": (
            "metric_on_hourly_sum_of_A_to_J_demand"
        ),
    }
    assert reference["tasks"]["24h"]["MSCMNet_W"]["total"] == {
        "MAE": 14.471,
        "MAPE": 0.026,
        "RMSE": 7.586,
        "NSE": 0.959,
    }
    assert reference["tasks"]["168h"]["MSCMNet_W"]["total"] == {
        "MAE": 14.950,
        "MAPE": 0.026,
        "RMSE": 7.756,
        "NSE": 0.960,
    }


def test_publisher_total_mae_and_aggregate_total_are_separate() -> None:
    true = np.asarray([[[1.0, 9.0], [2.0, 8.0]]])
    pred = np.asarray([[[2.0, 8.0], [4.0, 6.0]]])
    frame = evaluate_predictions(
        y_true=true,
        y_pred=pred,
        dma_names=["A", "B"],
    )
    total = frame.loc[frame["entity"] == "total"].iloc[0]
    assert total["MAE"] == pytest.approx(3.0)
    assert total["RMSE"] == pytest.approx(0.0)
    assert total["NSE"] != frame.loc[
        frame["entity"] == "A", "NSE"
    ].iloc[0]
    aggregate = evaluate_aggregate_total_predictions(
        y_true=true,
        y_pred=pred,
    )
    aggregate_total = aggregate.iloc[0]
    assert aggregate_total["entity"] == "aggregate_total"
    assert aggregate_total["MAE"] == pytest.approx(0.0)
    assert aggregate_total["RMSE"] == pytest.approx(0.0)


def test_comparison_uses_metric_direction() -> None:
    reference = load_paper_reference(REFERENCE_PATH)
    entities = list("ABCDEFGHIJ") + ["total"]
    rows = []
    for entity in entities:
        paper = reference["tasks"]["24h"]["MSCMNet_W"][entity]
        rows.append(
            {
                "entity": entity,
                "MAE": paper["MAE"] - 0.01,
                "MAPE": paper["MAPE"] - 0.001,
                "RMSE": paper["RMSE"] - 0.01,
                "NSE": paper["NSE"] + 0.01,
            }
        )
    comparison = compare_metrics_to_paper(
        task="24h",
        dcrnn_metrics=pd.DataFrame(rows),
        reference=reference,
    )
    primary = comparison.loc[comparison["primary_reference"]]
    assert primary["beats_paper"].all()
    assert (primary["improvement"] > 0.0).all()
    assert set(primary["direction"]) == {"lower", "higher"}
