#!/usr/bin/env python
"""Score six-model Que et al. reconstruction candidates against paper tables.

This is a reverse-engineering diagnostic, not an unbiased model-selection
procedure: paper test metrics are deliberately used to rank implementation
ambiguities.  The output labels that provenance explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


MODEL_DISPLAY = {
    "gru": "GRU",
    "lstm": "LSTM",
    "msnet": "MSNet",
    "mscmnet_m": "MSCMNet_M",
    "mscmnet_wm": "MSCMNet_WM",
    "mscmnet_w": "MSCMNet_W",
}
METRICS = ("MAE", "MAPE", "RMSE", "NSE")
DMA_METRICS = ("MAE", "RMSE")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _lookup(table: pd.DataFrame, task: str, series: str, metric: str) -> float:
    values = table.loc[
        (table["task"] == task)
        & (table["series"] == series)
        & (table["metric"] == metric),
        "value",
    ]
    if len(values) != 1:
        raise ValueError(f"Expected one value for {task}/{series}/{metric}")
    return float(values.iloc[0])


def score_run(metrics: pd.DataFrame, reference: dict) -> dict[str, float]:
    aggregate_errors: list[float] = []
    dma_relative_errors: list[float] = []
    correlations: list[float] = []
    output: dict[str, float] = {}
    for task in ("24h", "168h"):
        paper_task = reference[task]
        for metric in METRICS:
            actual = _lookup(metrics, task, "total", metric)
            expected = float(paper_task["total"][metric])
            gap = actual - expected
            output[f"{task}_{metric}"] = actual
            output[f"{task}_{metric}_paper"] = expected
            output[f"{task}_{metric}_gap"] = gap
            error = abs(gap) if metric == "NSE" else abs(gap) / abs(expected)
            aggregate_errors.append(error)
        for metric in DMA_METRICS:
            actual_dma = np.asarray(
                [_lookup(metrics, task, letter, metric) for letter in "ABCDEFGHIJ"]
            )
            paper_dma = np.asarray(
                [float(paper_task[letter][metric]) for letter in "ABCDEFGHIJ"]
            )
            dma_relative_errors.extend(
                np.abs(actual_dma - paper_dma) / np.maximum(np.abs(paper_dma), 1e-12)
            )
            correlations.append(_correlation(actual_dma, paper_dma))
    aggregate = float(np.mean(aggregate_errors))
    dma_relative = float(np.mean(dma_relative_errors))
    dma_correlation = float(np.mean(correlations))
    output.update(
        {
            "aggregate_paper_error": aggregate,
            "dma_mae_rmse_relative_error": dma_relative,
            "dma_mae_rmse_correlation": dma_correlation,
            "paper_match_score": (
                0.55 * aggregate
                + 0.35 * dma_relative
                + 0.10 * ((1.0 - dma_correlation) / 2.0)
            ),
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paper-metrics", type=Path, required=True)
    args = parser.parse_args()

    root = args.result_root.resolve()
    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str).fillna("")
    paper = yaml.safe_load(args.paper_metrics.read_text(encoding="utf-8"))["tasks"]
    rows: list[dict[str, object]] = []
    for item in manifest.to_dict("records"):
        run = root / item["case"] / item["model"] / f"seed_{item['seed']}"
        metrics_path = run / "metrics.csv"
        status_path = run / "status.json"
        if not metrics_path.is_file() or not status_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            continue
        model = item["model"]
        display = MODEL_DISPLAY[model]
        reference = {
            task: paper[task][display] for task in ("24h", "168h")
        }
        scores = score_run(pd.read_csv(metrics_path), reference)
        rows.append(
            {
                **item,
                **scores,
                "elapsed_seconds": float(status["elapsed_seconds"]),
                "train_samples": int(
                    status.get("train_samples", status.get("last_dma_train_samples", 0))
                ),
                "selection_provenance": "paper_test_reverse_engineering_diagnostic",
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(root / "all_case_paper_gaps.tsv", sep="\t", index=False)
    screening = summary.loc[summary["phase"] == "screening"].copy()
    if screening.empty:
        raise SystemExit("No completed screening cases found")
    winners = (
        screening.sort_values(["model", "paper_match_score", "case"])
        .groupby("model", as_index=False)
        .first()
        .sort_values("model")
    )
    if set(winners["model"]) != set(MODEL_DISPLAY):
        missing = sorted(set(MODEL_DISPLAY) - set(winners["model"]))
        raise SystemExit(f"Missing screening winner(s): {missing}")
    winners.to_csv(root / "selected_candidates.tsv", sep="\t", index=False)

    selected_names = set(winners["case"])
    robustness = pd.concat(
        [
            winners,
            summary.loc[
                (summary["phase"] == "robustness")
                & summary["candidate"].isin(selected_names)
            ],
        ],
        ignore_index=True,
    )
    if not robustness.empty:
        numeric = [
            *[f"{task}_{metric}" for task in ("24h", "168h") for metric in METRICS],
            "paper_match_score",
            "aggregate_paper_error",
            "dma_mae_rmse_relative_error",
            "dma_mae_rmse_correlation",
        ]
        grouped = robustness.groupby("model")[numeric].agg(["mean", "std", "min", "max"])
        grouped.columns = ["_".join(column) for column in grouped.columns]
        grouped.reset_index().to_csv(
            root / "selected_seed_robustness.tsv", sep="\t", index=False
        )


if __name__ == "__main__":
    main()
