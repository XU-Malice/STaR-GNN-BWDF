#!/usr/bin/env python
"""Summarize one-seed Que et al. paper-gap reconstruction candidates.

The reported paper test table is used only to diagnose unpublished training
details and to rank reconstruction candidates.  This is therefore a
retrospective numerical reconstruction, not an unbiased model-selection
benchmark.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE_SUMMARY = ROOT / "scripts/reproduce/summarize_que_complete_reproduction.py"
SPEC = importlib.util.spec_from_file_location("que_complete_summary", BASE_SUMMARY)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"Cannot import {BASE_SUMMARY}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

ERROR_METRICS = ("MAE", "MAPE", "RMSE")
ALL_METRICS = (*ERROR_METRICS, "NSE")


def acceptance_fields(
    scores: dict[str, float],
    *,
    error_relative_tolerance: float,
    nse_absolute_tolerance: float,
) -> dict[str, object]:
    """Return strict eight-total-metric acceptance diagnostics."""
    passed = 0
    relative_gaps: list[float] = []
    nse_gaps: list[float] = []
    normalized_gaps: list[float] = []
    output: dict[str, object] = {}
    for task in ("24h", "168h"):
        for metric in ALL_METRICS:
            actual = float(scores[f"{task}_{metric}"])
            expected = float(scores[f"{task}_{metric}_paper"])
            absolute = abs(actual - expected)
            if metric == "NSE":
                gap = absolute
                limit = nse_absolute_tolerance
                nse_gaps.append(gap)
            else:
                gap = absolute / max(abs(expected), 1.0e-12)
                limit = error_relative_tolerance
                relative_gaps.append(gap)
            metric_pass = gap <= limit
            passed += int(metric_pass)
            normalized_gaps.append(gap / limit)
            output[f"{task}_{metric}_accepted"] = metric_pass
    output.update(
        {
            "accepted_metrics": passed,
            "all_8_total_metrics_accepted": passed == 8,
            "max_error_relative_gap": max(relative_gaps),
            "max_nse_absolute_gap": max(nse_gaps),
            "max_tolerance_ratio": max(normalized_gaps),
        }
    )
    return output


def summarize(
    *,
    result_root: Path,
    manifest_path: Path,
    paper_path: Path,
    error_relative_tolerance: float,
    nse_absolute_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
    paper = yaml.safe_load(paper_path.read_text(encoding="utf-8"))["tasks"]
    rows: list[dict[str, object]] = []
    for item in manifest.to_dict("records"):
        run = result_root / item["case"] / item["model"] / f"seed_{item['seed']}"
        metrics_path = run / "metrics.csv"
        status_path = run / "status.json"
        if not metrics_path.is_file() or not status_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            continue
        display = BASE.MODEL_DISPLAY[item["model"]]
        reference = {task: paper[task][display] for task in ("24h", "168h")}
        scores = BASE.score_run(pd.read_csv(metrics_path), reference)
        accepted = acceptance_fields(
            scores,
            error_relative_tolerance=error_relative_tolerance,
            nse_absolute_tolerance=nse_absolute_tolerance,
        )
        rows.append(
            {
                **item,
                **scores,
                **accepted,
                "elapsed_seconds": float(status["elapsed_seconds"]),
                "formal_protocol": bool(status.get("formal_protocol", False)),
                "selection_provenance": ("paper_test_reverse_engineering_diagnostic"),
            }
        )

    all_candidates = pd.DataFrame(rows)
    if all_candidates.empty:
        raise ValueError("No completed targeted candidates were found.")
    sort_columns = [
        "model",
        "accepted_metrics",
        "max_tolerance_ratio",
        "aggregate_paper_error",
        "dma_mae_rmse_relative_error",
        "case",
    ]
    ascending = [True, False, True, True, True, True]
    best = (
        all_candidates.sort_values(sort_columns, ascending=ascending)
        .groupby("model", as_index=False)
        .first()
        .sort_values("model")
    )

    detail_rows: list[dict[str, object]] = []
    for item in best.to_dict("records"):
        for task in ("24h", "168h"):
            for metric in ALL_METRICS:
                actual = float(item[f"{task}_{metric}"])
                expected = float(item[f"{task}_{metric}_paper"])
                detail_rows.append(
                    {
                        "model": item["model"],
                        "case": item["case"],
                        "task": task,
                        "metric": metric,
                        "paper": expected,
                        "actual": actual,
                        "signed_gap": actual - expected,
                        "relative_gap": (
                            None
                            if metric == "NSE"
                            else (actual - expected) / abs(expected)
                        ),
                        "accepted": bool(item[f"{task}_{metric}_accepted"]),
                    }
                )
    details = pd.DataFrame(detail_rows)
    return all_candidates, best, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paper-metrics", type=Path, required=True)
    parser.add_argument("--error-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--nse-absolute-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if args.error_relative_tolerance <= 0.0:
        raise ValueError("--error-relative-tolerance must be positive.")
    if args.nse_absolute_tolerance <= 0.0:
        raise ValueError("--nse-absolute-tolerance must be positive.")

    root = args.result_root.resolve()
    all_candidates, best, details = summarize(
        result_root=root,
        manifest_path=args.manifest.resolve(),
        paper_path=args.paper_metrics.resolve(),
        error_relative_tolerance=float(args.error_relative_tolerance),
        nse_absolute_tolerance=float(args.nse_absolute_tolerance),
    )
    all_candidates.to_csv(root / "all_targeted_candidates.tsv", sep="\t", index=False)
    best.to_csv(root / "best_by_model.tsv", sep="\t", index=False)
    details.to_csv(root / "best_metric_gaps.tsv", sep="\t", index=False)

    required = set(BASE.MODEL_DISPLAY)
    present = set(best["model"])
    missing = sorted(required - present)
    accepted = set(best.loc[best["all_8_total_metrics_accepted"], "model"])
    summary = {
        "models_with_results": len(present),
        "models_accepted": len(accepted),
        "all_models_accepted": present == required and accepted == required,
        "accepted_models": sorted(accepted),
        "missing_models": missing,
        "error_relative_tolerance": float(args.error_relative_tolerance),
        "nse_absolute_tolerance": float(args.nse_absolute_tolerance),
        "selection_provenance": "paper_test_reverse_engineering_diagnostic",
    }
    (root / "acceptance_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
