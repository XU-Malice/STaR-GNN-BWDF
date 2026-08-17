#!/usr/bin/env python
"""Build submission-ready CSV/Markdown tables from evaluated checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from paper_release_lib import METRICS, PROJECT_ROOT, STAR_VARIANTS, TASKS, read_metrics


def _metric_path(root: Path, family: str, model: str, task: str, frozen: bool) -> Path:
    prefix = root / "models" if frozen else root
    return (
        prefix
        / family
        / model
        / task
        / "seed_0"
        / "evaluation"
        / "metrics_aggregate_total_common_46.csv"
    )


def _publisher_total_path(
    root: Path, family: str, model: str, task: str, frozen: bool
) -> Path:
    return _metric_path(root, family, model, task, frozen).with_name(
        "metrics_common_46.csv"
    )


def _read_publisher_total(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = next(
        (row for row in rows if row.get("entity", "").lower() == "total"),
        None,
    )
    if selected is None:
        raise ValueError(f"Missing publisher-convention total row: {path}")
    mape = float(selected["MAPE"])
    if mape < 1.0:
        mape *= 100.0
    return {
        "MAE": float(selected["MAE"]),
        "MAPE": mape,
        "RMSE": float(selected["RMSE"]),
        "NSE": float(selected["NSE"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {model} | {MAE:.6f} | {MAPE:.6f} | "
            "{RMSE:.6f} | {NSE:.6f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-layout", action="store_true")
    args = parser.parse_args()
    root = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    ablation: list[dict[str, Any]] = []
    for task in TASKS:
        for label in STAR_VARIANTS:
            values = read_metrics(
                _metric_path(root, "star_gnn", label, task, args.frozen_layout)
            )
            public_name = "DCRNN" if label == "Base" else label
            ablation.append({"task": task, "model": public_name, **values})

    comparison: list[dict[str, Any]] = []
    reference = yaml.safe_load(
        (PROJECT_ROOT / "configs/evaluation/mscmnet_paper_metrics.yaml").read_text(
            encoding="utf-8"
        )
    )
    for task in TASKS:
        for model in ("MSCMNet_WM", "MSCMNet_W"):
            values = dict(reference["tasks"][task][model]["total"])
            values["MAPE"] = 100.0 * float(values["MAPE"])
            comparison.append(
                {"task": task, "model": f"{model} (reported)", **values}
            )
        dcrnn = _read_publisher_total(
            _publisher_total_path(
                root, "star_gnn", "Base", task, args.frozen_layout
            )
        )
        comparison.append({"task": task, "model": "DCRNN", **dcrnn})
        stgcn = _read_publisher_total(
            _publisher_total_path(
                root, "baselines", "stgcn", task, args.frozen_layout
            )
        )
        comparison.append({"task": task, "model": "STGCN", **stgcn})
        full = _read_publisher_total(
            _publisher_total_path(
                root, "star_gnn", "Full", task, args.frozen_layout
            )
        )
        comparison.append({"task": task, "model": "STaR-GNN", **full})

    _write_csv(output / "table_ablation_common46.csv", ablation)
    _write_csv(output / "table_comparison_common46.csv", comparison)
    (output / "table_ablation_common46.md").write_text(
        _markdown(ablation), encoding="utf-8"
    )
    (output / "table_comparison_common46.md").write_text(
        _markdown(comparison), encoding="utf-8"
    )
    print(f"Paper tables: {output}")


if __name__ == "__main__":
    main()
