#!/usr/bin/env python
"""Build submission-ready paper tables from evaluated common-46 checkpoints.

Manuscript-facing model comparison and ablation tables use the MSCMNet
publisher-compatible ``total`` convention:

- total MAE = sum of DMA A--J MAEs;
- total MAPE/RMSE/NSE = metric on the hourly aggregate-demand series.

The pure aggregate-demand view is retained only as an internal/operational
artifact.  STaR-GNN DMA-level metrics are reported separately without any
cross-DMA aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from paper_release_lib import METRICS, PROJECT_ROOT, TASKS, better, read_metrics


LITERATURE_TOTALS = PROJECT_ROOT / "configs/evaluation/mscmnet_literature_totals.yaml"
MSCMNET_DETAILED = PROJECT_ROOT / "configs/evaluation/mscmnet_paper_metrics.yaml"
INTERNAL_MODELS = (
    ("STGCN", "baselines", "stgcn"),
    ("DCRNN", "star_gnn", "Base"),
    ("STaR-GNN", "star_gnn", "Full"),
)
PUBLISHER_MODELS = (
    ("DCRNN", "star_gnn", "Base"),
    ("STGCN", "baselines", "stgcn"),
    ("STaR-GNN", "star_gnn", "Full"),
)
PUBLISHER_ABLATION_MODELS = (
    ("STGCN", "baselines", "stgcn"),
    ("DCRNN", "star_gnn", "Base"),
    ("DCRNN + SAS-Norm", "star_gnn", "State"),
    ("DCRNN + FA-DPR", "star_gnn", "FA-DPR"),
    ("STaR-GNN", "star_gnn", "Full"),
)


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


def _read_dma_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row.get("entity", "") in list("ABCDEFGHIJ")]
    if [row["entity"] for row in selected] != list("ABCDEFGHIJ"):
        raise ValueError(f"DMA A-J rows missing or out of order: {path}")
    output: list[dict[str, float | str]] = []
    for row in selected:
        mape = float(row["MAPE"])
        if mape < 1.0:
            mape *= 100.0
        output.append(
            {
                "DMA": row["entity"],
                "MAE": float(row["MAE"]),
                "MAPE": mape,
                "RMSE": float(row["RMSE"]),
                "NSE": float(row["NSE"]),
            }
        )
    return output


def _load_literature_reference() -> dict[str, Any]:
    reference = yaml.safe_load(LITERATURE_TOTALS.read_text(encoding="utf-8"))
    paper = reference["paper"]
    expected_conventions = {
        "total_MAE": "sum_of_A_to_J_dma_mae",
        "total_MAPE_RMSE_NSE": "metric_on_hourly_sum_of_A_to_J_demand",
    }
    if paper.get("metric_conventions") != expected_conventions:
        raise ValueError(f"Unexpected literature metric conventions: {paper.get('metric_conventions')!r}")
    if paper.get("protocol") != "common_46" or int(paper.get("expected_sequences", 0)) != 46:
        raise ValueError("Literature comparison must use the common-46 protocol.")
    expected_models = ["GRU", "LSTM", "MSNet", "MSCMNet_WM", "MSCMNet_M", "MSCMNet_W"]
    if list(paper.get("model_order", [])) != expected_models:
        raise ValueError("Unexpected literature model order.")
    for task, horizon in (("24h", 24), ("168h", 168)):
        task_reference = reference["tasks"][task]
        if int(task_reference["horizon"]) != horizon:
            raise ValueError(f"Literature horizon mismatch for {task}.")
        for model in expected_models:
            values = task_reference[model]
            if set(values) != {"MAE", "MAPE", "RMSE", "NSE"}:
                raise ValueError(f"Incomplete literature metrics for {task}/{model}.")
    return reference


def _validate_against_detailed_mscmnet(reference: dict[str, Any]) -> None:
    detailed = yaml.safe_load(MSCMNET_DETAILED.read_text(encoding="utf-8"))
    for task in TASKS:
        for model in ("MSCMNet_WM", "MSCMNet_W"):
            dma_sum = sum(
                float(detailed["tasks"][task][model][dma]["MAE"])
                for dma in list("ABCDEFGHIJ")
            )
            detailed_total = float(detailed["tasks"][task][model]["total"]["MAE"])
            compact_total = float(reference["tasks"][task][model]["MAE"])
            if abs(dma_sum - detailed_total) > 0.005:
                raise ValueError(
                    f"Publisher total MAE is not the A-J sum for {task}/{model}: "
                    f"{detailed_total} vs {dma_sum}."
                )
            if abs(compact_total - detailed_total) > 1.0e-12:
                raise ValueError(
                    f"Literature total drift for {task}/{model}: "
                    f"{compact_total} vs {detailed_total}."
                )


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


def _markdown_dma(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Horizon | DMA | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {DMA} | {MAE:.6f} | {MAPE:.6f} | "
            "{RMSE:.6f} | {NSE:.6f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _build_internal(root: Path, frozen: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for public_name, family, model in INTERNAL_MODELS:
            values = read_metrics(_metric_path(root, family, model, task, frozen))
            rows.append({"task": task, "model": public_name, **values})
    return rows


def _build_literature(root: Path, frozen: bool, reference: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reported_models = list(reference["paper"]["model_order"])
    for task in TASKS:
        for model in reported_models:
            values = dict(reference["tasks"][task][model])
            values["MAPE"] = 100.0 * float(values["MAPE"])
            rows.append({"task": task, "model": f"{model} (reported)", **values})
        for public_name, family, model in PUBLISHER_MODELS:
            values = _read_publisher_total(
                _publisher_total_path(root, family, model, task, frozen)
            )
            rows.append({"task": task, "model": public_name, **values})
    return rows


def _build_ablation(root: Path, frozen: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for public_name, family, model in PUBLISHER_ABLATION_MODELS:
            values = _read_publisher_total(
                _publisher_total_path(root, family, model, task, frozen)
            )
            rows.append({"task": task, "model": public_name, **values})
    return rows


def _build_star_dma(root: Path, frozen: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        path = _publisher_total_path(root, "star_gnn", "Full", task, frozen)
        for values in _read_dma_rows(path):
            rows.append({"task": task, **values})
    return rows


def _validate_no_mae_mixing(internal: list[dict[str, Any]], literature: list[dict[str, Any]]) -> None:
    internal_lookup = {(row["task"], row["model"]): row for row in internal}
    literature_lookup = {(row["task"], row["model"]): row for row in literature}
    for task in TASKS:
        for model in ("DCRNN", "STGCN", "STaR-GNN"):
            internal_row = internal_lookup[(task, model)]
            literature_row = literature_lookup[(task, model)]
            for metric in ("MAPE", "RMSE", "NSE"):
                if abs(float(internal_row[metric]) - float(literature_row[metric])) > 1.0e-8:
                    raise ValueError(
                        f"Unexpected {metric} convention drift for {task}/{model}: "
                        f"{internal_row[metric]} vs {literature_row[metric]}."
                    )
            if abs(float(internal_row["MAE"]) - float(literature_row["MAE"])) < 1.0e-8:
                raise ValueError(f"MAE conventions accidentally collapsed for {task}/{model}.")


def _validate_ablation(ablation: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["task"], row["model"]): row for row in ablation}
    relations = (
        ("DCRNN + SAS-Norm", "DCRNN"),
        ("DCRNN + FA-DPR", "DCRNN"),
        ("STaR-GNN", "DCRNN + SAS-Norm"),
        ("STaR-GNN", "DCRNN + FA-DPR"),
    )
    details: dict[str, str] = {}
    passed = 0
    for task in TASKS:
        for left, right in relations:
            count = sum(
                int(better(metric, float(lookup[(task, left)][metric]), float(lookup[(task, right)][metric])))
                for metric in METRICS
            )
            passed += count
            details[f"{task}_{left}_vs_{right}"] = f"{count}/4"
    expected = {
        "24h_DCRNN + SAS-Norm_vs_DCRNN": "4/4",
        "24h_DCRNN + FA-DPR_vs_DCRNN": "4/4",
        "24h_STaR-GNN_vs_DCRNN + SAS-Norm": "4/4",
        "24h_STaR-GNN_vs_DCRNN + FA-DPR": "4/4",
        "168h_DCRNN + SAS-Norm_vs_DCRNN": "4/4",
        "168h_DCRNN + FA-DPR_vs_DCRNN": "3/4",
        "168h_STaR-GNN_vs_DCRNN + SAS-Norm": "3/4",
        "168h_STaR-GNN_vs_DCRNN + FA-DPR": "4/4",
    }
    if details != expected or passed != 30:
        raise ValueError(f"Publisher-compatible ablation audit drift: {passed}/32, {details}")
    return {
        "protocol": "common_46",
        "mae_convention": "sum_of_A_to_J_dma_mae",
        "other_metrics": "metric_on_hourly_sum_of_A_to_J_demand",
        "passed_relations": "30/32",
        "relations": details,
        "transparent_exceptions": [
            "FA-DPR 168h MAPE is slightly worse than DCRNN.",
            "STaR-GNN 168h sum-of-DMA MAE is slightly higher than SAS-Norm-only (12.233590 vs 12.207835).",
        ],
    }


def _write_convention_note(output: Path) -> None:
    text = (
        "# Metric conventions\n\n"
        "Manuscript-facing overall comparison and ablation tables use one "
        "publisher-compatible convention.\n\n"
        "- Total MAE: sum of DMA A--J MAEs.\n"
        "- Total MAPE/RMSE/NSE: metric on the hourly aggregate-demand series.\n"
        "- `table_literature_comparison_common46.*`: nine-model overall comparison.\n"
        "- `table_ablation_common46.*`: STGCN/DCRNN/SAS-Norm/FA-DPR/STaR-GNN "
        "publisher-compatible ablation.\n"
        "- `table_star_gnn_dma_common46.*`: STaR-GNN DMA A--J metrics, with no "
        "cross-DMA aggregation.\n"
        "- `table_internal_common46.*`: retained only for aggregate-demand "
        "operational diagnostics; do not use it for cross-paper MAE comparisons.\n"
    )
    (output / "METRIC_CONVENTIONS.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-layout", action="store_true")
    args = parser.parse_args()
    root = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference = _load_literature_reference()
    _validate_against_detailed_mscmnet(reference)

    internal = _build_internal(root, args.frozen_layout)
    literature = _build_literature(root, args.frozen_layout, reference)
    ablation = _build_ablation(root, args.frozen_layout)
    star_dma = _build_star_dma(root, args.frozen_layout)

    _validate_no_mae_mixing(internal, literature)
    ablation_audit = _validate_ablation(ablation)

    _write_csv(output / "table_internal_common46.csv", internal)
    _write_csv(output / "table_literature_comparison_common46.csv", literature)
    _write_csv(output / "table_comparison_common46.csv", literature)
    _write_csv(output / "table_ablation_common46.csv", ablation)
    _write_csv(output / "table_star_gnn_dma_common46.csv", star_dma)

    (output / "table_internal_common46.md").write_text(_markdown(internal), encoding="utf-8")
    (output / "table_literature_comparison_common46.md").write_text(_markdown(literature), encoding="utf-8")
    (output / "table_comparison_common46.md").write_text(_markdown(literature), encoding="utf-8")
    (output / "table_ablation_common46.md").write_text(_markdown(ablation), encoding="utf-8")
    (output / "table_star_gnn_dma_common46.md").write_text(_markdown_dma(star_dma), encoding="utf-8")
    (output / "table_ablation_audit.json").write_text(
        json.dumps(ablation_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_convention_note(output)

    print(f"Paper tables: {output}")
    print("Metric convention audit: PASS")
    print("Publisher-compatible ablation audit: 30/32 PASS")


if __name__ == "__main__":
    main()
