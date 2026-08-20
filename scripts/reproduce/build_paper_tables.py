#!/usr/bin/env python
"""Build manuscript-facing result tables from frozen common-46 evaluations.

Two MAE conventions are intentionally kept separate:

* manuscript / publisher-compatible total MAE = sum of DMA A--J MAEs;
* internal aggregate-demand MAE = MAE after summing A--J at each hour.

The manuscript ablation is strictly factorial and contains only DCRNN,
DCRNN + SAS-Norm, DCRNN + FA-DPR, and STaR-GNN. STGCN is an independent
graph baseline and belongs only to the overall comparison.

CSV/JSON audit artifacts retain full numeric precision. Human-facing Markdown
tables use a uniform three-decimal display precision; no metric or model is
rounded differently to alter apparent ranking.
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
    ("DCRNN", "star_gnn", "Base"),
    ("DCRNN + SAS-Norm", "star_gnn", "State"),
    ("DCRNN + FA-DPR", "star_gnn", "FA-DPR"),
    ("STaR-GNN", "star_gnn", "Full"),
)


def _evaluation_dir(root: Path, family: str, model: str, task: str, frozen: bool) -> Path:
    prefix = root / "models" if frozen else root
    return prefix / family / model / task / "seed_0" / "evaluation"


def _aggregate_path(root: Path, family: str, model: str, task: str, frozen: bool) -> Path:
    return _evaluation_dir(root, family, model, task, frozen) / "metrics_aggregate_total_common_46.csv"


def _publisher_path(root: Path, family: str, model: str, task: str, frozen: bool) -> Path:
    return _evaluation_dir(root, family, model, task, frozen) / "metrics_common_46.csv"


def _read_publisher_total(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next((item for item in rows if item.get("entity", "").lower() == "total"), None)
    if row is None:
        raise ValueError(f"Missing publisher total row: {path}")
    mape = float(row["MAPE"])
    if mape < 1.0:
        mape *= 100.0
    return {
        "MAE": float(row["MAE"]),
        "MAPE": mape,
        "RMSE": float(row["RMSE"]),
        "NSE": float(row["NSE"]),
    }


def _read_dma_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("entity") in list("ABCDEFGHIJ")]
    if [row["entity"] for row in rows] != list("ABCDEFGHIJ"):
        raise ValueError(f"DMA A-J rows missing or out of order: {path}")
    output: list[dict[str, Any]] = []
    for row in rows:
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
    expected = {
        "total_MAE": "sum_of_A_to_J_dma_mae",
        "total_MAPE_RMSE_NSE": "metric_on_hourly_sum_of_A_to_J_demand",
    }
    if paper.get("metric_conventions") != expected:
        raise ValueError("Unexpected literature metric conventions.")
    if paper.get("protocol") != "common_46" or int(paper.get("expected_sequences", 0)) != 46:
        raise ValueError("Literature comparison must use common_46 with 46 origins.")
    expected_models = ["GRU", "LSTM", "MSNet", "MSCMNet_WM", "MSCMNet_M", "MSCMNet_W"]
    if list(paper.get("model_order", [])) != expected_models:
        raise ValueError("Unexpected literature model order.")
    return reference


def _validate_reported_reference(reference: dict[str, Any]) -> None:
    detailed = yaml.safe_load(MSCMNET_DETAILED.read_text(encoding="utf-8"))
    for task in TASKS:
        for model in ("MSCMNet_WM", "MSCMNet_W"):
            dma_sum = sum(float(detailed["tasks"][task][model][dma]["MAE"]) for dma in list("ABCDEFGHIJ"))
            detailed_total = float(detailed["tasks"][task][model]["total"]["MAE"])
            compact_total = float(reference["tasks"][task][model]["MAE"])
            if abs(dma_sum - detailed_total) > 0.005:
                raise ValueError(f"Publisher MAE convention mismatch for {task}/{model}.")
            if abs(compact_total - detailed_total) > 1.0e-12:
                raise ValueError(f"Literature total drift for {task}/{model}.")


def _build_internal(root: Path, frozen: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for name, family, model in INTERNAL_MODELS:
            values = read_metrics(_aggregate_path(root, family, model, task, frozen))
            rows.append({"task": task, "model": name, **values})
    return rows


def _build_literature(root: Path, frozen: bool, reference: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for model in reference["paper"]["model_order"]:
            values = dict(reference["tasks"][task][model])
            values["MAPE"] = 100.0 * float(values["MAPE"])
            rows.append({"task": task, "model": f"{model} (reported)", **values})
        for name, family, model in PUBLISHER_MODELS:
            rows.append(
                {
                    "task": task,
                    "model": name,
                    **_read_publisher_total(_publisher_path(root, family, model, task, frozen)),
                }
            )
    return rows


def _build_ablation(root: Path, frozen: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for name, family, model in PUBLISHER_ABLATION_MODELS:
            rows.append(
                {
                    "task": task,
                    "model": name,
                    **_read_publisher_total(_publisher_path(root, family, model, task, frozen)),
                }
            )
    return rows


def _build_star_dma(root: Path, frozen: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        path = _publisher_path(root, "star_gnn", "Full", task, frozen)
        for values in _read_dma_rows(path):
            rows.append({"task": task, **values})
    return rows


def _validate_no_mae_mixing(internal: list[dict[str, Any]], literature: list[dict[str, Any]]) -> None:
    internal_lookup = {(row["task"], row["model"]): row for row in internal}
    literature_lookup = {(row["task"], row["model"]): row for row in literature}
    for task in TASKS:
        for model in ("DCRNN", "STGCN", "STaR-GNN"):
            internal_row = internal_lookup[(task, model)]
            paper_row = literature_lookup[(task, model)]
            for metric in ("MAPE", "RMSE", "NSE"):
                if abs(float(internal_row[metric]) - float(paper_row[metric])) > 1.0e-8:
                    raise ValueError(f"Unexpected {metric} convention drift for {task}/{model}.")
            if abs(float(internal_row["MAE"]) - float(paper_row["MAE"])) < 1.0e-8:
                raise ValueError(f"MAE conventions accidentally collapsed for {task}/{model}.")


def _validate_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_order = tuple(name for name, _, _ in PUBLISHER_ABLATION_MODELS)
    for task in TASKS:
        observed = tuple(row["model"] for row in rows if row["task"] == task)
        if observed != expected_order:
            raise ValueError(f"Ablation must contain exactly four factorial variants: {observed}")

    lookup = {(row["task"], row["model"]): row for row in rows}
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
            details[f"{task}_{left}_vs_{right}"] = f"{count}/4"
            passed += count
    if passed != 30:
        raise ValueError(f"Publisher-compatible factorial ablation audit drift: {passed}/32")
    return {
        "protocol": "common_46",
        "models": list(expected_order),
        "mae_convention": "sum_of_A_to_J_dma_mae",
        "other_metrics": "metric_on_hourly_sum_of_A_to_J_demand",
        "passed_relations": "30/32",
        "relations": details,
        "interpretation": {
            "24h": "STaR-GNN is best on all four metrics.",
            "168h": (
                "STaR-GNN is best on MAPE/RMSE/NSE; SAS-Norm-only has a "
                "0.025755 lower sum-of-DMA MAE (about 0.21%)."
            ),
        },
        "transparent_exceptions": [
            "FA-DPR 168h MAPE is slightly worse than DCRNN.",
            "STaR-GNN 168h sum-of-DMA MAE is 12.233590 versus 12.207835 for SAS-Norm-only.",
        ],
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
            "| {task} | {model} | {MAE:.3f} | {MAPE:.3f} | {RMSE:.3f} | {NSE:.3f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _markdown_dma(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Horizon | DMA | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {DMA} | {MAE:.3f} | {MAPE:.3f} | {RMSE:.3f} | {NSE:.3f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _write_conventions(output: Path) -> None:
    text = """# Metric conventions

Manuscript-facing overall comparison and factorial ablation use the same
publisher-compatible convention as the Que et al. (2024) supplementary tables.

- Total MAE: sum of DMA A--J MAEs.
- Total MAPE/RMSE/NSE: metric on the hourly aggregate-demand series.
- `table_literature_comparison_common46.*`: nine-model overall comparison; STGCN is an independent graph baseline.
- `table_ablation_common46.*`: exactly four factorial variants: DCRNN, DCRNN + SAS-Norm, DCRNN + FA-DPR, STaR-GNN.
- `table_star_gnn_dma_common46.*`: DMA A--J metrics without cross-DMA aggregation.
- `table_internal_common46.*`: aggregate-demand diagnostics only; do not mix its MAE with publisher-compatible MAE.

The 168 h publisher-compatible MAE of SAS-Norm-only (12.207835) and STaR-GNN
(12.233590) differs by only 0.025755 (about 0.21%). This small point-estimate
difference is reported transparently and is not used to claim that the full
model dominates SAS-Norm on every metric.
"""
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
    _validate_reported_reference(reference)
    internal = _build_internal(root, args.frozen_layout)
    literature = _build_literature(root, args.frozen_layout, reference)
    ablation = _build_ablation(root, args.frozen_layout)
    star_dma = _build_star_dma(root, args.frozen_layout)

    _validate_no_mae_mixing(internal, literature)
    audit = _validate_ablation(ablation)

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
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_conventions(output)

    print(f"Paper tables: {output}")
    print("Metric convention audit: PASS")
    print("Factorial ablation model-set audit: PASS (4 models, no STGCN)")
    print("Publisher-compatible factorial cell audit: 30/32 PASS")


if __name__ == "__main__":
    main()
