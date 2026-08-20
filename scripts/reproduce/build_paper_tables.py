#!/usr/bin/env python
"""Build submission-ready CSV/Markdown tables from evaluated checkpoints.

Two metric views are intentionally kept separate:

1. Internal/common-46 comparison: all metrics are computed on the hourly
   aggregate demand series obtained by summing DMA A--J first.
2. Literature comparison: matches the mixed ``total`` convention used in the
   MSCMNet supplementary tables, where total MAE is the sum of DMA-level MAEs
   while MAPE/RMSE/NSE are computed on the hourly aggregate demand series.

The script never mixes the two MAE definitions in one comparison column.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from paper_release_lib import METRICS, PROJECT_ROOT, STAR_VARIANTS, TASKS, read_metrics


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


def _load_literature_reference() -> dict[str, Any]:
    reference = yaml.safe_load(LITERATURE_TOTALS.read_text(encoding="utf-8"))
    paper = reference["paper"]
    expected_conventions = {
        "total_MAE": "sum_of_A_to_J_dma_mae",
        "total_MAPE_RMSE_NSE": "metric_on_hourly_sum_of_A_to_J_demand",
    }
    if paper.get("metric_conventions") != expected_conventions:
        raise ValueError(
            "Unexpected literature metric conventions: "
            f"{paper.get('metric_conventions')!r}"
        )
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
    """Cross-check the compact literature totals against detailed publisher data.

    The detailed audit file contains DMA A--J rows for MSCMNet_W and
    MSCMNet_WM. Their displayed total MAE must equal the A--J MAE sum within
    the three-decimal rounding used by the supplementary material.
    """
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


def _build_internal(root: Path, frozen: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for public_name, family, model in INTERNAL_MODELS:
            values = read_metrics(_metric_path(root, family, model, task, frozen))
            rows.append({"task": task, "model": public_name, **values})
    return rows


def _build_literature(
    root: Path,
    frozen: bool,
    reference: dict[str, Any],
) -> list[dict[str, Any]]:
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
        for label in STAR_VARIANTS:
            values = read_metrics(_metric_path(root, "star_gnn", label, task, frozen))
            public_name = "DCRNN" if label == "Base" else label
            rows.append({"task": task, "model": public_name, **values})
    return rows


def _validate_no_mae_mixing(
    internal: list[dict[str, Any]],
    literature: list[dict[str, Any]],
) -> None:
    internal_lookup = {(row["task"], row["model"]): row for row in internal}
    literature_lookup = {(row["task"], row["model"]): row for row in literature}
    for task in TASKS:
        for model in ("DCRNN", "STGCN", "STaR-GNN"):
            internal_row = internal_lookup[(task, model)]
            literature_row = literature_lookup[(task, model)]
            # MAPE/RMSE/NSE share the aggregate-demand convention in both views.
            for metric in ("MAPE", "RMSE", "NSE"):
                if abs(float(internal_row[metric]) - float(literature_row[metric])) > 1.0e-8:
                    raise ValueError(
                        f"Unexpected {metric} convention drift for {task}/{model}: "
                        f"{internal_row[metric]} vs {literature_row[metric]}."
                    )
            # MAE is intentionally different: aggregate-demand MAE internally,
            # sum-of-DMA MAEs for the publisher-compatible literature table.
            if abs(float(internal_row["MAE"]) - float(literature_row["MAE"])) < 1.0e-8:
                raise ValueError(
                    f"MAE conventions were accidentally collapsed for {task}/{model}."
                )


def _write_convention_note(output: Path) -> None:
    text = """# Metric conventions\n\n"
    text += "Two MAE definitions are intentionally retained and must not be mixed.\n\n"
    text += "- `table_internal_common46.*`: all four metrics are calculated on the hourly aggregate-demand series after summing DMA A--J.\n"
    text += "- `table_literature_comparison_common46.*`: follows Que et al. (2024) supplementary Tables S1-1--S1-8. Total MAE is the sum of DMA-level MAEs; MAPE, RMSE and NSE are calculated on the hourly aggregate-demand series.\n"
    text += "- `table_comparison_common46.*` is retained as a backward-compatible alias of the literature-comparison table.\n"
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

    ablation = _build_ablation(root, args.frozen_layout)
    internal = _build_internal(root, args.frozen_layout)
    literature = _build_literature(root, args.frozen_layout, reference)
    _validate_no_mae_mixing(internal, literature)

    _write_csv(output / "table_ablation_common46.csv", ablation)
    _write_csv(output / "table_internal_common46.csv", internal)
    _write_csv(output / "table_literature_comparison_common46.csv", literature)
    # Backward-compatible alias; this table now means literature comparison.
    _write_csv(output / "table_comparison_common46.csv", literature)

    (output / "table_ablation_common46.md").write_text(
        _markdown(ablation), encoding="utf-8"
    )
    (output / "table_internal_common46.md").write_text(
        _markdown(internal), encoding="utf-8"
    )
    (output / "table_literature_comparison_common46.md").write_text(
        _markdown(literature), encoding="utf-8"
    )
    (output / "table_comparison_common46.md").write_text(
        _markdown(literature), encoding="utf-8"
    )
    _write_convention_note(output)

    print(f"Paper tables: {output}")
    print("Metric convention audit: PASS")


if __name__ == "__main__":
    main()
