"""Comparison against the publisher-supplied MSCMNet result tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


LOWER_IS_BETTER = {"MAE", "MAPE", "RMSE"}
HIGHER_IS_BETTER = {"NSE"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_paper_reference(path: Path) -> dict[str, Any]:
    """Load and validate the auditable paper-reference YAML."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Paper reference does not exist: {path}")
    reference = yaml.safe_load(path.read_text(encoding="utf-8"))
    paper = reference["paper"]
    if paper["protocol"] != "common_46":
        raise ValueError("Paper comparison must use protocol=common_46.")
    if int(paper["expected_sequences"]) != 46:
        raise ValueError("Paper comparison must contain 46 sequences.")
    if paper["mape_unit"] != "fraction":
        raise ValueError("Paper MAPE must be stored as a fraction.")
    expected_conventions = {
        "dma_rows": "metric_per_dma",
        "total_MAE": "sum_of_A_to_J_dma_mae",
        "total_MAPE_RMSE_NSE": (
            "metric_on_hourly_sum_of_A_to_J_demand"
        ),
    }
    if paper.get("metric_conventions") != expected_conventions:
        raise ValueError(
            "Unexpected publisher metric conventions: "
            f"{paper.get('metric_conventions')!r}."
        )
    entities = [str(value) for value in paper["entity_order"]]
    metrics = [str(value) for value in paper["metric_order"]]
    if entities != list("ABCDEFGHIJ") + ["total"]:
        raise ValueError("Unexpected paper entity order.")
    if set(metrics) != LOWER_IS_BETTER | HIGHER_IS_BETTER:
        raise ValueError("Unexpected paper metric set.")

    models = [
        str(paper["primary_reference_model"]),
        *[str(value) for value in paper["secondary_reference_models"]],
    ]
    for task, expected_horizon in [("24h", 24), ("168h", 168)]:
        task_reference = reference["tasks"][task]
        if int(task_reference["horizon"]) != expected_horizon:
            raise ValueError(f"Paper horizon mismatch for {task}.")
        for model in models:
            for entity in entities:
                values = task_reference[model][entity]
                missing = set(metrics) - set(values)
                if missing:
                    raise ValueError(
                        f"Missing {task}/{model}/{entity}: {sorted(missing)}"
                    )
            dma_mae_sum = sum(
                float(task_reference[model][entity]["MAE"])
                for entity in list("ABCDEFGHIJ")
            )
            published_total_mae = float(
                task_reference[model]["total"]["MAE"]
            )
            # The table values are rounded to three decimals.  All published
            # totals equal the A-J sum within the resulting rounding error.
            if abs(dma_mae_sum - published_total_mae) > 0.005:
                raise ValueError(
                    "Publisher total MAE is not the A-J DMA MAE sum for "
                    f"{task}/{model}: {published_total_mae} vs "
                    f"{dma_mae_sum}."
                )
    reference["_metadata"] = {
        "path": str(path),
        "sha256": _sha256(path),
    }
    return reference


def compare_metrics_to_paper(
    *,
    task: str,
    dcrnn_metrics: pd.DataFrame,
    reference: dict[str, Any],
) -> pd.DataFrame:
    """Create a tidy, direction-aware DCRNN-versus-paper comparison.

    ``improvement`` and ``relative_improvement_percent`` are positive when
    DCRNN is better, regardless of metric direction.
    """
    if task not in {"24h", "168h"}:
        raise ValueError("task must be '24h' or '168h'.")
    required_columns = {"entity", "MAE", "MAPE", "RMSE", "NSE"}
    missing = required_columns - set(dcrnn_metrics.columns)
    if missing:
        raise ValueError(f"DCRNN metric columns are missing: {sorted(missing)}")

    paper = reference["paper"]
    expected_entities = [str(value) for value in paper["entity_order"]]
    actual_entities = dcrnn_metrics["entity"].astype(str).tolist()
    if actual_entities != expected_entities:
        raise ValueError(
            f"DCRNN entity order mismatch: {actual_entities} "
            f"!= {expected_entities}."
        )
    models = [
        str(paper["primary_reference_model"]),
        *[str(value) for value in paper["secondary_reference_models"]],
    ]
    rows: list[dict[str, Any]] = []
    for model in models:
        for _, dcrnn_row in dcrnn_metrics.iterrows():
            entity = str(dcrnn_row["entity"])
            for metric in paper["metric_order"]:
                metric = str(metric)
                dcrnn_value = float(dcrnn_row[metric])
                paper_value = float(
                    reference["tasks"][task][model][entity][metric]
                )
                if metric in LOWER_IS_BETTER:
                    improvement = paper_value - dcrnn_value
                    direction = "lower"
                else:
                    improvement = dcrnn_value - paper_value
                    direction = "higher"
                denominator = abs(paper_value)
                relative = (
                    None
                    if denominator < 1.0e-12
                    else 100.0 * improvement / denominator
                )
                rows.append(
                    {
                        "task": task,
                        "protocol": "common_46",
                        "entity": entity,
                        "metric": metric,
                        "direction": direction,
                        "dcrnn": dcrnn_value,
                        "paper_model": model,
                        "paper_value": paper_value,
                        "dcrnn_minus_paper": dcrnn_value - paper_value,
                        "improvement": improvement,
                        "relative_improvement_percent": relative,
                        "beats_paper": bool(improvement > 0.0),
                        "ties_paper": bool(abs(improvement) <= 1.0e-12),
                        "primary_reference": bool(
                            model == paper["primary_reference_model"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write_paper_comparison_report(
    *,
    comparison: pd.DataFrame,
    task: str,
    reference: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Write a compact Markdown report and return its JSON summary."""
    primary_model = str(reference["paper"]["primary_reference_model"])
    primary = comparison.loc[
        comparison["paper_model"] == primary_model
    ].copy()
    total = primary.loc[primary["entity"] == "total"].copy()
    summary = {
        "task": task,
        "protocol": "common_46",
        "sequences": 46,
        "primary_reference_model": primary_model,
        "primary_cells_beaten": int(primary["beats_paper"].sum()),
        "primary_cells_total": int(len(primary)),
        "primary_total_metrics_beaten": int(total["beats_paper"].sum()),
        "primary_total_metrics_total": int(len(total)),
        "paper_metric_conventions": dict(
            reference["paper"]["metric_conventions"]
        ),
        "paper_reference_sha256": reference["_metadata"]["sha256"],
    }

    table = total[
        [
            "metric",
            "direction",
            "dcrnn",
            "paper_value",
            "improvement",
            "relative_improvement_percent",
            "beats_paper",
        ]
    ].copy()
    table = table.rename(
        columns={
            "paper_value": primary_model,
            "beats_paper": "DCRNN_better",
        }
    )
    table_lines = [
        "| metric | direction | DCRNN | "
        f"{primary_model} | improvement | relative improvement (%) | "
        "DCRNN better |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for _, row in table.iterrows():
        relative = row["relative_improvement_percent"]
        relative_text = (
            ""
            if pd.isna(relative)
            else f"{float(relative):.6f}"
        )
        table_lines.append(
            f"| {row['metric']} | {row['direction']} | "
            f"{float(row['dcrnn']):.6f} | "
            f"{float(row[primary_model]):.6f} | "
            f"{float(row['improvement']):.6f} | "
            f"{relative_text} | {bool(row['DCRNN_better'])} |"
        )
    lines = [
        f"# DCRNN vs {primary_model}: {task}",
        "",
        "- Evaluation protocol: `common_46`",
        "- Test sequences: 46",
        "- MAPE unit: fraction",
        "- Publisher `total MAE`: sum of DMA A-J MAE values",
        "- Publisher `total MAPE/RMSE/NSE`: metric on the hourly "
        "sum of DMA A-J demand",
        "- Pure aggregate-demand metrics are stored separately in "
        "`metrics_aggregate_total_common_46.csv`",
        "- Positive `improvement` always means DCRNN is better",
        "",
        "## Publisher-table total comparison",
        "",
        *table_lines,
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "The full per-DMA comparison is stored in `paper_comparison.csv`.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def compare_named_model_metrics_to_paper(
    *,
    task: str,
    model_name: str,
    model_metrics: pd.DataFrame,
    reference: dict[str, Any],
) -> pd.DataFrame:
    """Compare any named baseline to the immutable paper reference.

    The existing DCRNN-specific function remains untouched so prior DCRNN
    outputs and tests keep their exact schema.  New baselines receive the
    same calculations with a model-specific value column.
    """
    name = str(model_name).strip().lower()
    if not name or not name.replace("_", "").isalnum():
        raise ValueError(
            f"Invalid comparison model name: {model_name!r}."
        )
    if task not in {"24h", "168h"}:
        raise ValueError("task must be '24h' or '168h'.")
    required = {"entity", "MAE", "MAPE", "RMSE", "NSE"}
    missing = required - set(model_metrics.columns)
    if missing:
        raise ValueError(
            f"{name} metric columns are missing: {sorted(missing)}"
        )

    paper = reference["paper"]
    expected_entities = [
        str(value) for value in paper["entity_order"]
    ]
    actual_entities = (
        model_metrics["entity"].astype(str).tolist()
    )
    if actual_entities != expected_entities:
        raise ValueError(
            f"{name} entity order mismatch: {actual_entities} "
            f"!= {expected_entities}."
        )
    paper_models = [
        str(paper["primary_reference_model"]),
        *[
            str(value)
            for value in paper["secondary_reference_models"]
        ],
    ]
    rows: list[dict[str, Any]] = []
    for paper_model in paper_models:
        for _, model_row in model_metrics.iterrows():
            entity = str(model_row["entity"])
            for metric_value in paper["metric_order"]:
                metric = str(metric_value)
                value = float(model_row[metric])
                paper_value = float(
                    reference["tasks"][task][paper_model][entity][
                        metric
                    ]
                )
                if metric in LOWER_IS_BETTER:
                    improvement = paper_value - value
                    direction = "lower"
                else:
                    improvement = value - paper_value
                    direction = "higher"
                denominator = abs(paper_value)
                rows.append(
                    {
                        "task": task,
                        "protocol": "common_46",
                        "entity": entity,
                        "metric": metric,
                        "direction": direction,
                        name: value,
                        "paper_model": paper_model,
                        "paper_value": paper_value,
                        f"{name}_minus_paper": value - paper_value,
                        "improvement": improvement,
                        "relative_improvement_percent": (
                            None
                            if denominator < 1.0e-12
                            else 100.0 * improvement / denominator
                        ),
                        "beats_paper": bool(improvement > 0.0),
                        "ties_paper": bool(
                            abs(improvement) <= 1.0e-12
                        ),
                        "primary_reference": bool(
                            paper_model
                            == paper["primary_reference_model"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write_named_model_comparison_report(
    *,
    comparison: pd.DataFrame,
    task: str,
    model_name: str,
    reference: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Write the paper-comparison report for a non-DCRNN baseline."""
    name = str(model_name).strip().lower()
    display_name = name.upper()
    if name not in comparison.columns:
        raise ValueError(
            f"Comparison table has no {name!r} value column."
        )
    primary_model = str(
        reference["paper"]["primary_reference_model"]
    )
    primary = comparison.loc[
        comparison["paper_model"] == primary_model
    ].copy()
    total = primary.loc[primary["entity"] == "total"].copy()
    summary = {
        "model": name,
        "task": task,
        "protocol": "common_46",
        "sequences": 46,
        "primary_reference_model": primary_model,
        "primary_cells_beaten": int(primary["beats_paper"].sum()),
        "primary_cells_total": int(len(primary)),
        "primary_total_metrics_beaten": int(
            total["beats_paper"].sum()
        ),
        "primary_total_metrics_total": int(len(total)),
        "paper_metric_conventions": dict(
            reference["paper"]["metric_conventions"]
        ),
        "paper_reference_sha256": reference["_metadata"]["sha256"],
    }
    lines = [
        f"# {display_name} vs {primary_model}: {task}",
        "",
        "- Evaluation protocol: `common_46`",
        "- Test sequences: 46",
        "- Test-time target input / teacher forcing: none",
        "- MAPE unit: fraction",
        "- Publisher `total MAE`: sum of DMA A-J MAE values",
        "- Publisher `total MAPE/RMSE/NSE`: metric on the hourly "
        "sum of DMA A-J demand",
        "- Positive `improvement` always means "
        f"{display_name} is better",
        "",
        "## Publisher-table total comparison",
        "",
        "| metric | direction | "
        f"{display_name} | {primary_model} | improvement | "
        f"relative improvement (%) | {display_name} better |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for _, row in total.iterrows():
        relative = row["relative_improvement_percent"]
        relative_text = (
            ""
            if pd.isna(relative)
            else f"{float(relative):.6f}"
        )
        lines.append(
            f"| {row['metric']} | {row['direction']} | "
            f"{float(row[name]):.6f} | "
            f"{float(row['paper_value']):.6f} | "
            f"{float(row['improvement']):.6f} | "
            f"{relative_text} | {bool(row['beats_paper'])} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(summary, indent=2, ensure_ascii=False),
            "```",
            "",
            "The full per-DMA comparison is stored in "
            "`paper_comparison.csv`.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary
