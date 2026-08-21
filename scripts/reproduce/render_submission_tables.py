#!/usr/bin/env python
"""Render manuscript-display tables from audited full-precision CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ("MAE", "MAPE", "RMSE", "NSE")
REPORTED_SUFFIX = " (reported)"


def _best(frame: pd.DataFrame, metric: str) -> float:
    if metric == "NSE":
        return float(frame[metric].max())
    return float(frame[metric].min())


def _fmt(value: float, *, bold: bool) -> str:
    text = f"{float(value):.3f}"
    return f"**{text}**" if bold else text


def _overall_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in ("24h", "168h"):
        task_frame = frame.loc[frame["task"] == task].copy()
        best = {metric: _best(task_frame, metric) for metric in METRICS}

        lines.append(f"| **{task.replace('h', ' h')}** | **Published reference models** |  |  |  |  |")
        for _, row in task_frame.loc[
            task_frame["model"].astype(str).str.endswith(REPORTED_SUFFIX)
        ].iterrows():
            name = str(row["model"]).replace(REPORTED_SUFFIX, "†")
            vals = [
                _fmt(row[m], bold=abs(float(row[m]) - best[m]) < 1e-12)
                for m in METRICS
            ]
            lines.append(
                f"| {task.replace('h', ' h')} | {name} | "
                f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |"
            )

        lines.append(f"|  | **Re-evaluated graph models** |  |  |  |  |")
        for _, row in task_frame.loc[
            ~task_frame["model"].astype(str).str.endswith(REPORTED_SUFFIX)
        ].iterrows():
            name = str(row["model"])
            vals = [
                _fmt(row[m], bold=abs(float(row[m]) - best[m]) < 1e-12)
                for m in METRICS
            ]
            lines.append(
                f"| {task.replace('h', ' h')} | {name} | "
                f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |"
            )

    lines += [
        "",
        "**Note.** † Results reported by Que et al. (2024). DCRNN, STGCN and "
        "STaR-GNN were evaluated under the common 46-origin protocol. "
        "All manuscript values use uniform three-decimal display precision; "
        "the source CSV retains full precision.",
    ]
    return "\n".join(lines) + "\n"


def _ablation_markdown(frame: pd.DataFrame) -> str:
    expected = (
        "DCRNN",
        "DCRNN + SAS-Norm",
        "DCRNN + FA-DPR",
        "STaR-GNN",
    )
    lines = [
        "| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in ("24h", "168h"):
        tf = frame.loc[frame["task"] == task].copy()
        if tuple(tf["model"].astype(str)) != expected:
            raise ValueError(f"Unexpected factorial ablation order for {task}")
        best = {metric: _best(tf, metric) for metric in METRICS}
        for _, row in tf.iterrows():
            vals = [
                _fmt(row[m], bold=abs(float(row[m]) - best[m]) < 1e-12)
                for m in METRICS
            ]
            lines.append(
                f"| {task.replace('h', ' h')} | {row['model']} | "
                f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |"
            )
    lines += [
        "",
        "**Note.** STGCN is an independent graph baseline and is excluded from "
        "the factorial ablation. At 168 h, SAS-Norm-only has the marginally "
        "lower publisher-compatible MAE (12.208 vs. 12.234), whereas the full "
        "STaR-GNN is best in MAPE, RMSE and NSE. The corresponding paired "
        "moving-block analysis is reported with Main Fig. 1.",
    ]
    return "\n".join(lines) + "\n"


def _dma_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "| Horizon | DMA | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {str(row['task']).replace('h', ' h')} | {row['DMA']} | "
            f"{float(row['MAE']):.3f} | {float(row['MAPE']):.3f} | "
            f"{float(row['RMSE']):.3f} | {float(row['NSE']):.3f} |"
        )
    lines += [
        "",
        "**Note.** Detailed DMA-level metrics are provided as Supplementary "
        "Table S1; spatial consistency of the improvements is summarized in "
        "Main Fig. 2b.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("paper/tables/literature"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/tables/submission"),
    )
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(source / "table_literature_comparison_common46.csv")
    ablation = pd.read_csv(source / "table_ablation_common46.csv")
    dma = pd.read_csv(source / "table_star_gnn_dma_common46.csv")

    (output / "table1_overall_performance.md").write_text(
        _overall_markdown(overall), encoding="utf-8"
    )
    (output / "table2_factorial_ablation.md").write_text(
        _ablation_markdown(ablation), encoding="utf-8"
    )
    (output / "tableS1_dma_metrics.md").write_text(
        _dma_markdown(dma), encoding="utf-8"
    )

    print("Submission tables: PASS")
    print("  Table 1 — overall forecasting performance")
    print("  Table 2 — factorial ablation")
    print("  Table S1 — detailed DMA-level metrics")


if __name__ == "__main__":
    main()
