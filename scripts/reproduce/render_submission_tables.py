#!/usr/bin/env python
"""Render manuscript-display tables from audited full-precision CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ("MAE", "MAPE", "RMSE", "NSE")
DMA_MODELS = (
    "GRU",
    "LSTM",
    "MSNet",
    "MSCMNet-WM",
    "MSCMNet-M",
    "MSCMNet-W",
    "DCRNN",
    "STGCN",
    "STaR-GNN",
)
REPORTED_SUFFIX = " (reported)"
DISPLAY_NAMES = {
    "MSCMNet_WM": "MSCMNet-WM",
    "MSCMNet_M": "MSCMNet-M",
    "MSCMNet_W": "MSCMNet-W",
}


def _display_name(name: str) -> str:
    base = name.removesuffix(REPORTED_SUFFIX)
    base = DISPLAY_NAMES.get(base, base)
    return base


def _best(frame: pd.DataFrame, metric: str) -> float:
    if metric == "NSE":
        return float(frame[metric].max())
    return float(frame[metric].min())


def _fmt(value: float, *, bold: bool) -> str:
    text = f"{float(value):.3f}"
    return f"**{text}**" if bold else text


def _overall_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "**Table 1. Overall forecasting performance of the comparison models "
        "for the 24 h and 168 h prediction horizons.**",
        "",
        "| Horizon | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in ("24h", "168h"):
        task_frame = frame.loc[frame["task"] == task].copy()
        best = {metric: _best(task_frame, metric) for metric in METRICS}

        for _, row in task_frame.loc[
            task_frame["model"].astype(str).str.endswith(REPORTED_SUFFIX)
        ].iterrows():
            name = _display_name(str(row["model"]))
            vals = [
                _fmt(row[m], bold=abs(float(row[m]) - best[m]) < 1e-12)
                for m in METRICS
            ]
            lines.append(
                f"| {task.replace('h', ' h')} | {name} | "
                f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |"
            )

        for _, row in task_frame.loc[
            ~task_frame["model"].astype(str).str.endswith(REPORTED_SUFFIX)
        ].iterrows():
            name = _display_name(str(row["model"]))
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
        "**Note.** Values for GRU, LSTM, MSNet and the MSCMNet variants were "
        "reported by Que et al. (2024). DCRNN, STGCN and STaR-GNN were "
        "evaluated using the present study's pipeline. "
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
        "**Table 2. Factorial ablation of SAS-Norm and FA-DPR for the 24 h "
        "and 168 h prediction horizons.**",
        "",
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
        "lower total MAE (12.208 vs. 12.234), whereas the full "
        "STaR-GNN is best in MAPE, RMSE and NSE. The corresponding paired "
        "moving-block analysis is reported with Main Fig. 4.",
    ]
    return "\n".join(lines) + "\n"


def _dma_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "**Table S1. DMA-level forecasting performance of all comparison "
        "models for the 24 h and 168 h prediction horizons.**",
        "",
        "| Horizon | DMA | Model | MAE ↓ | MAPE (%) ↓ | RMSE ↓ | NSE ↑ |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for task in ("24h", "168h"):
        for dma in tuple("ABCDEFGHIJ"):
            block = frame.loc[(frame["task"] == task) & (frame["DMA"] == dma)]
            block = block.assign(
                model=pd.Categorical(
                    block["model"], categories=DMA_MODELS, ordered=True
                )
            ).sort_values("model")
            if tuple(block["model"].astype(str)) != DMA_MODELS:
                raise ValueError(f"Unexpected DMA model set for {task}/{dma}")
            best = {metric: _best(block, metric) for metric in METRICS}
            for _, row in block.iterrows():
                vals = [
                    _fmt(row[m], bold=abs(float(row[m]) - best[m]) < 1e-12)
                    for m in METRICS
                ]
                lines.append(
                    f"| {task.replace('h', ' h')} | {dma} | {row['model']} | "
                    f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |"
                )
    lines += [
        "",
        "**Note.** MAPE is displayed as a percentage. Bold indicates the best "
        "value within each horizon–DMA block. The six recurrent and multi-scale "
        "model results are transcribed from the supplementary material of Que "
        "et al. (2024); its fractional MAPE values were multiplied by 100 for "
        "consistent display. The same results support the signed DMA-level "
        "comparisons in Main Figs. 2 and 3.",
    ]
    return "\n".join(lines) + "\n"


def _load_dma_comparison(release: Path) -> pd.DataFrame:
    model_paths = {
        "DCRNN": "models/star_gnn/Base/{task}/seed_0/evaluation/metrics_common_46.csv",
        "STGCN": "models/baselines/stgcn/{task}/seed_0/evaluation/metrics_common_46.csv",
        "STaR-GNN": "models/star_gnn/Full/{task}/seed_0/evaluation/metrics_common_46.csv",
    }
    rows: list[dict[str, object]] = []
    for task in ("24h", "168h"):
        for model, template in model_paths.items():
            path = release / template.format(task=task)
            metrics = pd.read_csv(path)
            metrics = metrics.loc[metrics["entity"].astype(str).isin(tuple("ABCDEFGHIJ"))]
            for _, row in metrics.iterrows():
                rows.append(
                    {
                        "task": task,
                        "DMA": str(row["entity"]),
                        "model": model,
                        "MAE": float(row["MAE"]),
                        "MAPE": 100.0 * float(row["MAPE"]),
                        "RMSE": float(row["RMSE"]),
                        "NSE": float(row["NSE"]),
                    }
                )
    return pd.DataFrame(rows)


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
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("results/paper/frozen_v1"),
    )
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(source / "table_literature_comparison_common46.csv")
    ablation = pd.read_csv(source / "table_ablation_common46.csv")
    graph_dma = _load_dma_comparison(args.release.resolve())
    temporal_dma = pd.read_csv(source / "table_temporal_models_dma.csv")
    expected_temporal = len(tuple("ABCDEFGHIJ")) * 2 * 6
    required_dma_columns = {"task", "DMA", "model", *METRICS}
    if set(temporal_dma.columns) != required_dma_columns:
        raise ValueError("Unexpected temporal DMA table schema")
    if len(temporal_dma) != expected_temporal:
        raise ValueError(
            f"Expected {expected_temporal} temporal DMA rows, "
            f"found {len(temporal_dma)}"
        )
    if temporal_dma.duplicated(["task", "DMA", "model"]).any():
        raise ValueError("Duplicate temporal DMA result rows")
    dma = pd.concat([temporal_dma, graph_dma], ignore_index=True)

    (output / "table1_overall_performance.md").write_text(
        _overall_markdown(overall), encoding="utf-8"
    )
    (output / "table2_factorial_ablation.md").write_text(
        _ablation_markdown(ablation), encoding="utf-8"
    )
    (output / "tableS1_dma_metrics.md").write_text(
        _dma_markdown(dma), encoding="utf-8"
    )
    dma.to_csv(
        source / "table_all_models_dma.csv",
        index=False,
        float_format="%.10f",
    )

    print("Submission tables: PASS")
    print("  Table 1 — overall forecasting performance")
    print("  Table 2 — factorial ablation")
    print("  Table S1 — detailed DMA-level metrics")


if __name__ == "__main__":
    main()
