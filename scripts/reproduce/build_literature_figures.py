#!/usr/bin/env python
"""Build manuscript-facing overall, factorial-ablation, and DMA figures.

The overall comparison contains reported sequence/multiscale models plus the
re-evaluated graph baselines.  The ablation is strictly factorial and contains
only DCRNN, DCRNN + SAS-Norm, DCRNN + FA-DPR, and STaR-GNN; STGCN is not an
ablation variant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ("MAE", "MAPE", "RMSE", "NSE")
OVERALL_MODELS = (
    "GRU (reported)",
    "LSTM (reported)",
    "MSNet (reported)",
    "MSCMNet_WM (reported)",
    "MSCMNet_M (reported)",
    "MSCMNet_W (reported)",
    "DCRNN",
    "STGCN",
    "STaR-GNN",
)
ABLATION_MODELS = (
    "DCRNN",
    "DCRNN + SAS-Norm",
    "DCRNN + FA-DPR",
    "STaR-GNN",
)
DMAS = tuple("ABCDEFGHIJ")


def _validate_metric_table(frame: pd.DataFrame, expected_models: tuple[str, ...], label: str) -> None:
    required = {"task", "model", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} table missing columns: {sorted(missing)}")
    for task in ("24h", "168h"):
        selected = frame.loc[frame["task"] == task]
        models = tuple(selected["model"].astype(str))
        if models != expected_models:
            raise ValueError(f"Unexpected {label} model order for {task}: {models}")
        if not np.isfinite(selected.loc[:, list(METRICS)].to_numpy(float)).all():
            raise ValueError(f"Non-finite metrics in {label} table for {task}")


def _validate_dma(frame: pd.DataFrame) -> None:
    required = {"task", "DMA", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DMA table missing columns: {sorted(missing)}")
    for task in ("24h", "168h"):
        selected = frame.loc[frame["task"] == task]
        if tuple(selected["DMA"].astype(str)) != DMAS:
            raise ValueError(f"Unexpected DMA order for {task}")


def _save(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_metric_panels(
    frame: pd.DataFrame,
    task: str,
    expected_models: tuple[str, ...],
    base: Path,
) -> None:
    selected = frame.loc[frame["task"] == task].copy()
    labels = [name.replace(" (reported)", "") for name in selected["model"].astype(str)]
    x = np.arange(len(labels))
    width = 13.0 if len(labels) >= 8 else 9.8
    fig, axes = plt.subplots(2, 2, figsize=(width, 7.8))
    for axis, metric in zip(axes.ravel(), METRICS):
        values = selected[metric].to_numpy(float)
        bars = axis.bar(x, values, edgecolor="black", linewidth=0.45)
        for bar, model in zip(bars, selected["model"].astype(str)):
            if model == "STaR-GNN":
                bar.set_hatch("//")
                bar.set_linewidth(1.1)
        axis.set_xticks(x, labels, rotation=32, ha="right")
        axis.set_ylabel(metric + (" (%)" if metric == "MAPE" else ""))
        axis.grid(axis="y", alpha=0.18)
    _save(fig, base)


def _plot_dma(frame: pd.DataFrame, base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.7), sharex=True)
    x = np.arange(len(DMAS))
    for axis, metric in zip(axes.ravel(), METRICS):
        for task, marker in (("24h", "o"), ("168h", "s")):
            selected = frame.loc[frame["task"] == task].set_index("DMA").loc[list(DMAS)]
            axis.plot(
                x,
                selected[metric].to_numpy(float),
                marker=marker,
                linewidth=1.7,
                label=task,
            )
        axis.set_xticks(x, DMAS)
        axis.set_ylabel(metric + (" (%)" if metric == "MAPE" else ""))
        axis.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("DMA")
    axes[1, 1].set_xlabel("DMA")
    _save(fig, base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overall-table",
        type=Path,
        default=Path("paper/tables/literature/table_literature_comparison_common46.csv"),
    )
    parser.add_argument(
        "--ablation-table",
        type=Path,
        default=Path("paper/tables/literature/table_ablation_common46.csv"),
    )
    parser.add_argument(
        "--dma-table",
        type=Path,
        default=Path("paper/tables/literature/table_star_gnn_dma_common46.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    overall = pd.read_csv(args.overall_table.resolve())
    ablation = pd.read_csv(args.ablation_table.resolve())
    dma = pd.read_csv(args.dma_table.resolve())
    _validate_metric_table(overall, OVERALL_MODELS, "overall")
    _validate_metric_table(ablation, ABLATION_MODELS, "factorial ablation")
    _validate_dma(dma)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for task in ("24h", "168h"):
        _plot_metric_panels(overall, task, OVERALL_MODELS, output / f"test_overall_{task}")
        _plot_metric_panels(ablation, task, ABLATION_MODELS, output / f"test_ablation_{task}")
    _plot_dma(dma, output / "test_star_gnn_dma_metrics")

    print(f"Manuscript figures: {output}")
    print("Overall figure audit: PASS")
    print("Factorial ablation figure audit: PASS (4 models, no STGCN)")


if __name__ == "__main__":
    main()
