#!/usr/bin/env python
"""Build manuscript-facing comparison, ablation, and DMA figures.

Overall and ablation figures use the MSCMNet publisher-compatible convention:
- MAE = sum of DMA A--J MAEs;
- MAPE/RMSE/NSE = metrics on the hourly aggregate-demand series.

The STaR-GNN DMA figure reports each DMA's own MAE/MAPE/RMSE/NSE for 24 h and
168 h, with no cross-DMA aggregation.
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
    "STGCN",
    "DCRNN",
    "DCRNN + SAS-Norm",
    "DCRNN + FA-DPR",
    "STaR-GNN",
)
DMAS = tuple("ABCDEFGHIJ")


def _validate_metric_table(
    frame: pd.DataFrame,
    expected_models: tuple[str, ...],
    label: str,
) -> None:
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
        if not np.isfinite(selected.loc[:, list(METRICS)].to_numpy(float)).all():
            raise ValueError(f"Non-finite DMA metrics for {task}")


def _save(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_metric_panels(
    frame: pd.DataFrame,
    task: str,
    expected_models: tuple[str, ...],
    title: str,
    base: Path,
) -> None:
    selected = frame.loc[frame["task"] == task].copy()
    labels = [name.replace(" (reported)", "") for name in selected["model"].astype(str)]
    x = np.arange(len(labels))
    width = 13.0 if len(labels) >= 8 else 10.8
    fig, axes = plt.subplots(2, 2, figsize=(width, 8.5))
    for axis, metric in zip(axes.ravel(), METRICS):
        values = selected[metric].to_numpy(float)
        bars = axis.bar(x, values, edgecolor="black", linewidth=0.45)
        for bar, model in zip(bars, selected["model"].astype(str)):
            if model == "STaR-GNN":
                bar.set_hatch("//")
                bar.set_linewidth(1.2)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_ylabel(metric + (" (%)" if metric == "MAPE" else ""))
        axis.set_title(metric + (" (higher is better)" if metric == "NSE" else " (lower is better)"))
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=14)
    _save(fig, base)


def _plot_dma(frame: pd.DataFrame, base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    x = np.arange(len(DMAS))
    for axis, metric in zip(axes.ravel(), METRICS):
        for task, marker in (("24h", "o"), ("168h", "s")):
            selected = frame.loc[frame["task"] == task].set_index("DMA").loc[list(DMAS)]
            axis.plot(
                x,
                selected[metric].to_numpy(float),
                marker=marker,
                linewidth=1.8,
                label=task,
            )
        axis.set_xticks(x, DMAS)
        axis.set_ylabel(metric + (" (%)" if metric == "MAPE" else ""))
        axis.set_title(metric + (" (higher is better)" if metric == "NSE" else " (lower is better)"))
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("DMA")
    axes[1, 1].set_xlabel("DMA")
    fig.suptitle("STaR-GNN DMA-level performance", fontsize=14)
    _save(fig, base)


def _patch_report(report: Path) -> None:
    if not report.is_file():
        return
    text = report.read_text(encoding="utf-8")
    old_overall = (
        "- `figures/test_overall_24h.*`、`test_overall_168h.*`："
        "STGCN/DCRNN/Full 总体对比。"
    )
    new_overall = (
        "- `figures/test_overall_24h.*`、`test_overall_168h.*`："
        "GRU/LSTM/MSNet/MSCMNet + DCRNN/STGCN/STaR-GNN 的 publisher-compatible 总体比较。"
    )
    old_ablation = (
        "- `figures/test_ablation_24h.*`、`test_ablation_168h.*`：四单元消融。"
    )
    new_ablation = (
        "- `figures/test_ablation_24h.*`、`test_ablation_168h.*`："
        "STGCN/DCRNN/SAS-Norm/FA-DPR/STaR-GNN 的 publisher-compatible 消融与图模型对比。"
    )
    text = text.replace(old_overall, new_overall).replace(old_ablation, new_ablation)
    marker = "## 7. 图件索引"
    dma_line = (
        "- `figures/test_star_gnn_dma_metrics.*`：STaR-GNN 在 DMA A--J 上的 "
        "24 h/168 h MAE、MAPE、RMSE、NSE。\n"
    )
    if marker in text and "test_star_gnn_dma_metrics" not in text:
        insert_at = text.find(marker)
        # Add the DMA figure description to the figure-index section, not before it.
        section_start = text.find("\n", insert_at) + 1
        text = text[:section_start] + "\n" + dma_line + text[section_start:]
    report.write_text(text, encoding="utf-8")


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
    parser.add_argument("--report", type=Path, default=Path("paper/reports/TEST_RESULTS_CN.md"))
    args = parser.parse_args()

    overall_path = args.overall_table.resolve()
    ablation_path = args.ablation_table.resolve()
    dma_path = args.dma_table.resolve()
    output = args.output.resolve()
    report = args.report.resolve()
    for path in (overall_path, ablation_path, dma_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required paper table not found: {path}")
    output.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(overall_path)
    ablation = pd.read_csv(ablation_path)
    dma = pd.read_csv(dma_path)
    _validate_metric_table(overall, OVERALL_MODELS, "overall")
    _validate_metric_table(ablation, ABLATION_MODELS, "ablation")
    _validate_dma(dma)

    for task in ("24h", "168h"):
        _plot_metric_panels(
            overall,
            task,
            OVERALL_MODELS,
            f"Publisher-compatible overall comparison ({task})",
            output / f"test_overall_{task}",
        )
        _plot_metric_panels(
            ablation,
            task,
            ABLATION_MODELS,
            f"Publisher-compatible ablation and graph-model comparison ({task})",
            output / f"test_ablation_{task}",
        )
    _plot_dma(dma, output / "test_star_gnn_dma_metrics")
    _patch_report(report)

    print(f"Manuscript figures: {output}")
    print("Publisher-compatible figure audit: PASS")


if __name__ == "__main__":
    main()
