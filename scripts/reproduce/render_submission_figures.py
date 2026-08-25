#!/usr/bin/env python
"""Render the canonical Journal of Hydrology submission result figures.

The four main figures follow the manuscript's claim sequence: overall
performance, DMA-level consistency, component evidence, and a concrete
week-ahead forecast. MAE, MAPE, RMSE and NSE are carried through the first
three stages instead of treating MAE as a proxy for all forecasting quality.

The ablation figure reports paired improvements relative to DCRNN with small
horizontal offsets and confidence intervals.  This makes the very similar
SAS-Norm and full-model results legible without distorting their values.

Figures are rebuilt from frozen predictions and audited tables. No training,
checkpoint selection, or hyperparameter tuning occurs here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from manuscript_plot_style import (
    HERO_BLUE,
    MODEL_COLORS,
    MODEL_LINESTYLES,
    MODEL_MARKERS,
    OBSERVED_BLACK,
    ZERO_GRAY,
    add_panel_label,
    apply_publication_style,
    light_to_hero_cmap,
    save_publication_figure,
    style_axis,
)


TASKS = ("24h", "168h")
DMAS = tuple("ABCDEFGHIJ")
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

PUBLIC_TO_INTERNAL = {
    "DCRNN": "DCRNN",
    "STGCN": "STGCN",
    "DCRNN + SAS-Norm": "State",
    "DCRNN + FA-DPR": "FA-DPR",
    "STaR-GNN": "Full",
}
ABLATION_MODELS = (
    "DCRNN",
    "DCRNN + SAS-Norm",
    "DCRNN + FA-DPR",
    "STaR-GNN",
)
BASELINE_MODELS = ("DCRNN", "STGCN")
TRAJECTORY_MODELS = ("DCRNN", "STGCN", "STaR-GNN")

REPORTED_MODELS = (
    "GRU (reported)",
    "LSTM (reported)",
    "MSNet (reported)",
    "MSCMNet_WM (reported)",
    "MSCMNet_M (reported)",
    "MSCMNet_W (reported)",
)


def _evaluation_dir(release: Path, public_model: str, task: str) -> Path:
    model = PUBLIC_TO_INTERNAL[public_model]
    prefix = release / "models" if (release / "models").is_dir() else release
    if model == "DCRNN":
        family, name = "star_gnn", "Base"
    elif model in ("State", "FA-DPR", "Full"):
        family, name = "star_gnn", model
    elif model == "STGCN":
        family, name = "baselines", "stgcn"
    else:
        raise ValueError(public_model)
    path = prefix / family / name / task / "seed_0" / "evaluation"
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _load_common_predictions(
    release: Path, public_model: str, task: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_path = _evaluation_dir(release, "DCRNN", task) / "predictions.npz"
    with np.load(baseline_path, allow_pickle=False) as payload:
        common = payload["common_46_indices"].astype(np.int64)
    if common.shape != (46,):
        raise ValueError(f"{task}: expected 46 common indices, got {common.shape}")

    path = _evaluation_dir(release, public_model, task) / "predictions.npz"
    with np.load(path, allow_pickle=False) as payload:
        if "target" in payload.files and "prediction" in payload.files:
            truth = payload["target"]
            prediction = payload["prediction"]
        elif "y_true" in payload.files and "y_pred" in payload.files:
            truth = payload["y_true"]
            prediction = payload["y_pred"]
        else:
            raise ValueError(f"Unknown prediction schema: {path}")
        indices = payload["common_46_indices"].astype(np.int64)

    if not np.array_equal(indices, common):
        raise ValueError(f"common-46 index drift for {public_model}/{task}")
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError(
            f"Unexpected prediction shape for {public_model}/{task}: "
            f"{truth.shape} vs {prediction.shape}"
        )
    return (
        truth[common].astype(np.float64),
        prediction[common].astype(np.float64),
        common,
    )


def _ensure_same_truth(
    reference: np.ndarray | None, current: np.ndarray, label: str
) -> np.ndarray:
    if reference is None:
        return current
    if not np.allclose(reference, current, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"Ground-truth drift: {label}")
    return reference


def _publisher_mae_per_origin(
    truth: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    return np.mean(np.abs(prediction - truth), axis=1).sum(axis=1)


def _metric_per_origin(
    truth: np.ndarray,
    prediction: np.ndarray,
    metric: str,
    *,
    day: int | None = None,
) -> np.ndarray:
    """Return one metric per forecast origin using manuscript conventions."""
    if day is not None:
        if not 1 <= day <= 7:
            raise ValueError(day)
        sl = slice((day - 1) * 24, day * 24)
        truth = truth[:, sl, :]
        prediction = prediction[:, sl, :]
    if metric == "MAE":
        return np.mean(np.abs(prediction - truth), axis=1).sum(axis=1)

    aggregate_truth = truth.sum(axis=2)
    aggregate_prediction = prediction.sum(axis=2)
    error = aggregate_prediction - aggregate_truth
    if metric == "MAPE":
        denominator = np.maximum(np.abs(aggregate_truth), 1.0e-12)
        return 100.0 * np.mean(np.abs(error) / denominator, axis=1)
    if metric == "RMSE":
        return np.sqrt(np.mean(error**2, axis=1))
    if metric == "NSE":
        numerator = np.sum(error**2, axis=1)
        centered = aggregate_truth - aggregate_truth.mean(axis=1, keepdims=True)
        denominator = np.sum(centered**2, axis=1)
        return 1.0 - numerator / np.maximum(denominator, 1.0e-12)
    raise ValueError(metric)


def _paired_improvement(
    baseline: np.ndarray,
    candidate: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Direction-aligned improvement: positive always favors the candidate."""
    if metric == "NSE":
        return candidate - baseline
    denominator = np.maximum(np.abs(baseline), 1.0e-12)
    return 100.0 * (baseline - candidate) / denominator


def _improvement_label(metric: str) -> str:
    return "NSE improvement ($\\Delta$NSE)" if metric == "NSE" else f"{metric} reduction (%)"


def _moving_block_indices(
    n: int, block_length: int, iterations: int, seed: int
) -> np.ndarray:
    if not 1 <= block_length <= n:
        raise ValueError("Invalid block length")
    rng = np.random.default_rng(seed)
    blocks_per_draw = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=(iterations, blocks_per_draw))
    offsets = np.arange(block_length, dtype=np.int64)
    idx = (starts[..., None] + offsets) % n
    return idx.reshape(iterations, -1)[:, :n]


def _block_ci(
    values: np.ndarray, indices: np.ndarray
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(values.shape)
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _load_dma_metrics(release: Path, model: str, task: str) -> pd.DataFrame:
    path = _evaluation_dir(release, model, task) / "metrics_common_46.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["entity"].astype(str).isin(DMAS)].copy()
    if tuple(frame["entity"].astype(str)) != DMAS:
        raise ValueError(f"DMA order drift: {model}/{task}")
    return frame


def _load_overall(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"task", "model", "MAE", "MAPE", "RMSE", "NSE"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Overall table missing columns: {sorted(missing)}")
    return frame


def _derive_daywise(
    release: Path,
    *,
    indices: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    truth, base_pred, common = _load_common_predictions(release, "DCRNN", "168h")
    for model in ABLATION_MODELS[1:]:
        model_truth, pred, model_common = _load_common_predictions(
            release, model, "168h"
        )
        _ensure_same_truth(truth, model_truth, model)
        if not np.array_equal(common, model_common):
            raise ValueError(f"Origin index drift: {model}")
        for metric in METRICS:
            for day in range(1, 8):
                baseline = _metric_per_origin(
                    truth, base_pred, metric, day=day
                )
                candidate = _metric_per_origin(
                    truth, pred, metric, day=day
                )
                values = _paired_improvement(baseline, candidate, metric)
                mean, lo, hi = _block_ci(values, indices)
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "day": day,
                        "mean_improvement": mean,
                        "ci95_lower": lo,
                        "ci95_upper": hi,
                        "wins": int((values > 0).sum()),
                        "n_origins": int(values.size),
                    }
                )
    return pd.DataFrame(rows)


def _main_figure3_ablation(
    daywise: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.35), sharex=False)
    offsets = {
        "DCRNN + SAS-Norm": -0.12,
        "DCRNN + FA-DPR": 0.0,
        "STaR-GNN": 0.12,
    }
    for ax, metric, panel in zip(axes.flat, METRICS, "abcd"):
        for model in ABLATION_MODELS[1:]:
            sel = daywise.loc[
                (daywise["model"] == model) & (daywise["metric"] == metric)
            ].sort_values("day")
            x = sel["day"].to_numpy(float) + offsets[model]
            y = sel["mean_improvement"].to_numpy(float)
            lo = sel["ci95_lower"].to_numpy(float)
            hi = sel["ci95_upper"].to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                marker=MODEL_MARKERS[model],
                markersize=4.0,
                linewidth=2.0 if model == "STaR-GNN" else 1.25,
                elinewidth=0.85,
                capsize=2.0,
                label=model,
                zorder=5 if model == "STaR-GNN" else 3,
            )
        ax.axhline(0.0, color=ZERO_GRAY, linewidth=0.8, zorder=0)
        ax.set_xticks(range(1, 8))
        ax.set_xlabel("Forecast day")
        ax.set_ylabel(_improvement_label(metric))
        ax.set_title(
            metric,
            pad=7,
            fontweight="bold",
        )
        style_axis(ax, ygrid=True)
        add_panel_label(ax, panel)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=1.25,
        handlelength=2.5,
    )
    fig.subplots_adjust(top=0.86, wspace=0.36, hspace=0.34, bottom=0.10)
    save_publication_figure(fig, output / "main_fig3_ablation_leadtime")


def _derive_origin_improvements(
    release: Path,
    *,
    indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for task in TASKS:
        star_truth, star_pred, common = _load_common_predictions(release, "STaR-GNN", task)
        for baseline in BASELINE_MODELS:
            base_truth, base_pred, base_common = _load_common_predictions(
                release, baseline, task
            )
            _ensure_same_truth(star_truth, base_truth, f"{task}/{baseline}")
            if not np.array_equal(common, base_common):
                raise ValueError(f"Origin index drift: {task}/{baseline}")
            for metric in METRICS:
                base = _metric_per_origin(base_truth, base_pred, metric)
                star = _metric_per_origin(star_truth, star_pred, metric)
                improvement = _paired_improvement(base, star, metric)
                mean, lo, hi = _block_ci(improvement, indices)
                for pos, (common_index, value) in enumerate(
                    zip(common, improvement)
                ):
                    rows.append(
                        {
                            "task": task,
                            "baseline": baseline,
                            "metric": metric,
                            "origin_position": pos,
                            "common_index": int(common_index),
                            "improvement": float(value),
                        }
                    )
                summary.append(
                    {
                        "task": task,
                        "baseline": baseline,
                        "metric": metric,
                        "mean_improvement": mean,
                        "ci95_lower": lo,
                        "ci95_upper": hi,
                        "wins": int((improvement > 0).sum()),
                        "losses": int((improvement < 0).sum()),
                        "ties": int(np.isclose(improvement, 0.0).sum()),
                        "n_origins": int(improvement.size),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def _derive_dma_improvement(release: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        star = _load_dma_metrics(release, "STaR-GNN", task).set_index("entity")
        for baseline in BASELINE_MODELS:
            base = _load_dma_metrics(release, baseline, task).set_index("entity")
            for dma in DMAS:
                for metric in METRICS:
                    base_value = float(base.loc[dma, metric])
                    star_value = float(star.loc[dma, metric])
                    if metric == "MAPE":
                        base_value *= 100.0
                        star_value *= 100.0
                    improvement = float(
                        _paired_improvement(
                            np.array([base_value]),
                            np.array([star_value]),
                            metric,
                        )[0]
                    )
                    rows.append(
                        {
                            "DMA": dma,
                            "task": task,
                            "baseline": baseline,
                            "metric": metric,
                            "improvement": improvement,
                        }
                    )
    return pd.DataFrame(rows)


def _main_figure2(
    paired: pd.DataFrame,
    paired_summary: pd.DataFrame,
    dma: pd.DataFrame,
    output: Path,
) -> None:
    del paired  # detailed paired values remain available in the exported CSV
    groups = [("24h", "DCRNN"), ("24h", "STGCN"),
              ("168h", "DCRNN"), ("168h", "STGCN")]
    labels = ["DCRNN", "STGCN", "DCRNN", "STGCN"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.45))
    cmap = light_to_hero_cmap()

    origin_win = np.zeros((4, 4), dtype=float)
    origin_text: list[list[str]] = [["" for _ in groups] for _ in METRICS]
    dma_win = np.zeros((4, 4), dtype=float)
    dma_text: list[list[str]] = [["" for _ in groups] for _ in METRICS]
    for i, metric in enumerate(METRICS):
        for j, (task, baseline) in enumerate(groups):
            row = paired_summary.loc[
                (paired_summary["task"] == task)
                & (paired_summary["baseline"] == baseline)
                & (paired_summary["metric"] == metric)
            ].iloc[0]
            wins, n = int(row["wins"]), int(row["n_origins"])
            origin_win[i, j] = 100.0 * wins / n
            mean = float(row["mean_improvement"])
            unit = "" if metric == "NSE" else "%"
            origin_text[i][j] = f"{mean:+.2f}{unit}\n{wins}/{n}"

            vals = dma.loc[
                (dma["task"] == task)
                & (dma["baseline"] == baseline)
                & (dma["metric"] == metric),
                "improvement",
            ].to_numpy(float)
            wins_dma = int((vals > 0).sum())
            dma_win[i, j] = 100.0 * wins_dma / len(vals)
            dma_text[i][j] = f"{vals.mean():+.2f}{unit}\n{wins_dma}/{len(vals)}"

    images = []
    for ax, matrix, text_matrix, title, panel in (
        (axes[0], origin_win, origin_text, "Forecast-origin consistency", "a"),
        (axes[1], dma_win, dma_text, "DMA-level consistency", "b"),
    ):
        images.append(ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto"))
        ax.set_yticks(np.arange(4), METRICS)
        ax.set_xticks(np.arange(4), labels)
        ax.set_xlabel("Baseline model")
        ax.set_ylabel("Evaluation metric")
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        ax.axvline(1.5, color="white", linewidth=3.0)
        ax.text(0.5, 1.03, "24 h", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.text(2.5, 1.03, "168 h", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_title(title, pad=20, fontweight="bold")
        for i in range(4):
            for j in range(4):
                color = "white" if matrix[i, j] >= 72 else "#202020"
                ax.text(j, i, text_matrix[i][j], ha="center", va="center",
                        fontsize=7.0, color=color, linespacing=1.15)
        add_panel_label(ax, panel)
    fig.subplots_adjust(wspace=0.26, bottom=0.17, top=0.82, left=0.09, right=0.86)
    cax = fig.add_axes([0.90, 0.17, 0.018, 0.65])
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label("Share of comparisons improved (%)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    fig.text(
        0.5,
        0.015,
        "Cell text: mean error reduction (%) or mean $\\Delta$NSE; wins / comparisons",
             ha="center", va="bottom", fontsize=7.6, color="#4B4B4B")
    save_publication_figure(fig, output / "main_fig3_temporal_spatial_robustness")


def _load_dma_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"task", "DMA", "model", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DMA table missing columns: {sorted(missing)}")
    frame = frame.loc[
        frame["task"].isin(TASKS)
        & frame["DMA"].isin(DMAS)
        & frame["model"].isin(DMA_MODELS)
    ].copy()
    expected = len(TASKS) * len(DMAS) * len(DMA_MODELS)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} DMA rows, found {len(frame)}")
    counts = frame.groupby(["task", "DMA"])["model"].nunique()
    if not (counts == len(DMA_MODELS)).all():
        raise ValueError("Every horizon-DMA block must contain all nine models")
    return frame


def _derive_dma_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    long = frame.melt(
        id_vars=["task", "DMA", "model"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    )
    long["rank"] = long.groupby(["task", "DMA", "metric"])["value"].rank(
        method="min", ascending=True
    )
    nse = long["metric"].eq("NSE")
    long.loc[nse, "rank"] = long.loc[nse].groupby(
        ["task", "DMA", "metric"]
    )["value"].rank(method="min", ascending=False)
    return long


def _derive_dma_pairwise(frame: pd.DataFrame) -> pd.DataFrame:
    long = frame.melt(
        id_vars=["task", "DMA", "model"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    )
    wide = long.pivot(
        index=["task", "DMA", "metric"], columns="model", values="value"
    )
    rows: list[dict[str, Any]] = []
    for (task, dma, metric), values in wide.iterrows():
        star = float(values["STaR-GNN"])
        for baseline in DMA_MODELS[:-1]:
            base = float(values[baseline])
            if metric == "NSE":
                improvement = star - base
            else:
                improvement = 100.0 * (base - star) / max(abs(base), 1.0e-12)
            rows.append(
                {
                    "task": task,
                    "DMA": dma,
                    "metric": metric,
                    "baseline": baseline,
                    "baseline_family": (
                        "graph" if baseline in BASELINE_MODELS else "sequence"
                    ),
                    "baseline_value": base,
                    "star_value": star,
                    "improvement": improvement,
                    "star_better": bool(improvement > 0.0),
                }
            )
    return pd.DataFrame(rows)


def _main_figure2_dma_ranks(
    ranks: pd.DataFrame,
    output: Path,
) -> None:
    """Show all-model spatial consistency without mixing DMA-specific scales."""
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 5.35))
    cmap = light_to_hero_cmap().reversed()
    norm = plt.Normalize(1.0, float(len(DMA_MODELS)))
    image = None

    panel = iter("abcdefgh")
    for row, task in enumerate(TASKS):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            block = ranks.loc[
                (ranks["task"] == task) & (ranks["metric"] == metric)
            ].pivot(index="model", columns="DMA", values="rank")
            matrix = block.reindex(index=DMA_MODELS, columns=DMAS).to_numpy(float)
            if np.isnan(matrix).any():
                raise ValueError(f"Incomplete rank matrix: {task}/{metric}")
            image = ax.imshow(
                matrix,
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )
            ax.set_xticks(np.arange(len(DMAS)), DMAS)
            ax.set_yticks(np.arange(len(DMA_MODELS)), DMA_MODELS)
            if col != 0:
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.set_ylabel("Model")
            ax.set_xlabel("DMA")
            ax.set_title(
                f"{task.replace('h', ' h')} | {metric}",
                pad=7,
                fontweight="bold",
            )
            ax.tick_params(length=0, labelsize=6.7)
            ax.spines[:].set_visible(False)
            ax.axhline(5.5, color="white", linewidth=2.2)
            ax.axhline(7.5, color="white", linewidth=2.2)
            ax.add_patch(
                Rectangle(
                    (-0.49, len(DMA_MODELS) - 1.49),
                    len(DMAS) - 0.02,
                    0.98,
                    fill=False,
                    edgecolor=HERO_BLUE,
                    linewidth=1.25,
                    clip_on=False,
                )
            )
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    rgba = cmap(norm(matrix[i, j]))
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    ax.text(
                        j,
                        i,
                        f"{int(matrix[i, j])}",
                        ha="center",
                        va="center",
                        fontsize=5.25,
                        color="white" if luminance < 0.52 else "#202020",
                    )
            add_panel_label(ax, next(panel))

    if image is None:
        raise RuntimeError("No DMA rank panels were rendered")
    fig.subplots_adjust(
        left=0.13,
        right=0.91,
        bottom=0.09,
        top=0.94,
        wspace=0.13,
        hspace=0.30,
    )
    cax = fig.add_axes([0.93, 0.15, 0.018, 0.72])
    cbar = fig.colorbar(image, cax=cax, ticks=np.arange(1, 10))
    cbar.set_label("Within-DMA rank (1 = best)", fontsize=8.2)
    cbar.ax.tick_params(labelsize=7.0)
    save_publication_figure(fig, output / "main_fig2_dma_performance")


def _main_figure2_dma_summary(
    ranks: pd.DataFrame,
    output: Path,
) -> None:
    """Summarize all-model rank distributions and STaR-GNN spatial coverage."""
    fig = plt.figure(figsize=(7.4, 4.65))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[3.2, 0.9],
        hspace=0.42,
        wspace=0.20,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    family_colors = {
        **{model: "#D5DCE3" for model in DMA_MODELS[:6]},
        "DCRNN": MODEL_COLORS["DCRNN"],
        "STGCN": MODEL_COLORS["STGCN"],
        "STaR-GNN": HERO_BLUE,
    }
    rng = np.random.default_rng(20260825)
    for ax, task, panel in zip(axes, TASKS, ("a", "b")):
        distributions = [
            ranks.loc[
                (ranks["task"] == task) & (ranks["model"] == model), "rank"
            ].to_numpy(float)
            for model in DMA_MODELS
        ]
        boxes = ax.boxplot(
            distributions,
            vert=False,
            positions=np.arange(len(DMA_MODELS)),
            widths=0.52,
            whis=(0, 100),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#202020", "linewidth": 1.0},
            whiskerprops={"color": "#777777", "linewidth": 0.8},
            capprops={"color": "#777777", "linewidth": 0.8},
        )
        for model, box in zip(DMA_MODELS, boxes["boxes"]):
            box.set_facecolor(family_colors[model])
            box.set_edgecolor(HERO_BLUE if model == "STaR-GNN" else "#777777")
            box.set_linewidth(1.35 if model == "STaR-GNN" else 0.8)

        for pos, (model, values) in enumerate(zip(DMA_MODELS, distributions)):
            y = pos + rng.uniform(-0.13, 0.13, size=len(values))
            ax.scatter(
                values,
                y,
                s=14 if model == "STaR-GNN" else 9,
                marker="D" if model == "STaR-GNN" else "o",
                color=family_colors[model],
                edgecolor=HERO_BLUE if model == "STaR-GNN" else "#707070",
                linewidth=0.45,
                alpha=0.82 if model == "STaR-GNN" else 0.42,
                zorder=4 if model == "STaR-GNN" else 3,
            )

        star = ranks.loc[
            (ranks["task"] == task) & (ranks["model"] == "STaR-GNN"), "rank"
        ]
        first = int((star == 1).sum())
        ax.set_xlim(0.65, 9.35)
        ax.set_xticks(np.arange(1, 10))
        ax.set_yticks(np.arange(len(DMA_MODELS)), DMA_MODELS)
        ax.invert_yaxis()
        ax.set_xlabel("Within-DMA rank (1 = best)")
        if ax is axes[0]:
            ax.set_ylabel("Model")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(
            f"{task.replace('h', ' h')}\nSTaR-GNN: {first}/40 first-place ranks",
            pad=5,
            fontweight="bold",
            fontsize=8.8,
            linespacing=1.15,
        )
        ax.xaxis.grid(True, color="#E4E4E4", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        add_panel_label(ax, panel)

    ax_c = fig.add_subplot(gs[1, :])
    coverage = np.zeros((len(TASKS), len(DMAS)), dtype=float)
    for i, task in enumerate(TASKS):
        for j, dma in enumerate(DMAS):
            coverage[i, j] = float(
                (
                    (ranks["task"] == task)
                    & (ranks["DMA"] == dma)
                    & (ranks["model"] == "STaR-GNN")
                    & (ranks["rank"] == 1)
                ).sum()
            )
    image = ax_c.imshow(
        coverage,
        cmap=light_to_hero_cmap(),
        vmin=0,
        vmax=4,
        aspect="auto",
        interpolation="nearest",
    )
    ax_c.set_xticks(np.arange(len(DMAS)), DMAS)
    ax_c.set_yticks(np.arange(len(TASKS)), [task.replace("h", " h") for task in TASKS])
    ax_c.set_xlabel("DMA")
    ax_c.set_ylabel("Horizon")
    ax_c.set_title(
        "Number of metrics ranked first by STaR-GNN",
        pad=6,
        fontweight="bold",
    )
    ax_c.tick_params(length=0)
    ax_c.spines[:].set_visible(False)
    for i in range(coverage.shape[0]):
        for j in range(coverage.shape[1]):
            ax_c.text(
                j,
                i,
                f"{int(coverage[i, j])}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if coverage[i, j] >= 3 else "#202020",
            )
    add_panel_label(ax_c, "c")
    cbar = fig.colorbar(image, ax=ax_c, fraction=0.018, pad=0.018, ticks=np.arange(5))
    cbar.set_label("First-place metrics (out of 4)", fontsize=7.8)
    cbar.ax.tick_params(labelsize=7.0)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.94, bottom=0.10)
    save_publication_figure(fig, output / "main_fig2_dma_performance")


def _diurnal_profile(
    release: Path,
    *,
    indices: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    truth_ref: np.ndarray | None = None
    for model in TRAJECTORY_MODELS:
        truth, pred, _ = _load_common_predictions(release, model, "168h")
        truth_ref = _ensure_same_truth(truth_ref, truth, model)
        aggregate_truth = truth.sum(axis=2)
        aggregate_pred = pred.sum(axis=2)
        abs_error = np.abs(aggregate_pred - aggregate_truth).reshape(46, 7, 24)
        origin_profile = abs_error.mean(axis=1)
        for hour in range(24):
            mean, lo, hi = _block_ci(origin_profile[:, hour], indices)
            rows.append(
                {
                    "model": model,
                    "hour_within_day": hour,
                    "mean_abs_error": mean,
                    "ci95_lower": lo,
                    "ci95_upper": hi,
                }
            )
    return pd.DataFrame(rows)


def _select_representative(
    release: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth_star, pred_star, common = _load_common_predictions(
        release, "STaR-GNN", "168h"
    )
    star_mae = _publisher_mae_per_origin(truth_star, pred_star)
    median_mae = float(np.median(star_mae))
    pos = int(np.argmin(np.abs(star_mae - median_mae)))
    common_index = int(common[pos])

    truth_ref: np.ndarray | None = None
    data: dict[str, np.ndarray] = {}
    total_mae: dict[str, float] = {}
    for model in TRAJECTORY_MODELS:
        truth, pred, model_common = _load_common_predictions(release, model, "168h")
        truth_ref = _ensure_same_truth(truth_ref, truth, model)
        if int(model_common[pos]) != common_index:
            raise ValueError("Representative common-index drift")
        data[model] = pred[pos].sum(axis=1)
        total_mae[model] = float(
            _publisher_mae_per_origin(
                truth[pos : pos + 1], pred[pos : pos + 1]
            )[0]
        )
    if truth_ref is None:
        raise RuntimeError("No representative truth")
    observed = truth_ref[pos].sum(axis=1)
    frame = pd.DataFrame(
        {
            "forecast_hour": np.arange(1, 169),
            "observed_total": observed,
            "DCRNN": data["DCRNN"],
            "STGCN": data["STGCN"],
            "STaR-GNN": data["STaR-GNN"],
        }
    )
    meta = {
        "selection_rule": (
            "STaR-GNN origin whose 168 h total MAE is closest "
            "to the median among the 46 common test origins"
        ),
        "selected_origin_position_zero_based": pos,
        "selected_common_index": common_index,
        "median_star_total_mae": median_mae,
        "selected_star_total_mae": float(star_mae[pos]),
        "selected_origin_model_total_mae": total_mae,
    }
    return frame, meta


def _main_figure4_week(
    diurnal: pd.DataFrame,
    trajectory: pd.DataFrame,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(7.4, 5.0))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 2.15],
        height_ratios=[1.65, 1.0],
        wspace=0.38,
        hspace=0.38,
    )
    ax_d = fig.add_subplot(gs[:, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1], sharex=ax_t)

    for model in TRAJECTORY_MODELS:
        sel = diurnal.loc[diurnal["model"] == model].sort_values("hour_within_day")
        x = sel["hour_within_day"].to_numpy(int)
        y = sel["mean_abs_error"].to_numpy(float)
        lo = sel["ci95_lower"].to_numpy(float)
        hi = sel["ci95_upper"].to_numpy(float)
        lw = 2.35 if model == "STaR-GNN" else 1.45
        ax_d.plot(
            x, y,
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=lw,
            label=model,
        )
        ax_d.fill_between(
            x, lo, hi,
            color=MODEL_COLORS[model],
            alpha=0.10 if model == "STaR-GNN" else 0.06,
            linewidth=0,
        )
    ax_d.set_xticks([0, 6, 12, 18, 23])
    ax_d.set_xlabel("Hour within forecast day")
    ax_d.set_ylabel("Aggregate absolute error (L s$^{-1}$)")
    ax_d.set_title("Diurnal error profile", pad=8, fontweight="bold")
    style_axis(ax_d, ygrid=True)
    add_panel_label(ax_d, "a")

    hours = trajectory["forecast_hour"].to_numpy(int)
    observed = trajectory["observed_total"].to_numpy(float)
    ax_t.plot(
        hours, observed,
        color=OBSERVED_BLACK,
        linewidth=2.25,
        label="Observed",
        zorder=5,
    )
    for model in TRAJECTORY_MODELS:
        values = trajectory[model].to_numpy(float)
        lw = 2.15 if model == "STaR-GNN" else 1.25
        ax_t.plot(
            hours, values,
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=lw,
            label=model,
            zorder=4 if model == "STaR-GNN" else 2,
        )
        ax_e.plot(
            hours, np.abs(values - observed),
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=lw,
            label=model,
        )

    for boundary in range(24, 168, 24):
        for ax in (ax_t, ax_e):
            ax.axvline(
                boundary + 0.5,
                color="#D0D0D0",
                linewidth=0.55,
                zorder=0,
            )

    ax_t.set_ylabel("Aggregate demand (L s$^{-1}$)")
    ax_t.set_title("Week-ahead demand", pad=8, fontweight="bold")
    style_axis(ax_t, ygrid=False)
    day_ticks = np.arange(12, 169, 24)
    ax_t.set_xticks(day_ticks, [str(i) for i in range(1, 8)])
    add_panel_label(ax_t, "b")

    ax_e.set_ylabel("Absolute error\n(L s$^{-1}$)")
    ax_e.set_xlabel("Forecast day")
    ax_e.set_xticks(day_ticks, [str(i) for i in range(1, 8)])
    ax_e.set_title("Hourly errors", pad=7, fontweight="bold")
    style_axis(ax_e, ygrid=True)
    ax_t.set_xlabel("")
    add_panel_label(ax_e, "c")

    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.66, 1.01),
        ncol=4,
        columnspacing=1.0,
        handlelength=2.5,
    )
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.09, right=0.98)
    save_publication_figure(fig, output / "main_fig4_week_ahead_dynamics")


def _main_figure_overall(
    overall: pd.DataFrame,
    output: Path,
) -> None:
    baselines = list(REPORTED_MODELS) + ["DCRNN", "STGCN"]
    clean_names = [
        name.replace(" (reported)", "")
        .replace("MSCMNet_WM", "MSCMNet-WM")
        .replace("MSCMNet_M", "MSCMNet-M")
        .replace("MSCMNet_W", "MSCMNet-W")
        for name in baselines
    ]
    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        row: dict[str, Any] = {"baseline": baseline}
        for task in TASKS:
            tf = overall.loc[overall["task"] == task].set_index("model")
            star = tf.loc["STaR-GNN"]
            base = tf.loc[baseline]
            for metric in ("MAE", "MAPE", "RMSE"):
                row[f"{task}_{metric}"] = 100.0 * (
                    float(base[metric]) - float(star[metric])
                ) / float(base[metric])
            row[f"{task}_NSE"] = float(star["NSE"]) - float(base["NSE"])
        rows.append(row)
    df = pd.DataFrame(rows)

    cols = [
        "24h_MAE", "24h_MAPE", "24h_RMSE",
        "168h_MAE", "168h_MAPE", "168h_RMSE",
    ]
    labels = [
        "MAE", "MAPE", "RMSE",
        "MAE", "MAPE", "RMSE",
    ]
    matrix = df[cols].to_numpy(float)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.4, 3.6),
        gridspec_kw={"width_ratios": [2.8, 1.0]},
    )
    ax_h, ax_n = axes
    cmap = light_to_hero_cmap()
    vmax = float(np.ceil(matrix.max() / 5.0) * 5.0)
    image = ax_h.imshow(
        matrix,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    ax_h.set_yticks(np.arange(len(clean_names)), clean_names)
    ax_h.set_xticks(np.arange(6), labels)
    ax_h.set_xlabel("Error metric")
    ax_h.set_ylabel("Baseline model")
    ax_h.set_title(
        "Relative reduction in forecasting errors",
        pad=17,
        fontweight="bold",
    )
    ax_h.tick_params(length=0)
    ax_h.spines[:].set_visible(False)
    ax_h.axvline(2.5, color="white", linewidth=3.0)
    ax_h.axhline(5.5, color="white", linewidth=2.5)
    ax_h.text(
        1.0, 1.025, "24 h",
        transform=ax_h.get_xaxis_transform(),
        ha="center", va="bottom", fontweight="bold", fontsize=8.5,
    )
    ax_h.text(
        4.0, 1.025, "168 h",
        transform=ax_h.get_xaxis_transform(),
        ha="center", va="bottom", fontweight="bold", fontsize=8.5,
    )
    norm = plt.Normalize(0.0, vmax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            r, g, b, _ = cmap(norm(value))
            lum = 0.299*r + 0.587*g + 0.114*b
            ax_h.text(
                j, i, f"{value:.1f}",
                ha="center", va="center", fontsize=6.8,
                color="white" if lum < 0.52 else "#202020",
            )
    cb = fig.colorbar(image, ax=ax_h, fraction=0.035, pad=0.02)
    cb.set_label("Error reduction (%)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    add_panel_label(ax_h, "a")

    y = np.arange(len(baselines))
    n24 = df["24h_NSE"].to_numpy(float)
    n168 = df["168h_NSE"].to_numpy(float)
    ax_n.axvline(0.0, color=ZERO_GRAY, linewidth=0.8)
    ax_n.scatter(n24, y - 0.12, color="#8FB6D5", s=18, label="24 h")
    ax_n.scatter(n168, y + 0.12, color=HERO_BLUE, s=18, marker="s", label="168 h")
    for i in range(len(y)):
        ax_n.plot(
            [n24[i], n168[i]],
            [y[i]-0.12, y[i]+0.12],
            color="#B5B5B5", linewidth=0.65, zorder=0,
        )
    ax_n.set_yticks(y, clean_names)
    ax_n.yaxis.tick_right()
    ax_n.tick_params(axis="y", labelsize=6.7, pad=3)
    ax_n.invert_yaxis()
    ax_n.set_xlabel("Absolute NSE improvement ($\\Delta$NSE)")
    ax_n.set_ylabel("Baseline model")
    ax_n.yaxis.set_label_position("right")
    ax_n.set_title(
        "Improvement in NSE",
        pad=17,
        fontweight="bold",
    )
    style_axis(ax_n, ygrid=False)
    ax_n.legend(loc="lower right")
    add_panel_label(ax_n, "b")

    fig.subplots_adjust(wspace=0.34, bottom=0.19, top=0.88, right=0.90)
    save_publication_figure(fig, output / "main_fig1_overall_performance")


def _supp_figure_s1_dma(
    dma: pd.DataFrame,
    output: Path,
) -> None:
    """Detailed DMA-level improvements for all four metrics."""
    groups = [("24h", "DCRNN"), ("24h", "STGCN"),
              ("168h", "DCRNN"), ("168h", "STGCN")]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), sharey=True)
    for ax, metric, panel in zip(axes.flat, METRICS, "abcd"):
        matrix = np.zeros((len(DMAS), len(groups)), dtype=float)
        for i, dma_name in enumerate(DMAS):
            for j, (task, baseline) in enumerate(groups):
                value = dma.loc[
                    (dma["DMA"] == dma_name)
                    & (dma["task"] == task)
                    & (dma["baseline"] == baseline)
                    & (dma["metric"] == metric),
                    "improvement",
                ]
                if len(value) != 1:
                    raise ValueError(
                        f"DMA cell missing: {dma_name}/{task}/{baseline}/{metric}"
                    )
                matrix[i, j] = float(value.iloc[0])
        absmax = max(float(np.max(np.abs(matrix))), 1.0e-9)
        image = ax.imshow(
            matrix,
            cmap="RdBu",
            vmin=-absmax,
            vmax=absmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_yticks(np.arange(len(DMAS)), DMAS)
        ax.set_xticks(np.arange(4), ["DCRNN", "STGCN", "DCRNN", "STGCN"])
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        ax.axvline(1.5, color="white", linewidth=3.0)
        ax.text(0.5, 1.02, "24 h", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.2, fontweight="bold")
        ax.text(2.5, 1.02, "168 h", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.2, fontweight="bold")
        ax.set_title(_improvement_label(metric), pad=19, fontweight="bold")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                number = f"{matrix[i, j]:+.2f}" if metric == "NSE" else f"{matrix[i, j]:+.1f}"
                ax.text(j, i, number, ha="center", va="center",
                        fontsize=6.7,
                        color="white" if abs(matrix[i, j]) > 0.58 * absmax else "#202020")
        fig.colorbar(image, ax=ax, fraction=0.042, pad=0.025).ax.tick_params(
            labelsize=7.0
        )
        add_panel_label(ax, panel)
    axes[0, 0].set_ylabel("DMA")
    axes[1, 0].set_ylabel("DMA")
    fig.supxlabel("Baseline model", y=0.015, fontsize=9)
    fig.subplots_adjust(wspace=0.30, hspace=0.32, bottom=0.10, top=0.94)
    save_publication_figure(fig, output / "supp_figS1_dma_improvement")


def _supp_figure_s2(
    release: Path,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, task, label in zip(axes, TASKS, ("a", "b")):
        truth_ref: np.ndarray | None = None
        for model in TRAJECTORY_MODELS:
            truth, pred, _ = _load_common_predictions(release, model, task)
            truth_ref = _ensure_same_truth(truth_ref, truth, f"{task}/{model}")
            values = np.sort(_publisher_mae_per_origin(truth, pred))
            p = np.arange(1, values.size + 1) / values.size
            ax.step(
                values,
                p,
                where="post",
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                linewidth=2.25 if model == "STaR-GNN" else 1.45,
                label=model,
            )
        ax.set_xlabel("Per-origin total MAE (L s$^{-1}$)")
        style_axis(ax, ygrid=True)
        add_panel_label(ax, label)
        ax.text(
            0.98, 0.04, task.replace("h", " h"),
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8.5, fontweight="bold",
        )
    axes[0].set_ylabel("Empirical cumulative probability")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
    )
    fig.subplots_adjust(top=0.82, bottom=0.20, wspace=0.27)
    save_publication_figure(fig, output / "supp_figS2_origin_ecdf")


def _write_audit(
    path: Path,
    *,
    daywise: pd.DataFrame,
    ranks: pd.DataFrame,
    pairwise: pd.DataFrame,
    representative: dict[str, Any],
    block_length: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> None:
    audit = {
        "figure_architecture": {
            "main_fig1": "Overall four-metric performance against eight baselines.",
            "main_fig2": (
                "Within-DMA ranks for nine models, two horizons and all four "
                "evaluation metrics."
            ),
            "main_fig3": (
                "Four-metric factorial ablation and lead-time stability; "
                "paired improvements relative to DCRNN."
            ),
            "main_fig4": (
                "Scale-to-instance week-ahead dynamics; population diurnal "
                "error profile plus deterministic representative trajectory."
            ),
        },
        "block_bootstrap": {
            "block_length_origins": block_length,
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "rationale": (
                "Ordered forecast origins preserve weekly dependence; the 168 h "
                "origins also overlap strongly because they start 24 h apart."
            ),
        },
        "main_fig2_dma": {
            task: {
                "first_place_cells": int(
                    (
                        (ranks["task"] == task)
                        & (ranks["model"] == "STaR-GNN")
                        & (ranks["rank"] == 1)
                    ).sum()
                ),
                "n_dma_metric_cells": int(len(DMAS) * len(METRICS)),
                "pairwise_wins": int(
                    pairwise.loc[pairwise["task"] == task, "star_better"].sum()
                ),
                "n_pairwise_comparisons": int(
                    (pairwise["task"] == task).sum()
                ),
                "graph_baseline_wins": int(
                    pairwise.loc[
                        (pairwise["task"] == task)
                        & (pairwise["baseline_family"] == "graph"),
                        "star_better",
                    ].sum()
                ),
                "n_graph_baseline_comparisons": int(
                    (
                        (pairwise["task"] == task)
                        & (pairwise["baseline_family"] == "graph")
                    ).sum()
                ),
            }
            for task in TASKS
        },
        "main_fig3_daywise_summary": daywise.to_dict(orient="records"),
        "main_fig4_representative": representative,
    }
    path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("results/paper/frozen_v1"),
    )
    parser.add_argument(
        "--overall-table",
        type=Path,
        default=Path(
            "paper/tables/literature/table_literature_comparison_common46.csv"
        ),
    )
    parser.add_argument(
        "--dma-table",
        type=Path,
        default=Path("paper/tables/literature/table_all_models_dma.csv"),
    )
    parser.add_argument(
        "--main-output",
        type=Path,
        default=Path("paper/figures/submission"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("paper/tables/manuscript/submission"),
    )
    parser.add_argument("--block-length", type=int, default=7)
    parser.add_argument("--bootstrap-iterations", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    args = parser.parse_args()

    apply_publication_style()

    release = args.release.resolve()
    overall_path = args.overall_table.resolve()
    dma_path = args.dma_table.resolve()
    main_output = args.main_output.resolve()
    audit_output = args.audit_output.resolve()

    if not release.is_dir():
        raise FileNotFoundError(release)
    if not overall_path.is_file():
        raise FileNotFoundError(overall_path)
    if not dma_path.is_file():
        raise FileNotFoundError(dma_path)
    if args.bootstrap_iterations < 1000:
        raise ValueError("Use at least 1000 bootstrap iterations")

    main_output.mkdir(parents=True, exist_ok=True)
    audit_output.mkdir(parents=True, exist_ok=True)

    indices = _moving_block_indices(
        46,
        args.block_length,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )

    overall = _load_overall(overall_path)
    _main_figure_overall(overall, main_output)

    dma_results = _load_dma_results(dma_path)
    ranks = _derive_dma_ranks(dma_results)
    pairwise = _derive_dma_pairwise(dma_results)
    ranks.to_csv(
        audit_output / "main_fig2_dma_ranks.csv",
        index=False,
        float_format="%.9f",
    )
    pairwise.to_csv(
        audit_output / "main_fig2_dma_pairwise_improvement.csv",
        index=False,
        float_format="%.9f",
    )
    _main_figure2_dma_summary(ranks, main_output)

    daywise = _derive_daywise(release, indices=indices)
    daywise.to_csv(
        audit_output / "main_fig3_daywise_paired_improvement.csv",
        index=False,
        float_format="%.9f",
    )
    _main_figure3_ablation(daywise, main_output)

    diurnal = _diurnal_profile(release, indices=indices)
    diurnal.to_csv(
        audit_output / "main_fig4_diurnal_aggregate_error.csv",
        index=False,
        float_format="%.9f",
    )
    trajectory, representative = _select_representative(release)
    trajectory.to_csv(
        audit_output / "main_fig4_representative_trajectory.csv",
        index=False,
        float_format="%.9f",
    )
    (audit_output / "main_fig4_representative_selection.json").write_text(
        json.dumps(representative, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _main_figure4_week(diurnal, trajectory, main_output)

    _write_audit(
        audit_output / "submission_figure_audit.json",
        daywise=daywise,
        ranks=ranks,
        pairwise=pairwise,
        representative=representative,
        block_length=args.block_length,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )

    print("Submission figure renderer: PASS")
    print("Main figures:")
    print("  Main Fig. 1 — overall four-metric performance")
    print("  Main Fig. 2 — all-model DMA-level performance ranks")
    print("  Main Fig. 3 — four-metric ablation and lead-time stability")
    print("  Main Fig. 4 — week-ahead demand dynamics")
    print(
        f"Block bootstrap: length={args.block_length}, "
        f"iterations={args.bootstrap_iterations}, seed={args.bootstrap_seed}"
    )


if __name__ == "__main__":
    main()
