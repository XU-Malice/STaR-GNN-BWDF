#!/usr/bin/env python
"""Render the canonical Journal of Hydrology submission result figures.

The renderer implements a claim-driven evidence architecture:

Main Figure 1
    Mechanism / ablation: absolute day-wise publisher-compatible MAE with
    moving-block bootstrap uncertainty + Day-1-to-Day-7 degradation.

Main Figure 2
    Temporal and spatial robustness: paired per-origin MAE improvements +
    DMA-level improvement heatmap.

Main Figure 3
    Week-ahead dynamics: population-level diurnal aggregate-error profile +
    a deterministic representative 168 h trajectory + its absolute error.

Supplementary Figure S1
    Relative improvement over all literature/re-evaluated baselines.

Supplementary Figure S2
    ECDF of per-origin publisher-compatible MAE for DCRNN, STGCN, STaR-GNN.

All figures are rebuilt from frozen common-46 predictions and audited tables.
No training, checkpoint selection, or hyperparameter tuning occurs here.
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


def _publisher_mae_per_origin_day(
    truth: np.ndarray, prediction: np.ndarray, day: int
) -> np.ndarray:
    if not 1 <= day <= 7:
        raise ValueError(day)
    sl = slice((day - 1) * 24, day * 24)
    return np.mean(np.abs(prediction[:, sl, :] - truth[:, sl, :]), axis=1).sum(axis=1)


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
    truth_ref: np.ndarray | None = None
    for model in ABLATION_MODELS:
        truth, pred, _ = _load_common_predictions(release, model, "168h")
        truth_ref = _ensure_same_truth(truth_ref, truth, model)
        for day in range(1, 8):
            values = _publisher_mae_per_origin_day(truth, pred, day)
            mean, lo, hi = _block_ci(values, indices)
            rows.append(
                {
                    "model": model,
                    "day": day,
                    "mean_MAE": mean,
                    "ci95_lower": lo,
                    "ci95_upper": hi,
                }
            )
    return pd.DataFrame(rows)


def _main_figure1(
    daywise: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharex=True)
    ax_abs, ax_deg = axes

    degradation_rows: list[dict[str, Any]] = []
    for model in ABLATION_MODELS:
        sel = daywise.loc[daywise["model"] == model].sort_values("day")
        x = sel["day"].to_numpy(int)
        y = sel["mean_MAE"].to_numpy(float)
        lo = sel["ci95_lower"].to_numpy(float)
        hi = sel["ci95_upper"].to_numpy(float)
        lw = 2.35 if model == "STaR-GNN" else 1.55
        z = 5 if model == "STaR-GNN" else 3

        ax_abs.plot(
            x, y,
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            marker=MODEL_MARKERS[model],
            markersize=4.3,
            linewidth=lw,
            label=model,
            zorder=z,
        )
        ax_abs.fill_between(
            x, lo, hi,
            color=MODEL_COLORS[model],
            alpha=0.10 if model == "STaR-GNN" else 0.07,
            linewidth=0,
            zorder=1,
        )

        day1 = float(y[0])
        rel = 100.0 * (y - day1) / day1
        for day, mean_mae, value in zip(x, y, rel):
            degradation_rows.append(
                {
                    "model": model,
                    "day": int(day),
                    "mean_MAE": float(mean_mae),
                    "relative_to_day1_pct": float(value),
                }
            )
        ax_deg.plot(
            x, rel,
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            marker=MODEL_MARKERS[model],
            markersize=4.3,
            linewidth=lw,
            label=model,
            zorder=z,
        )
        ax_deg.annotate(
            f"{rel[-1]:+.1f}%",
            xy=(7, rel[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.3,
            color=MODEL_COLORS[model],
            fontweight="bold" if model == "STaR-GNN" else "normal",
        )

    for ax in axes:
        ax.set_xticks(range(1, 8))
        ax.set_xlabel("Forecast day")
        style_axis(ax, ygrid=True)
    ax_abs.set_ylabel("Publisher-compatible MAE (L s$^{-1}$)")
    ax_deg.set_ylabel("MAE change from Day 1 (%)")
    ax_deg.axhline(0.0, color=ZERO_GRAY, linewidth=0.8, zorder=0)

    add_panel_label(ax_abs, "a")
    add_panel_label(ax_deg, "b")

    handles, labels = ax_abs.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        columnspacing=1.2,
        handlelength=2.7,
    )
    fig.subplots_adjust(top=0.80, wspace=0.32, bottom=0.18)
    save_publication_figure(fig, output / "main_fig1_ablation_leadtime")
    return pd.DataFrame(degradation_rows)


def _derive_origin_improvements(
    release: Path,
    *,
    indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for task in TASKS:
        star_truth, star_pred, common = _load_common_predictions(release, "STaR-GNN", task)
        star = _publisher_mae_per_origin(star_truth, star_pred)
        for baseline in BASELINE_MODELS:
            base_truth, base_pred, base_common = _load_common_predictions(
                release, baseline, task
            )
            _ensure_same_truth(star_truth, base_truth, f"{task}/{baseline}")
            if not np.array_equal(common, base_common):
                raise ValueError(f"Origin index drift: {task}/{baseline}")
            base = _publisher_mae_per_origin(base_truth, base_pred)
            diff = base - star
            mean, lo, hi = _block_ci(diff, indices)
            for pos, (common_index, value) in enumerate(zip(common, diff)):
                rows.append(
                    {
                        "task": task,
                        "baseline": baseline,
                        "origin_position": pos,
                        "common_index": int(common_index),
                        "mae_improvement": float(value),
                    }
                )
            summary.append(
                {
                    "task": task,
                    "baseline": baseline,
                    "mean_improvement": mean,
                    "ci95_lower": lo,
                    "ci95_upper": hi,
                    "wins": int((diff > 0).sum()),
                    "losses": int((diff < 0).sum()),
                    "ties": int(np.isclose(diff, 0.0).sum()),
                    "n_origins": int(diff.size),
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
                base_mae = float(base.loc[dma, "MAE"])
                star_mae = float(star.loc[dma, "MAE"])
                rows.append(
                    {
                        "DMA": dma,
                        "task": task,
                        "baseline": baseline,
                        "MAE_reduction_pct": 100.0
                        * (base_mae - star_mae)
                        / base_mae,
                    }
                )
    return pd.DataFrame(rows)


def _main_figure2(
    paired: pd.DataFrame,
    paired_summary: pd.DataFrame,
    dma: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.4, 3.75),
        gridspec_kw={"width_ratios": [1.05, 1.15]},
    )
    ax_pair, ax_dma = axes

    groups = [
        ("24h", "DCRNN"),
        ("24h", "STGCN"),
        ("168h", "DCRNN"),
        ("168h", "STGCN"),
    ]
    labels = [
        "24 h\nDCRNN",
        "24 h\nSTGCN",
        "168 h\nDCRNN",
        "168 h\nSTGCN",
    ]
    rng = np.random.default_rng(20260821)
    for x, (task, baseline) in enumerate(groups):
        vals = paired.loc[
            (paired["task"] == task) & (paired["baseline"] == baseline),
            "mae_improvement",
        ].to_numpy(float)
        jitter = rng.uniform(-0.13, 0.13, size=vals.size)
        point_color = "#AFC4D8" if task == "24h" else "#7398BB"
        ax_pair.scatter(
            np.full(vals.size, x) + jitter,
            vals,
            s=11,
            color=point_color,
            alpha=0.62,
            edgecolors="none",
            zorder=2,
        )
        row = paired_summary.loc[
            (paired_summary["task"] == task)
            & (paired_summary["baseline"] == baseline)
        ].iloc[0]
        mean = float(row["mean_improvement"])
        lo = float(row["ci95_lower"])
        hi = float(row["ci95_upper"])
        ax_pair.errorbar(
            x,
            mean,
            yerr=np.array([[mean - lo], [hi - mean]]),
            fmt="D",
            markersize=5,
            color=HERO_BLUE,
            ecolor=HERO_BLUE,
            elinewidth=1.6,
            capsize=3.5,
            capthick=1.2,
            zorder=4,
        )
        y_top = max(float(vals.max()), hi)
        ax_pair.annotate(
            f'{int(row["wins"])}/{int(row["n_origins"])}',
            xy=(x, y_top),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=HERO_BLUE,
            fontweight="bold",
        )

    ax_pair.axhline(0.0, color=ZERO_GRAY, linewidth=0.9)
    ax_pair.set_xticks(range(len(groups)), labels)
    ax_pair.set_ylabel(
        "Publisher-compatible MAE improvement\n"
        "(baseline − STaR-GNN; L s$^{-1}$)"
    )
    style_axis(ax_pair, ygrid=True)
    add_panel_label(ax_pair, "a")

    col_keys = [
        ("24h", "DCRNN"),
        ("24h", "STGCN"),
        ("168h", "DCRNN"),
        ("168h", "STGCN"),
    ]
    matrix = np.zeros((len(DMAS), len(col_keys)), dtype=float)
    for i, dma_name in enumerate(DMAS):
        for j, (task, baseline) in enumerate(col_keys):
            val = dma.loc[
                (dma["DMA"] == dma_name)
                & (dma["task"] == task)
                & (dma["baseline"] == baseline),
                "MAE_reduction_pct",
            ]
            if len(val) != 1:
                raise ValueError(f"DMA cell missing: {dma_name}/{task}/{baseline}")
            matrix[i, j] = float(val.iloc[0])

    cmap = light_to_hero_cmap()
    vmin = 0.0
    vmax = float(np.ceil(matrix.max() / 5.0) * 5.0)
    image = ax_dma.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    ax_dma.set_yticks(np.arange(len(DMAS)), DMAS)
    ax_dma.set_ylabel("DMA")
    ax_dma.set_xticks(
        np.arange(4),
        ["DCRNN", "STGCN", "DCRNN", "STGCN"],
        rotation=0,
    )
    ax_dma.tick_params(length=0)
    ax_dma.spines[:].set_visible(False)
    ax_dma.axvline(1.5, color="white", linewidth=3.0)
    ax_dma.text(
        0.5, 1.025, "24 h",
        transform=ax_dma.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
    )
    ax_dma.text(
        2.5, 1.025, "168 h",
        transform=ax_dma.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
    )
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            r, g, b, _ = cmap(norm(value))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if luminance < 0.52 else "#202020"
            ax_dma.text(
                j, i, f"{value:.1f}",
                ha="center", va="center",
                fontsize=7.0,
                color=color,
            )
    cbar = fig.colorbar(image, ax=ax_dma, fraction=0.048, pad=0.025)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label("MAE reduction (%)", fontsize=8.5)
    add_panel_label(ax_dma, "b")

    fig.subplots_adjust(wspace=0.38, bottom=0.18, top=0.93)
    save_publication_figure(fig, output / "main_fig2_temporal_spatial_robustness")


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
    publisher_mae: dict[str, float] = {}
    for model in TRAJECTORY_MODELS:
        truth, pred, model_common = _load_common_predictions(release, model, "168h")
        truth_ref = _ensure_same_truth(truth_ref, truth, model)
        if int(model_common[pos]) != common_index:
            raise ValueError("Representative common-index drift")
        data[model] = pred[pos].sum(axis=1)
        publisher_mae[model] = float(
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
            "STaR-GNN origin whose 168 h publisher-compatible MAE is closest "
            "to the median among the 46 common test origins"
        ),
        "selected_origin_position_zero_based": pos,
        "selected_common_index": common_index,
        "median_star_publisher_mae": median_mae,
        "selected_star_publisher_mae": float(star_mae[pos]),
        "selected_origin_model_publisher_mae": publisher_mae,
    }
    return frame, meta


def _main_figure3(
    diurnal: pd.DataFrame,
    trajectory: pd.DataFrame,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(7.4, 4.65))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 2.15],
        height_ratios=[1.65, 1.0],
        wspace=0.38,
        hspace=0.12,
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
    style_axis(ax_t, ygrid=False)
    ax_t.tick_params(axis="x", labelbottom=False)
    add_panel_label(ax_t, "b")

    ax_e.set_ylabel("Absolute error\n(L s$^{-1}$)")
    ax_e.set_xlabel("Forecast day")
    day_ticks = np.arange(12, 169, 24)
    ax_e.set_xticks(day_ticks, [str(i) for i in range(1, 8)])
    style_axis(ax_e, ygrid=True)
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
    save_publication_figure(fig, output / "main_fig3_week_ahead_dynamics")


def _supp_figure_s1(
    overall: pd.DataFrame,
    output: Path,
) -> None:
    baselines = list(REPORTED_MODELS) + ["DCRNN", "STGCN"]
    clean_names = [name.replace(" (reported)", "†") for name in baselines]
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
    ax_n.set_yticks(y, [""] * len(y))
    ax_n.invert_yaxis()
    ax_n.set_xlabel("NSE gain")
    style_axis(ax_n, ygrid=False)
    ax_n.legend(loc="lower right")
    add_panel_label(ax_n, "b")

    fig.subplots_adjust(wspace=0.24, bottom=0.17, top=0.92)
    save_publication_figure(fig, output / "supp_figS1_relative_improvement")


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
        ax.set_xlabel("Per-origin publisher-compatible MAE (L s$^{-1}$)")
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
    degradation: pd.DataFrame,
    paired_summary: pd.DataFrame,
    dma: pd.DataFrame,
    representative: dict[str, Any],
    block_length: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> None:
    day7 = degradation.loc[degradation["day"] == 7].set_index("model")
    audit = {
        "figure_architecture": {
            "main_fig1": (
                "Ablation mechanism and lead-time stability; four factorial "
                "variants only, no STGCN."
            ),
            "main_fig2": (
                "Temporal and spatial robustness; paired forecast-origin "
                "improvements plus DMA-level improvement."
            ),
            "main_fig3": (
                "Scale-to-instance week-ahead dynamics; population diurnal "
                "error profile plus deterministic representative trajectory."
            ),
            "supp_figS1": "Overall relative improvement over eight baselines.",
            "supp_figS2": "Per-origin ECDF as supplementary distributional evidence.",
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
        "main_fig1_day7_relative_to_day1_pct": {
            model: float(day7.loc[model, "relative_to_day1_pct"])
            for model in ABLATION_MODELS
        },
        "main_fig2_origin_summary": paired_summary.to_dict(orient="records"),
        "main_fig2_dma": {
            "all_positive": bool((dma["MAE_reduction_pct"] > 0).all()),
            "n_cells": int(len(dma)),
            "min_reduction_pct": float(dma["MAE_reduction_pct"].min()),
            "max_reduction_pct": float(dma["MAE_reduction_pct"].max()),
        },
        "main_fig3_representative": representative,
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
        "--main-output",
        type=Path,
        default=Path("paper/figures/submission"),
    )
    parser.add_argument(
        "--supp-output",
        type=Path,
        default=Path("paper/figures/supplementary"),
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
    main_output = args.main_output.resolve()
    supp_output = args.supp_output.resolve()
    audit_output = args.audit_output.resolve()

    if not release.is_dir():
        raise FileNotFoundError(release)
    if not overall_path.is_file():
        raise FileNotFoundError(overall_path)
    if args.bootstrap_iterations < 1000:
        raise ValueError("Use at least 1000 bootstrap iterations")

    main_output.mkdir(parents=True, exist_ok=True)
    supp_output.mkdir(parents=True, exist_ok=True)
    audit_output.mkdir(parents=True, exist_ok=True)

    indices = _moving_block_indices(
        46,
        args.block_length,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )

    daywise = _derive_daywise(release, indices=indices)
    daywise.to_csv(
        audit_output / "main_fig1_daywise_block_ci.csv",
        index=False,
        float_format="%.9f",
    )
    degradation = _main_figure1(daywise, main_output)
    degradation.to_csv(
        audit_output / "main_fig1_day7_degradation.csv",
        index=False,
        float_format="%.9f",
    )

    paired, paired_summary = _derive_origin_improvements(
        release,
        indices=indices,
    )
    paired.to_csv(
        audit_output / "main_fig2_origin_paired_improvement.csv",
        index=False,
        float_format="%.9f",
    )
    paired_summary.to_csv(
        audit_output / "main_fig2_origin_paired_summary.csv",
        index=False,
        float_format="%.9f",
    )
    dma = _derive_dma_improvement(release)
    dma.to_csv(
        audit_output / "main_fig2_dma_improvement.csv",
        index=False,
        float_format="%.9f",
    )
    _main_figure2(paired, paired_summary, dma, main_output)

    diurnal = _diurnal_profile(release, indices=indices)
    diurnal.to_csv(
        audit_output / "main_fig3_diurnal_aggregate_error.csv",
        index=False,
        float_format="%.9f",
    )
    trajectory, representative = _select_representative(release)
    trajectory.to_csv(
        audit_output / "main_fig3_representative_trajectory.csv",
        index=False,
        float_format="%.9f",
    )
    (audit_output / "main_fig3_representative_selection.json").write_text(
        json.dumps(representative, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _main_figure3(diurnal, trajectory, main_output)

    overall = _load_overall(overall_path)
    _supp_figure_s1(overall, supp_output)
    _supp_figure_s2(release, supp_output)

    _write_audit(
        audit_output / "submission_figure_audit.json",
        degradation=degradation,
        paired_summary=paired_summary,
        dma=dma,
        representative=representative,
        block_length=args.block_length,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )

    print("Submission figure renderer: PASS")
    print("Main figures:")
    print("  Main Fig. 1 — ablation and lead-time stability")
    print("  Main Fig. 2 — temporal and spatial robustness")
    print("  Main Fig. 3 — week-ahead demand dynamics")
    print("Supplementary figures:")
    print("  Fig. S1 — relative improvement over all baselines")
    print("  Fig. S2 — per-origin ECDF")
    print(
        f"Block bootstrap: length={args.block_length}, "
        f"iterations={args.bootstrap_iterations}, seed={args.bootstrap_seed}"
    )


if __name__ == "__main__":
    main()
