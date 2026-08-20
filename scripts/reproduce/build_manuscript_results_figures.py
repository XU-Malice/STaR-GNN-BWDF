#!/usr/bin/env python
"""Build manuscript-level result figures for Journal of Hydrology submission.

This script intentionally separates *precise reporting* from *scientific
visualization*:

- precise absolute metrics stay in the manuscript tables;
- figures expose relative gains, long-horizon degradation, robustness across
  test origins, spatial consistency across DMAs, and representative forecast
  behavior.

All derived figures are rebuilt from the frozen common-46 test artifacts and
existing audited publisher-compatible tables. No training, checkpoint
selection, or hyper-parameter tuning is performed here.

Manuscript figures
------------------
1. Relative improvement over eight baselines (error reduction + NSE gain).
2. Day-1--Day-7 publisher-compatible MAE with deterministic bootstrap 95% CI.
3. ECDF of per-origin publisher-compatible MAE for graph/ablation models.
4. DMA-level MAE reduction versus DCRNN and STGCN.
5. Representative 168 h aggregate-demand trajectory chosen by a pre-specified
   median-error rule (no cherry-picking), with absolute-error panel.

Each figure also writes a CSV/JSON audit artifact under
``paper/tables/manuscript`` so numerical content can be checked independently
of the rendered PNG/PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASKS = ("24h", "168h")
DMAS = tuple("ABCDEFGHIJ")
OVERALL_BASELINES = (
    "GRU (reported)",
    "LSTM (reported)",
    "MSNet (reported)",
    "MSCMNet_WM (reported)",
    "MSCMNet_M (reported)",
    "MSCMNet_W (reported)",
    "DCRNN",
    "STGCN",
)
GRAPH_MODELS = ("STGCN", "DCRNN", "State", "FA-DPR", "Full")
PUBLIC_LABEL = {
    "STGCN": "STGCN",
    "DCRNN": "DCRNN",
    "State": "DCRNN + SAS-Norm",
    "FA-DPR": "DCRNN + FA-DPR",
    "Full": "STaR-GNN",
}
LINE_STYLES = {
    "STGCN": "--",
    "DCRNN": "-.",
    "State": ":",
    "FA-DPR": (0, (5, 2)),
    "Full": "-",
}
MARKERS = {
    "STGCN": "s",
    "DCRNN": "^",
    "State": "D",
    "FA-DPR": "v",
    "Full": "o",
}

MODEL_COLORS = {
    "DCRNN": "#1f77b4",
    "STGCN": "#ff7f0e",
    "State": "#d62728",
    "FA-DPR": "#9467bd",
    "Full": "#2ca02c",
}


def _evaluation_dir(release: Path, model: str, task: str) -> Path:
    prefix = release / "models" if (release / "models").is_dir() else release
    if model == "DCRNN":
        family, name = "star_gnn", "Base"
    elif model in ("State", "FA-DPR", "Full"):
        family, name = "star_gnn", model
    elif model == "STGCN":
        family, name = "baselines", "stgcn"
    else:
        raise ValueError(f"Unsupported frozen model: {model}")
    path = prefix / family / name / task / "seed_0" / "evaluation"
    if not path.is_dir():
        raise FileNotFoundError(f"Missing frozen evaluation directory: {path}")
    return path


def _load_common_predictions(
    release: Path,
    model: str,
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_path = _evaluation_dir(release, "DCRNN", task) / "predictions.npz"
    with np.load(baseline_path, allow_pickle=False) as baseline:
        common = baseline["common_46_indices"].astype(np.int64)
    if common.shape != (46,):
        raise ValueError(f"{task}: expected 46 common indices, got {common.shape}")

    path = _evaluation_dir(release, model, task) / "predictions.npz"
    with np.load(path, allow_pickle=False) as payload:
        if "target" in payload.files and "prediction" in payload.files:
            truth = payload["target"]
            prediction = payload["prediction"]
        elif "y_true" in payload.files and "y_pred" in payload.files:
            truth = payload["y_true"]
            prediction = payload["y_pred"]
        else:
            raise ValueError(f"Unknown predictions.npz schema: {path}")
        if "common_46_indices" not in payload.files:
            raise ValueError(f"Missing common_46_indices: {path}")
        indices = payload["common_46_indices"].astype(np.int64)

    if not np.array_equal(indices, common):
        raise ValueError(f"common-46 indices drift for {model}/{task}")
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError(
            f"Unexpected prediction shape for {model}/{task}: "
            f"{truth.shape} vs {prediction.shape}"
        )
    selected_truth = truth[common].astype(np.float64)
    selected_prediction = prediction[common].astype(np.float64)
    return selected_truth, selected_prediction, common


def _save(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _ensure_same_truth(reference: np.ndarray | None, current: np.ndarray, label: str) -> np.ndarray:
    if reference is None:
        return current
    if not np.allclose(reference, current, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"Ground-truth drift across models: {label}")
    return reference


def _publisher_mae_per_origin(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Per-origin publisher-compatible MAE: sum of DMA-level MAEs."""
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError("Expected [origin, horizon, dma] arrays")
    return np.mean(np.abs(prediction - truth), axis=1).sum(axis=1)


def _publisher_mae_per_origin_day(
    truth: np.ndarray,
    prediction: np.ndarray,
    day: int,
) -> np.ndarray:
    if not 1 <= day <= 7:
        raise ValueError(day)
    window = slice((day - 1) * 24, day * 24)
    return np.mean(np.abs(prediction[:, window, :] - truth[:, window, :]), axis=1).sum(axis=1)


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size != 46:
        raise ValueError(f"Expected 46 origin values, got {values.shape}")
    samples = rng.choice(values, size=(iterations, values.size), replace=True)
    means = samples.mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _load_overall(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"task", "model", "MAE", "MAPE", "RMSE", "NSE"}
    if required - set(frame.columns):
        raise ValueError(f"Overall table missing columns: {required - set(frame.columns)}")
    for task in TASKS:
        selected = frame.loc[frame["task"] == task]
        expected = (*OVERALL_BASELINES, "STaR-GNN")
        if tuple(selected["model"].astype(str)) != expected:
            raise ValueError(f"Unexpected overall model order for {task}")
    return frame


def _load_dma_metrics(release: Path, model: str, task: str) -> pd.DataFrame:
    path = _evaluation_dir(release, model, task) / "metrics_common_46.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["entity"].astype(str).isin(DMAS)].copy()
    if tuple(frame["entity"].astype(str)) != DMAS:
        raise ValueError(f"DMA order drift: {model}/{task}")
    return frame


def _figure1_relative_improvement(
    overall: pd.DataFrame,
    figure_dir: Path,
    table_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for baseline in OVERALL_BASELINES:
        row: dict[str, Any] = {
            "baseline": baseline.replace(" (reported)", ""),
            "source": (
                "reported_Que_et_al_2024"
                if baseline.endswith(" (reported)")
                else "common_46_re_evaluated"
            ),
        }
        for task in TASKS:
            task_frame = overall.loc[overall["task"] == task].set_index("model")
            base = task_frame.loc[baseline]
            star = task_frame.loc["STaR-GNN"]
            for metric in ("MAE", "MAPE", "RMSE"):
                row[f"{task}_{metric}_reduction_pct"] = 100.0 * (
                    float(base[metric]) - float(star[metric])
                ) / float(base[metric])
            row[f"{task}_NSE_gain"] = float(star["NSE"]) - float(base["NSE"])
        rows.append(row)
    derived = pd.DataFrame(rows)
    derived.to_csv(
        table_dir / "fig1_relative_improvement.csv",
        index=False,
        float_format="%.9f",
    )

    error_cols = [
        "24h_MAE_reduction_pct",
        "24h_MAPE_reduction_pct",
        "24h_RMSE_reduction_pct",
        "168h_MAE_reduction_pct",
        "168h_MAPE_reduction_pct",
        "168h_RMSE_reduction_pct",
    ]
    error_labels = [
        "24 h\nMAE",
        "24 h\nMAPE",
        "24 h\nRMSE",
        "168 h\nMAE",
        "168 h\nMAPE",
        "168 h\nRMSE",
    ]
    matrix = derived[error_cols].to_numpy(float)

    fig = plt.figure(figsize=(13.5, 6.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.3])
    ax_hm = fig.add_subplot(grid[0, 0])
    ax_nse = fig.add_subplot(grid[0, 1])

    limit = max(5.0, float(np.nanmax(np.abs(matrix))))
    image = ax_hm.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    display_labels = [
        f"{name}†" if source == "reported_Que_et_al_2024" else name
        for name, source in zip(derived["baseline"], derived["source"])
    ]
    ax_hm.set_xticks(np.arange(len(error_labels)), error_labels)
    ax_hm.set_yticks(np.arange(len(derived)), display_labels)
    ax_hm.set_title("(a) Relative reduction in error metrics")
    ax_hm.set_xlabel("Forecast horizon and metric")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax_hm.text(j, i, f"{value:+.1f}%", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax_hm, fraction=0.035, pad=0.02)
    cbar.set_label("Relative reduction (%)")

    y = np.arange(len(derived))
    nse24 = derived["24h_NSE_gain"].to_numpy(float)
    nse168 = derived["168h_NSE_gain"].to_numpy(float)
    ax_nse.axvline(0.0, linewidth=0.8, color="black")
    ax_nse.scatter(nse24, y - 0.12, marker="o", label="24 h")
    ax_nse.scatter(nse168, y + 0.12, marker="s", label="168 h")
    for i in range(len(y)):
        ax_nse.plot(
            [nse24[i], nse168[i]],
            [y[i] - 0.12, y[i] + 0.12],
            linewidth=0.8,
            alpha=0.5,
        )
    ax_nse.set_yticks(y, [""] * len(y))
    ax_nse.invert_yaxis()
    ax_nse.set_xlabel("ΔNSE (STaR-GNN − baseline)")
    ax_nse.set_title("(b) Absolute NSE gain")
    ax_nse.grid(axis="x", alpha=0.25)
    ax_nse.legend(frameon=False)
    fig.suptitle(
        "Relative performance improvement of STaR-GNN over competing models",
        fontsize=14,
    )
    fig.text(
        0.01,
        0.005,
        "† Reported results from Que et al. (2024); DCRNN and STGCN are "
        "re-evaluated on the common-46 protocol.",
        fontsize=8,
        ha="left",
    )
    _save(fig, figure_dir / "manuscript_fig1_relative_improvement")


def _figure2_day1_day7(
    release: Path,
    figure_dir: Path,
    table_dir: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> None:
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    reference_truth: np.ndarray | None = None
    for model in GRAPH_MODELS:
        truth, pred, _ = _load_common_predictions(release, model, "168h")
        if truth.shape != (46, 168, 10):
            raise ValueError(f"Unexpected 168h common shape: {model} {truth.shape}")
        reference_truth = _ensure_same_truth(reference_truth, truth, model)
        for day in range(1, 8):
            origin_values = _publisher_mae_per_origin_day(truth, pred, day)
            mean, lower, upper = _bootstrap_mean_ci(
                origin_values,
                rng,
                bootstrap_iterations,
            )
            rows.append(
                {
                    "model": PUBLIC_LABEL[model],
                    "internal_model": model,
                    "day": day,
                    "mean_MAE": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    derived = pd.DataFrame(rows)
    derived.to_csv(
        table_dir / "fig2_day1_day7_publisher_mae_ci.csv",
        index=False,
        float_format="%.9f",
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.9))
    for model in GRAPH_MODELS:
        selected = derived.loc[derived["internal_model"] == model].sort_values("day")
        x = selected["day"].to_numpy(int)
        y = selected["mean_MAE"].to_numpy(float)
        lo = selected["ci95_lower"].to_numpy(float)
        hi = selected["ci95_upper"].to_numpy(float)
        ax.plot(
            x,
            y,
            label=PUBLIC_LABEL[model],
            marker=MARKERS[model],
            linestyle=LINE_STYLES[model],
            linewidth=2.2 if model == "Full" else 1.5,
            color=MODEL_COLORS[model],
        )
        ax.fill_between(x, lo, hi, alpha=0.10, color=MODEL_COLORS[model])
    ax.set_xticks(range(1, 8), [f"Day {i}" for i in range(1, 8)])
    ax.set_xlabel("Forecast day within the 168 h horizon")
    ax.set_ylabel("Publisher-compatible MAE")
    ax.set_title("Long-horizon error evolution across seven forecast days")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    _save(fig, figure_dir / "manuscript_fig2_day1_day7_publisher_mae")

    metadata = {
        "protocol": "common_46",
        "n_origins": 46,
        "horizon_hours": 168,
        "day_width_hours": 24,
        "mae_convention": "sum_of_DMA_level_MAE_within_each_day",
        "aggregation": "mean_across_46_origins",
        "ci": "nonparametric bootstrap 95% CI of origin-level mean",
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
    }
    (table_dir / "fig2_day1_day7_publisher_mae_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def _figure3_origin_ecdf(
    release: Path,
    figure_dir: Path,
    table_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        truth_ref: np.ndarray | None = None
        for model in GRAPH_MODELS:
            truth, pred, common = _load_common_predictions(release, model, task)
            truth_ref = _ensure_same_truth(truth_ref, truth, f"{model}/{task}")
            values = _publisher_mae_per_origin(truth, pred)
            for rank_index, (origin_index, value) in enumerate(
                zip(common.tolist(), values.tolist())
            ):
                rows.append(
                    {
                        "task": task,
                        "model": PUBLIC_LABEL[model],
                        "internal_model": model,
                        "origin_position": rank_index,
                        "common_index": int(origin_index),
                        "publisher_MAE": float(value),
                    }
                )
    derived = pd.DataFrame(rows)
    derived.to_csv(
        table_dir / "fig3_origin_publisher_mae.csv",
        index=False,
        float_format="%.9f",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)
    for panel_index, (ax, task) in enumerate(zip(axes, TASKS)):
        for model in GRAPH_MODELS:
            values = derived.loc[
                (derived["task"] == task) & (derived["internal_model"] == model),
                "publisher_MAE",
            ].to_numpy(float)
            x, y = _ecdf(values)
            ax.step(
                x,
                y,
                where="post",
                label=PUBLIC_LABEL[model],
                linestyle=LINE_STYLES[model],
                linewidth=2.2 if model == "Full" else 1.5,
            color=MODEL_COLORS[model],
            )
        ax.set_xlabel("Per-origin publisher-compatible MAE")
        ax.set_title(f"({chr(97 + panel_index)}) {task.replace('h', ' h')}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Distribution of forecast errors across the 46 common test origins",
        fontsize=14,
    )
    _save(fig, figure_dir / "manuscript_fig3_origin_ecdf")


def _figure4_dma_improvement(
    release: Path,
    figure_dir: Path,
    table_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        star = _load_dma_metrics(release, "Full", task).set_index("entity")
        for baseline in ("DCRNN", "STGCN"):
            base = _load_dma_metrics(release, baseline, task).set_index("entity")
            for dma in DMAS:
                base_mae = float(base.loc[dma, "MAE"])
                star_mae = float(star.loc[dma, "MAE"])
                rows.append(
                    {
                        "DMA": dma,
                        "task": task,
                        "baseline": baseline,
                        "MAE_reduction_pct": 100.0 * (base_mae - star_mae) / base_mae,
                    }
                )
    derived = pd.DataFrame(rows)
    derived.to_csv(
        table_dir / "fig4_dma_mae_improvement.csv",
        index=False,
        float_format="%.9f",
    )

    column_keys = [
        ("24h", "DCRNN"),
        ("24h", "STGCN"),
        ("168h", "DCRNN"),
        ("168h", "STGCN"),
    ]
    column_labels = [
        "24 h\nvs DCRNN",
        "24 h\nvs STGCN",
        "168 h\nvs DCRNN",
        "168 h\nvs STGCN",
    ]
    matrix = np.zeros((len(DMAS), len(column_keys)), dtype=float)
    for i, dma in enumerate(DMAS):
        for j, (task, baseline) in enumerate(column_keys):
            selected = derived.loc[
                (derived["DMA"] == dma)
                & (derived["task"] == task)
                & (derived["baseline"] == baseline),
                "MAE_reduction_pct",
            ]
            if len(selected) != 1:
                raise ValueError(f"Missing DMA improvement: {dma}/{task}/{baseline}")
            matrix[i, j] = float(selected.iloc[0])

    limit = max(5.0, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(7.8, 7.1))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(column_labels)), column_labels)
    ax.set_yticks(np.arange(len(DMAS)), DMAS)
    ax.set_xlabel("Forecast horizon and baseline")
    ax.set_ylabel("DMA")
    ax.set_title("DMA-level MAE reduction of STaR-GNN")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.1f}%", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("MAE reduction relative to baseline (%)")
    _save(fig, figure_dir / "manuscript_fig4_dma_mae_improvement")


def _figure5_representative_trajectory(
    release: Path,
    figure_dir: Path,
    table_dir: Path,
) -> None:
    truth_star, pred_star, common = _load_common_predictions(release, "Full", "168h")
    star_origin_mae = _publisher_mae_per_origin(truth_star, pred_star)
    median_value = float(np.median(star_origin_mae))
    selected_position = int(np.argmin(np.abs(star_origin_mae - median_value)))
    selected_common_index = int(common[selected_position])

    truth_ref: np.ndarray | None = None
    model_series: dict[str, np.ndarray] = {}
    selected_model_mae: dict[str, float] = {}
    for model in ("STGCN", "DCRNN", "Full"):
        truth, pred, model_common = _load_common_predictions(release, model, "168h")
        if int(model_common[selected_position]) != selected_common_index:
            raise ValueError("Representative origin index drift")
        truth_ref = _ensure_same_truth(truth_ref, truth, model)
        aggregate_pred = pred[selected_position].sum(axis=1)
        model_series[model] = aggregate_pred
        selected_model_mae[PUBLIC_LABEL[model]] = float(
            _publisher_mae_per_origin(
                truth[selected_position : selected_position + 1],
                pred[selected_position : selected_position + 1],
            )[0]
        )
    if truth_ref is None:
        raise RuntimeError("No trajectory truth loaded")
    aggregate_truth = truth_ref[selected_position].sum(axis=1)

    hours = np.arange(1, 169)
    trajectory = pd.DataFrame(
        {
            "forecast_hour": hours,
            "observed_total": aggregate_truth,
            "STGCN": model_series["STGCN"],
            "DCRNN": model_series["DCRNN"],
            "STaR-GNN": model_series["Full"],
        }
    )
    trajectory.to_csv(
        table_dir / "fig5_representative_168h_trajectory.csv",
        index=False,
        float_format="%.9f",
    )

    metadata = {
        "selection_rule": (
            "STaR-GNN origin whose publisher-compatible 168h MAE is closest "
            "to the median among the 46 common test origins"
        ),
        "selected_origin_position_zero_based": selected_position,
        "selected_common_index": selected_common_index,
        "median_star_publisher_mae": median_value,
        "selected_star_publisher_mae": float(star_origin_mae[selected_position]),
        "selected_origin_model_publisher_mae": selected_model_mae,
        "note": (
            "Selection is deterministic and based only on STaR-GNN median-error "
            "proximity, not visual appearance."
        ),
    }
    (table_dir / "fig5_representative_168h_selection.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    top, bottom = axes
    top.plot(hours, aggregate_truth, label="Observed", linewidth=2.4, color="black")
    for model in ("STGCN", "DCRNN", "Full"):
        top.plot(
            hours,
            model_series[model],
            label=PUBLIC_LABEL[model],
            linestyle=LINE_STYLES[model],
            linewidth=2.0 if model == "Full" else 1.4,
            color=MODEL_COLORS[model],
        )
        bottom.plot(
            hours,
            np.abs(model_series[model] - aggregate_truth),
            label=PUBLIC_LABEL[model],
            linestyle=LINE_STYLES[model],
            linewidth=2.0 if model == "Full" else 1.3,
            color=MODEL_COLORS[model],
        )
    for boundary in range(24, 168, 24):
        top.axvline(boundary + 0.5, linewidth=0.7, alpha=0.25, color="black")
        bottom.axvline(boundary + 0.5, linewidth=0.7, alpha=0.25, color="black")
    top.set_ylabel("Aggregate demand")
    bottom.set_ylabel("Absolute error")
    bottom.set_xlabel("Forecast hour")
    top.set_title("Representative 168 h forecast trajectory (median-error selection rule)")
    top.legend(ncol=4, frameon=False)
    top.grid(alpha=0.20)
    bottom.grid(alpha=0.20)
    day_ticks = np.arange(12, 169, 24)
    bottom.set_xticks(day_ticks, [f"Day {i}" for i in range(1, 8)])
    _save(fig, figure_dir / "manuscript_fig5_representative_168h_trajectory")


def _write_readme(table_dir: Path) -> None:
    text = (
        "# Manuscript result-figure audit artifacts\n\n"
        "These files are generated by `scripts/reproduce/build_manuscript_results_figures.py`.\n\n"
        "- `fig1_relative_improvement.csv`: relative reductions in MAE/MAPE/RMSE and absolute NSE gains versus eight baselines.\n"
        "- `fig2_day1_day7_publisher_mae_ci.csv`: day-wise publisher-compatible MAE and bootstrap 95% CI for the 168 h task.\n"
        "- `fig2_day1_day7_publisher_mae_metadata.json`: deterministic bootstrap settings and metric definition.\n"
        "- `fig3_origin_publisher_mae.csv`: per-origin publisher-compatible MAE for 46 common test origins.\n"
        "- `fig4_dma_mae_improvement.csv`: DMA-level STaR-GNN MAE reduction versus DCRNN/STGCN.\n"
        "- `fig5_representative_168h_trajectory.csv`: observed/predicted aggregate-demand trajectory for the pre-specified median-error origin.\n"
        "- `fig5_representative_168h_selection.json`: exact selection rule, origin index, and model errors.\n"
    )
    (table_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("results/paper/frozen_v1"),
        help="Frozen evaluation root containing predictions.npz and DMA metrics.",
    )
    parser.add_argument(
        "--overall-table",
        type=Path,
        default=Path("paper/tables/literature/table_literature_comparison_common46.csv"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("paper/figures"),
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("paper/tables/manuscript"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    args = parser.parse_args()

    release = args.release.resolve()
    overall_path = args.overall_table.resolve()
    figure_dir = args.figure_output.resolve()
    table_dir = args.table_output.resolve()
    if not release.is_dir():
        raise FileNotFoundError(release)
    if not overall_path.is_file():
        raise FileNotFoundError(overall_path)
    if args.bootstrap_iterations < 1000:
        raise ValueError("Use at least 1000 bootstrap iterations for manuscript CI")
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    overall = _load_overall(overall_path)
    _figure1_relative_improvement(overall, figure_dir, table_dir)
    _figure2_day1_day7(
        release,
        figure_dir,
        table_dir,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )
    _figure3_origin_ecdf(release, figure_dir, table_dir)
    _figure4_dma_improvement(release, figure_dir, table_dir)
    _figure5_representative_trajectory(release, figure_dir, table_dir)
    _write_readme(table_dir)

    print(f"Manuscript figures: {figure_dir}")
    print(f"Manuscript audit tables: {table_dir}")
    print("Manuscript scientific-figure audit: PASS")


if __name__ == "__main__":
    main()
