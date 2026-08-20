#!/usr/bin/env python
"""Render the final Journal of Hydrology manuscript Figure 2 and Figure 3.

Figure 2 is now a pure factorial-ablation figure:

(a) Day-wise publisher-compatible MAE reduction relative to DCRNN for
    SAS-Norm, FA-DPR, and the full STaR-GNN;
(b) Day-wise MAE change relative to Day 1 for all four factorial variants.

STGCN is intentionally excluded from Figure 2 because it is an independent
baseline, not an ablation variant.  Figure 3 keeps DCRNN/STGCN/STaR-GNN and
therefore answers a different question: robustness across common test origins.

The script also quantifies the small 168 h Full-vs-SAS publisher-MAE
difference using a moving-block bootstrap over ordered test origins.  A block
length of seven origins is used because 168 h forecasts started one day apart
strongly overlap; this is more defensible than treating the 46 origins as
independent observations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ABLATION_MODELS = (
    "DCRNN",
    "DCRNN + SAS-Norm",
    "DCRNN + FA-DPR",
    "STaR-GNN",
)
MODULE_MODELS = (
    "DCRNN + SAS-Norm",
    "DCRNN + FA-DPR",
    "STaR-GNN",
)
ECDF_MODELS = ("DCRNN", "STGCN", "STaR-GNN")
LINESTYLES = {
    "DCRNN": "--",
    "STGCN": ":",
    "DCRNN + SAS-Norm": "-.",
    "DCRNN + FA-DPR": (0, (5, 2)),
    "STaR-GNN": "-",
}
MARKERS = {
    "DCRNN": "^",
    "STGCN": "s",
    "DCRNN + SAS-Norm": "D",
    "DCRNN + FA-DPR": "v",
    "STaR-GNN": "o",
}

MODEL_COLORS = {
    "DCRNN": "#1f77b4",
    "STGCN": "#ff7f0e",
    "DCRNN + SAS-Norm": "#d62728",
    "DCRNN + FA-DPR": "#9467bd",
    "STaR-GNN": "#2ca02c",
}


def _save(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_daywise(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"model", "day", "mean_MAE", "ci95_lower", "ci95_upper"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Day-wise audit table missing columns: {sorted(missing)}")
    for model in ABLATION_MODELS:
        selected = frame.loc[frame["model"] == model].sort_values("day")
        if selected["day"].tolist() != list(range(1, 8)):
            raise ValueError(f"Unexpected Day 1--7 rows for {model}")
        if not np.isfinite(selected[["mean_MAE", "ci95_lower", "ci95_upper"]].to_numpy(float)).all():
            raise ValueError(f"Non-finite day-wise values for {model}")
    return frame


def _daywise_reduction_vs_dcrnn(frame: pd.DataFrame) -> pd.DataFrame:
    base = (
        frame.loc[frame["model"] == "DCRNN", ["day", "mean_MAE"]]
        .sort_values("day")
        .set_index("day")["mean_MAE"]
    )
    rows: list[dict[str, float | int | str]] = []
    for model in MODULE_MODELS:
        selected = frame.loc[frame["model"] == model].sort_values("day")
        for _, row in selected.iterrows():
            day = int(row["day"])
            model_mae = float(row["mean_MAE"])
            dcrnn_mae = float(base.loc[day])
            rows.append(
                {
                    "model": model,
                    "day": day,
                    "model_MAE": model_mae,
                    "dcrnn_MAE": dcrnn_mae,
                    "MAE_reduction_vs_DCRNN_pct": 100.0 * (dcrnn_mae - model_mae) / dcrnn_mae,
                }
            )
    return pd.DataFrame(rows)


def _daywise_degradation(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model in ABLATION_MODELS:
        selected = frame.loc[frame["model"] == model].sort_values("day")
        day1 = float(selected.iloc[0]["mean_MAE"])
        for _, row in selected.iterrows():
            mean_mae = float(row["mean_MAE"])
            rows.append(
                {
                    "model": model,
                    "day": int(row["day"]),
                    "mean_MAE": mean_mae,
                    "relative_to_day1_pct": 100.0 * (mean_mae - day1) / day1,
                }
            )
    return pd.DataFrame(rows)


def _plot_ablation(
    reduction: pd.DataFrame,
    degradation: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    ax_gain, ax_deg = axes

    for model in MODULE_MODELS:
        selected = reduction.loc[reduction["model"] == model].sort_values("day")
        ax_gain.plot(
            selected["day"].to_numpy(int),
            selected["MAE_reduction_vs_DCRNN_pct"].to_numpy(float),
            label=model,
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            linewidth=2.2 if model == "STaR-GNN" else 1.6,
            color=MODEL_COLORS[model],
        )
    ax_gain.axhline(0.0, linewidth=0.8, color="black", alpha=0.55)
    ax_gain.set_xticks(range(1, 8), [f"Day {i}" for i in range(1, 8)])
    ax_gain.set_xlabel("Forecast day within the 168 h horizon")
    ax_gain.set_ylabel("MAE reduction relative to DCRNN (%)")
    ax_gain.set_title("(a) Day-wise contribution of proposed modules")
    ax_gain.grid(axis="y", alpha=0.18)
    ax_gain.legend(frameon=False, fontsize=8)

    for model in ABLATION_MODELS:
        selected = degradation.loc[degradation["model"] == model].sort_values("day")
        ax_deg.plot(
            selected["day"].to_numpy(int),
            selected["relative_to_day1_pct"].to_numpy(float),
            label=model,
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            linewidth=2.2 if model == "STaR-GNN" else 1.5,
            color=MODEL_COLORS[model],
        )
    ax_deg.axhline(0.0, linewidth=0.8, color="black", alpha=0.55)
    ax_deg.set_xticks(range(1, 8), [f"Day {i}" for i in range(1, 8)])
    ax_deg.set_xlabel("Forecast day within the 168 h horizon")
    ax_deg.set_ylabel("MAE change relative to Day 1 (%)")
    ax_deg.set_title("(b) Long-horizon error degradation")
    ax_deg.grid(axis="y", alpha=0.18)
    ax_deg.legend(frameon=False, fontsize=8)

    _save(fig, figure_dir / "manuscript_fig2_day1_day7_publisher_mae")


def _load_origin_errors(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"task", "model", "common_index", "publisher_MAE"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Origin-error table missing columns: {sorted(missing)}")
    for task in ("24h", "168h"):
        for model in set(ECDF_MODELS) | set(ABLATION_MODELS):
            selected = frame.loc[(frame["task"] == task) & (frame["model"] == model)]
            if len(selected) != 46 or selected["common_index"].nunique() != 46:
                raise ValueError(f"Expected 46 unique origins for {task}/{model}")
    return frame


def _paired_series(frame: pd.DataFrame, task: str, left: str, right: str) -> tuple[pd.Series, pd.Series]:
    task_frame = frame.loc[frame["task"] == task]
    left_series = (
        task_frame.loc[task_frame["model"] == left]
        .set_index("common_index")["publisher_MAE"]
        .sort_index()
    )
    right_series = (
        task_frame.loc[task_frame["model"] == right]
        .set_index("common_index")["publisher_MAE"]
        .sort_index()
    )
    if not left_series.index.equals(right_series.index):
        raise ValueError(f"Paired common-index drift: {task}/{left}/{right}")
    return left_series, right_series


def _moving_block_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    block_length: int,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size != 46:
        raise ValueError(f"Expected 46 ordered paired differences, got {values.shape}")
    if not 1 <= block_length <= values.size:
        raise ValueError("Invalid block length")
    rng = np.random.default_rng(seed)
    blocks_per_draw = math.ceil(values.size / block_length)
    starts = rng.integers(0, values.size, size=(iterations, blocks_per_draw))
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[..., None] + offsets) % values.size
    indices = indices.reshape(iterations, -1)[:, : values.size]
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _full_vs_sas_audit(
    origins: pd.DataFrame,
    *,
    block_length: int,
    iterations: int,
    seed: int,
) -> dict[str, float | int | str | list[float]]:
    full, sas = _paired_series(origins, "168h", "STaR-GNN", "DCRNN + SAS-Norm")
    diff = full.to_numpy(float) - sas.to_numpy(float)
    mean_diff, ci_low, ci_high = _moving_block_bootstrap_mean_ci(
        diff,
        block_length=block_length,
        iterations=iterations,
        seed=seed,
    )
    full_mean = float(full.mean())
    sas_mean = float(sas.mean())
    return {
        "comparison": "STaR-GNN minus DCRNN + SAS-Norm",
        "task": "168h",
        "metric": "publisher-compatible sum-of-DMA MAE",
        "n_origins": int(len(full)),
        "full_mean_MAE": full_mean,
        "sas_mean_MAE": sas_mean,
        "mean_difference_full_minus_sas": mean_diff,
        "relative_difference_percent_of_sas": 100.0 * mean_diff / sas_mean,
        "full_lower_origins": int((full < sas).sum()),
        "sas_lower_origins": int((sas < full).sum()),
        "moving_block_bootstrap": {
            "block_length_origins": int(block_length),
            "iterations": int(iterations),
            "seed": int(seed),
            "ci95_mean_difference": [ci_low, ci_high],
            "contains_zero": bool(ci_low <= 0.0 <= ci_high),
            "rationale": (
                "168 h forecasts are started 24 h apart and overlap strongly; "
                "ordered moving blocks preserve local dependence better than an IID bootstrap."
            ),
        },
    }


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def _paired_win_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for task in ("24h", "168h"):
        star, _ = _paired_series(frame, task, "STaR-GNN", "DCRNN")
        for baseline in ("DCRNN", "STGCN"):
            star, base = _paired_series(frame, task, "STaR-GNN", baseline)
            wins = int((star < base).sum())
            losses = int((star > base).sum())
            ties = int(np.isclose(star.to_numpy(float), base.to_numpy(float)).sum())
            rows.append(
                {
                    "task": task,
                    "baseline": baseline,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "n_origins": len(star),
                    "win_rate_pct": 100.0 * wins / len(star),
                    "median_star_MAE": float(star.median()),
                    "median_baseline_MAE": float(base.median()),
                }
            )
    return pd.DataFrame(rows)


def _plot_ecdf(frame: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    for ax, task, panel in zip(axes, ("24h", "168h"), ("a", "b")):
        for model in ECDF_MODELS:
            values = frame.loc[
                (frame["task"] == task) & (frame["model"] == model),
                "publisher_MAE",
            ].to_numpy(float)
            x, y = _ecdf(values)
            ax.step(
                x,
                y,
                where="post",
                label=model,
                linestyle=LINESTYLES[model],
                linewidth=2.2 if model == "STaR-GNN" else 1.6,
            color=MODEL_COLORS[model],
            )
        ax.set_xlabel("Per-origin publisher-compatible MAE")
        ax.set_title(f"({panel}) {task.replace('h', ' h')}")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[1].legend(frameon=False, loc="lower right")
    _save(fig, figure_dir / "manuscript_fig3_origin_ecdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", type=Path, default=Path("paper/tables/manuscript"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--block-bootstrap-iterations", type=int, default=50000)
    parser.add_argument("--block-bootstrap-length", type=int, default=7)
    parser.add_argument("--block-bootstrap-seed", type=int, default=20260820)
    args = parser.parse_args()

    table_dir = args.table_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    daywise_path = table_dir / "fig2_day1_day7_publisher_mae_ci.csv"
    origin_path = table_dir / "fig3_origin_publisher_mae.csv"
    for path in (daywise_path, origin_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required first-stage audit file not found: {path}. "
                "Run build_manuscript_results_figures.py first."
            )
    figure_dir.mkdir(parents=True, exist_ok=True)

    daywise = _load_daywise(daywise_path)
    reduction = _daywise_reduction_vs_dcrnn(daywise)
    degradation = _daywise_degradation(daywise)
    reduction.to_csv(
        table_dir / "fig2_ablation_daywise_reduction_vs_dcrnn.csv",
        index=False,
        float_format="%.9f",
    )
    degradation.to_csv(
        table_dir / "fig2_day1_day7_degradation.csv",
        index=False,
        float_format="%.9f",
    )
    _plot_ablation(reduction, degradation, figure_dir)

    origins = _load_origin_errors(origin_path)
    full_sas = _full_vs_sas_audit(
        origins,
        block_length=args.block_bootstrap_length,
        iterations=args.block_bootstrap_iterations,
        seed=args.block_bootstrap_seed,
    )
    (table_dir / "fig2_full_vs_sas_block_bootstrap.json").write_text(
        json.dumps(full_sas, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    win_rates = _paired_win_rates(origins)
    win_rates.to_csv(
        table_dir / "fig3_origin_win_rates.csv",
        index=False,
        float_format="%.9f",
    )
    _plot_ecdf(origins, figure_dir)

    day7 = degradation.loc[degradation["day"] == 7].set_index("model")
    audit = {
        "figure2": {
            "models": list(ABLATION_MODELS),
            "stgcn_in_ablation": False,
            "day7_relative_to_day1_pct": {
                model: float(day7.loc[model, "relative_to_day1_pct"])
                for model in ABLATION_MODELS
            },
            "full_vs_sas_168h": full_sas,
            "interpretation_guardrail": (
                "Treat the 168 h publisher-MAE difference between Full and SAS-Norm-only "
                "as a small point-estimate difference, not as evidence that one dominates "
                "the other. Emphasize Full's lower MAPE/RMSE, higher NSE, lower aggregate-demand "
                "MAE, and smaller Day-1-to-Day-7 degradation."
            ),
        },
        "figure3": {
            "models": list(ECDF_MODELS),
            "paired_win_rates": win_rates.to_dict(orient="records"),
            "interpretation_guardrail": (
                "Figure 3 is a baseline robustness analysis, not an ablation figure."
            ),
        },
    }
    (table_dir / "manuscript_empirical_figure_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Refined manuscript Figure 2 and Figure 3: PASS")
    print("Figure 2 factorial-model audit: PASS (4 models, no STGCN)")
    print("Day 7 MAE change relative to Day 1:")
    for model in ABLATION_MODELS:
        value = float(day7.loc[model, "relative_to_day1_pct"])
        print(f"  {model}: {value:+.2f}%")
    ci = full_sas["moving_block_bootstrap"]["ci95_mean_difference"]
    print(
        "168 h Full - SAS publisher MAE: "
        f"{float(full_sas['mean_difference_full_minus_sas']):+.6f}; "
        f"moving-block 95% CI [{float(ci[0]):+.6f}, {float(ci[1]):+.6f}]"
    )
    print("Paired STaR-GNN win rates across common origins:")
    for _, row in win_rates.iterrows():
        print(
            f"  {row['task']} vs {row['baseline']}: "
            f"{int(row['wins'])}/{int(row['n_origins'])} "
            f"({float(row['win_rate_pct']):.1f}%)"
        )


if __name__ == "__main__":
    main()
