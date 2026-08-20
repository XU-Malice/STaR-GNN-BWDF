#!/usr/bin/env python
"""Refine manuscript result figures after empirical audit.

This script is intentionally a second-stage renderer. It reads the audit CSVs
created by ``build_manuscript_results_figures.py`` and applies the final
Journal of Hydrology / spatio-temporal forecasting presentation decisions:

- Figure 2 separates *absolute long-horizon accuracy* from *relative error
  degradation* so the baseline comparison and ablation mechanism are not
  visually conflated.
- Figure 3 restricts the ECDF to the two primary graph baselines and STaR-GNN;
  ablation variants remain in Table 2 and Figure 2(b).

No model predictions are changed and no new metric convention is introduced.
The script also writes compact audit files used by the manuscript text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DAYWISE_BASELINE_MODELS = ("DCRNN", "STGCN", "STaR-GNN")
DAYWISE_ABLATION_MODELS = (
    "DCRNN",
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


def _save(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _load_daywise(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "model",
        "day",
        "mean_MAE",
        "ci95_lower",
        "ci95_upper",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Day-wise audit table missing columns: {sorted(missing)}")
    for model in set(DAYWISE_BASELINE_MODELS) | set(DAYWISE_ABLATION_MODELS):
        selected = frame.loc[frame["model"] == model].sort_values("day")
        if selected["day"].tolist() != list(range(1, 8)):
            raise ValueError(f"Unexpected Day 1--7 rows for {model}")
        values = selected[["mean_MAE", "ci95_lower", "ci95_upper"]].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite day-wise values for {model}")
    return frame


def _daywise_degradation(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model in DAYWISE_ABLATION_MODELS:
        selected = frame.loc[frame["model"] == model].sort_values("day")
        day1 = float(selected.iloc[0]["mean_MAE"])
        for _, row in selected.iterrows():
            mean = float(row["mean_MAE"])
            rows.append(
                {
                    "model": model,
                    "day": int(row["day"]),
                    "mean_MAE": mean,
                    "relative_to_day1_pct": 100.0 * (mean - day1) / day1,
                }
            )
    return pd.DataFrame(rows)


def _plot_daywise(
    frame: pd.DataFrame,
    degradation: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5))
    ax_abs, ax_rel = axes

    for model in DAYWISE_BASELINE_MODELS:
        selected = frame.loc[frame["model"] == model].sort_values("day")
        x = selected["day"].to_numpy(int)
        y = selected["mean_MAE"].to_numpy(float)
        lo = selected["ci95_lower"].to_numpy(float)
        hi = selected["ci95_upper"].to_numpy(float)
        ax_abs.plot(
            x,
            y,
            label=model,
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            linewidth=2.2 if model == "STaR-GNN" else 1.6,
        )
        ax_abs.fill_between(x, lo, hi, alpha=0.10)

    ax_abs.set_xticks(range(1, 8), [f"Day {i}" for i in range(1, 8)])
    ax_abs.set_xlabel("Forecast day within the 168 h horizon")
    ax_abs.set_ylabel("Publisher-compatible MAE")
    ax_abs.set_title("(a) Absolute long-horizon forecasting error")
    ax_abs.grid(alpha=0.25)
    ax_abs.legend(frameon=False)

    for model in DAYWISE_ABLATION_MODELS:
        selected = degradation.loc[degradation["model"] == model].sort_values("day")
        ax_rel.plot(
            selected["day"].to_numpy(int),
            selected["relative_to_day1_pct"].to_numpy(float),
            label=model,
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            linewidth=2.2 if model == "STaR-GNN" else 1.5,
        )

    ax_rel.axhline(0.0, linewidth=0.8, color="black", alpha=0.55)
    ax_rel.set_xticks(range(1, 8), [f"Day {i}" for i in range(1, 8)])
    ax_rel.set_xlabel("Forecast day within the 168 h horizon")
    ax_rel.set_ylabel("MAE change relative to Day 1 (%)")
    ax_rel.set_title("(b) Relative error degradation of ablation variants")
    ax_rel.grid(alpha=0.25)
    ax_rel.legend(frameon=False, fontsize=8)

    fig.suptitle("Long-horizon forecasting behavior over the 168 h horizon", fontsize=14)
    _save(fig, figure_dir / "manuscript_fig2_day1_day7_publisher_mae")


def _load_origin_errors(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"task", "model", "common_index", "publisher_MAE"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Origin-error audit table missing columns: {sorted(missing)}")
    for task in ("24h", "168h"):
        for model in ECDF_MODELS:
            selected = frame.loc[(frame["task"] == task) & (frame["model"] == model)]
            if len(selected) != 46:
                raise ValueError(f"Expected 46 origins for {task}/{model}, got {len(selected)}")
            if selected["common_index"].nunique() != 46:
                raise ValueError(f"Duplicate common indices for {task}/{model}")
    return frame


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def _paired_win_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for task in ("24h", "168h"):
        task_frame = frame.loc[frame["task"] == task]
        star = (
            task_frame.loc[task_frame["model"] == "STaR-GNN"]
            .set_index("common_index")["publisher_MAE"]
            .sort_index()
        )
        for baseline in ("DCRNN", "STGCN"):
            base = (
                task_frame.loc[task_frame["model"] == baseline]
                .set_index("common_index")["publisher_MAE"]
                .sort_index()
            )
            if not star.index.equals(base.index):
                raise ValueError(f"Paired common-index drift for {task}/{baseline}")
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
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
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
            )
        ax.set_xlabel("Per-origin publisher-compatible MAE")
        ax.set_title(f"({panel}) {task.replace('h', ' h')}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Forecast-error distributions across the 46 common test origins", fontsize=14)
    _save(fig, figure_dir / "manuscript_fig3_origin_ecdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("paper/tables/manuscript"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("paper/figures"),
    )
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
    degradation = _daywise_degradation(daywise)
    degradation.to_csv(
        table_dir / "fig2_day1_day7_degradation.csv",
        index=False,
        float_format="%.9f",
    )
    _plot_daywise(daywise, degradation, figure_dir)

    origins = _load_origin_errors(origin_path)
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
            "day7_relative_to_day1_pct": {
                model: float(day7.loc[model, "relative_to_day1_pct"])
                for model in DAYWISE_ABLATION_MODELS
            },
            "interpretation_guardrail": (
                "SAS-Norm is the primary contributor to low long-horizon MAE; "
                "STaR-GNN should not be claimed to dominate SAS-Norm on 168 h MAE."
            ),
        },
        "figure3": {
            "paired_win_rates": win_rates.to_dict(orient="records"),
            "interpretation_guardrail": (
                "ECDF is restricted to DCRNN, STGCN, and STaR-GNN so sample-level "
                "robustness is not conflated with ablation analysis."
            ),
        },
    }
    (table_dir / "manuscript_empirical_figure_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Refined manuscript Figure 2 and Figure 3: PASS")
    print("Day 7 MAE change relative to Day 1:")
    for model in DAYWISE_ABLATION_MODELS:
        value = float(day7.loc[model, "relative_to_day1_pct"])
        print(f"  {model}: {value:+.2f}%")
    print("Paired STaR-GNN win rates across common origins:")
    for _, row in win_rates.iterrows():
        print(
            f"  {row['task']} vs {row['baseline']}: "
            f"{int(row['wins'])}/{int(row['n_origins'])} "
            f"({float(row['win_rate_pct']):.1f}%)"
        )


if __name__ == "__main__":
    main()
