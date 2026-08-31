#!/usr/bin/env python
"""Render the data-quality and error-distribution supplementary figures.

The figures complement, rather than duplicate, the seven result figures:

* Fig. S1 documents representative missing-value and outlier treatment;
* Fig. S2 summarizes the observed weekly demand structure that motivates
  day-state and intra-day pattern separation;
* Fig. S3 reports the empirical distribution of per-origin MAE under the
  common-46 protocol.

No model is selected or trained here.  Fig. S1 loads the public BWDF source
through the repository-pinned ``wf4bwdf`` package and compares it with the
leakage-safe preprocessing output.  Only metadata, not the raw series excerpt,
is written to the audit directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manuscript_plot_style import (
    BASELINE_LIGHT,
    HERO_BLUE,
    MODEL_COLORS,
    MODEL_LINESTYLES,
    OBSERVED_BLACK,
    add_panel_label,
    apply_publication_style,
    save_publication_figure,
    style_axis,
)


DMAS = tuple("ABCDEFGHIJ")
MODELS = ("DCRNN", "STGCN", "STaR-GNN")
OUTLIER_ORANGE = "#C95A43"
WINDOW_HOURS = 168


def _rename_dma_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if len(result.columns) != len(DMAS):
        raise ValueError(f"Expected ten DMA columns, got {len(result.columns)}")
    result.columns = [f"DMA {dma}" for dma in DMAS]
    return result


def _load_raw_demand(wf4bwdf_repo: Path | None) -> pd.DataFrame:
    if wf4bwdf_repo is not None:
        source = (wf4bwdf_repo / "src").resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        sys.path.insert(0, str(source))
    import wf4bwdf  # type: ignore

    raw = wf4bwdf.load_complete_dataset()["dma-inflows"]
    raw = raw.loc["2021-01-01":"2023-03-05"]
    return _rename_dma_columns(raw)


def _best_window(
    indicator: pd.Series,
    *,
    observed: pd.Series | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    counts = indicator.astype(int).rolling(
        WINDOW_HOURS, min_periods=WINDOW_HOURS
    ).sum()
    if observed is not None:
        observed_counts = observed.astype(int).rolling(
            WINDOW_HOURS, min_periods=WINDOW_HOURS
        ).sum()
        # Preserve enough measured values to show the original temporal shape.
        counts = counts.where(observed_counts >= WINDOW_HOURS // 2)
    if counts.dropna().empty:
        raise ValueError("No eligible representative window")
    end = counts.idxmax()
    start = end - pd.Timedelta(hours=WINDOW_HOURS - 1)
    return start, end, int(counts.loc[end])


def _format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=25)


def _render_cleaning(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    outlier_mask: pd.DataFrame,
    missing_profile: pd.DataFrame,
    outlier_profile: pd.DataFrame,
    output: Path,
    audit: Path,
) -> dict[str, object]:
    demand_missing = missing_profile.loc[
        missing_profile["table"] == "dma-inflows"
    ].copy()
    missing_numeric = demand_missing.loc[
        demand_missing["missing_ratio_before"].idxmax(), "column"
    ]
    missing_dma = f"DMA {DMAS[int(str(missing_numeric).split()[-1]) - 1]}"

    outlier_numeric = outlier_profile.loc[
        outlier_profile["outlier_count"].idxmax(), "dma_column"
    ]
    outlier_dma = f"DMA {DMAS[int(str(outlier_numeric).split()[-1]) - 1]}"

    missing_start, missing_end, missing_count = _best_window(
        raw[missing_dma].isna(), observed=raw[missing_dma].notna()
    )
    outlier_start, outlier_end, outlier_count = _best_window(
        outlier_mask[outlier_dma]
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.7), sharex=False)
    panels = (
        (axes[0, 0], missing_dma, missing_start, missing_end, "raw_missing"),
        (axes[0, 1], missing_dma, missing_start, missing_end, "clean_missing"),
        (axes[1, 0], outlier_dma, outlier_start, outlier_end, "raw_outlier"),
        (axes[1, 1], outlier_dma, outlier_start, outlier_end, "clean_outlier"),
    )

    row_limits: dict[str, tuple[float, float]] = {}
    for dma, start, end in (
        (missing_dma, missing_start, missing_end),
        (outlier_dma, outlier_start, outlier_end),
    ):
        values = pd.concat([raw.loc[start:end, dma], clean.loc[start:end, dma]])
        lo, hi = float(values.min()), float(values.max())
        pad = max((hi - lo) * 0.08, 0.2)
        row_limits[dma] = (lo - pad, hi + pad)

    for label, (ax, dma, start, end, kind) in zip("abcd", panels):
        raw_slice = raw.loc[start:end, dma]
        clean_slice = clean.loc[start:end, dma]
        if kind.startswith("raw"):
            ax.plot(
                raw_slice.index,
                raw_slice,
                color=OBSERVED_BLACK,
                linewidth=1.25,
                label="Raw demand",
            )
        else:
            ax.plot(
                clean_slice.index,
                clean_slice,
                color=HERO_BLUE,
                linewidth=1.55,
                label="Cleaned demand",
            )

        if kind == "raw_missing":
            missing_times = raw_slice.index[raw_slice.isna()]
            ymin, ymax = row_limits[dma]
            ax.vlines(
                missing_times,
                ymin,
                ymin + 0.035 * (ymax - ymin),
                color=OUTLIER_ORANGE,
                linewidth=0.65,
                alpha=0.85,
                label="Missing observation",
            )
        elif kind == "raw_outlier":
            mask_slice = outlier_mask.loc[start:end, dma].astype(bool)
            ax.scatter(
                raw_slice.index[mask_slice],
                raw_slice.loc[mask_slice],
                marker="x",
                s=14,
                linewidths=0.8,
                color=OUTLIER_ORANGE,
                label="IQR-identified outlier",
                zorder=4,
            )

        ax.set_ylim(*row_limits[dma])
        ax.set_ylabel("Water demand (L s$^{-1}$)")
        ax.set_title(
            f"{dma}: {'before' if kind.startswith('raw') else 'after'} cleaning",
            pad=7,
            fontweight="bold",
        )
        _format_time_axis(ax)
        style_axis(ax, ygrid=True)
        add_panel_label(ax, label)

    fig.subplots_adjust(top=0.94, bottom=0.13, left=0.09, right=0.98,
                        hspace=0.52, wspace=0.24)
    save_publication_figure(
        fig, output / "supp_figS1_data_cleaning"
    )

    missing_row = demand_missing.loc[
        demand_missing["column"] == missing_numeric
    ].iloc[0]
    outlier_row = outlier_profile.loc[
        outlier_profile["dma_column"] == outlier_numeric
    ].iloc[0]
    metadata: dict[str, object] = {
        "selection_rule": {
            "missing_dma": "highest raw missing-data rate across the ten DMAs",
            "missing_window": (
                "168 h window with the most missing values among windows "
                "retaining at least 84 observed hours"
            ),
            "outlier_dma": "highest IQR-identified outlier count across the ten DMAs",
            "outlier_window": "168 h window with the most identified outliers",
        },
        "missing_example": {
            "dma": missing_dma,
            "missing_rate_full_period": float(
                missing_row["missing_ratio_before"]
            ),
            "window_start": missing_start.isoformat(),
            "window_end": missing_end.isoformat(),
            "window_missing_count": missing_count,
        },
        "outlier_example": {
            "dma": outlier_dma,
            "outlier_count_full_period": int(outlier_row["outlier_count"]),
            "window_start": outlier_start.isoformat(),
            "window_end": outlier_end.isoformat(),
            "window_outlier_count": outlier_count,
        },
        "raw_values_redistributed": False,
    }
    (audit / "supp_figS1_data_cleaning_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def _weekly_profile(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    train = clean.loc[:"2022-12-15 23:00:00"]
    renamed = _rename_dma_columns(train)
    means = renamed.mean()
    lowest = str(means.idxmin())
    highest = str(means.idxmax())
    median_mean = float(means.median())
    middle = str((means - median_mean).abs().sort_values().index[0])
    selected = (lowest, middle, highest)

    hour_of_week = train.index.weekday * 24 + train.index.hour
    rows: list[dict[str, object]] = []
    for dma in selected:
        grouped = renamed[dma].groupby(hour_of_week)
        for hour in range(168):
            values = grouped.get_group(hour)
            rows.append(
                {
                    "dma": dma,
                    "hour_of_week": hour,
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "n_observations": int(values.shape[0]),
                }
            )
    metadata = {
        "selection_rule": (
            "DMAs with the lowest, closest-to-median, and highest mean "
            "training-period demand"
        ),
        "selected_dmas": list(selected),
        "training_end": "2022-12-15T23:00:00",
    }
    return pd.DataFrame(rows), metadata


def _render_weekly_patterns(
    profile: pd.DataFrame,
    output: Path,
) -> None:
    selected = profile["dma"].drop_duplicates().tolist()
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.5), sharex=True)
    for label, ax, dma in zip("abc", axes, selected):
        block = profile.loc[profile["dma"] == dma].sort_values("hour_of_week")
        x = block["hour_of_week"].to_numpy(int)
        median = block["median"].to_numpy(float)
        q25 = block["q25"].to_numpy(float)
        q75 = block["q75"].to_numpy(float)
        ax.axvspan(120, 168, color="#F2F2F2", zorder=0)
        ax.fill_between(x, q25, q75, color=HERO_BLUE, alpha=0.16, linewidth=0)
        ax.plot(x, median, color=HERO_BLUE, linewidth=1.55)
        for boundary in range(24, 168, 24):
            ax.axvline(boundary, color="#D6D6D6", linewidth=0.55, zorder=0)
        ax.set_ylabel("Demand\n(L s$^{-1}$)")
        ax.set_title(dma, loc="left", pad=4, fontweight="bold")
        style_axis(ax, ygrid=True)
        add_panel_label(ax, label)

    axes[-1].set_xlim(0, 167)
    axes[-1].set_xticks(
        np.arange(12, 168, 24),
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )
    axes[-1].set_xlabel("Day of week")
    fig.subplots_adjust(top=0.97, bottom=0.11, left=0.11, right=0.98, hspace=0.38)
    save_publication_figure(
        fig, output / "supp_figS2_weekly_demand_patterns"
    )


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values.astype(float))
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def _render_origin_ecdf(source: pd.DataFrame, output: Path) -> pd.DataFrame:
    required = {"task", "model", "origin_position", "publisher_MAE"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Origin MAE source missing columns: {sorted(missing)}")
    filtered = source.loc[source["model"].isin(MODELS)].copy()
    expected = {(task, model) for task in ("24h", "168h") for model in MODELS}
    observed = set(zip(filtered["task"], filtered["model"]))
    if observed != expected:
        raise ValueError(f"Origin ECDF groups drift: {observed}")
    counts = filtered.groupby(["task", "model"]).size()
    if not (counts == 46).all():
        raise ValueError(f"Origin ECDF is not common-46: {counts.to_dict()}")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for label, ax, task in zip("ab", axes, ("24h", "168h")):
        for model in MODELS:
            values = filtered.loc[
                (filtered["task"] == task) & (filtered["model"] == model),
                "publisher_MAE",
            ].to_numpy(float)
            x, y = _ecdf(values)
            ax.step(
                x, y,
                where="post",
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                linewidth=2.0 if model == "STaR-GNN" else 1.45,
                label=model,
            )
        ax.set_xlabel("Per-origin MAE (L s$^{-1}$)")
        ax.set_title(f"{task[:-1]} h", pad=7, fontweight="bold")
        ax.set_ylim(0.0, 1.02)
        style_axis(ax, ygrid=True)
        add_panel_label(ax, label)
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[1].legend(loc="lower right")
    fig.subplots_adjust(top=0.93, bottom=0.18, left=0.10, right=0.98, wspace=0.18)
    save_publication_figure(
        fig, output / "supp_figS3_origin_mae_ecdf"
    )
    return filtered.sort_values(["task", "model", "origin_position"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/data_build"),
    )
    parser.add_argument(
        "--wf4bwdf-repo",
        type=Path,
        help=(
            "Optional local checkout pinned to the commit in data/README.md; "
            "otherwise import the installed pinned wf4bwdf dependency"
        ),
    )
    parser.add_argument(
        "--origin-mae",
        type=Path,
        default=Path("paper/tables/manuscript/fig3_origin_publisher_mae.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/submission"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("paper/tables/manuscript/submission"),
    )
    args = parser.parse_args()

    apply_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_raw_demand(args.wf4bwdf_repo)
    clean = pd.read_parquet(args.data_dir / "demand_hourly.parquet")
    interpolated = pd.read_parquet(
        args.data_dir / "demand_interpolated_before_outliers.parquet"
    )
    outlier_mask = pd.read_parquet(
        args.data_dir / "demand_outlier_mask.parquet"
    )
    clean = _rename_dma_columns(clean)
    interpolated = _rename_dma_columns(interpolated)
    outlier_mask = _rename_dma_columns(outlier_mask).astype(bool)
    if not clean.index.equals(raw.index):
        raise ValueError("Raw and cleaned demand indices do not match")
    if not interpolated.index.equals(clean.index):
        raise ValueError("Interpolated and cleaned demand indices do not match")

    missing_profile = pd.read_csv(args.data_dir / "missing_profile.csv")
    outlier_profile = pd.read_csv(
        args.data_dir / "demand_outlier_profile.csv"
    )
    cleaning_metadata = _render_cleaning(
        raw,
        clean,
        outlier_mask,
        missing_profile,
        outlier_profile,
        args.output_dir,
        args.audit_dir,
    )

    weekly, weekly_metadata = _weekly_profile(clean)
    weekly.to_csv(
        args.audit_dir / "supp_figS2_weekly_demand_patterns.csv",
        index=False,
        float_format="%.9f",
    )
    (args.audit_dir / "supp_figS2_weekly_demand_patterns_metadata.json").write_text(
        json.dumps(weekly_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _render_weekly_patterns(weekly, args.output_dir)

    origin = pd.read_csv(args.origin_mae)
    ecdf_source = _render_origin_ecdf(origin, args.output_dir)
    ecdf_source.to_csv(
        args.audit_dir / "supp_figS3_origin_mae_ecdf.csv",
        index=False,
        float_format="%.9f",
    )

    audit = {
        "supp_figS1": cleaning_metadata,
        "supp_figS2": weekly_metadata,
        "supp_figS3": {
            "tasks": ["24h", "168h"],
            "models": list(MODELS),
            "origins_per_model_task": 46,
            "metric": "publisher-compatible per-origin MAE",
        },
    }
    (args.audit_dir / "supplementary_figure_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Supplementary figures S1--S3: PASS")


if __name__ == "__main__":
    main()
