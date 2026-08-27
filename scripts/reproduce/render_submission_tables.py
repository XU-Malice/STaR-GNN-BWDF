#!/usr/bin/env python
"""Render manuscript-display tables from audited full-precision CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from render_submission_figures import (
    _derive_origin_improvements,
    _moving_block_indices,
)


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
        "**Note.** MAE is the sum of the ten DMA-level MAEs; MAPE, RMSE and "
        "NSE are evaluated on aggregate demand. Values for GRU, LSTM, MSNet "
        "and the MSCMNet variants were reported by Que et al. (2024). DCRNN, "
        "STGCN and STaR-GNN were evaluated using the present study's pipeline. "
        "Cross-source comparisons are used for performance positioning, "
        "whereas paired inference is restricted to the latter three models. "
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
        "**Note.** MAE is the sum of the ten DMA-level MAEs; MAPE, RMSE and "
        "NSE are evaluated on aggregate demand. STGCN is an independent graph "
        "baseline and is excluded from the factorial ablation. At 168 h, "
        "SAS-Norm-only has the marginally lower MAE (12.208 vs. 12.234), "
        "whereas the full "
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
        "comparisons in Main Fig. 2 and the absolute best-baseline comparison "
        "in Main Fig. 3.",
    ]
    return "\n".join(lines) + "\n"


def _dma_local_margin_markdown(frame: pd.DataFrame) -> str:
    """Render comparator identities and signed margins complementing Fig. 3."""
    lines = [
        "**Table S2. DMA-level margins of STaR-GNN relative to the locally "
        "strongest competing model.**",
        "",
        "| Horizon | DMA | MAE competitor | ΔMAE (%) | MAPE competitor | "
        "ΔMAPE (%) | RMSE competitor | ΔRMSE (%) | NSE competitor | ΔNSE |",
        "|---|---|---|---:|---|---:|---|---:|---|---:|",
    ]
    for task in ("24h", "168h"):
        for dma in tuple("ABCDEFGHIJ"):
            block = frame.loc[(frame["task"] == task) & (frame["DMA"] == dma)]
            star = block.loc[block["model"] == "STaR-GNN"].iloc[0]
            competitors = block.loc[block["model"] != "STaR-GNN"]
            cells: list[str] = []
            for metric in METRICS:
                if metric == "NSE":
                    best = competitors.loc[competitors[metric].idxmax()]
                    margin = float(star[metric]) - float(best[metric])
                    margin_text = f"{margin:+.3f}"
                else:
                    best = competitors.loc[competitors[metric].idxmin()]
                    denominator = max(abs(float(best[metric])), 1.0e-12)
                    margin = 100.0 * (
                        float(best[metric]) - float(star[metric])
                    ) / denominator
                    margin_text = f"{margin:+.1f}"
                cells.extend([str(best["model"]), margin_text])
            lines.append(
                f"| {task.replace('h', ' h')} | {dma} | "
                + " | ".join(cells)
                + " |"
            )
    lines += [
        "",
        "**Note.** For each horizon, DMA and metric, the comparator is selected "
        "independently as the best-performing non-STaR-GNN model among the eight "
        "alternatives. For MAE, MAPE and RMSE, "
        "\\(\\Delta E=(E_{competitor}-E_{STaR-GNN})/E_{competitor}\\times100\\%\\); "
        "for NSE, "
        "\\(\\Delta\\mathrm{NSE}=\\mathrm{NSE}_{STaR-GNN}-\\mathrm{NSE}_{competitor}\\). "
        "Positive values favor STaR-GNN; negative values identify local losses. "
        "All values are retained, including cases in which STaR-GNN is not best. "
        "Source provenance for the underlying absolute metrics is provided in "
        "Supplementary Table S1.",
    ]
    return "\n".join(lines) + "\n"


def _origin_robustness_markdown(frame: pd.DataFrame) -> str:
    """Render paired forecast-origin and difficult-window evidence."""
    lines = [
        "**Table S3. Forecast-origin robustness of STaR-GNN relative to the "
        "same-protocol graph baselines.**",
        "",
        "| Horizon | Baseline | Metric | Mean paired effect | 95% block CI | "
        "Wins / 46 | High-variability mean | Wins / 12 | Spearman ρ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in ("24h", "168h"):
        for baseline in ("DCRNN", "STGCN"):
            for metric in METRICS:
                row = frame.loc[
                    (frame["task"] == task)
                    & (frame["baseline"] == baseline)
                    & (frame["metric"] == metric)
                ].iloc[0]
                if metric == "NSE":
                    effect = f"{float(row['mean_improvement']):+.4f}"
                    ci = (
                        f"[{float(row['ci95_lower']):+.4f}, "
                        f"{float(row['ci95_upper']):+.4f}]"
                    )
                    difficult = (
                        f"{float(row['high_variability_mean_improvement']):+.4f}"
                    )
                else:
                    effect = f"{float(row['mean_improvement']):+.2f}%"
                    ci = (
                        f"[{float(row['ci95_lower']):+.2f}%, "
                        f"{float(row['ci95_upper']):+.2f}%]"
                    )
                    difficult = (
                        f"{float(row['high_variability_mean_improvement']):+.2f}%"
                    )
                lines.append(
                    f"| {task.replace('h', ' h')} | {baseline} | {metric} | "
                    f"{effect} | {ci} | {int(row['wins'])}/46 | {difficult} | "
                    f"{int(row['high_variability_wins'])}/12 | "
                    f"{float(row['spearman_difficulty_improvement']):+.3f} |"
                )
    lines += [
        "",
        "**Note.** Error-metric effects are per-origin relative reductions; "
        "NSE effects are absolute differences. Positive values favor "
        "STaR-GNN. Confidence intervals are from an ordered seven-origin "
        "moving-block bootstrap (50,000 iterations); adjacent 168 h windows "
        "overlap and are not treated as independent. Demand difficulty is "
        "defined from observations only: for each origin, each DMA's mean "
        "absolute hourly ramp is normalized by its mean demand, and the median "
        "over ten DMAs is used. The highest horizon-specific quartile contains "
        "12 origins. Spearman ρ describes the association between this "
        "difficulty index and paired improvement and is not a causal estimate.",
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
    parser.add_argument("--block-length", type=int, default=7)
    parser.add_argument("--bootstrap-iterations", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
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
    (output / "tableS2_dma_local_margin.md").write_text(
        _dma_local_margin_markdown(dma), encoding="utf-8"
    )
    bootstrap_indices = _moving_block_indices(
        46,
        args.block_length,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )
    _, origin_summary = _derive_origin_improvements(
        args.release.resolve(), indices=bootstrap_indices
    )
    origin_summary.to_csv(
        output / "tableS3_forecast_origin_robustness.csv",
        index=False,
        float_format="%.9f",
    )
    (output / "tableS3_forecast_origin_robustness.md").write_text(
        _origin_robustness_markdown(origin_summary), encoding="utf-8"
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
    print("  Table S2 — DMA-level local competitor identities and margins")
    print("  Table S3 — forecast-origin and high-variability robustness")


if __name__ == "__main__":
    main()
