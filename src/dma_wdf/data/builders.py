#!/usr/bin/env python
"""Build model-ready tensor samples from processed parquet data.

Supports multiple model families:
  - MSNet / MSCMNet (per-DMA branch tensors with weather + temporal)
  - GRU / LSTM baselines (demand-only, per-DMA)
  - Share-STGNN (multi-modal past + future + node metadata)

Usage::

    python -m dma_wdf.data.builders msnet \\
        --data-dir data/processed/data_build/ \\
        --arch-config configs/models/msnet_architecture.yaml \\
        --output-dir data/processed/samples/msnet/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dma_wdf.data.loader import read_parquet_artifacts
from dma_wdf.utils.config import parse_timestamp, read_yaml, load_config_with_inheritance
from dma_wdf.data.outlier_detection import preprocess_demand
from dma_wdf.data.sliding_window import (
    build_branch_array,
    build_windows,
    combine_past,
    daily_starts,
    make_eval_index,
    make_train_starts,
    target_24h,
    target_168h,
)


def build_demand_only_samples(
    *,
    demand: pd.DataFrame,
    dma_column: str,
    input_weeks: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    tz: Any,
) -> dict[str, np.ndarray]:
    """Build demand-only samples for GRU/LSTM baselines.

    History = ``input_weeks * 7 * 24`` hours of the DMA's own demand.
    No weather or temporal features.

    Args:
        demand: Full demand DataFrame (all DMAs, paper period).
        dma_column: Column name for this DMA.
        input_weeks: Number of weeks of history.
        train_start, train_end: Training period bounds.
        test_start, test_end: Test period bounds.
        tz: Timezone.

    Returns:
        Dictionary with keys: ``x_train``, ``y_train_24h``,
        ``x_test_eval``, ``y_test_eval_24h``, ``y_test_eval_168h``.
    """
    series = demand[dma_column].astype(float)
    history_hours = input_weeks * 7 * 24

    train_starts = make_train_starts(
        train_start=train_start,
        train_end=train_end,
        input_weeks=input_weeks,
        target_hours=24,
        tz=tz,
    )
    eval_index = make_eval_index(
        test_start=test_start,
        test_end=test_end,
        tz=tz,
    )

    # Train samples.
    x_train = []
    for forecast_start in train_starts:
        history_start = forecast_start - pd.Timedelta(hours=history_hours)
        window = series.loc[
            (series.index >= history_start)
            & (series.index < forecast_start)
        ].to_numpy(dtype=np.float32)
        if len(window) != history_hours:
            raise ValueError(
                f"Expected {history_hours} history rows for {forecast_start}, "
                f"got {len(window)}"
            )
        x_train.append(window.reshape(history_hours, 1))

    y_train_24h = []
    for forecast_start in train_starts:
        y_train_24h.append(
            series.loc[
                (series.index >= forecast_start)
                & (series.index < forecast_start + pd.Timedelta(hours=24))
            ].to_numpy(dtype=np.float32)
        )

    # Test samples.
    x_test = []
    y_test_24h = []
    y_test_168h = []
    for _, row in eval_index.iterrows():
        fs = row["forecast_start"]
        history_start = fs - pd.Timedelta(hours=history_hours)
        window = series.loc[
            (series.index >= history_start) & (series.index < fs)
        ].to_numpy(dtype=np.float32)
        x_test.append(window.reshape(history_hours, 1))
        y_test_24h.append(
            series.loc[
                (series.index >= fs)
                & (series.index < fs + pd.Timedelta(hours=24))
            ].to_numpy(dtype=np.float32)
        )
        y_test_168h.append(
            series.loc[
                (series.index >= fs)
                & (series.index < fs + pd.Timedelta(hours=168))
            ].to_numpy(dtype=np.float32)
        )

    return {
        "x_train": np.stack(x_train).astype(np.float32),
        "y_train_24h": np.stack(y_train_24h).astype(np.float32),
        "x_test_eval": np.stack(x_test).astype(np.float32),
        "y_test_eval_24h": np.stack(y_test_24h).astype(np.float32),
        "y_test_eval_168h": np.stack(y_test_168h).astype(np.float32),
    }


def build_branch_samples(
    *,
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    temporal: pd.DataFrame,
    dma_column: str,
    dma_letter: str,
    input_weeks: int,
    weather_columns: list[str],
    temporal_columns: list[str],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    outlier_multiplier: float = 1.5,
    tz: Any = None,
) -> dict[str, np.ndarray]:
    """Build per-DMA branch tensors for MSNet / MSCMNet models.

    Each branch input is a ``(days, 24, N_features)`` block with:
    ``own_demand + weather + temporal`` features.

    Args:
        demand: Full demand DataFrame (all DMAs).
        weather: Weather DataFrame.
        temporal: Temporal features DataFrame.
        dma_column: Column name for this DMA.
        dma_letter: DMA letter (A-J) for labeling.
        input_weeks: Weeks of history for this DMA.
        weather_columns: Weather feature column names.
        temporal_columns: Temporal feature column names.
        train_start, train_end: Training period bounds.
        test_start, test_end: Test period bounds.
        outlier_multiplier: IQR multiplier (default 1.5).
        tz: Timezone.

    Returns:
        Dictionary with keys: ``x_train``, ``x_test_eval``,
        ``feature_columns``.
    """
    input_days = input_weeks * 7

    # Build per-DMA feature DataFrame.
    feature = pd.DataFrame(index=demand.index)
    feature["own_dma_demand"] = demand[dma_column].astype(float)
    for col in weather_columns:
        feature[col] = weather[col].astype(float)
    for col in temporal_columns:
        feature[col] = temporal[col].astype(float)

    # Forecast starts.
    train_forecast_starts = make_train_starts(
        train_start=train_start,
        train_end=train_end,
        input_weeks=input_weeks,
        target_hours=24,
        tz=tz,
    )
    eval_index = make_eval_index(
        test_start=test_start,
        test_end=test_end,
        tz=tz,
    )

    x_train = build_branch_array(feature, train_forecast_starts, input_days)
    x_test = build_branch_array(
        feature, pd.DatetimeIndex(eval_index["forecast_start"]), input_days
    )

    return {
        "x_train": x_train,
        "x_test_eval": x_test,
        "feature_columns": np.array(list(feature.columns)),
        "dma_letter": dma_letter,
        "dma_column": dma_column,
    }


def build_share_stgnn_samples(
    *,
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    temporal: pd.DataFrame,
    dma_columns: list[str],
    dma_letters: list[str],
    weather_columns: list[str],
    temporal_columns: list[str],
    node_meta_config: dict[str, Any],
    history_hours: int = 672,
    multi_step_hours: int = 168,
    outlier_multiplier: float = 1.5,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    tz: Any = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build Share-STGNN multi-modal tensors.

    Produces a single ``.npz`` file with past/future demand, weather,
    temporal, node metadata, and static correlation matrix.

    Args:
        demand: Demand DataFrame.
        weather: Weather DataFrame.
        temporal: Temporal features DataFrame.
        dma_columns: Ordered list of DMA column names.
        dma_letters: Ordered list of DMA letters (A-J).
        weather_columns: Weather feature names.
        temporal_columns: Temporal feature names.
        node_meta_config: Dict with ``users`` (list of ints) and
            ``land_types`` (list of land-use strings).
        history_hours: History window length (default 672 = 4 weeks).
        multi_step_hours: Multi-step horizon (default 168 = 7 days).
        outlier_multiplier: IQR multiplier.
        train_start, train_end: Training period bounds.
        test_start, test_end: Test period bounds.
        tz: Timezone.
        output_dir: If provided, save ``.npz`` and manifest files.

    Returns:
        Status dictionary with shapes and file paths.
    """
    # Land type encoding.
    land_type_ids = {
        "hospital": 0,
        "residential_countryside": 1,
        "suburban_residential_commercial": 2,
        "residential_commercial_city": 3,
        "facility_office_sport": 4,
        "residential_city": 5,
        "city_centre": 6,
        "commercial_industrial_port": 7,
    }

    # Preprocess outliers.
    demand_clean, outlier_profile = preprocess_demand(
        demand[dma_columns], multiplier=outlier_multiplier
    )

    # Forecast starts.
    history_days = history_hours // 24
    train_forecast_starts = daily_starts(
        train_start + pd.Timedelta(days=history_days), train_end, tz
    )
    train_rollout_starts = daily_starts(
        train_start + pd.Timedelta(days=history_days),
        train_end - pd.Timedelta(hours=multi_step_hours - 1),
        tz,
    )
    test_forecast_starts = daily_starts(
        test_start + pd.Timedelta(days=history_days),
        test_end - pd.Timedelta(hours=multi_step_hours - 1),
        tz,
    )

    # Build windows.
    x_demand_train_past, _ = build_windows(
        demand_clean[dma_columns], train_forecast_starts, history_hours, 24
    )
    x_demand_test_past, _ = build_windows(
        demand_clean[dma_columns], test_forecast_starts, history_hours, 24
    )
    x_weather_train_past, x_weather_future_168h = build_windows(
        weather[weather_columns], train_rollout_starts, history_hours, multi_step_hours
    )
    x_time_train_past, x_time_future_168h = build_windows(
        temporal[temporal_columns], train_rollout_starts, history_hours, multi_step_hours
    )
    x_weather_test_past, x_weather_test_future_168h = build_windows(
        weather[weather_columns], test_forecast_starts, history_hours, multi_step_hours
    )
    x_time_test_past, x_time_test_future_168h = build_windows(
        temporal[temporal_columns], test_forecast_starts, history_hours, multi_step_hours
    )

    # Build targets.
    y_test_24h = np.stack(
        [target_24h(demand_clean[dma_columns], ts) for ts in test_forecast_starts]
    ).astype(np.float32)
    y_test_168h = np.stack(
        [target_168h(demand_clean[dma_columns], ts) for ts in test_forecast_starts]
    ).astype(np.float32)

    # Node metadata.
    train_slice = demand_clean.loc[
        (demand_clean.index >= train_start) & (demand_clean.index <= train_end),
        dma_columns,
    ]
    train_mean = train_slice.mean(axis=0).to_numpy(dtype=np.float32)
    train_std = train_slice.std(axis=0).to_numpy(dtype=np.float32)
    train_std[train_std <= 1.0e-6] = 1.0

    users = np.asarray(node_meta_config["users"], dtype=np.float32)
    land_type = np.asarray(
        [land_type_ids[str(v)] for v in node_meta_config["land_types"]],
        dtype=np.float32,
    )
    raw = np.stack([users, land_type, train_mean, train_std], axis=1).astype(np.float32)

    # Normalize node meta.
    normalized = raw.copy()
    normalized[:, 0] = np.log1p(normalized[:, 0])
    normalized[:, 2] = np.log1p(np.maximum(normalized[:, 2], 0.0))
    normalized[:, 3] = np.log1p(np.maximum(normalized[:, 3], 0.0))
    for col in range(4):
        mean = normalized[:, col].mean()
        std = normalized[:, col].std()
        if std <= 1.0e-6:
            std = 1.0
        normalized[:, col] = (normalized[:, col] - mean) / std

    # Static correlation matrix.
    corr = train_slice.corr().fillna(0.0).to_numpy(dtype=np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Combine past features.
    x_past_train = combine_past(x_demand_train_past, x_weather_train_past, x_time_train_past)
    x_past_test = combine_past(x_demand_test_past, x_weather_test_past, x_time_test_past)

    shapes = {
        "x_past_train": "x".join(map(str, x_past_train.shape)),
        "x_past_test": "x".join(map(str, x_past_test.shape)),
        "y_test_168h": "x".join(map(str, y_test_168h.shape)),
        "node_meta": "x".join(map(str, normalized.shape)),
        "static_corr": "x".join(map(str, corr.shape)),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_file = output_dir / "share_stgnn_samples.npz"
        np.savez_compressed(
            sample_file,
            x_past_train=x_past_train,
            x_past_test=x_past_test,
            y_test_24h=y_test_24h,
            y_test_168h=y_test_168h,
            node_meta=normalized.astype(np.float32),
            static_corr=corr,
            dma_columns=np.array(dma_columns),
            dma_letters=np.array(dma_letters),
            weather_columns=np.array(weather_columns),
            temporal_columns=np.array(temporal_columns),
        )
        outlier_profile.to_csv(output_dir / "demand_outlier_profile.csv", index=False)

    return {
        "status": "completed",
        "shapes": shapes,
        "output_dir": str(output_dir) if output_dir else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build model-ready tensor samples."
    )
    subparsers = parser.add_subparsers(dest="command", help="Model type")

    # --- MSNet ---
    msnet_parser = subparsers.add_parser("msnet", help="Build MSNet branch samples.")
    msnet_parser.add_argument("--data-dir", type=Path, required=True)
    msnet_parser.add_argument("--arch-config", type=Path, required=True)
    msnet_parser.add_argument("--split-config", type=Path, required=True)
    msnet_parser.add_argument("--output-dir", type=Path, required=True)

    # --- Share-STGNN ---
    share_parser = subparsers.add_parser(
        "share_stgnn", help="Build Share-STGNN samples."
    )
    share_parser.add_argument("--data-dir", type=Path, required=True)
    share_parser.add_argument("--config", type=Path, required=True)
    share_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "msnet":
        _build_msnet(args)
    elif args.command == "share_stgnn":
        _build_share_stgnn(args)
    else:
        parser.print_help()


def _build_msnet(args: Any) -> None:
    """Internal: build MSNet branch samples from CLI args."""
    data = read_parquet_artifacts(args.data_dir)
    arch_cfg = read_yaml(args.arch_config)
    split_cfg = read_yaml(args.split_config)
    tz = data["demand"].index.tz
    train_start = parse_timestamp(split_cfg["split"]["train_start"], tz)
    train_end = parse_timestamp(split_cfg["split"]["train_end_inclusive"], tz)
    test_start = parse_timestamp(split_cfg["split"]["test_start"], tz)
    test_end = parse_timestamp(split_cfg["split"]["test_end_inclusive"], tz)

    # Preprocess demand.
    demand_clean, outlier_profile = preprocess_demand(
        data["demand"][arch_cfg["target"]["dma_columns"]],
        multiplier=1.5,
    )

    # Build targets.
    max_history_weeks = int(arch_cfg["sampling"]["max_history_weeks"])
    max_history_days = max_history_weeks * 7
    train_starts = daily_starts(
        train_start + pd.Timedelta(days=max_history_days), train_end, tz
    )
    test_starts = daily_starts(
        test_start + pd.Timedelta(days=max_history_days),
        test_end - pd.Timedelta(hours=int(arch_cfg["sampling"]["multi_step_hours"]) - 1),
        tz,
    )

    y_train_24h = np.stack(
        [target_24h(demand_clean, ts) for ts in train_starts]
    ).astype(np.float32)
    y_test_24h = np.stack(
        [target_24h(demand_clean, ts) for ts in test_starts]
    ).astype(np.float32)
    y_test_168h = np.stack(
        [target_168h(demand_clean, ts) for ts in test_starts]
    ).astype(np.float32)

    # Save targets.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "msnet_targets.npz",
        y_train_24h=y_train_24h,
        y_test_eval_24h=y_test_24h,
        y_test_eval_168h=y_test_168h,
    )

    # Build per-DMA branches.
    branch_dir = args.output_dir / "branches_npz"
    branch_dir.mkdir(parents=True, exist_ok=True)
    weather_cols = list(arch_cfg["features"]["weather_columns"])
    temporal_cols = list(arch_cfg["features"]["temporal_columns"])

    for i, (dma_letter, dma_col) in enumerate(
        zip(arch_cfg["target"]["dmas"], arch_cfg["target"]["dma_columns"])
    ):
        input_weeks = int(arch_cfg["table3"]["dma_input_weeks"][i])
        branch = build_branch_samples(
            demand=demand_clean,
            weather=data["weather"],
            temporal=data["temporal"],
            dma_column=dma_col,
            dma_letter=dma_letter,
            input_weeks=input_weeks,
            weather_columns=weather_cols,
            temporal_columns=temporal_cols,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            tz=tz,
        )
        fname = f"MSNet_DMA_{dma_letter}_{dma_col.replace(' ', '_')}_branch.npz"
        np.savez_compressed(branch_dir / fname, **branch)

    # Manifest.
    outlier_profile.to_csv(args.output_dir / "demand_outlier_profile.csv", index=False)
    status = {
        "status": "completed",
        "output_dir": str(args.output_dir),
        "train_samples": int(len(train_starts)),
        "test_eval_sequences": int(len(test_starts)),
    }
    (args.output_dir / "status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


def _build_share_stgnn(args: Any) -> None:
    """Internal: build Share-STGNN samples from CLI args."""
    data = read_parquet_artifacts(args.data_dir)
    cfg = load_config_with_inheritance(
        args.data_dir.parent.parent, args.config
    )
    split_cfg = read_yaml(
        args.data_dir.parent.parent / cfg["source"]["paper_split_config"]
    )
    tz = data["demand"].index.tz

    train_start = parse_timestamp(split_cfg["split"]["train_start"], tz)
    train_end = parse_timestamp(split_cfg["split"]["train_end_inclusive"], tz)
    test_start = parse_timestamp(split_cfg["split"]["test_start"], tz)
    test_end = parse_timestamp(split_cfg["split"]["test_end_inclusive"], tz)

    result = build_share_stgnn_samples(
        demand=data["demand"],
        weather=data["weather"],
        temporal=data["temporal"],
        dma_columns=list(cfg["target"]["dma_columns"]),
        dma_letters=list(cfg["target"]["dmas"]),
        weather_columns=list(cfg["features"]["weather_columns"]),
        temporal_columns=list(cfg["features"]["temporal_columns"]),
        node_meta_config=cfg["node_meta"],
        history_hours=int(cfg["sampling"]["history_hours"]),
        multi_step_hours=int(cfg["sampling"]["multi_step_hours"]),
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        tz=tz,
        output_dir=args.output_dir,
    )
    (args.output_dir / "status.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
