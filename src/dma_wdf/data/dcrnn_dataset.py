"""Leakage-aware DCRNN dataset construction for the BWDF protocol.

This module is the only bridge between the processed parquet artifacts and
the DCRNN model contract. It deliberately keeps training and testing separate:

* :func:`prepare_dcrnn_training_data` reads only official-training-period
  windows, fits scalers on the fitting portion, and returns fitting/validation
  tensors.
* :func:`prepare_dcrnn_test_data` requires already-fitted scaler states and
  constructs official-test-period tensors plus the three evaluation protocols.

The fixed Pearson graph is never rebuilt here.  Both tasks use the graph
artifact referenced by ``configs/model/dcrnn.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dma_wdf.data.graph import load_graph
from dma_wdf.data.loader import read_parquet_artifacts
from dma_wdf.data.sliding_window import (
    build_windows,
    combine_past,
    make_eval_index,
)
from dma_wdf.utils.config import parse_timestamp, read_yaml


TEMPORAL_FEATURE_ORDER = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "time_zone_standard",
    "weekday",
    "holiday",
]


@dataclass(frozen=True)
class ZScoreScaler:
    """Serializable feature-wise Z-score scaler."""

    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]
    fit_start: str
    fit_end: str
    fit_rows: int

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        feature_names: list[str],
        fit_start: pd.Timestamp,
        fit_end: pd.Timestamp,
    ) -> "ZScoreScaler":
        if fit_end < fit_start:
            raise ValueError("fit_end must not precede fit_start.")
        missing = [name for name in feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"Scaler features are missing: {missing}")
        fit_frame = frame.loc[
            (frame.index >= fit_start) & (frame.index <= fit_end),
            feature_names,
        ]
        if fit_frame.empty:
            raise ValueError("Scaler fitting frame is empty.")
        values = fit_frame.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Scaler fitting values contain NaN/Inf.")
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std[std <= 1.0e-6] = 1.0
        return cls(
            mean=mean,
            std=std,
            feature_names=tuple(feature_names),
            fit_start=str(fit_start),
            fit_end=str(fit_end),
            fit_rows=int(len(fit_frame)),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != len(self.feature_names):
            raise ValueError(
                "Scaler feature dimension mismatch: "
                f"{array.shape[-1]} != {len(self.feature_names)}."
            )
        return ((array - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != len(self.feature_names):
            raise ValueError(
                "Scaler feature dimension mismatch: "
                f"{array.shape[-1]} != {len(self.feature_names)}."
            )
        return (array * self.std + self.mean).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": list(self.feature_names),
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
            "fit_rows": self.fit_rows,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ZScoreScaler":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            feature_names=tuple(str(v) for v in state["feature_names"]),
            fit_start=str(state["fit_start"]),
            fit_end=str(state["fit_end"]),
            fit_rows=int(state["fit_rows"]),
        )


@dataclass(frozen=True)
class DCRNNWindowSubset:
    """One model-ready fitting, validation, or test subset."""

    x_past: np.ndarray
    y_scaled: np.ndarray
    y_raw: np.ndarray
    future_exog: np.ndarray
    forecast_starts: tuple[str, ...]

    @property
    def num_samples(self) -> int:
        return int(self.x_past.shape[0])


@dataclass(frozen=True)
class DCRNNTrainingData:
    fit: DCRNNWindowSubset
    validation: DCRNNWindowSubset
    demand_scaler: ZScoreScaler
    weather_scaler: ZScoreScaler
    input_feature_names: tuple[str, ...]
    future_exog_feature_names: tuple[str, ...]
    dma_columns: tuple[str, ...]
    purged_forecast_starts: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def input_dim(self) -> int:
        return len(self.input_feature_names)

    @property
    def future_exog_dim(self) -> int:
        return len(self.future_exog_feature_names)


@dataclass(frozen=True)
class DCRNNTestData:
    test: DCRNNWindowSubset
    protocol_indices: dict[str, np.ndarray]
    dma_columns: tuple[str, ...]
    metadata: dict[str, Any]


def encode_temporal_features(temporal: pd.DataFrame) -> pd.DataFrame:
    """Encode hour/day-of-week cyclically using a fixed seven-column order."""
    required = [
        "hour",
        "day_of_week",
        "time_zone_standard",
        "weekday",
        "holiday",
    ]
    missing = [name for name in required if name not in temporal.columns]
    if missing:
        raise ValueError(f"Temporal columns are missing: {missing}")

    hour = temporal["hour"].astype(float)
    day_of_week = temporal["day_of_week"].astype(float)
    encoded = pd.DataFrame(index=temporal.index)
    encoded["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    encoded["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    encoded["day_of_week_sin"] = np.sin(
        2.0 * np.pi * day_of_week / 7.0
    )
    encoded["day_of_week_cos"] = np.cos(
        2.0 * np.pi * day_of_week / 7.0
    )
    for name in ["time_zone_standard", "weekday", "holiday"]:
        encoded[name] = temporal[name].astype(float)
    encoded = encoded[TEMPORAL_FEATURE_ORDER].astype(np.float32)
    if not np.isfinite(encoded.to_numpy()).all():
        raise ValueError("Encoded temporal features contain NaN/Inf.")
    return encoded


def load_sample_index(
    path: Path,
    *,
    timezone: Any,
    expected_horizon: int,
    expected_stride: int,
) -> pd.DataFrame:
    """Load a saved sample index and restore timezone-aware columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Sample index does not exist: {path}")
    frame = pd.read_csv(path)
    required = {
        "forecast_start",
        "forecast_end",
        "history_start",
        "history_end",
        "horizon_hours",
        "stride_hours",
        "split",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Sample index columns are missing: {sorted(missing)}")
    for name in [
        "forecast_start",
        "forecast_end",
        "history_start",
        "history_end",
    ]:
        parsed = pd.to_datetime(frame[name], utc=True)
        frame[name] = parsed.dt.tz_convert(timezone)
    if set(frame["horizon_hours"].astype(int)) != {int(expected_horizon)}:
        raise ValueError("Sample index horizon does not match task config.")
    if set(frame["stride_hours"].astype(int)) != {int(expected_stride)}:
        raise ValueError("Sample index stride does not match task config.")
    if not frame["forecast_start"].is_monotonic_increasing:
        raise ValueError("Sample index forecast starts are not monotonic.")
    return frame


def split_development_index(
    train_index: pd.DataFrame,
    *,
    validation_samples: int,
    purge_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split official training origins into fit/purge/validation."""
    validation_samples = int(validation_samples)
    purge_samples = int(purge_samples)
    if validation_samples <= 0 or purge_samples < 0:
        raise ValueError("Invalid validation/purge sample counts.")
    n_fit = len(train_index) - validation_samples - purge_samples
    if n_fit <= 0:
        raise ValueError("Not enough development samples for requested split.")

    fit = train_index.iloc[:n_fit].copy()
    purge = train_index.iloc[n_fit : n_fit + purge_samples].copy()
    validation = train_index.iloc[n_fit + purge_samples :].copy()
    if len(validation) != validation_samples:
        raise AssertionError("Validation sample count mismatch.")

    last_fit_label_end = fit.iloc[-1]["forecast_end"]
    first_validation_origin = validation.iloc[0]["forecast_start"]
    if last_fit_label_end >= first_validation_origin:
        raise ValueError(
            "Fitting and validation label windows overlap: "
            f"{last_fit_label_end} >= {first_validation_origin}."
        )
    return fit, purge, validation


def build_evaluation_protocol_indices(
    test_index: pd.DataFrame,
    *,
    official_test_start: pd.Timestamp,
    official_test_end: pd.Timestamp,
    history_hours: int,
    timezone: Any,
) -> dict[str, np.ndarray]:
    """Return operational, strict-within-test, and common-46 indices."""
    starts = pd.DatetimeIndex(test_index["forecast_start"])
    operational = np.arange(len(starts), dtype=np.int64)
    strict = np.flatnonzero(
        np.asarray(
            [
                start - pd.Timedelta(hours=history_hours)
                >= official_test_start
                for start in starts
            ],
            dtype=bool,
        )
    ).astype(np.int64)

    common_index = make_eval_index(
        test_start=official_test_start,
        test_end=official_test_end,
        max_history_weeks=history_hours // (7 * 24),
        horizon_hours=168,
        stride_hours=24,
        tz=timezone,
    )
    common_starts = {
        pd.Timestamp(value)
        for value in common_index["forecast_start"]
    }
    common = np.asarray(
        [
            index
            for index, start in enumerate(starts)
            if pd.Timestamp(start) in common_starts
        ],
        dtype=np.int64,
    )
    return {
        "operational": operational,
        "strict_within_test": strict,
        "common_46": common,
    }


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _validate_hourly_frames(
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    temporal: pd.DataFrame,
) -> None:
    if not demand.index.equals(weather.index) or not demand.index.equals(
        temporal.index
    ):
        raise ValueError("Demand, weather, and temporal indices are not aligned.")
    for name, frame in [
        ("demand", demand),
        ("weather", weather),
        ("temporal", temporal),
    ]:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{name} must have a DatetimeIndex.")
        if frame.index.tz is None:
            raise ValueError(f"{name} index must be timezone-aware.")
        if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise ValueError(f"{name} index must be sorted and unique.")
        if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name} contains NaN/Inf.")


def _load_context(
    *,
    project_root: Path,
    config: dict[str, Any],
    data_dir: Path | None,
) -> dict[str, Any]:
    source = config["source"]
    resolved_data_dir = (
        data_dir.resolve()
        if data_dir is not None
        else _resolve(project_root, source["data_dir"]).resolve()
    )
    artifacts = read_parquet_artifacts(resolved_data_dir)
    demand = artifacts["demand"]
    weather = artifacts["weather"]
    temporal = artifacts["temporal"]
    _validate_hourly_frames(demand, weather, temporal)

    dma_columns = [str(value) for value in config["graph"]["dma_columns"]]
    weather_columns = [
        str(value)
        for value in config["features"]["weather_columns"]
    ]
    if list(demand.columns) != dma_columns:
        raise ValueError(
            "Processed demand order does not match fixed graph order: "
            f"{list(demand.columns)} != {dma_columns}."
        )
    if list(weather.columns) != weather_columns:
        raise ValueError(
            "Processed weather order does not match config: "
            f"{list(weather.columns)} != {weather_columns}."
        )

    split_cfg = read_yaml(_resolve(project_root, source["split_config"]))
    timezone = demand.index.tz
    train_start = parse_timestamp(
        split_cfg["split"]["train_start"],
        timezone,
    )
    train_end = parse_timestamp(
        split_cfg["split"]["train_end_inclusive"],
        timezone,
    )
    test_start = parse_timestamp(
        split_cfg["split"]["test_start"],
        timezone,
    )
    test_end = parse_timestamp(
        split_cfg["split"]["test_end_inclusive"],
        timezone,
    )
    sample_index = load_sample_index(
        _resolve(project_root, source["sample_index"]),
        timezone=timezone,
        expected_horizon=int(config["task"]["horizon"]),
        expected_stride=int(config["task"]["stride_hours"]),
    )
    temporal_encoded = encode_temporal_features(temporal)

    graph_path = _resolve(
        project_root,
        config["graph"]["artifact_path"],
    )
    graph = load_graph(graph_path)
    if int(graph["fit_rows"]) != 17136:
        raise ValueError("Fixed graph must use all 17136 training rows.")
    if graph["dma_columns"] != dma_columns:
        raise ValueError("Fixed graph DMA order does not match dataset.")

    return {
        "data_dir": resolved_data_dir,
        "demand": demand[dma_columns],
        "weather": weather[weather_columns],
        "temporal": temporal_encoded,
        "dma_columns": dma_columns,
        "weather_columns": weather_columns,
        "sample_index": sample_index,
        "timezone": timezone,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "graph_path": graph_path,
        "graph": graph,
    }


def _build_window_subset(
    *,
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    temporal: pd.DataFrame,
    sample_index: pd.DataFrame,
    history_hours: int,
    horizon: int,
    demand_scaler: ZScoreScaler,
    weather_scaler: ZScoreScaler,
    num_nodes: int,
) -> DCRNNWindowSubset:
    starts = pd.DatetimeIndex(sample_index["forecast_start"])
    x_demand, y_raw = build_windows(
        demand,
        starts,
        history_hours,
        horizon,
    )
    x_weather, _ = build_windows(
        weather,
        starts,
        history_hours,
        horizon,
    )
    x_temporal, future_temporal = build_windows(
        temporal,
        starts,
        history_hours,
        horizon,
    )

    x_demand_scaled = demand_scaler.transform(x_demand)
    x_weather_scaled = weather_scaler.transform(x_weather)
    y_scaled = demand_scaler.transform(y_raw)
    x_past = combine_past(
        x_demand_scaled,
        x_weather_scaled,
        x_temporal,
    )
    future_exog = np.repeat(
        future_temporal[:, :, None, :],
        num_nodes,
        axis=2,
    ).astype(np.float32)

    arrays = [x_past, y_scaled, y_raw, future_exog]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Model-ready subset contains NaN/Inf.")
    return DCRNNWindowSubset(
        x_past=x_past.astype(np.float32),
        y_scaled=y_scaled.astype(np.float32),
        y_raw=y_raw.astype(np.float32),
        future_exog=future_exog,
        forecast_starts=tuple(str(value) for value in starts),
    )


def prepare_dcrnn_training_data(
    *,
    project_root: Path,
    config: dict[str, Any],
    data_dir: Path | None = None,
) -> DCRNNTrainingData:
    """Build fitting/validation tensors without constructing test windows."""
    project_root = project_root.resolve()
    context = _load_context(
        project_root=project_root,
        config=config,
        data_dir=data_dir,
    )
    task = config["task"]
    split = config["split"]
    history_hours = int(task["history_hours"])
    horizon = int(task["horizon"])

    train_index = context["sample_index"].loc[
        context["sample_index"]["split"] == "train"
    ].reset_index(drop=True)
    expected_development = int(split["expected_development_samples"])
    if len(train_index) != expected_development:
        raise ValueError(
            f"Development samples={len(train_index)}, "
            f"expected={expected_development}."
        )
    fit_index, purge_index, validation_index = split_development_index(
        train_index,
        validation_samples=int(split["validation_samples"]),
        purge_samples=int(split["purge_samples"]),
    )
    if len(fit_index) != int(split["expected_fit_samples"]):
        raise ValueError("Fitting sample count does not match config.")

    fit_end = pd.Timestamp(fit_index.iloc[-1]["forecast_end"])
    demand_train = context["demand"].loc[
        context["train_start"] : context["train_end"]
    ]
    weather_train = context["weather"].loc[
        context["train_start"] : context["train_end"]
    ]
    temporal_train = context["temporal"].loc[
        context["train_start"] : context["train_end"]
    ]
    demand_scaler = ZScoreScaler.fit(
        demand_train,
        feature_names=context["dma_columns"],
        fit_start=context["train_start"],
        fit_end=fit_end,
    )
    weather_scaler = ZScoreScaler.fit(
        weather_train,
        feature_names=context["weather_columns"],
        fit_start=context["train_start"],
        fit_end=fit_end,
    )

    fit = _build_window_subset(
        demand=demand_train,
        weather=weather_train,
        temporal=temporal_train,
        sample_index=fit_index,
        history_hours=history_hours,
        horizon=horizon,
        demand_scaler=demand_scaler,
        weather_scaler=weather_scaler,
        num_nodes=len(context["dma_columns"]),
    )
    validation = _build_window_subset(
        demand=demand_train,
        weather=weather_train,
        temporal=temporal_train,
        sample_index=validation_index,
        history_hours=history_hours,
        horizon=horizon,
        demand_scaler=demand_scaler,
        weather_scaler=weather_scaler,
        num_nodes=len(context["dma_columns"]),
    )

    input_features = (
        ["demand"]
        + context["weather_columns"]
        + TEMPORAL_FEATURE_ORDER
    )
    metadata = {
        "task": str(task["name"]),
        "horizon": horizon,
        "history_hours": history_hours,
        "development_samples": len(train_index),
        "fit_samples": fit.num_samples,
        "purge_samples": len(purge_index),
        "validation_samples": validation.num_samples,
        "last_fit_origin": str(fit_index.iloc[-1]["forecast_start"]),
        "last_fit_label_end": str(fit_index.iloc[-1]["forecast_end"]),
        "first_validation_origin": str(
            validation_index.iloc[0]["forecast_start"]
        ),
        "first_validation_label_end": str(
            validation_index.iloc[0]["forecast_end"]
        ),
        "labels_overlap": False,
        "graph_artifact_path": str(context["graph_path"]),
        "graph_fit_rows": int(context["graph"]["fit_rows"]),
        "graph_shared_by_tasks": True,
    }
    return DCRNNTrainingData(
        fit=fit,
        validation=validation,
        demand_scaler=demand_scaler,
        weather_scaler=weather_scaler,
        input_feature_names=tuple(input_features),
        future_exog_feature_names=tuple(TEMPORAL_FEATURE_ORDER),
        dma_columns=tuple(context["dma_columns"]),
        purged_forecast_starts=tuple(
            str(value)
            for value in purge_index["forecast_start"]
        ),
        metadata=metadata,
    )


def prepare_dcrnn_test_data(
    *,
    project_root: Path,
    config: dict[str, Any],
    demand_scaler: ZScoreScaler,
    weather_scaler: ZScoreScaler,
    data_dir: Path | None = None,
) -> DCRNNTestData:
    """Build official-test tensors using scaler states from a checkpoint."""
    project_root = project_root.resolve()
    context = _load_context(
        project_root=project_root,
        config=config,
        data_dir=data_dir,
    )
    task = config["task"]
    split = config["split"]
    history_hours = int(task["history_hours"])
    horizon = int(task["horizon"])
    test_index = context["sample_index"].loc[
        context["sample_index"]["split"] == "test"
    ].reset_index(drop=True)
    expected_test = int(split["expected_test_candidates"])
    if len(test_index) != expected_test:
        raise ValueError(
            f"Test candidates={len(test_index)}, expected={expected_test}."
        )

    test = _build_window_subset(
        demand=context["demand"],
        weather=context["weather"],
        temporal=context["temporal"],
        sample_index=test_index,
        history_hours=history_hours,
        horizon=horizon,
        demand_scaler=demand_scaler,
        weather_scaler=weather_scaler,
        num_nodes=len(context["dma_columns"]),
    )
    protocols = build_evaluation_protocol_indices(
        test_index,
        official_test_start=context["test_start"],
        official_test_end=context["test_end"],
        history_hours=history_hours,
        timezone=context["timezone"],
    )
    expected_protocols = {
        "operational": int(split["expected_operational_samples"]),
        "strict_within_test": int(split["expected_strict_samples"]),
        "common_46": int(split["expected_common_samples"]),
    }
    for name, expected in expected_protocols.items():
        if len(protocols[name]) != expected:
            raise ValueError(
                f"{name} samples={len(protocols[name])}, "
                f"expected={expected}."
            )
    metadata = {
        "task": str(task["name"]),
        "horizon": horizon,
        "history_hours": history_hours,
        "test_candidates": test.num_samples,
        "protocol_counts": {
            name: int(len(indices))
            for name, indices in protocols.items()
        },
        "teacher_forcing_required": False,
    }
    return DCRNNTestData(
        test=test,
        protocol_indices=protocols,
        dma_columns=tuple(context["dma_columns"]),
        metadata=metadata,
    )
