"""Data loading utilities for the DMA-WDF preprocessing pipeline.

Handles loading from the ``wf4bwdf`` package and reading pre-built
parquet artifacts from previous pipeline runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _import_wf4bwdf() -> Any:
    """Import wf4bwdf with a helpful error on failure."""
    try:
        import wf4bwdf

        return wf4bwdf
    except ImportError:
        raise ImportError(
            "The wf4bwdf package is required to load the raw BWDF dataset.\n"
            "Install it with:\n"
            "  pip install wf4bwdf@git+https://github.com/WaterFutures/wf4bwdf.git\n"
            "Or from a local clone:\n"
            "  pip install -e /path/to/wf4bwdf"
        ) from None


def load_raw_dataset() -> dict[str, pd.DataFrame]:
    """Load the complete BWDF dataset from the wf4bwdf package.

    Returns a dictionary with these keys:

    ==================  ===========================================
    Key                 Content
    ==================  ===========================================
    ``dma_inflows``     10 DMA inflow columns (L/s), hourly,
                         timezone-aware (Europe/Rome).
    ``weather``         4 weather columns (Rain, Temperature,
                         Humidity, Windspeed), hourly.
    ``calendar``        Calendar information (CEST, Holiday,
                         evaluation weeks).
    ``dma_properties``  Static DMA metadata (population, category,
                         mean flow).
    ==================  ===========================================

    Returns:
        Dictionary of DataFrames with cleaned keys.
    """
    wf4bwdf = _import_wf4bwdf()
    dataset = wf4bwdf.load_complete_dataset()
    return {
        "dma_inflows": dataset["dma-inflows"],
        "weather": dataset["weather"],
        "calendar": dataset["calendar"],
        "dma_properties": dataset["dma-properties"],
    }


def read_parquet_artifacts(
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Read pre-built parquet files from a pipeline output directory.

    Expects these files in ``data_dir``:

    - ``demand_hourly.parquet``
    - ``weather_hourly.parquet``
    - ``temporal_hourly.parquet``
    - ``dma_properties.csv`` (optional)

    Args:
        data_dir: Directory containing the parquet files.

    Returns:
        Dictionary with keys ``demand``, ``weather``, ``temporal``,
        and optionally ``dma_properties``.

    Raises:
        FileNotFoundError: If a required file is missing.
    """
    required = {
        "demand": "demand_hourly.parquet",
        "weather": "weather_hourly.parquet",
        "temporal": "temporal_hourly.parquet",
    }
    result: dict[str, pd.DataFrame] = {}
    for key, fname in required.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}. "
                f"Run the data build pipeline first."
            )
        result[key] = pd.read_parquet(path)

    props_path = data_dir / "dma_properties.csv"
    if props_path.exists():
        result["dma_properties"] = pd.read_csv(props_path)

    return result


def select_period(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Filter a DataFrame to rows between ``start`` and ``end`` inclusive.

    Args:
        df: DataFrame with a DatetimeIndex.
        start: Start timestamp (inclusive).
        end: End timestamp (inclusive).

    Returns:
        A copy of the filtered DataFrame.
    """
    return df.loc[(df.index >= start) & (df.index <= end)].copy()
