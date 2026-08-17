"""Weather feature renaming and processing.

Handles the mapping from raw wf4bwdf weather column names to the
cleaned names used by the MSCMNet paper.
"""

from __future__ import annotations

import pandas as pd


def rename_weather(
    weather: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    """Rename weather columns according to a mapping dictionary.

    The ``mapping`` has cleaned names as keys and raw column names as
    values::

        {
            "rainfall_depth": "Rain",
            "air_temperature": "Temperature",
            "air_humidity": "Humidity",
            "windspeed": "Windspeed",
        }

    Args:
        weather: DataFrame with raw weather columns.
        mapping: ``{clean_name: raw_name}`` dictionary.

    Returns:
        DataFrame containing only the renamed columns, ordered by
        the mapping keys.

    Raises:
        ValueError: If any raw column name from ``mapping`` is
            missing from ``weather``.
    """
    rename_map = {
        raw: clean for clean, raw in mapping.items() if raw in weather.columns
    }
    missing = {
        clean: raw for clean, raw in mapping.items() if raw not in weather.columns
    }
    if missing:
        raise ValueError(
            f"Missing weather columns in data: {missing}. "
            f"Available columns: {list(weather.columns)}"
        )
    return weather.rename(columns=rename_map)[list(mapping.keys())]


# Default weather mapping from the MSCMNet paper.
PAPER_WEATHER_MAPPING: dict[str, str] = {
    "rainfall_depth": "Rain",
    "air_temperature": "Temperature",
    "air_humidity": "Humidity",
    "windspeed": "Windspeed",
}
