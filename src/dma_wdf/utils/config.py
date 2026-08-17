"""YAML configuration loading with inheritance support."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration as a dictionary.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    Values from ``override`` take precedence over ``base``.
    The ``base_config`` key is skipped during merging (used only
    for config inheritance, not passed through to the result).
    Neither input dictionary is modified.

    Args:
        base: The base configuration dictionary.
        override: The overriding configuration dictionary.

    Returns:
        A new dictionary with merged values.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config_with_inheritance(
    root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Load a YAML configuration that may inherit from a base config.

    If the loaded config contains a ``base_config`` key, that file
    is loaded first (resolved relative to ``root``), and the current
    config is deep-merged on top.  This is recursive — a base config
    may itself reference another base.

    Args:
        root: Project root directory for resolving relative paths.
        config_path: Path to the configuration file to load.

    Returns:
        Fully resolved configuration dictionary.
    """
    resolved = config_path if config_path.is_absolute() else root / config_path
    cfg = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    base_config = cfg.get("base_config")
    if not base_config:
        return cfg
    base_path = Path(str(base_config))
    if not base_path.is_absolute():
        base_path = root / base_path
    base = load_config_with_inheritance(root, base_path)
    return deep_merge(base, cfg)


def parse_timestamp(value: str, tz: Any = None) -> pd.Timestamp:
    """Parse a timestamp string and localize or convert to the given timezone.

    Args:
        value: ISO-format timestamp string.
        tz: Target timezone (e.g. ``"Europe/Rome"``).  If the parsed
            timestamp is timezone-naive and ``tz`` is provided, it is
            localized; if it is already timezone-aware, it is converted.

    Returns:
        Timezone-aware pandas Timestamp.
    """
    ts = pd.Timestamp(value)
    if tz is None:
        return ts
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)
