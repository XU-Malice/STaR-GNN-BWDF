"""Utility functions for generating data quality reports.

These are pure functions: they take data, return text or write files,
and have no side effects beyond the file system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def profile_table(name: str, df: pd.DataFrame) -> dict[str, Any]:
    """Profile a DataFrame: shape, time range, columns.

    Args:
        name: Display name for the table.
        df: DataFrame to profile (should have a DatetimeIndex for time plots).

    Returns:
        Dictionary with keys ``table``, ``rows``, ``cols``, ``start``,
        ``end``, and ``columns``.
    """
    return {
        "table": name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "start": str(df.index.min()) if len(df) else "",
        "end": str(df.index.max()) if len(df) else "",
        "columns": "|".join(map(str, df.columns)),
    }


def missing_rows(
    table: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Compare per-column NaN counts before and after an operation.

    Args:
        table: Table name for labeling.
        before: DataFrame before the operation.
        after: DataFrame after the operation.

    Returns:
        List of per-column records with keys ``table``, ``column``,
        ``missing_before``, ``missing_after``, ``missing_ratio_before``,
        ``missing_ratio_after``.
    """
    rows: list[dict[str, Any]] = []
    for column in before.columns:
        missing_before = int(before[column].isna().sum())
        missing_after = int(after[column].isna().sum())
        rows.append(
            {
                "table": table,
                "column": str(column),
                "missing_before": missing_before,
                "missing_after": missing_after,
                "missing_ratio_before": float(before[column].isna().mean()),
                "missing_ratio_after": float(after[column].isna().mean()),
            }
        )
    return rows


def feature_summary(
    feature_set: str,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Generate per-column summary statistics.

    Args:
        feature_set: Label for this feature group (e.g. "demand", "weather").
        df: DataFrame to summarize.

    Returns:
        List of records with keys ``feature_set``, ``column``, ``dtype``,
        ``missing``, ``unique``, ``min``, ``max``.
    """
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        rows.append(
            {
                "feature_set": feature_set,
                "column": str(column),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "min": float(numeric.min()) if numeric.notna().any() else "",
                "max": float(numeric.max()) if numeric.notna().any() else "",
            }
        )
    return rows


def write_markdown_report(
    path: Path,
    title: str,
    sections: dict[str, str],
) -> None:
    """Write a structured markdown report.

    Args:
        path: Output file path.
        title: Report title (H1 heading).
        sections: Dictionary mapping section headings to section body text.
    """
    lines = [f"# {title}", ""]
    for heading, body in sections.items():
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quality_checks_json(
    path: Path,
    checks: list[dict[str, Any]],
    all_passed: bool,
) -> None:
    """Write quality check results as JSON.

    Args:
        path: Output file path.
        checks: List of check records (each with ``check``, ``passed``,
            ``observed``, ``expected`` keys).
        all_passed: Whether all checks passed.
    """
    payload = {
        "all_passed": all_passed,
        "checks": checks,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def quality_check_row(
    check: str,
    observed: Any,
    expected: Any,
    passed: bool | None = None,
) -> dict[str, Any]:
    """Build a single quality check record.

    Args:
        check: Description of the check.
        observed: Observed value.
        expected: Expected value.
        passed: Whether the check passed.  If ``None``, auto-computed
            as ``observed == expected``.

    Returns:
        Record dictionary.
    """
    if passed is None:
        passed = bool(observed == expected)
    return {
        "check": check,
        "passed": passed,
        "observed": str(observed) if not isinstance(observed, (int, float, bool)) else observed,
        "expected": str(expected) if not isinstance(expected, (int, float, bool)) else expected,
    }
