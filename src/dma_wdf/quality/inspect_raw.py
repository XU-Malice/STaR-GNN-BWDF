#!/usr/bin/env python
"""Inspect raw data from the wf4bwdf package.

This is a **standalone** script — it does not depend on any other
DMA-WDF modules.  Run it directly to profile the raw BWDF dataset.

Usage::

    python -m dma_wdf.quality.inspect_raw [--output-dir data/reports/raw/]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _try_import_wf4bwdf() -> Any:
    """Attempt to import wf4bwdf with a helpful error message."""
    try:
        import wf4bwdf  # noqa: F811

        return wf4bwdf
    except ImportError:
        print(
            "ERROR: wf4bwdf is not installed.\n"
            "Install it with:\n"
            "  pip install wf4bwdf@git+https://github.com/WaterFutures/wf4bwdf.git\n"
            "Or, if already installed in the old BWDF project:\n"
            "  pip install -e /path/to/wf4bwdf",
            file=sys.stderr,
        )
        sys.exit(1)


def profile_dataframe(name: str, df: pd.DataFrame) -> dict[str, Any]:
    """Return a dictionary of basic profiling info for a DataFrame."""
    info: dict[str, Any] = {
        "table": name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
    }
    if isinstance(df.index, pd.DatetimeIndex):
        info["index_freq"] = str(df.index.freq) if df.index.freq else "irregular"
        info["index_tz"] = str(df.index.tz)
    return info


def profile_missing(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column NaN summary."""
    rows = []
    for col in df.columns:
        rows.append(
            {
                "table": name,
                "column": str(col),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean()) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def inspect_official(output_dir: Path | None = None) -> dict[str, Any]:
    """Load and profile the complete wf4bwdf dataset.

    Args:
        output_dir: If provided, write CSV/JSON reports here.

    Returns:
        Status dictionary with profile summaries.
    """
    wf4bwdf = _try_import_wf4bwdf()
    dataset = wf4bwdf.load_complete_dataset()

    tables: dict[str, pd.DataFrame] = {
        "dma-inflows": dataset["dma-inflows"],
        "weather": dataset["weather"],
        "calendar": dataset["calendar"],
        "dma-properties": dataset["dma-properties"],
    }

    profiles = [profile_dataframe(name, df) for name, df in tables.items()]
    profile_df = pd.DataFrame(profiles)

    missing_dfs = [profile_missing(name, df) for name, df in tables.items()]
    missing_df = pd.concat(missing_dfs, ignore_index=True)

    # Evaluation week summary.
    calendar = dataset["calendar"]
    eval_weeks: list[dict[str, Any]] = []
    if "Evaluation week" in calendar.columns:
        for (iteration, week_num), group in calendar.groupby(
            ["Iteration", "Dataset week number"], dropna=True
        ):
            if group["Evaluation week"].any():
                eval_weeks.append(
                    {
                        "iteration": int(iteration),
                        "dataset_week": int(week_num),
                        "start": str(group.index.min()),
                        "end": str(group.index.max()),
                        "hours": int(len(group)),
                    }
                )

    print("=" * 60)
    print("wf4bwdf Official Dataset Inspection")
    print("=" * 60)
    print(f"\nTables found: {len(tables)}")
    for name, df in tables.items():
        print(f"  {name}: {len(df)} rows × {len(df.columns)} cols")
        print(f"    index: {df.index.min()} → {df.index.max()}")
        if df.columns.nlevels == 1:
            print(f"    columns: {list(df.columns)[:5]}...")

    print(f"\nTotal missing values:")
    print(missing_df[missing_df["missing"] > 0].to_string(index=False))
    print(f"\nEvaluation weeks: {len(eval_weeks)}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_df.to_csv(output_dir / "table_profiles.csv", index=False)
        missing_df.to_csv(output_dir / "missing_profile.csv", index=False)
        pd.DataFrame(eval_weeks).to_csv(output_dir / "evaluation_weeks.csv", index=False)
        (output_dir / "inspection_report.json").write_text(
            json.dumps(
                {
                    "tables": profiles,
                    "evaluation_weeks": eval_weeks,
                    "total_missing": int(missing_df["missing"].sum()),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nReports written to {output_dir}")

    return {
        "status": "completed",
        "tables": profiles,
        "evaluation_week_count": len(eval_weeks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect raw wf4bwdf data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV/JSON reports.",
    )
    args = parser.parse_args()
    result = inspect_official(output_dir=args.output_dir)
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
