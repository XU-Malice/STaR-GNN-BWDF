"""Verify evaluation indices match the paper protocol.

Three protocol tiers:

1. **operational**: all test candidates with available history
   - 24h: 80, 168h: 74

2. **strict_within_test**: history AND labels must be inside test period
   - 24h: 52 (= 80 - 28 within-test history days)
   - 168h: 46 (= 80 - 28 - 7 + 1)

3. **common_46**: shared forecast origins between 24h and 168h tasks
   - Both: exactly 46
   - Based on the paper's 168h eval derivation
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from dma_wdf.utils.config import read_yaml, parse_timestamp
from dma_wdf.data.sliding_window import build_sample_index, make_eval_index

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_protocol():
    """Load paper split config and parse timestamps."""
    config_path = _PROJECT_ROOT / "configs" / "data" / "paper_split.yaml"
    cfg = read_yaml(config_path)
    tz = "Europe/Rome"
    paper_start = parse_timestamp(cfg["paper_period"]["start"], tz)
    paper_end = parse_timestamp(cfg["paper_period"]["end"], tz)
    train_end = parse_timestamp(cfg["split"]["train_end_inclusive"], tz)
    test_start = parse_timestamp(cfg["split"]["test_start"], tz)
    test_end = parse_timestamp(cfg["split"]["test_end_inclusive"], tz)
    return {
        "paper_start": paper_start, "paper_end": paper_end,
        "train_end": train_end, "test_start": test_start, "test_end": test_end,
        "tz": tz, "cfg": cfg,
    }


def _build_test_starts(horizon_hours: int) -> pd.DatetimeIndex:
    """Return test forecast starts from build_sample_index."""
    proto = _load_protocol()
    si = build_sample_index(
        paper_start=proto["paper_start"], paper_end=proto["paper_end"],
        train_end=proto["train_end"], test_start=proto["test_start"],
        test_end=proto["test_end"],
        horizon_hours=horizon_hours, stride_hours=24,
        max_history_weeks=4, tz=proto["tz"],
    )
    test_si = si[si["split"] == "test"]
    return pd.DatetimeIndex(test_si["forecast_start"])


# ---------------------------------------------------------------------------
# Sample counts
# ---------------------------------------------------------------------------

class TestSampleCounts:
    def test_24h_train_686(self):
        proto = _load_protocol()
        si = build_sample_index(
            paper_start=proto["paper_start"], paper_end=proto["paper_end"],
            train_end=proto["train_end"], test_start=proto["test_start"],
            test_end=proto["test_end"],
            horizon_hours=24, stride_hours=24, max_history_weeks=4, tz=proto["tz"],
        )
        assert int((si["split"] == "train").sum()) == 686

    def test_24h_test_80(self):
        si = build_sample_index(
            paper_start=_load_protocol()["paper_start"], paper_end=_load_protocol()["paper_end"],
            train_end=_load_protocol()["train_end"], test_start=_load_protocol()["test_start"],
            test_end=_load_protocol()["test_end"],
            horizon_hours=24, stride_hours=24, max_history_weeks=4, tz=_load_protocol()["tz"],
        )
        assert int((si["split"] == "test").sum()) == 80

    def test_168h_train_680(self):
        proto = _load_protocol()
        si = build_sample_index(
            paper_start=proto["paper_start"], paper_end=proto["paper_end"],
            train_end=proto["train_end"], test_start=proto["test_start"],
            test_end=proto["test_end"],
            horizon_hours=168, stride_hours=24, max_history_weeks=4, tz=proto["tz"],
        )
        assert int((si["split"] == "train").sum()) == 680

    def test_168h_test_74(self):
        proto = _load_protocol()
        si = build_sample_index(
            paper_start=proto["paper_start"], paper_end=proto["paper_end"],
            train_end=proto["train_end"], test_start=proto["test_start"],
            test_end=proto["test_end"],
            horizon_hours=168, stride_hours=24, max_history_weeks=4, tz=proto["tz"],
        )
        assert int((si["split"] == "test").sum()) == 74


# ---------------------------------------------------------------------------
# Operational protocol
# ---------------------------------------------------------------------------

class TestOperationalProtocol:
    """All test candidates with available history."""

    def test_24h_operational_80(self):
        starts = _build_test_starts(24)
        assert len(starts) == 80

    def test_168h_operational_74(self):
        starts = _build_test_starts(168)
        assert len(starts) == 74


# ---------------------------------------------------------------------------
# Strict within-test protocol
# ---------------------------------------------------------------------------

def _count_strict_within_test(horizon_hours: int) -> int:
    """Count test origins whose history AND labels are fully inside the test period."""
    proto = _load_protocol()
    test_start = proto["test_start"]
    history_hours = 4 * 7 * 24  # 672h
    starts = _build_test_starts(horizon_hours)
    count = 0
    for ts in starts:
        history_start = ts - pd.Timedelta(hours=history_hours)
        label_end = ts + pd.Timedelta(hours=horizon_hours - 1)
        if history_start >= test_start and label_end <= proto["test_end"]:
            count += 1
    return count


class TestStrictWithinTest:
    def test_24h_strict_52(self):
        n = _count_strict_within_test(24)
        assert n == 52, f"Expected 52 strict_within_test samples for 24h, got {n}"

    def test_168h_strict_46(self):
        n = _count_strict_within_test(168)
        assert n == 46, f"Expected 46 strict_within_test samples for 168h, got {n}"

    def test_46_derivation(self):
        """46 = 80 test days - 28 history - 7 horizon + 1."""
        assert 80 - 28 - 7 + 1 == 46


# ---------------------------------------------------------------------------
# Common 46 protocol
# ---------------------------------------------------------------------------

class TestCommon46:
    """Shared forecast origins between 24h and 168h paper evaluation."""

    def test_common_46_origins_match(self):
        """The 46 paper eval forecast starts must be identical for both tasks."""
        proto = _load_protocol()
        eval_idx = make_eval_index(
            test_start=proto["test_start"], test_end=proto["test_end"],
            max_history_weeks=4, horizon_hours=168, stride_hours=24, tz=proto["tz"],
        )
        paper_46 = set(str(ts) for ts in eval_idx["forecast_start"].tolist())
        assert len(paper_46) == 46

        # 24h test starts — the common_46 subset.
        starts_24 = _build_test_starts(24)
        common_24 = [ts for ts in starts_24 if str(ts) in paper_46]
        assert len(common_24) == 46, f"24h: expected 46 common origins, got {len(common_24)}"

        # 168h test starts — the common_46 subset.
        starts_168 = _build_test_starts(168)
        common_168 = [ts for ts in starts_168 if str(ts) in paper_46]
        assert len(common_168) == 46, f"168h: expected 46 common origins, got {len(common_168)}"

        # Same timestamps.
        assert [str(ts) for ts in common_24] == [str(ts) for ts in common_168], \
            "Common 46 origins differ between 24h and 168h tasks"

    def test_46_is_subset_of_operational(self):
        """Common 46 must be a subset of operational samples for both tasks."""
        proto = _load_protocol()
        eval_idx = make_eval_index(
            test_start=proto["test_start"], test_end=proto["test_end"],
            max_history_weeks=4, horizon_hours=168, stride_hours=24, tz=proto["tz"],
        )
        paper_46 = set(str(ts) for ts in eval_idx["forecast_start"].tolist())

        for h in [24, 168]:
            starts = _build_test_starts(h)
            op_set = set(str(ts) for ts in starts)
            for ts in paper_46:
                assert ts in op_set, f"{h}h: common_46 origin {ts} not in operational set"


# ---------------------------------------------------------------------------
# CSV file validation
# ---------------------------------------------------------------------------

class TestReadSampleIndexFiles:
    def test_sample_index_24h_exists(self):
        path = _PROJECT_ROOT / "data" / "processed" / "data_build" / "sample_index_single_step_24h.csv"
        assert path.exists()

    def test_sample_index_168h_exists(self):
        path = _PROJECT_ROOT / "data" / "processed" / "data_build" / "sample_index_multi_step_168h.csv"
        assert path.exists()

    def test_sample_index_24h_counts(self):
        path = _PROJECT_ROOT / "data" / "processed" / "data_build" / "sample_index_single_step_24h.csv"
        si = pd.read_csv(path)
        assert int((si["split"] == "train").sum()) == 686
        assert int((si["split"] == "test").sum()) == 80

    def test_sample_index_168h_counts(self):
        path = _PROJECT_ROOT / "data" / "processed" / "data_build" / "sample_index_multi_step_168h.csv"
        si = pd.read_csv(path)
        assert int((si["split"] == "train").sum()) == 680
        assert int((si["split"] == "test").sum()) == 74
