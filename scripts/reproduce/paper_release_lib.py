"""Shared, side-effect-free definitions for the STaR-GNN paper release."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "configs/paper/protocol.yaml"
TASKS = ("24h", "168h")
METRICS = ("MAE", "MAPE", "RMSE", "NSE")
STAR_VARIANTS = {
    "Base": "backbone",
    "State": "dssn_sasr",
    "FA-DPR": "fa_dpr",
    "Full": "full",
}


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty metric file: {path}")
    row = rows[-1]
    mape_key = "MAPE_percent" if "MAPE_percent" in row else "MAPE"
    mape = float(row[mape_key])
    if mape_key == "MAPE" and mape < 1.0:
        mape *= 100.0
    return {
        "MAE": float(row["MAE"]),
        "MAPE": mape,
        "RMSE": float(row["RMSE"]),
        "NSE": float(row["NSE"]),
    }


def better(metric: str, left: float, right: float) -> bool:
    return left > right if metric == "NSE" else left < right


def release_model_dir(
    release_root: Path,
    family: str,
    name: str,
    task: str,
) -> Path:
    return release_root / "models" / family / name / task / "seed_0"
def check_test_summary(path: Path, checkpoint: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"Evaluation is not completed: {path}")
    counts = payload.get("protocol_counts", {})
    if int(counts.get("common_46", -1)) != 46:
        raise ValueError(f"common_46 count is not 46: {path}")
    if payload.get("test_targets_used_for_training_or_selection") is not False:
        raise ValueError(f"Test-selection guard is absent or false: {path}")
    recorded = payload.get("checkpoint_sha256")
    actual = sha256(checkpoint)
    if recorded != actual:
        raise ValueError(
            f"Checkpoint/evaluation hash mismatch: {checkpoint}: "
            f"{recorded!r} != {actual!r}"
        )
    return payload
