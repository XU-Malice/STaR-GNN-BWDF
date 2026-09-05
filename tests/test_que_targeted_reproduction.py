from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "scripts/reproduce/summarize_que_targeted_reproduction.py"
SPEC = importlib.util.spec_from_file_location("que_targeted_summary", SUMMARY)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _scores(relative_error: float = 0.0, nse_error: float = 0.0):
    values = {}
    for task in ("24h", "168h"):
        for metric in MODULE.ALL_METRICS:
            expected = 1.0
            error = nse_error if metric == "NSE" else relative_error
            values[f"{task}_{metric}"] = expected + error
            values[f"{task}_{metric}_paper"] = expected
    return values


def test_acceptance_requires_all_eight_total_metrics() -> None:
    accepted = MODULE.acceptance_fields(
        _scores(relative_error=0.049, nse_error=0.009),
        error_relative_tolerance=0.05,
        nse_absolute_tolerance=0.01,
    )
    assert accepted["accepted_metrics"] == 8
    assert accepted["all_8_total_metrics_accepted"] is True

    rejected = MODULE.acceptance_fields(
        _scores(relative_error=0.051, nse_error=0.009),
        error_relative_tolerance=0.05,
        nse_absolute_tolerance=0.01,
    )
    assert rejected["accepted_metrics"] == 2
    assert rejected["all_8_total_metrics_accepted"] is False


def test_runner_is_one_seed_and_covers_all_six_models() -> None:
    runner = (ROOT / "scripts/train/run_que_targeted_reproduction_gpu6.sh").read_text(
        encoding="utf-8"
    )
    assert "EXTRA_SEEDS" not in runner
    assert "robustness" not in runner
    for model in (
        "gru",
        "lstm",
        "msnet",
        "mscmnet_m",
        "mscmnet_wm",
        "mscmnet_w",
    ):
        assert f"|{model}|" in runner
    assert "ALREADY_ACCEPTED" in runner
    assert "--learning-rate-scale" in runner
    assert "--best-epoch-scale" in runner


def test_lstm_s3_2_nonzero_weight_decays_are_preserved() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/model/mscmnet_baselines.yaml").read_text(encoding="utf-8")
    )
    assert config["models"]["lstm"]["weight_decays"] == [
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.01,
        0.0001,
        0.001,
        0.001,
        0.0001,
        0.0001,
    ]
