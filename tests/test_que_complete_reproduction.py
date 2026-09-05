from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce/summarize_que_complete_reproduction.py"
SPEC = importlib.util.spec_from_file_location("que_complete_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_paper_metrics_have_zero_match_score() -> None:
    reference = {
        task: {
            letter: {metric: 1.0 for metric in MODULE.METRICS}
            for letter in [*"ABCDEFGHIJ", "total"]
        }
        for task in ("24h", "168h")
    }
    rows = [
        {"task": task, "series": series, "metric": metric, "value": 1.0}
        for task in ("24h", "168h")
        for series in [*"ABCDEFGHIJ", "total"]
        for metric in MODULE.METRICS
    ]
    score = MODULE.score_run(pd.DataFrame(rows), reference)
    assert score["aggregate_paper_error"] == 0.0
    assert score["dma_mae_rmse_relative_error"] == 0.0
    assert np.isclose(score["dma_mae_rmse_correlation"], 0.0)
    # Constant DMA vectors have undefined correlation and receive the explicit
    # neutral fallback penalty rather than producing NaN.
    assert np.isclose(score["paper_match_score"], 0.05)


def test_paper_metric_file_covers_all_six_models() -> None:
    import yaml

    paper = yaml.safe_load(
        (ROOT / "configs/evaluation/mscmnet_paper_metrics.yaml").read_text(
            encoding="utf-8"
        )
    )["tasks"]
    expected = set(MODULE.MODEL_DISPLAY.values())
    for task in ("24h", "168h"):
        assert expected.issubset(paper[task])
        for model in expected:
            assert set([*"ABCDEFGHIJ", "total"]).issubset(paper[task][model])


def test_gpu_runner_does_not_expand_a_local_before_assignment() -> None:
    runner = (
        ROOT / "scripts/train/run_que_complete_reproduction_gpu6.sh"
    ).read_text(encoding="utf-8")
    assert 'local output_root="${RESULT_ROOT}' not in runner
    assert 'output_root="${RESULT_ROOT}/${case_name}"' in runner
    assert 'run="${output_root}/${model}/seed_${seed}"' in runner
