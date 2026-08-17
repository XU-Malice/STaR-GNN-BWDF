"""DCRNN/Base 冻结工件唯一化的隔离回归测试。"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("24h", "168h")
STAR_MODELS = ("Base", "State", "FA-DPR", "Full")


def _write_run(run: Path, *, indices: bool) -> None:
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True)
    (run / "checkpoint_best.pt").write_bytes(b"checkpoint")
    (evaluation / "test_summary.json").write_text(
        '{"status":"completed"}\n', encoding="utf-8"
    )
    for name in (
        "metrics_aggregate_total_common_46.csv",
        "metrics_common_46.csv",
    ):
        (evaluation / name).write_text(
            "entity,MAE,MAPE,RMSE,NSE\ntotal,1,0.01,2,0.9\n",
            encoding="utf-8",
        )
    arrays: dict[str, np.ndarray] = {
        "target": np.zeros((46, 24, 10), dtype=np.float32),
        "prediction": np.zeros((46, 24, 10), dtype=np.float32),
    }
    if indices:
        common = np.arange(46, dtype=np.int64)
        arrays.update(
            operational_indices=common,
            strict_within_test_indices=common,
            common_46_indices=common,
        )
    np.savez_compressed(evaluation / "predictions.npz", **arrays)


def test_consolidation_keeps_only_canonical_dcrnn(tmp_path: Path) -> None:
    release = tmp_path / "frozen_v1"
    release.mkdir()
    source_manifest = json.loads(
        (ROOT / "results/paper/frozen_v1/MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    (release / "MANIFEST.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    for model in STAR_MODELS:
        for task in TASKS:
            _write_run(
                release / "models/star_gnn" / model / task / "seed_0",
                indices=False,
            )
    for task in TASKS:
        _write_run(
            release / "models/baselines/stgcn" / task / "seed_0",
            indices=False,
        )
        _write_run(
            release / "models/baselines/dcrnn" / task / "seed_0",
            indices=True,
        )

    rows = [
        {"task": task, "model": model, "MAE": "1"}
        for task in TASKS
        for model in ("Base", "State", "FA-DPR", "Full", "STGCN", "DCRNN")
    ]
    with (release / "results_common46.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    subprocess.run(
        [
            sys.executable,
            "scripts/reproduce/consolidate_dcrnn_base_release.py",
            "--release",
            str(release),
        ],
        cwd=ROOT,
        check=True,
    )

    assert not (release / "models/baselines/dcrnn").exists()
    assert len(list(release.rglob("checkpoint_best.pt"))) == 10
    with (release / "results_common46.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 10
    for model in STAR_MODELS:
        for task in TASKS:
            prediction = (
                release
                / "models/star_gnn"
                / model
                / task
                / "seed_0/evaluation/predictions.npz"
            )
            with np.load(prediction, allow_pickle=False) as payload:
                assert payload["common_46_indices"].shape == (46,)
    assert (release / "DCRNN_BASE_UNIFICATION.json").is_file()
    assert (release / "CHECKSUMS.sha256").is_file()
