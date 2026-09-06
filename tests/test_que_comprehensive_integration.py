"""Real small CPU fits protect the artifact contract used by the long queue.

These are functional tests, not paper experiments. They deliberately use tiny
hidden states and one epoch, while retaining all ten DMAs, 46 origins,
the actual window builders and the full seven-day recursive prediction path.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location(
        "que_integration_trainer", ROOT / "scripts/train/train_temporal_baselines.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield module
    torch.set_num_threads(previous)


def _inputs():
    # 35 train days (7 joint samples after 28-day history), 80 test days
    # (46 common origins after 28-day history and 7-day target).
    index = pd.date_range("2021-01-01", periods=115 * 24, freq="h", tz="UTC")
    time = np.arange(len(index), dtype=float)
    cfg = yaml.safe_load((ROOT / "configs/model/mscmnet_baselines.yaml").read_text())
    demand = pd.DataFrame({name: 20 + 5 * j + np.sin(time / 24 + j) + time * .002
                           for j, name in enumerate(cfg["protocol"]["dma_columns"])}, index=index)
    weather = pd.DataFrame({name: 4 + j + np.cos(time / (13 + j))
                            for j, name in enumerate(cfg["protocol"]["weather_columns"])}, index=index)
    temporal = pd.DataFrame({name: (time.astype(int) // (j + 1)) % (j + 2)
                             for j, name in enumerate(cfg["protocol"]["temporal_columns"])}, index=index)
    bounds = {"train_start": index[0], "train_end": index[35 * 24 - 1],
              "test_start": index[35 * 24], "test_end": index[-1]}
    cfg["protocol"]["expected_train_samples_joint"] = 7
    cfg["cam"]["channel_sizes"] = [1, 1, 1]
    cfg["training"].update(batch_size=2, deterministic_algorithms=True,
                           best_epoch_scale=1.0, learning_rate_scale=1.0)
    return cfg, {"demand": demand, "weather": weather, "temporal": temporal}, bounds


@pytest.mark.parametrize("model", ["gru", "lstm", "msnet", "mscmnet_m", "mscmnet_wm", "mscmnet_w"])
def test_real_fit_writes_frozen_full_horizons_and_scaler_evidence(trainer, tmp_path, model):
    cfg, frames, bounds = _inputs()
    model_cfg = cfg["models"][model]
    model_cfg["input_weeks"] = [1] * 10
    if model in ("gru", "lstm"):
        model_cfg["hidden_sizes"] = [[2]] * 10
        model_cfg["layers"] = [1] * 10
        model_cfg["best_epochs"] = [1] * 10
        if model == "lstm":
            cfg["training"].update(recurrent_layout="daily_vectors", scaler_fit_scope="train_rows", normalization="minmax")
    else:
        model_cfg.update(hidden_sizes=[2] * 10, lstm_layers=[1] * 10, best_epoch=1)
        if "fc1" in model_cfg:
            model_cfg["fc1"].update(nodes=3, dropout=0.0)
            model_cfg.update(correction_mode="residual", zero_init_correction=True)
        if "fc2" in model_cfg:
            model_cfg["fc2"].update(hidden_size=2, lstm_layers=1, nodes=3, dropout=0.0, share_supervision_weight=.05)
        if model in ("mscmnet_m", "mscmnet_wm"):
            model_cfg["correction_layout"] = "hourwise_shared"
            cfg["training"].update(scaler_fit_scope="train_rows", demand_scaling="shared")
        if model == "mscmnet_m":
            cfg["training"]["normalization"] = "minmax"

    result = trainer.run_one_model(
        canonical=model, config=cfg, frames=frames, bounds=bounds,
        data_audit={"synthetic_fixture": True}, device=torch.device("cpu"),
        output_root=tmp_path, seed=20240604, overwrite=False,
        max_epochs_override=None, max_train_batches=None, train_stride_hours=24,
        literature_config=None, preflight={"synthetic_fixture": True},
    )
    run = tmp_path / model / "seed_20240604"
    assert result["status"] == "completed"
    assert result["paper_reproduction_verified"] is False
    # This legacy flag checks epoch/stride/loss only, not agreement with S3.
    assert result["formal_protocol_scope"] == "legacy_epoch_stride_loss_check_only"
    assert len(result["checkpoint_files"]) == (10 if model in ("gru", "lstm") else 1)
    for checkpoint in result["checkpoint_files"]:
        assert (run / checkpoint).stat().st_size > 0
    scalers = json.loads((run / "scaler_audit.json").read_text())
    assert scalers["scaler_fit_scope"] == cfg["training"]["scaler_fit_scope"]
    assert scalers["demand_scaling"] == cfg["training"]["demand_scaling"]
    with np.load(run / "predictions_common46.npz", allow_pickle=False) as arrays:
        from dma_wdf.data.reproduction_metrics import validate_prediction_bundle
        validate_prediction_bundle(arrays)
        np.testing.assert_array_equal(arrays["y_pred_24h"], arrays["y_pred_168h"][:, :24])
        assert arrays["y_pred_168h"].shape == (46, 168, 10)
    metrics = pd.read_csv(run / "metrics.csv")
    assert len(metrics) == 88
    assert np.isfinite(metrics["value"]).all()
    loss = pd.read_csv(run / "loss_curve.csv")
    assert len(loss) == (10 if model in ("gru", "lstm") else 1)
    assert np.isfinite(loss["train_loss"]).all()

    # Feed real trainer artifacts to the actual long-queue validator.
    spec = importlib.util.spec_from_file_location(
        "que_integration_runner", ROOT / "scripts/train/run_que_comprehensive_reconstruction.py"
    )
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    options = cfg["training"]
    case = runner.make_case(
        model, normalization=options["normalization"], optimizer=options["optimizer"],
        batch_size=options["batch_size"], recurrent_layout=options["recurrent_layout"],
        scaler_fit_scope=options["scaler_fit_scope"], demand_scaling=options["demand_scaling"],
        cam_channel_sizes=cfg["cam"]["channel_sizes"],
        correction_layout=model_cfg.get("correction_layout", "global_flat"),
        correction_mode=model_cfg.get("correction_mode", "direct"),
        zero_init_correction=model_cfg.get("zero_init_correction", False),
        share_weight=model_cfg.get("fc2", {}).get("share_supervision_weight", 0.0),
    )
    with np.load(run / "predictions_common46.npz", allow_pickle=False) as arrays:
        evaluation = {"forecast_starts": arrays["forecast_starts"].tolist(),
                      "truths": {f"{h}h": {"array_sha256": runner.life.array_digest(arrays[f"y_true_{h}h"])}
                                 for h in (24, 168)}}
    request = {"signature": "synthetic-functionality-only", "case": case,
               "model_config": model_cfg, "evaluation": evaluation}
    runner.life.atomic_json(run / "request_signature.json", request)
    valid, reason = runner.validate_case(run, case, request, require_receipt=False)
    assert valid, reason
    runner.life.atomic_json(run / "completion_receipt.json", {
        "request_sha256": runner.life.digest(request), "files": runner.evidence_hashes(run, result),
    })
    valid, reason = runner.validate_case(run, case, request)
    assert valid, reason
