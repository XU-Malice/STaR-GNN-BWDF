from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train/run_que_protocol_audit.py"
SPEC = importlib.util.spec_from_file_location("que_protocol_runner", SCRIPT)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def origin_fixture():
    return pd.date_range("2023-01-13T00:00:00+01:00", periods=46, freq="D")


def evaluation_fixture():
    arrays = np.ones((46, 168, 10), dtype=np.float32)
    return {
        "forecast_starts": [str(x) for x in origin_fixture()],
        "truths": {f"{horizon}h": {"array_sha256": RUNNER.array_digest(arrays[:, :horizon])} for horizon in (24, 168)},
    }


def complete_run(tmp_path, model="msnet"):
    case = next(case for case in RUNNER.make_cases() if case["model"] == model)
    run = tmp_path / model
    run.mkdir()
    config = {
        "training": {key: case[key] for key in ("normalization", "optimizer", "batch_size", "loss", "learning_rate_scale", "best_epoch_scale")},
        "seed": case["seed"], "train_stride_hours": 24,
        "max_epochs_override": None, "max_train_batches": None,
        "cam": {"attention_update": "replace", "attention_scaling": "none", "temporal_layout": "per_day_vectors"},
        "model": {"best_epoch": 2, "best_epochs": [2] * 10},
    }
    if model.startswith("mscmnet_"):
        config["model"].update(correction_mode="direct", zero_init_correction=False)
    RUNNER.atomic_json(run / "request_signature.json", {"signature": "sig", "model_config": config["model"], "evaluation": evaluation_fixture()})
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    checkpoints = [f"checkpoint_{index}.pt" for index in range(10 if model in ("gru", "lstm") else 1)]
    for name in checkpoints:
        (run / name).write_bytes(b"fixture-checkpoint")
    RUNNER.atomic_json(run / "status.json", {"status": "completed", "model": model, "seed": case["seed"], "single_frozen_checkpoint_for_24h_and_168h": True, "checkpoint_files": checkpoints})
    arrays = np.ones((46, 168, 10), dtype=np.float32)
    np.savez_compressed(run / "predictions_common46.npz", y_true_24h=arrays[:, :24], y_pred_24h=arrays[:, :24], y_true_168h=arrays, y_pred_168h=arrays, dma_letters=np.asarray(list("ABCDEFGHIJ")), forecast_starts=np.asarray([str(x) for x in origin_fixture()]))
    with (run / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "series", "metric", "value"])
        for task, series, metric in itertools.product(("24h", "168h"), [*"ABCDEFGHIJ", "total"], RUNNER.METRICS):
            writer.writerow([task, series, metric, 1.0])
    if model in ("gru", "lstm"):
        (run / "loss_curve.csv").write_text("dma,epoch,train_loss\n" + "".join(f"{letter},{epoch},0.1\n" for letter in "ABCDEFGHIJ" for epoch in (1, 2)))
    else:
        (run / "loss_curve.csv").write_text("epoch,train_loss\n1,0.1\n2,0.1\n")
    return run, case


def test_fixed_plan_has_24_cases_one_seed_no_target_adaptation():
    cases = RUNNER.make_cases()
    assert len(cases) == 24
    assert len({case["case"] for case in cases}) == 24
    assert {case["seed"] for case in cases} == {20240604}
    assert {case["model"] for case in cases} == set(RUNNER.MODELS)
    assert all(case["correction_mode"] == "direct" and case["fc2_share_supervision_weight"] == 0 for case in cases)
    assert all(case["best_epoch_scale"] == case["learning_rate_scale"] == 1 for case in cases)


def test_end_audit_requires_recomputed_metric_agreement_and_all_models():
    summary = {
        "status": "completed", "valid_sources": 24, "invalid_sources": 0,
        "stored_metric_roundoff_exceeded": 0, "first_day_discrepancy_sources": 0,
        "models_with_raw_predictions": list(RUNNER.MODELS),
        "truth_groups_per_task": {"24h": 1, "168h": 1},
        "truth_origin_group_mismatch": False,
        "pooled_total_blocked_models_even_with_rounding": ["gru", "lstm"],
    }
    assert RUNNER.strict_audit_passes(summary, 24)
    for key, bad in (("stored_metric_roundoff_exceeded", 1), ("valid_sources", 23),
                     ("models_with_raw_predictions", ["gru"]),
                     ("first_day_discrepancy_sources", 1)):
        assert not RUNNER.strict_audit_passes({**summary, key: bad}, 24)


def test_dry_run_needs_no_gpu_or_optional_imports():
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--dry-run"], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["case_count"] == 24


def test_commands_do_not_override_published_epochs_or_decay(tmp_path):
    args = argparse.Namespace(data_dir=tmp_path / "data", device="cuda:0")
    for case in RUNNER.make_cases():
        command = RUNNER.command_for(case, args, tmp_path / "results")
        assert "--max-epochs" not in command
        assert "--max-train-batches" not in command
        assert "--recurrent-weight-decay" not in command
        assert "--joint-weight-decay" not in command
        assert "--zero-init-correction" not in command
        if case["model"] not in ("gru", "lstm"):
            assert command[command.index("--cam-attention-scaling") + 1] == "none"


@pytest.mark.parametrize("model", RUNNER.MODELS)
def test_complete_case_validates_actual_artifacts(tmp_path, model):
    run, case = complete_run(tmp_path, model)
    assert RUNNER.validate_case(run, case, "sig") == (True, "validated_artifacts")


@pytest.mark.parametrize("model", ("gru", "lstm"))
def test_recurrent_builder_origins_match_audit_without_rewriting_evidence(tmp_path, model):
    from dma_wdf.data.mscmnet_dataset import build_independent_temporal_samples

    run, case = complete_run(tmp_path, model)
    bounds = {
        "train_start": pd.Timestamp("2021-01-01T00:00:00+01:00"),
        "train_end": pd.Timestamp("2022-12-15T23:00:00+01:00"),
        "test_start": pd.Timestamp("2022-12-16T00:00:00+01:00"),
        "test_end": pd.Timestamp("2023-03-05T23:00:00+01:00"),
    }
    demand = pd.DataFrame(
        {"DMA A": 1.0},
        index=pd.date_range(bounds["train_start"], bounds["test_end"], freq="h"),
    )
    samples = build_independent_temporal_samples(
        demand=demand, bounds=bounds, dma_column="DMA A", input_weeks=1,
        train_stride_hours=24,
    )
    with np.load(run / "predictions_common46.npz") as source:
        arrays = dict(source)
    arrays["forecast_starts"] = samples["test_forecast_start"]
    assert "T" in arrays["forecast_starts"][0]
    assert " " in evaluation_fixture()["forecast_starts"][0]
    np.savez_compressed(run / "predictions_common46.npz", **arrays)
    before = (run / "predictions_common46.npz").read_bytes()
    assert RUNNER.validate_case(run, case, "sig") == (True, "validated_artifacts")
    assert (run / "predictions_common46.npz").read_bytes() == before


def test_actual_one_hour_origin_shift_is_still_rejected(tmp_path):
    run, case = complete_run(tmp_path)
    with np.load(run / "predictions_common46.npz") as source:
        arrays = dict(source)
    arrays["forecast_starts"] = np.asarray([
        value.isoformat() for value in origin_fixture() + pd.Timedelta(hours=1)
    ])
    np.savez_compressed(run / "predictions_common46.npz", **arrays)
    assert RUNNER.validate_case(run, case, "sig")[1] == "origins_do_not_match_audited_data"


def test_signature_mismatch_disallows_resume(tmp_path):
    run, case = complete_run(tmp_path)
    assert RUNNER.validate_case(run, case, "different")[0] is False


def test_missing_checkpoint_disallows_resume(tmp_path):
    run, case = complete_run(tmp_path)
    (run / "checkpoint_0.pt").unlink()
    assert RUNNER.validate_case(run, case, "sig")[1] == "missing_or_invalid_checkpoint"


def test_metric_duplicate_cannot_hide_missing_key(tmp_path):
    run, case = complete_run(tmp_path)
    content = (run / "metrics.csv").read_text().splitlines()
    content[-1] = content[-2]
    (run / "metrics.csv").write_text("\n".join(content) + "\n")
    assert RUNNER.validate_case(run, case, "sig")[1] == "invalid_metric_keyset_or_values"


def test_wrong_resolved_config_disallows_resume(tmp_path):
    run, case = complete_run(tmp_path)
    config = yaml.safe_load((run / "resolved_config.yaml").read_text())
    config["training"]["optimizer"] = "other"
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    assert RUNNER.validate_case(run, case, "sig")[1] == "resolved_config_mismatch:optimizer"


def test_truncated_epochs_disallow_resume(tmp_path):
    run, case = complete_run(tmp_path)
    (run / "loss_curve.csv").write_text("epoch,train_loss\n1,0.1\n")
    assert RUNNER.validate_case(run, case, "sig")[1] == "loss_curve_wrong_published_epochs"


def test_first_recursive_day_must_equal_direct_prediction(tmp_path):
    run, case = complete_run(tmp_path)
    with np.load(run / "predictions_common46.npz") as saved:
        arrays = dict(saved)
    arrays["y_pred_168h"] = arrays["y_pred_168h"].copy()
    arrays["y_pred_168h"][0, 0, 0] += 1
    np.savez_compressed(run / "predictions_common46.npz", **arrays)
    assert RUNNER.validate_case(run, case, "sig")[1] == "recursive_first_day_mismatch"


def test_matching_shape_but_wrong_truth_is_not_reusable(tmp_path):
    run, case = complete_run(tmp_path)
    with np.load(run / "predictions_common46.npz") as saved:
        arrays = dict(saved)
    arrays["y_true_24h"] = arrays["y_true_24h"].copy() + 1
    arrays["y_true_168h"] = arrays["y_true_168h"].copy() + 1
    np.savez_compressed(run / "predictions_common46.npz", **arrays)
    assert RUNNER.validate_case(run, case, "sig")[1] == "truth_does_not_match_audited_data"


def test_46_unique_but_wrong_origins_are_not_reusable(tmp_path):
    run, case = complete_run(tmp_path)
    with np.load(run / "predictions_common46.npz") as saved:
        arrays = dict(saved)
    arrays["forecast_starts"] = np.asarray([f"wrong-{x}" for x in range(46)])
    np.savez_compressed(run / "predictions_common46.npz", **arrays)
    assert RUNNER.validate_case(run, case, "sig")[1] == "origins_do_not_match_audited_data"


def test_source_and_data_signature_detect_edits(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("x = 1\n")
    data = tmp_path / "data"
    data.mkdir()
    (data / "sample.csv").write_text("1\n")
    original = RUNNER.fingerprints(tmp_path, data)
    (tmp_path / "src/a.py").write_text("x = 2\n")
    modified = RUNNER.fingerprints(tmp_path, data)
    assert original["source_sha256"] != modified["source_sha256"]
    assert original["data_sha256"] == modified["data_sha256"]
    (data / "sample.csv").write_text("2\n")
    assert modified["data_sha256"] != RUNNER.fingerprints(tmp_path, data)["data_sha256"]


def test_child_failure_is_reported_not_passed(tmp_path):
    supervisor = RUNNER.ChildSupervisor(tmp_path, dict(os.environ))
    assert supervisor.run([sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "child.log") == 7
    assert supervisor.child is None


def test_children_have_own_process_group(tmp_path):
    supervisor = RUNNER.ChildSupervisor(tmp_path, dict(os.environ))
    supervisor.run([sys.executable, "-c", "import os; print(str(os.getpid()) + ',' + str(os.getpgrp()))"], tmp_path / "child.log")
    pid, group = map(int, (tmp_path / "child.log").read_text().splitlines()[-1].split(","))
    assert pid == group
    assert group != os.getpgrp()


def test_term_stops_owned_training_child(tmp_path):
    marker = tmp_path / "child.pid"
    child_code = f"import os,time; open({str(marker)!r},'w').write(str(os.getpid())); time.sleep(30)"
    launcher_code = (
        "import importlib.util,os,pathlib,signal,sys; "
        f"s=importlib.util.spec_from_file_location('runner',{str(SCRIPT)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"r=m.ChildSupervisor(pathlib.Path({str(tmp_path)!r}),dict(os.environ)); "
        "signal.signal(signal.SIGTERM,r.signal_handler); "
        "\ntry:\n"
        f" r.run([sys.executable,'-c',{child_code!r}],pathlib.Path({str(tmp_path / 'log')!r}))\n"
        "except m.InterruptedRun as e:\n sys.exit(128+e.signum)\n"
    )
    launcher = subprocess.Popen([sys.executable, "-c", launcher_code], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        child = int(marker.read_text())
        launcher.send_signal(signal.SIGTERM)
        assert launcher.wait(timeout=15) == 143
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
    finally:
        if launcher.poll() is None:
            launcher.terminate()
            launcher.wait(timeout=15)


def test_global_gpu_lock_path_is_not_run_tag_scoped():
    source = SCRIPT.read_text()
    assert 'f"que_gpu_{args.gpu_id' in source
    assert "LOCK_EX | fcntl.LOCK_NB" in source


def test_lock_conflict_does_not_overwrite_active_status(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, "PROJECT_ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    status = tmp_path / "results/active/queue_status.json"
    RUNNER.atomic_json(status, {"status": "running", "owner": "other"})
    original = status.read_bytes()
    with (tmp_path / "logs/que_gpu_cpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert RUNNER.main(["--run-tag", "active", "--device", "cpu", "--audit-only"]) == 1
    assert status.read_bytes() == original


def fake_project(tmp_path, monkeypatch):
    monkeypatch.setattr(RUNNER, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(RUNNER, "PREFLIGHT_TESTS", ())
    case = next(case for case in RUNNER.make_cases() if case["model"] == "msnet")
    monkeypatch.setattr(RUNNER, "make_cases", lambda: [case])
    monkeypatch.setattr(RUNNER, "MODELS", ("msnet",))
    data = tmp_path / "data"
    data.mkdir()
    (data / "data.csv").write_text("fixture\n")
    configs = tmp_path / "configs/model"
    configs.mkdir(parents=True)
    (configs / "mscmnet_baselines.yaml").write_text(yaml.safe_dump({"models": {"msnet": {"best_epoch": 2, "best_epochs": [2] * 10}}}))
    paper = tmp_path / "paper.yaml"
    paper.write_text("{}\n")
    RUNNER.atomic_json(tmp_path / "results/fixture_queue/audit_data_protocol/paper_data_statistics.json", {"common_evaluation": evaluation_fixture()})
    RUNNER.atomic_json(tmp_path / "results/fixture_queue/audit_new/audit_summary.json", {
        "status": "completed", "valid_sources": 1, "invalid_sources": 0,
        "stored_metric_roundoff_exceeded": 0, "first_day_discrepancy_sources": 0,
        "models_with_raw_predictions": ["msnet"],
        "truth_groups_per_task": {"24h": 1, "168h": 1},
        "truth_origin_group_mismatch": False,
    })
    argv = ["--run-tag", "fixture_queue", "--device", "cpu", "--data-dir", str(data), "--paper-config", str(paper)]
    return case, argv


def test_preflight_failure_blocks_all_training(tmp_path, monkeypatch):
    _case, argv = fake_project(tmp_path, monkeypatch)
    calls = []

    def fake_run(self, command, log_path):
        calls.append(command)
        return 1 if "pytest" in command else 0

    monkeypatch.setattr(RUNNER.ChildSupervisor, "run", fake_run)
    assert RUNNER.main(argv) == 1
    assert all("scripts/train/train_temporal_baselines.py" not in command for command in calls)
    status = json.loads((tmp_path / "results/fixture_queue/queue_status.json").read_text())
    assert status["status"] == "failed"
    assert status["technical_success"] is False


def test_training_failure_exits_nonzero_and_does_not_claim_success(tmp_path, monkeypatch):
    _case, argv = fake_project(tmp_path, monkeypatch)

    def fake_run(self, command, log_path):
        return 7 if "scripts/train/train_temporal_baselines.py" in command else 0

    monkeypatch.setattr(RUNNER.ChildSupervisor, "run", fake_run)
    assert RUNNER.main(argv) == 1
    status = json.loads((tmp_path / "results/fixture_queue/queue_status.json").read_text())
    assert status["failed_cases"] == 1
    assert status["cases"][0]["exit_code"] == 7
    assert status["reproduction_claim"] == "not_established"


def test_complete_same_signature_is_reused_without_retraining(tmp_path, monkeypatch):
    case, argv = fake_project(tmp_path, monkeypatch)
    trained = []

    def fake_run(self, command, log_path):
        if "scripts/train/train_temporal_baselines.py" in command:
            trained.append(command)
            output = Path(command[command.index("--output-root") + 1])
            output.mkdir(parents=True, exist_ok=True)
            run, _ = complete_run(output)
            destination = run / f"seed_{case['seed']}"
            destination.mkdir()
            for child in list(run.iterdir()):
                if child != destination:
                    child.rename(destination / child.name)
        return 0

    monkeypatch.setattr(RUNNER.ChildSupervisor, "run", fake_run)
    assert RUNNER.main(argv) == 0
    assert RUNNER.main(argv) == 0
    assert len(trained) == 1
    status = json.loads((tmp_path / "results/fixture_queue/queue_status.json").read_text())
    assert status["cases"][0]["technical_status"] == "PASS(existing)"
    assert status["reproduction_claim"] == "not_established"


def test_bundle_failure_is_persisted_as_nonzero(tmp_path, monkeypatch):
    _case, argv = fake_project(tmp_path, monkeypatch)
    monkeypatch.setattr(RUNNER.ChildSupervisor, "run", lambda self, command, log_path: 0)

    def broken_bundle(*_args):
        raise OSError("disk full fixture")

    monkeypatch.setattr(RUNNER, "make_bundle", broken_bundle)
    assert RUNNER.main([*argv, "--audit-only"]) == 1
    status = json.loads((tmp_path / "results/fixture_queue/queue_status.json").read_text())
    assert status["exit_code"] == 1
    assert status["technical_success"] is False
    assert status["bundle_status"] == "failed"


def test_bundle_includes_hashed_historical_raw_evidence(tmp_path):
    root = tmp_path / "project"
    historical = root / "results/historical"
    historical.mkdir(parents=True)
    run, _ = complete_run(historical)
    new_root = root / "results/new"
    logs = root / "logs/new"
    logs.mkdir(parents=True)
    source = {"source_id": "0123456789abcdef", "source_path": str(run / "predictions_common46.npz"), "npz_sha256": RUNNER.file_digest(run / "predictions_common46.npz"), "status_sha256": RUNNER.file_digest(run / "status.json"), "config_sha256": RUNNER.file_digest(run / "resolved_config.yaml")}
    RUNNER.atomic_json(new_root / "audit_existing/provenance.json", [source])
    bundle = RUNNER.make_bundle(root, new_root, logs)
    with tarfile.open(bundle) as archive:
        names = archive.getnames()
        assert "evidence/0123456789abcdef/predictions_common46.npz" in names
        assert not any(name.endswith(".pt") for name in names)
        assert "evidence/index.json" in names
    (run / "status.json").write_text("{}")
    with pytest.raises(ValueError, match="changed before packaging"):
        RUNNER.make_bundle(root, new_root, logs)
