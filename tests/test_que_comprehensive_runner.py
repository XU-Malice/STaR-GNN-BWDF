from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("que_comprehensive_runner", ROOT / "scripts/train/run_que_comprehensive_reconstruction.py")
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def make_record(case, value=1.0):
    return {"case": case["case"], "model": case["model"], "stage": case["stage"], "settings": case, "technical_status": "PASS", "scores": {
        mode: {"balanced_distance": value, "q95": value, "worst_ratio": value, "total_mean_ratio": value,
               "dma_mean_ratio": value, "total8_passed": 8, "dma80_passed": 80, "all88_within_tolerance": True}
        for mode in ("pooled", "origin_mean")}}


def arrays_fixture():
    t = np.arange(46 * 168 * 10, dtype=np.float32).reshape(46, 168, 10)
    truth = 12 + np.sin(t / 97) + .05 * t / t.max()
    pred = truth + .01 * np.cos(t / 23)
    starts = [x.isoformat() for x in pd.date_range("2023-01-13T00:00:00+01:00", periods=46, freq="D")]
    return {"y_true_24h": truth[:, :24], "y_true_168h": truth, "y_pred_24h": pred[:, :24], "y_pred_168h": pred,
            "forecast_starts": np.asarray(starts), "dma_letters": np.asarray(list("ABCDEFGHIJ"))}


def evaluation_fixture():
    arrays = arrays_fixture()
    return {"forecast_starts": [x.replace("T", " ") for x in arrays["forecast_starts"].tolist()],
            "truths": {f"{h}h": {"array_sha256": RUNNER.life.array_digest(arrays[f"y_true_{h}h"])} for h in (24, 168)}}


def model_fixture(case):
    model = {"best_epoch": 2, "best_epochs": [2] * 10}
    if case["model"] not in ("gru", "lstm"):
        model["correction_layout"] = case["correction_layout"]
    if case["model"].startswith("mscmnet_"):
        model.update(correction_mode=case["correction_mode"], zero_init_correction=case["zero_init_correction"])
    if case["model"] in ("mscmnet_w", "mscmnet_wm"):
        model["fc2"] = {"share_supervision_weight": case["share_weight"]}
    return model


def build_artifacts(run: Path, case, request=None, receipt=True):
    run.mkdir(parents=True, exist_ok=True)
    request = request or {"signature": "test", "case": case, "model_config": model_fixture(case), "evaluation": evaluation_fixture()}
    config = {"training": {key: case[key] for key in ("normalization", "optimizer", "batch_size", "recurrent_layout", "scaler_fit_scope", "demand_scaling", "loss", "best_epoch_scale", "learning_rate_scale")},
              "model": request["model_config"], "seed": RUNNER.SEED, "train_stride_hours": 24, "max_epochs_override": case["max_epochs"], "max_train_batches": None,
              "cam": {"attention_update": "replace", "attention_scaling": case["cam_attention_scaling"], "temporal_layout": case["cam_temporal_layout"], "channel_sizes": case["cam_channel_sizes"]}}
    if case["joint_weight_decay"] is not None:
        config["training"]["joint_weight_decay_override"] = case["joint_weight_decay"]
    if case["independent_weight_decay"] is not None:
        config["training"]["independent_weight_decay_override"] = case["independent_weight_decay"]
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    count = 10 if case["model"] in ("gru", "lstm") else 1
    names = [f"checkpoint_{i}.pt" for i in range(count)]
    for name in names:
        (run / name).write_bytes(b"test-only-checkpoint")
    status = {"status": "completed", "model": case["model"], "seed": RUNNER.SEED, "single_frozen_checkpoint_for_24h_and_168h": True, "checkpoint_files": names}
    RUNNER.life.atomic_json(run / "status.json", status)
    RUNNER.life.atomic_json(run / "request_signature.json", request)
    RUNNER.life.atomic_json(run / "scaler_audit.json", {"scaler_fit_scope": case["scaler_fit_scope"], "demand_scaling": case["demand_scaling"], "normalization": case["normalization"], "test_values_used_for_fit": False, "scalers": [{"mean": [1.0], "std": [2.0]}, {"mean": [], "std": []}]})
    arrays = arrays_fixture()
    np.savez_compressed(run / "predictions_common46.npz", **arrays)
    with (run / "metrics.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=["task", "series", "metric", "value"], extrasaction="ignore")
        w.writeheader(); w.writerows(RUNNER.metric_rows(arrays, "pooled"))
    epochs = RUNNER.actual_epochs(2, case)
    losses = [{"epoch": epoch, "train_loss": .1} for epoch in range(1, epochs + 1)]
    if count == 10:
        losses = [{**r, "dma": letter} for letter in "ABCDEFGHIJ" for r in losses]
    with (run / "loss_curve.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(losses[0]))
        w.writeheader(); w.writerows(losses)
    if receipt:
        RUNNER.life.atomic_json(run / "completion_receipt.json", {"request_sha256": RUNNER.life.digest(request), "files": RUNNER.evidence_hashes(run, status)})
    return request


def test_finite_three_stage_budget_single_seed_and_no_axis_search():
    a = RUNNER.stage_a_cases()
    assert len(a) == 95
    assert {c["recurrent_layout"] for c in a} == {"hourly"}
    assert {c["cam_attention_scaling"] for c in a} == {"sqrt_dim"}
    assert {c["best_epoch_scale"] for c in a} == {.35, 1.0}
    assert sum(c["correction_mode"] == "residual" for c in a) == 14
    assert sum(c["share_weight"] == .05 for c in a) == 4
    records = [make_record(case, i) for i, case in enumerate(a)]
    b, _ = RUNNER.adaptive_cases(records, a, "B")
    assert len(b) <= 168
    assert {c["learning_rate_scale"] for c in b} == {.3, 1., 3.}
    assert {c["best_epoch_scale"] for c in b} == {.35, .5, 1., 2., 4.}
    records += [make_record(case, i + .01) for i, case in enumerate(b)]
    c, _ = RUNNER.adaptive_cases(records, a + b, "C")
    assert len(c) <= 152
    all_cases = a + b + c
    assert len(all_cases) <= 415
    assert len({RUNNER.setting_key(case) for case in all_cases}) == len(all_cases)
    assert {case["seed"] for case in all_cases} == {20240604}
    assert all(case["best_epoch_scale"] == 1 for case in c if case["max_epochs"] is not None)
    assert {case["max_epochs"] for case in c if case["max_epochs"]} == {50, 100}


def test_whole_table_mode_rankings_and_diverse_parents():
    cases = [RUNNER.make_case("gru", optimizer="adam", normalization="zscore"), RUNNER.make_case("gru", optimizer="adamw", normalization="minmax"), RUNNER.make_case("gru", batch_size=4)]
    records = [make_record(c, i + 1) for i, c in enumerate(cases)]
    records[1]["scores"]["origin_mean"]["balanced_distance"] = .01
    assert [p["case"] for p in RUNNER.choose_diverse(records, "gru")] == [cases[0]["case"], cases[1]["case"]]
    records[0]["technical_status"] = "FAIL"
    assert cases[0]["case"] not in [p["case"] for p in RUNNER.choose_diverse(records, "gru")]


def test_dry_run_and_status_without_site_packages_or_cuda():
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/train/run_que_comprehensive_reconstruction.py"), "--dry-run", "--budget-hours", "0"], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["stage_a_count"] == 95 and plan["maximum_case_count"] == 415
    assert plan["time_budget_hours"] == 0
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/train/run_que_comprehensive_reconstruction.py"), "--status", "--run-tag", "unit_test_absent"], capture_output=True, text=True, check=True)
    assert "尚未生成状态" in result.stdout


@pytest.mark.parametrize("model", RUNNER.MODELS)
def test_complete_artifacts_validate_and_real_recomputed_scores(tmp_path, model):
    case = RUNNER.make_case(model, best_epoch_scale=.5)
    run = tmp_path / model
    request = build_artifacts(run, case)
    assert RUNNER.validate_case(run, case, request) == (True, "validated_artifacts")
    paper = yaml.safe_load((ROOT / "configs/evaluation/mscmnet_paper_metrics.yaml").read_text())
    scores = RUNNER.score_case(run, case, paper)
    assert set(scores) == {"pooled", "origin_mean"}
    assert all(np.isfinite(s["balanced_distance"]) for s in scores.values())
    assert len(list(csv.DictReader((run / "paper_gaps.tsv").open(), delimiter="\t"))) == 176
    assert RUNNER.validate_case(run, case, request)[0]  # scoring does not mutate hashed training evidence


@pytest.mark.parametrize("mutation", ["truth", "origins", "prediction_prefix", "scaler", "checkpoint", "metrics", "loss", "config", "request"])
def test_cache_never_accepts_altered_evidence(tmp_path, mutation):
    case = RUNNER.make_case("gru")
    run = tmp_path / "run"
    request = build_artifacts(run, case)
    if mutation in ("truth", "origins", "prediction_prefix"):
        arrays = arrays_fixture()
        if mutation == "truth":
            arrays["y_true_168h"] = arrays["y_true_168h"] + 1
            arrays["y_true_24h"] = arrays["y_true_168h"][:, :24]
        elif mutation == "origins":
            arrays["forecast_starts"][0] = "2023-01-13T01:00:00+01:00"
        else:
            arrays["y_pred_24h"] = arrays["y_pred_24h"] + 1
        np.savez_compressed(run / "predictions_common46.npz", **arrays)
    elif mutation == "scaler":
        (run / "scaler_audit.json").write_text('{"scaler_fit_scope":"windows","demand_scaling":"per_dma","scalers":[{"std":[-1]}]}')
    elif mutation == "checkpoint":
        (run / "checkpoint_0.pt").write_bytes(b"altered")
    elif mutation == "metrics":
        with (run / "metrics.csv").open("a") as stream: stream.write("24h,A,MAE,99\n")
    elif mutation == "loss":
        (run / "loss_curve.csv").write_text("dma,epoch,train_loss\nA,1,.1\n")
    elif mutation == "config":
        config = yaml.safe_load((run / "resolved_config.yaml").read_text()); config["training"]["recurrent_layout"] = "daily_vectors"
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    elif mutation == "request":
        (run / "request_signature.json").write_text("{}")
    assert not RUNNER.validate_case(run, case, request)[0]


def test_command_records_exploratory_overrides_and_epoch_policy(tmp_path):
    args = argparse.Namespace(device="cpu", data_dir=tmp_path)
    case = RUNNER.make_case("mscmnet_w", max_epochs=100, best_epoch_scale=1, learning_rate_scale=.3, correction_mode="residual", zero_init_correction=True, share_weight=.05, joint_weight_decay=0., loss="huber", correction_layout="hourwise_shared")
    cmd = RUNNER.command_for(case, args, tmp_path)
    for flag, value in (("--max-epochs", "100"), ("--best-epoch-scale", "1"), ("--learning-rate-scale", "0.3"), ("--loss", "huber"), ("--joint-weight-decay", "0.0"), ("--correction-layout", "hourwise_shared")):
        assert cmd[cmd.index(flag) + 1] == value
    assert "--zero-init-correction" in cmd and "--allow-cpu" in cmd
    index = cmd.index("--cam-channel-sizes")
    assert cmd[index + 1:index + 4] == ["16", "16", "1"]
    assert cmd[index + 4] == "--correction-layout"
    assert RUNNER.actual_epochs(77, RUNNER.make_case("gru", best_epoch_scale=.25)) == 19


def test_compact_bundle_preserves_prediction_evidence_excludes_checkpoints(tmp_path):
    root = tmp_path / "project"; result = root / "results/run"; log = root / "logs/run"
    result.mkdir(parents=True); log.mkdir(parents=True)
    (result / "predictions_common46.npz").write_bytes(b"raw")
    (result / "checkpoint.pt").write_bytes(b"weights")
    (result / "manifest.json").write_text("{}")
    bundle = RUNNER.make_bundle(root, result, log)
    with tarfile.open(bundle) as archive:
        names = archive.getnames()
    assert any(n.endswith("predictions_common46.npz") for n in names)
    assert not any(n.endswith(".pt") for n in names)


def test_status_keeps_generated_denominator_and_final_budget_distinct(tmp_path, capsys):
    result = tmp_path / "results"; result.mkdir()
    RUNNER.life.atomic_json(result / "queue_status.json", {"status": "running", "case_count": 88, "finished_cases": 10, "passed_cases": 9, "failed_cases": 1, "maximum_case_count": 400, "cases": []})
    assert RUNNER.print_status(result, tmp_path) == 0
    text = capsys.readouterr().out
    assert "10/88" in text and "78" in text and "400" in text and "不是最终任务数" in text


def mock_queue_environment(tmp_path, monkeypatch):
    root = tmp_path / "project"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "input.txt").write_text("fixed-data")
    (root / "dummy.txt").write_text("fixed-source")
    (root / "configs/model").mkdir(parents=True)
    (root / "configs/evaluation").mkdir(parents=True)
    bases = [RUNNER.make_case(model) for model in RUNNER.MODELS]
    variants_b = [RUNNER.make_case(model, stage="B", normalization="minmax") for model in RUNNER.MODELS]
    variants_c = [RUNNER.make_case(model, stage="C", learning_rate_scale=2.) for model in RUNNER.MODELS]
    registry = {c["case"]: c for c in bases + variants_b + variants_c}
    published = {c["model"]: model_fixture(c) for c in bases}
    (root / "configs/model/mscmnet_baselines.yaml").write_text(yaml.safe_dump({"models": published}))
    paper = root / "configs/evaluation/mscmnet_paper_metrics.yaml"
    paper.write_bytes((ROOT / "configs/evaluation/mscmnet_paper_metrics.yaml").read_bytes())
    monkeypatch.setattr(RUNNER, "PROJECT_ROOT", root)
    monkeypatch.setattr(RUNNER, "stage_a_cases", lambda: bases)
    def adapt(records, existing, stage):
        selected = {model: [{"case": model}] for model in RUNNER.MODELS}
        return (variants_b if stage == "B" else variants_c), selected
    monkeypatch.setattr(RUNNER, "adaptive_cases", adapt)
    monkeypatch.setattr(RUNNER.life, "fingerprints", lambda root, data: {"source": {"dummy.txt": RUNNER.life.file_digest(root / "dummy.txt")}, "data": {"input.txt": RUNNER.life.file_digest(data / "input.txt")}})
    calls = {"training": [], "assembly": 0, "preflight": 0, "command_preflights": []}
    def fake_validate_commands(training_script, commands):
        calls["command_preflights"].append(commands)
        return [{"status": "passed", "command": command} for command in commands]
    monkeypatch.setattr(RUNNER, "validate_commands", fake_validate_commands)
    def assemble(queue, result_root, paper_config):
        calls["assembly"] += 1
        queue["recurrent_assembly"] = {"test": "independently_unit_tested"}
    monkeypatch.setattr(RUNNER, "run_recurrent_assembly", assemble)
    class FakeSupervisor(RUNNER.life.ChildSupervisor):
        def run(self, cmd, log):
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("mock child operation\n")
            if "scripts/reproduce/audit_que_data_protocol.py" in cmd:
                output = Path(cmd[cmd.index("--output-root") + 1]); output.mkdir(parents=True, exist_ok=True)
                RUNNER.life.atomic_json(output / "paper_data_statistics.json", {"common_evaluation": evaluation_fixture()})
            elif "scripts/train/train_temporal_baselines.py" in cmd:
                output = Path(cmd[cmd.index("--output-root") + 1])
                case = registry[output.name]
                calls["training"].append(case["case"])
                build_artifacts(output / case["model"] / f"seed_{RUNNER.SEED}", case, receipt=False)
            elif "scripts/reproduce/audit_que_saved_predictions.py" in cmd:
                output = Path(cmd[cmd.index("--output-root") + 1]); output.mkdir(parents=True, exist_ok=True)
                RUNNER.life.atomic_json(output / "audit_summary.json", {"status": "completed", "valid_sources": 18, "invalid_sources": 0,
                    "stored_metric_roundoff_exceeded": 0, "first_day_discrepancy_sources": 0, "models_with_raw_predictions": list(RUNNER.MODELS),
                    "truth_groups_per_task": {"24h": 1, "168h": 1}, "truth_origin_group_mismatch": False})
            else:
                calls["preflight"] += 1
            return 0
    monkeypatch.setattr(RUNNER.life, "ChildSupervisor", FakeSupervisor)
    args = ["--device", "cpu", "--data-dir", str(data), "--paper-config", str(paper), "--run-tag", "test_run", "--budget-hours", "0"]
    return root, args, calls


def test_full_tiny_queue_all_stages_resume_without_retraining(tmp_path, monkeypatch):
    root, args, calls = mock_queue_environment(tmp_path, monkeypatch)
    assert RUNNER.main(args) == 0
    assert len(calls["training"]) == 18
    assert calls["assembly"] == 1
    assert [len(commands) for commands in calls["command_preflights"]] == [6, 6, 6]
    result = root / "results/test_run"
    status = json.loads((result / "queue_status.json").read_text())
    assert status["status"] == "completed" and status["finished_cases"] == 18 and status["stage_c_selected"]
    assert status["paper_reproduction_verified"] is False
    assert (result / "stage_b_manifest.json").exists() and (result / "stage_c_manifest.json").exists()
    assert all(json.loads((result / f"stage_{stage}_command_preflight.json").read_text())["status"] == "passed" for stage in "abc")
    original_manifest = (result / "manifest.json").read_bytes()
    assert RUNNER.main(args) == 0
    assert len(calls["training"]) == 18
    assert (result / "manifest.json").read_bytes() == original_manifest
    resumed = json.loads((result / "queue_status.json").read_text())
    assert all(r["technical_status"] == "PASS(existing)" for r in resumed["cases"])


def test_changed_source_resume_refuses_without_overwriting_existing_status(tmp_path, monkeypatch):
    root, args, calls = mock_queue_environment(tmp_path, monkeypatch)
    assert RUNNER.main(args) == 0
    status = root / "results/test_run/queue_status.json"
    before = status.read_bytes()
    (root / "dummy.txt").write_text("changed-source")
    assert RUNNER.main(args) == 1
    assert status.read_bytes() == before
    assert len(calls["training"]) == 18


def test_budget_pause_packages_partial_without_claiming_complete(tmp_path, monkeypatch):
    root, args, calls = mock_queue_environment(tmp_path, monkeypatch)
    args[-1] = "0.00000000001"
    assert RUNNER.main(args) == 0
    status = json.loads((root / "results/test_run/queue_status.json").read_text())
    assert status["status"] == "paused_time_budget" and not status["technical_success"]
    assert len(calls["training"]) == 0
    assert (root.parent / "test_run_compact.tar.gz").exists()


def test_interrupted_training_cleans_owned_supervisor_and_keeps_state(tmp_path, monkeypatch):
    root, args, calls = mock_queue_environment(tmp_path, monkeypatch)
    parent = RUNNER.life.ChildSupervisor
    class InterruptedSupervisor(parent):
        def run(self, cmd, log):
            if "scripts/train/train_temporal_baselines.py" in cmd:
                raise RUNNER.life.InterruptedRun(signal.SIGTERM)
            return super().run(cmd, log)
    monkeypatch.setattr(RUNNER.life, "ChildSupervisor", InterruptedSupervisor)
    assert RUNNER.main(args) == 143
    status = json.loads((root / "results/test_run/queue_status.json").read_text())
    assert status["status"] == "interrupted" and status["failed_cases"] == 0
    assert status["active_case"] is None
    assert status["cases"][0]["technical_status"] == "running"


def test_retry_failed_prior_case_keeps_frozen_adaptive_plans(tmp_path, monkeypatch):
    root, args, calls = mock_queue_environment(tmp_path, monkeypatch)
    parent = RUNNER.life.ChildSupervisor
    fail_once = [True]
    class OnceFailedSupervisor(parent):
        def run(self, cmd, log):
            rc = super().run(cmd, log)
            if "scripts/train/train_temporal_baselines.py" in cmd and fail_once[0]:
                fail_once[0] = False
                return 1
            return rc
    monkeypatch.setattr(RUNNER.life, "ChildSupervisor", OnceFailedSupervisor)
    assert RUNNER.main(args) == 1
    result = root / "results/test_run"
    frozen_b = (result / "stage_b_manifest.json").read_bytes()
    frozen_c = (result / "stage_c_manifest.json").read_bytes()
    def forbidden_reranking(*args, **kwargs):
        raise AssertionError("Frozen plans must never be regenerated on resume")
    monkeypatch.setattr(RUNNER, "adaptive_cases", forbidden_reranking)
    assert RUNNER.main(args) == 0
    assert len(calls["training"]) == 19  # precisely the one failed case was retried
    assert (result / "stage_b_manifest.json").read_bytes() == frozen_b
    assert (result / "stage_c_manifest.json").read_bytes() == frozen_c


def test_invalid_stage_command_stops_before_any_gpu_or_training(tmp_path, monkeypatch):
    root, args, calls = mock_queue_environment(tmp_path, monkeypatch)
    def reject(training_script, commands):
        assert len(commands) == 6
        assert training_script == root / "scripts/train/train_temporal_baselines.py"
        raise ValueError("--cam-channel-sizes: expected 3 arguments")
    def forbidden_gpu(*args, **kwargs):
        raise AssertionError("No GPU preflight may run when actual CLI validation fails")
    monkeypatch.setattr(RUNNER, "validate_commands", reject)
    monkeypatch.setattr(RUNNER.life, "gpu_preflight", forbidden_gpu)
    assert RUNNER.main(args) == 1
    assert calls["training"] == []
    result = root / "results/test_run"
    status = json.loads((result / "queue_status.json").read_text())
    assert status["status"] == "failed" and status["cases"] == []
    assert status["active_case"] is None
    report = json.loads((result / "stage_a_command_preflight.json").read_text())
    assert report["status"] == "failed" and len(report["commands"]) == 6
    assert "expected 3 arguments" in report["error"]
    assert not (result / "stage_b_manifest.json").exists()


def test_unexpected_training_exit_two_stops_on_first_case(tmp_path, monkeypatch):
    root, args, calls = mock_queue_environment(tmp_path, monkeypatch)
    parent = RUNNER.life.ChildSupervisor
    class ArgumentFailureSupervisor(parent):
        def run(self, cmd, log):
            if "scripts/train/train_temporal_baselines.py" in cmd:
                calls["training"].append(cmd)
                log.write_text("training CLI error\n")
                return 2
            return super().run(cmd, log)
    monkeypatch.setattr(RUNNER.life, "ChildSupervisor", ArgumentFailureSupervisor)
    assert RUNNER.main(args) == 1
    assert len(calls["training"]) == 1
    result = root / "results/test_run"
    status = json.loads((result / "queue_status.json").read_text())
    assert status["status"] == "failed" and status["failed_cases"] == 1
    assert status["cases"][0]["exit_code"] == 2 and status["active_case"] is None
    assert "stopped immediately" in status["error"]
    assert not (result / "stage_b_manifest.json").exists()


def test_reuse_sweep_precedes_training_and_resume_recovers_source(tmp_path, monkeypatch):
    root, args, calls = mock_queue_environment(tmp_path, monkeypatch)
    source = root / "results/old_run"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text('{"unit_test_source":true}\n')
    source_before = (source / "manifest.json").read_bytes()
    imported = []
    def prepare_reuse(source_root, destination_root, current_manifest, current_evaluation, runner):
        assert source_root == source
        assert current_evaluation == evaluation_fixture()
        assert runner.validate_case is RUNNER.validate_case
        assert current_manifest["reuse_source"]["root"] == str(source)
        return {"destination": destination_root}
    def import_case(context, case, expected):
        if case["model"] not in ("gru", "lstm"):
            return None
        run = context["destination"] / "cases" / case["case"] / case["model"] / f"seed_{RUNNER.SEED}"
        assert not run.exists()
        build_artifacts(run, case, expected)
        metadata = {"source_root": str(source), "source_case": case["case"], "source_manifest_signature": "unit_test",
                    "source_training_git_commit": "original_training_commit", "training_seconds_saved": 100.0, "reused_utc": RUNNER.life.utc_now()}
        RUNNER.life.atomic_json(run / "reused_source_provenance.json", metadata)
        status = json.loads((run / "status.json").read_text())
        RUNNER.life.atomic_json(run / "completion_receipt.json", {"request_sha256": RUNNER.life.digest(expected), "files": RUNNER.evidence_hashes(run, status)})
        imported.append(case["case"])
        return metadata
    helper = SimpleNamespace(prepare_reuse=prepare_reuse, import_case=import_case)
    monkeypatch.setattr(RUNNER, "load_helper", lambda name: helper)
    parent = RUNNER.life.ChildSupervisor
    class VerifiedReuseSupervisor(parent):
        def run(self, cmd, log):
            if "scripts/train/train_temporal_baselines.py" in cmd:
                assert len(imported) == 2, "All reusable Stage A cases must be imported before first training"
            return super().run(cmd, log)
    monkeypatch.setattr(RUNNER.life, "ChildSupervisor", VerifiedReuseSupervisor)
    assert RUNNER.main(args + ["--reuse-from", str(source)]) == 0
    assert len(calls["training"]) == 16
    result = root / "results/test_run"
    status = json.loads((result / "queue_status.json").read_text())
    reused = [record for record in status["cases"] if record.get("reuse")]
    assert len(reused) == 2 and all(record["technical_status"] == "PASS(reused)" for record in reused)
    assert all(record["elapsed_seconds"] == 0 for record in reused)
    assert sum(record["reuse"]["training_seconds_saved"] for record in reused) == 200
    original_manifest = (result / "manifest.json").read_bytes()
    assert RUNNER.main(args) == 0  # reuse source persists without repeating the flag
    assert len(calls["training"]) == 16 and len(imported) == 2
    assert (result / "manifest.json").read_bytes() == original_manifest
    assert (source / "manifest.json").read_bytes() == source_before


def test_reused_provenance_is_bound_to_completion_receipt(tmp_path):
    case = RUNNER.make_case("gru")
    run = tmp_path / "run"
    request = build_artifacts(run, case)
    RUNNER.life.atomic_json(run / "reused_source_provenance.json", {"source_training_git_commit": "original"})
    status = json.loads((run / "status.json").read_text())
    RUNNER.life.atomic_json(run / "completion_receipt.json", {"request_sha256": RUNNER.life.digest(request), "files": RUNNER.evidence_hashes(run, status)})
    assert RUNNER.validate_case(run, case, request)[0]
    RUNNER.life.atomic_json(run / "reused_source_provenance.json", {"source_training_git_commit": "changed"})
    assert not RUNNER.validate_case(run, case, request)[0]
