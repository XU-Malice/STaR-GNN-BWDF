"""Bounded recurrent follow-ups preserve the total-eight selection protocol."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FOLLOWUP = load("que_total_followup_training_test", ROOT / "scripts/train/que_total_followup_training.py")
RUNNER = load("que_total_followup_runner_test", ROOT / "scripts/train/run_que_comprehensive_reconstruction.py")
RUNTIME = load("que_total_followup_runtime_test", ROOT / "scripts/train/que_shared_gpu_runtime.py")
TOTAL_KEYS = [(task, metric) for task in ("24h", "168h") for metric in ("MAE", "MAPE", "RMSE", "NSE")]


def record(case, ratios=1.0, **metadata):
    if isinstance(ratios, (int, float)):
        ratios = [ratios] * 8
    assert len(ratios) == 8
    return {
        "case": case["case"], "model": case["model"], "settings": copy.deepcopy(case),
        "metrics": [{"task": task, "series": "total", "metric": metric,
                     "tolerance_ratio": ratio, "mode": "pooled"}
                    for (task, metric), ratio in zip(TOTAL_KEYS, ratios)],
        **metadata,
    }


def two_parent_records():
    return [record(RUNNER.make_case(model, normalization=normalization,
                                    learning_rate_scale=2.0, best_epoch_scale=2.0), ratio)
            for model in ("gru", "lstm")
            for normalization, ratio in (("zscore", 1.0), ("minmax", 2.0))]


def test_parents_rank_by_worst_then_mean_of_all_eight_totals():
    worst_first = RUNNER.make_case("gru", optimizer="adam")
    second = RUNNER.make_case("gru", normalization="minmax")
    first = RUNNER.make_case("gru", batch_size=16)
    records = [record(worst_first, [2.0] + [0.0] * 7),
               record(second, [1.5] * 8),
               record(first, [1.5] + [1.0] * 7)]
    # Poison cached aggregate/DMA/origin-mean scores; none is the selection input.
    for index, item in enumerate(records):
        item.update(worst_ratio=-100 + index, mean_ratio=-100 + index,
                    dma_mean_ratio=-100 + index,
                    scores={mode: {"balanced_distance": -100 + index,
                                   "total_mean_ratio": -100 + index,
                                   "dma_mean_ratio": -100 + index}
                            for mode in ("pooled", "origin_mean")})
    generated = FOLLOWUP.generate_followups(records, RUNNER)
    assert generated
    assert {case["parent_case"] for case in generated} == {first["case"], second["case"]}
    assert generated[0]["parent_case"] == first["case"]


def test_parent_selection_uses_two_distinct_training_settings():
    first = RUNNER.make_case("gru")
    duplicate = RUNNER.make_case("gru", stage="B")
    second = RUNNER.make_case("gru", normalization="minmax")
    third = RUNNER.make_case("gru", optimizer="adam")
    records = [record(first, 1), record(duplicate, 1.1), record(second, 2), record(third, 3)]
    generated = FOLLOWUP.generate_followups(records, RUNNER)
    assert {case["parent_case"] for case in generated} == {first["case"], second["case"]}
    assert len({RUNNER.setting_key(case) for case in generated}) == len(generated)


@pytest.mark.parametrize("outlier_index", range(8), ids=[f"{task}-{metric}" for task, metric in TOTAL_KEYS])
def test_each_total_metric_in_both_horizons_participates_in_parent_ranking(outlier_index):
    ratios = [.1] * 8
    ratios[outlier_index] = 8
    misleading = RUNNER.make_case("gru", optimizer="adam")
    first = RUNNER.make_case("gru")
    second = RUNNER.make_case("gru", normalization="minmax")
    generated = FOLLOWUP.generate_followups(
        [record(misleading, ratios), record(first, 2), record(second, 3)], RUNNER)
    assert {case["parent_case"] for case in generated} == {first["case"], second["case"]}


@pytest.mark.parametrize("status", ["FAIL", "FAIL(existing)", "failed", "incomplete", "running"])
def test_failed_or_unfinished_records_never_become_parents(status):
    good = RUNNER.make_case("gru")
    bad = RUNNER.make_case("gru", normalization="minmax")
    generated = FOLLOWUP.generate_followups([record(bad, 0, technical_status=status), record(good, 2)], RUNNER)
    assert generated
    assert {case["parent_case"] for case in generated} == {good["case"]}


def test_failed_settings_are_still_excluded_from_new_followups():
    parent = record(RUNNER.make_case("gru"))
    initial = FOLLOWUP.generate_followups([parent], RUNNER)
    assert initial
    failed_case = copy.deepcopy(initial[0])
    failed = record(failed_case, .01, technical_status="FAIL")
    generated = FOLLOWUP.generate_followups([parent, failed], RUNNER)
    assert RUNNER.setting_key(failed_case) not in {RUNNER.setting_key(case) for case in generated}
    assert all(case["parent_case"] == parent["case"] for case in generated)


@pytest.mark.parametrize("cap", [0, 1, 7, 24, 25, 1000])
def test_followup_budget_is_hard_capped_per_model_and_stage_identity_is_signed(cap):
    records = two_parent_records()
    before = copy.deepcopy(records)
    generated = FOLLOWUP.generate_followups(records, RUNNER, max_per_model=cap)
    assert records == before
    assert len(generated) <= min(cap, 24) * 2
    for model in ("gru", "lstm"):
        assert sum(case["model"] == model for case in generated) <= min(cap, 24)
    if cap:
        assert {case["model"] for case in generated} == {"gru", "lstm"}
    assert len({RUNNER.setting_key(case) for case in generated}) == len(generated)
    original = {RUNNER.setting_key(item["settings"]) for item in records}
    assert not original.intersection(RUNNER.setting_key(case) for case in generated)
    for case in generated:
        assert case["seed"] == 20240604 and case["stage"] == "F"
        assert case["case"] == f"F_{case['model']}_{RUNNER.setting_key(case)[:12]}"


@pytest.mark.parametrize("cap", [-1, 1.5, "24"])
def test_invalid_followup_budget_is_rejected(cap):
    with pytest.raises((TypeError, ValueError)):
        FOLLOWUP.generate_followups(two_parent_records(), RUNNER, max_per_model=cap)


def test_seed_is_enforced_before_any_followups_are_generated():
    item = record(RUNNER.make_case("gru", seed=123))
    with pytest.raises(ValueError, match="(?i)seed"):
        FOLLOWUP.generate_followups([item], RUNNER)


def test_only_recurrent_models_generate_followups():
    records = [record(RUNNER.make_case(model)) for model in RUNNER.MODELS]
    generated = FOLLOWUP.generate_followups(records, RUNNER)
    assert {case["model"] for case in generated} == {"gru", "lstm"}


def test_single_parent_variations_are_relative_and_keep_the_protocol_fixed():
    parent = RUNNER.make_case("gru", learning_rate_scale=2.0, best_epoch_scale=2.0)
    generated = FOLLOWUP.generate_followups([record(parent)], RUNNER)
    assert {case["learning_rate_scale"] for case in generated} == {1.0, 1.5, 2.0, 2.5, 3.0}
    assert {case["best_epoch_scale"] for case in generated} == {1.3, 1.6, 2.0, 2.4, 3.0}
    assert {case["batch_size"] for case in generated} == {2, 4, 8, 16, 32}
    assert {case["loss"] for case in generated} == {"mse", "mae", "huber"}
    paired = {(case["learning_rate_scale"], case["best_epoch_scale"]) for case in generated}
    assert {(1.0, 1.3), (1.5, 1.6), (2.5, 2.4), (3.0, 3.0)} <= paired
    variable_keys = {"learning_rate_scale", "best_epoch_scale", "max_epochs", "batch_size", "loss"}
    fixed_keys = set(RUNNER.TRAINING_KEYS) - variable_keys
    for case in generated:
        assert all(case[key] == parent[key] for key in fixed_keys)
        assert case["max_epochs"] is None


def test_epoch_override_refinements_remain_relative_to_the_parent():
    parent = RUNNER.make_case("lstm", max_epochs=100, best_epoch_scale=1.0,
                              learning_rate_scale=2.0)
    generated = FOLLOWUP.generate_followups([record(parent)], RUNNER)
    assert {case["max_epochs"] for case in generated} == {65, 80, 100, 120, 150}
    assert all(case["best_epoch_scale"] == 1.0 for case in generated)
    paired = {(case["learning_rate_scale"], case["max_epochs"]) for case in generated}
    assert {(1.0, 65), (1.5, 80), (2.5, 120), (3.0, 150)} <= paired


def test_every_generated_command_passes_real_trainer_and_shared_wrapper_parsers(tmp_path, monkeypatch):
    records = two_parent_records()
    generated = FOLLOWUP.generate_followups(records, RUNNER)
    data_dir, output_root = tmp_path / "data_must_not_be_read", tmp_path / "must_not_be_created"
    options = argparse.Namespace(device="cuda:0", data_dir=data_dir, allow_shared_gpu=True,
                                 shared_memory_limit_gib=6.0, shared_headroom_gib=2.0)
    monkeypatch.setattr(RUNNER.life, "gpu_preflight", lambda *args: pytest.fail("CLI preflight accessed GPU"))
    monkeypatch.setattr(RUNTIME, "wait_for_memory", lambda *args: pytest.fail("CLI preflight waited for GPU"))
    commands = []
    def actual_validator(script, batch):
        commands.extend(batch)
        return RUNNER.validate_commands(script, batch)
    reports = FOLLOWUP.preflight_followups(
        generated, tool_root=ROOT, data_dir=data_dir, output_root=output_root,
        runner=RUNNER, command_validator=actual_validator)
    assert len(reports) == len(generated) and generated
    for case, command, report in zip(generated, commands, reports):
        assert report["status"] == "PASS"
        assert report["scope"] == "actual_trainer_cli_and_configuration"
        assert report["model"] == case["model"]
        for key in ("loss", "learning_rate_scale", "best_epoch_scale", "batch_size", "recurrent_layout"):
            assert report["training"][key] == case[key]
        wrapped = RUNNER.launch_command(command, options, tmp_path / "resource_report.json")
        parsed = RUNTIME.parse_arguments(wrapped[3:])
        assert parsed.trainer_args == command[3:]
        assert parsed.memory_limit_gib == 6.0 and parsed.headroom_gib == 2.0
        assert parsed.trainer_args[parsed.trainer_args.index("--seed") + 1] == "20240604"
        assert parsed.trainer_args[parsed.trainer_args.index("--device") + 1] == "cuda:0"
        assert command[command.index("--config") + 1] == str(ROOT / "configs/model/mscmnet_baselines.yaml")
        assert command[command.index("--split-config") + 1] == str(ROOT / "configs/data/paper_split.yaml")
        assert "--overwrite" not in command
    assert not data_dir.exists() and not output_root.exists()
    assert not (tmp_path / "resource_report.json").exists()


@pytest.fixture
def execution(tmp_path):
    artifacts = load("que_total_followup_artifacts_test", ROOT / "tests/test_que_comprehensive_runner.py")
    case = RUNNER.make_case("gru", stage="F", max_epochs=2)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "input.txt").write_text("fixed synthetic input")
    output_root = tmp_path / "followups"
    manifest = {"signatures": {}, "paper_sha256": "injected-source-check"}
    manifest["signature"] = RUNNER.life.digest(manifest)
    evaluation = artifacts.evaluation_fixture()
    published = artifacts.yaml.safe_load((ROOT / "configs/model/mscmnet_baselines.yaml").read_text())["models"]
    request = {
        "signature": RUNNER.life.digest({"manifest": manifest["signature"], "settings": RUNNER.setting_key(case)}),
        "case": copy.deepcopy(case), "model_config": RUNNER.expected_model_config(published, case),
        "evaluation": evaluation,
    }
    calls = {"commands": [], "waits": [], "checks": [], "sleeps": [], "partial_files": []}
    codes = [0]
    resource_statuses = []
    def wait(gpu_id, required, poll, deadline, callback):
        calls["waits"].append((gpu_id, required))
        return {"gpu_id": gpu_id, "free_mib": 9000, "processes": [{"pid": 12345, "process_name": "unrelated-model"}]}
    def run(command, log):
        calls["commands"].append(command)
        number = len(calls["commands"]) - 1
        code = codes[number]
        parsed = RUNTIME.parse_arguments(command[3:])
        status = (resource_statuses[number] if resource_statuses else
                  "completed" if code == 0 else "resource_oom" if code == 75 else "failed")
        RUNNER.life.atomic_json(parsed.report, {"status": status})
        case_root = Path(parsed.trainer_args[parsed.trainer_args.index("--output-root") + 1])
        run_path = case_root / case["model"] / f"seed_{RUNNER.SEED}"
        if code == 0:
            artifacts.build_artifacts(run_path, case, request=request, receipt=False)
            # Model the unchanged trainer, which does not emit lifecycle sidecars.
            (run_path / "request_signature.json").unlink()
        else:
            run_path.mkdir(parents=True)
            partial = run_path / "partial_checkpoint.pt"
            partial.write_bytes(f"incomplete attempt {number}".encode())
            calls["partial_files"].append((partial, partial.read_bytes()))
        return code
    kwargs = dict(
        tool_root=ROOT, data_dir=data_dir, output_root=output_root,
        paper_config=ROOT / "configs/evaluation/mscmnet_paper_metrics.yaml",
        evaluation=evaluation, campaign_manifest=manifest, runner=RUNNER,
        supervisor=SimpleNamespace(run=run),
        resource_helper=SimpleNamespace(wait_for_memory=wait, wrap_command=RUNTIME.wrap_command,
                                        GpuWaitExpired=RUNTIME.GpuWaitExpired),
        source_check=lambda: calls["checks"].append(True),
        command_validator=lambda script, commands: [{"status": "PASS"} for command in commands],
        clock=lambda: 100.0, sleep=calls["sleeps"].append, deadline=1000.0,
    )
    return SimpleNamespace(case=case, kwargs=kwargs, request=request, artifacts=artifacts,
                           calls=calls, codes=codes, resource_statuses=resource_statuses,
                           run=output_root / "cases" / case["case"] / case["model"] / f"seed_{RUNNER.SEED}")


def snapshot(path):
    return {str(item.relative_to(path)): item.read_bytes() for item in path.rglob("*") if item.is_file()}


def test_execution_validates_real_artifacts_and_resumes_without_training_or_overwrite(execution):
    record = FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert record["technical_status"] == "PASS", record.get("validation")
    assert record["run"] == str(execution.run)
    assert record["request"] == execution.request
    assert RUNNER.validate_case(execution.run, execution.case, execution.request) == (True, "validated_artifacts")
    assert [(row["task"], row["metric"]) for row in record["metrics"]] == TOTAL_KEYS
    assert all(row["series"] == "total" and row["mode"] == "pooled" for row in record["metrics"])
    assert record["raw_array_hashes"] == {
        key: RUNNER.life.array_digest(value) for key, value in execution.artifacts.arrays_fixture().items()
    }
    assert record["completion_receipt"]["request_sha256"] == RUNNER.life.digest(execution.request)
    assert record["paper_reproduction_verified"] is False
    assert execution.calls["waits"] == [("7", 8192)]
    before = snapshot(execution.run)
    resumed = FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert resumed["technical_status"] == "PASS(existing)", resumed.get("validation")
    assert resumed["completion_receipt"] == record["completion_receipt"]
    assert snapshot(execution.run) == before
    assert len(execution.calls["commands"]) == 1 and len(execution.calls["waits"]) == 1
    assert all("--overwrite" not in command for command in execution.calls["commands"])


def test_incomplete_existing_output_is_preserved_without_implicit_retraining(execution):
    execution.run.mkdir(parents=True)
    (execution.run / "checkpoint_partial.pt").write_bytes(b"prior failed training evidence")
    (execution.run / "status.json").write_text('{"status":"failed"}')
    before = snapshot(execution.run)
    record = FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert record["technical_status"] == "FAIL(existing)"
    assert snapshot(execution.run) == before
    assert not execution.calls["commands"] and not execution.calls["waits"]


def test_tampered_completed_output_is_preserved_and_not_retrained(execution):
    assert FOLLOWUP.execute_followup(execution.case, **execution.kwargs)["technical_status"] == "PASS"
    (execution.run / "checkpoint_0.pt").write_bytes(b"altered saved evidence")
    before = snapshot(execution.run)
    record = FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert record["technical_status"] == "FAIL(existing)"
    assert snapshot(execution.run) == before
    assert len(execution.calls["commands"]) == 1 and len(execution.calls["waits"]) == 1


def test_verified_resource_retry_preserves_failed_output_and_exact_training_arguments(execution):
    execution.codes[:] = [75, 0]
    record = FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert record["technical_status"] == "PASS", record.get("validation")
    assert len(record["resource_attempts"]) == 2
    trainer_args = [RUNTIME.parse_arguments(command[3:]).trainer_args for command in execution.calls["commands"]]
    assert trainer_args[0] == trainer_args[1]
    assert all("--overwrite" not in args for args in trainer_args)
    archived = Path(record["resource_attempts"][0]["preserved_output"])
    partial = archived / execution.case["model"] / f"seed_{RUNNER.SEED}" / "partial_checkpoint.pt"
    assert partial.read_bytes() == b"incomplete attempt 0"
    assert not (execution.run / "partial_checkpoint.pt").exists()
    assert execution.calls["sleeps"] == [30]
    assert execution.calls["waits"] == [("7", 8192), ("7", 8192)]


def test_resource_retries_stop_at_the_cap_without_erasing_any_attempt(execution):
    execution.codes[:] = [75, 75, 75]
    record = FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert record["technical_status"] == "FAIL"
    assert "retries_exhausted" in record["validation"]
    assert len(execution.calls["commands"]) == len(record["resource_attempts"]) == 3
    assert len(execution.calls["sleeps"]) == 2
    for number, attempt in enumerate(record["resource_attempts"][:2]):
        archived = Path(attempt["preserved_output"]) / execution.case["model"] / f"seed_{RUNNER.SEED}"
        assert (archived / "partial_checkpoint.pt").read_bytes() == f"incomplete attempt {number}".encode()
    assert (execution.run / "partial_checkpoint.pt").read_bytes() == b"incomplete attempt 2"


@pytest.mark.parametrize("exit_code,resource_status", [(1, "failed"), (2, "failed"), (75, "failed")])
def test_only_verified_resource_failures_are_retried(execution, exit_code, resource_status):
    execution.codes[:] = [exit_code]
    execution.resource_statuses[:] = [resource_status]
    record = FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert record["technical_status"] == "FAIL"
    assert len(execution.calls["commands"]) == 1
    assert not execution.calls["sleeps"]
    assert (execution.run / "partial_checkpoint.pt").read_bytes() == b"incomplete attempt 0"


def test_cli_preflight_failure_stops_before_resource_admission(execution):
    def reject(script, commands):
        raise ValueError("invalid trainer command")
    execution.kwargs["command_validator"] = reject
    record = FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert record["technical_status"] == "FAIL"
    assert "invalid trainer command" in record["validation"]
    assert not execution.calls["commands"] and not execution.calls["waits"]
    assert not execution.run.exists()


def test_expired_budget_pauses_without_creating_training_output(execution):
    execution.kwargs["deadline"] = 100.0
    record = FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert record["technical_status"] == "PAUSED"
    assert not execution.calls["commands"] and not execution.calls["waits"]
    assert not execution.run.exists()


def test_source_change_during_resource_wait_stops_before_training(execution):
    changed = False
    def check():
        if changed:
            raise RuntimeError("frozen source changed during wait")
    def wait(gpu_id, required, poll, deadline, callback):
        nonlocal changed
        changed = True
        callback({"free_mib": 4000})
    execution.kwargs["source_check"] = check
    execution.kwargs["resource_helper"].wait_for_memory = wait
    record = FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert record["technical_status"] == "FAIL"
    assert "frozen source changed during wait" in record["validation"]
    assert not execution.calls["commands"] and not execution.run.exists()


@pytest.fixture
def signal_registry(monkeypatch):
    # Exercise handler ownership without modifying the process's real handlers
    # or sending signals to pytest or any unrelated process.
    original = {FOLLOWUP.signal.SIGINT: object(), FOLLOWUP.signal.SIGTERM: object()}
    active = original.copy()
    def replace(signum, handler):
        previous = active[signum]
        active[signum] = handler
        return previous
    monkeypatch.setattr(FOLLOWUP.signal, "signal", replace)
    return SimpleNamespace(original=original, active=active)


@pytest.mark.parametrize("fail_in_child", [False, True], ids=["success", "exception"])
def test_supervisor_signal_handlers_are_installed_then_restored(execution, signal_registry, fail_in_child):
    original_run = execution.kwargs["supervisor"].run
    handler = lambda signum, frame: None
    def run(command, log):
        assert all(value is handler for value in signal_registry.active.values())
        if fail_in_child:
            raise RuntimeError("child failed before producing outputs")
        return original_run(command, log)
    execution.kwargs["supervisor"] = SimpleNamespace(run=run, signal_handler=handler)
    record = FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert record["technical_status"] == ("FAIL" if fail_in_child else "PASS")
    assert signal_registry.active == signal_registry.original


@pytest.mark.parametrize("signum", [FOLLOWUP.signal.SIGINT, FOLLOWUP.signal.SIGTERM], ids=["SIGINT", "SIGTERM"])
def test_owned_child_interruption_propagates_after_saving_failure_and_restoring_handlers(
        execution, signal_registry, signum):
    cleanup = []
    class Supervisor(RUNNER.life.ChildSupervisor):
        def terminate_owned(self):
            cleanup.append(True)
        def run(self, command, log):
            assert signal_registry.active[signum] == self.signal_handler
            # Invoke the registered lifecycle handler directly; no OS signal.
            signal_registry.active[signum](signum, None)
    execution.kwargs["supervisor"] = Supervisor(ROOT, {})
    with pytest.raises(RUNNER.life.InterruptedRun) as interrupted:
        FOLLOWUP.execute_followup(execution.case, **execution.kwargs)
    assert interrupted.value.signum == signum
    assert cleanup == [True]
    assert signal_registry.active == signal_registry.original
    record_path = execution.kwargs["output_root"] / "followup_records" / f"{execution.case['case']}.json"
    record = json.loads(record_path.read_text())
    assert record["technical_status"] == "FAIL"
    assert "owned_followup_interrupted:InterruptedRun" in record["validation"]
    assert "completion_receipt" not in record and "metrics" not in record
