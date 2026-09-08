"""Shared resource scheduling preserves numerical candidates and cached work."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARTIFACTS = load("que_shared_queue_artifacts", ROOT / "tests/test_que_comprehensive_runner.py")
RUNNER = ARTIFACTS.RUNNER
RUNTIME = load("que_shared_queue_runtime", ROOT / "scripts/train/que_shared_gpu_runtime.py")


def shared_options(tmp_path):
    return argparse.Namespace(device="cuda:0", gpu_id="7", data_dir=tmp_path,
                              allow_shared_gpu=True, minimum_free_mib=8192,
                              shared_memory_limit_gib=6.0, shared_headroom_gib=2.0,
                              gpu_poll_seconds=30.0, shared_resource_retries=2)


def test_shared_command_changes_only_resource_threshold_for_all_models(tmp_path):
    args = shared_options(tmp_path)
    ordinary = argparse.Namespace(device="cuda:0", data_dir=tmp_path)
    cases = RUNNER.stage_a_cases()
    for case in cases:
        before = dict(case)
        command = RUNNER.command_for(case, args, tmp_path)
        direct = RUNNER.command_for(case, ordinary, tmp_path)
        index = command.index("--minimum-free-gib")
        assert command[index + 1] == "6.0"
        assert command[:index] + command[index + 2:] == direct
        wrapped = RUNNER.launch_command(command, args, tmp_path / "runtime.json")
        parsed = RUNTIME.parse_arguments(wrapped[3:])
        assert parsed.trainer_args == command[3:]
        assert parsed.memory_limit_gib == 6 and parsed.headroom_gib == 2
        assert case == before


def execution_fixture(tmp_path):
    case = RUNNER.make_case("msnet")
    args = shared_options(tmp_path)
    result, logs = tmp_path / "results", tmp_path / "logs"
    result.mkdir(); logs.mkdir()
    case_root = result / "cases" / case["case"]
    run = case_root / "msnet" / f"seed_{RUNNER.SEED}"
    record = {"case": case["case"], "model": case["model"], "stage": "A", "settings": case}
    queue = {"cases": [record], "case_count": 95, "active_case": case["case"]}
    return case, args, case_root, run, record, queue, result, logs


@pytest.mark.parametrize("codes,expected", [([75, 0], 0), ([75, 75, 75], 75), ([1], 1), ([2], 2)])
def test_shared_retries_are_bounded_and_numerical_arguments_unchanged(tmp_path, monkeypatch, codes, expected):
    case, args, case_root, run, record, queue, result, logs = execution_fixture(tmp_path)
    snapshots = {"gpu_id": "7", "gpu_uuid": "GPU-test", "free_mib": 8838,
                 "processes": [{"pid": 2501102, "process_name": "VLLM::EngineCore"}]}
    observed = []
    def wait(gpu_id, required, poll, deadline, callback):
        assert gpu_id == "7" and required == 8192
        callback({**snapshots, "free_mib": 4000})
        assert queue["status"] == "waiting_gpu" and record["technical_status"] == "waiting_gpu"
        return snapshots
    helper = SimpleNamespace(wait_for_memory=wait, GpuWaitExpired=RUNTIME.GpuWaitExpired,
                             wrap_command=RUNTIME.wrap_command)
    monkeypatch.setattr(RUNNER, "load_helper", lambda name: helper)
    monkeypatch.setattr(RUNNER.time, "sleep", lambda seconds: None)
    commands = []
    class Supervisor:
        def run(self, command, log):
            commands.append(command)
            code = codes[len(commands) - 1]
            report = Path(command[command.index("--report") + 1])
            RUNNER.life.atomic_json(report, {"status": "completed" if code == 0 else "resource_oom" if code == 75 else "failed"})
            return code
    rc, elapsed = RUNNER.execute_candidate(case, args, case_root, run, record, queue,
                                           result, logs, Supervisor(), float("inf"), lambda: observed.append(True))
    assert rc == expected and elapsed >= 0
    assert len(commands) == len(codes) and len(record["resource_attempts"]) == len(codes)
    trainer_args = [RUNTIME.parse_arguments(command[3:]).trainer_args for command in commands]
    assert all(value == trainer_args[0] for value in trainer_args)
    assert observed and queue["failed_cases"] == 0  # caller handles final result
    assert "gpu_wait" not in queue


def test_resource_budget_pause_does_not_become_model_failure(tmp_path, monkeypatch):
    case, args, case_root, run, record, queue, result, logs = execution_fixture(tmp_path)
    def wait(*values):
        values[-1]({"free_mib": 4000, "gpu_id": "7"})
        raise RUNTIME.GpuWaitExpired("time budget")
    helper = SimpleNamespace(wait_for_memory=wait, GpuWaitExpired=RUNTIME.GpuWaitExpired)
    monkeypatch.setattr(RUNNER, "load_helper", lambda name: helper)
    with pytest.raises(RUNNER.QueueBudgetPause):
        RUNNER.execute_candidate(case, args, case_root, run, record, queue, result, logs,
                                 SimpleNamespace(run=lambda *args: pytest.fail("No training during wait")),
                                 float("inf"), lambda: None)
    assert record["technical_status"] == "waiting_gpu" and queue["failed_cases"] == 0


def test_source_change_while_waiting_stops_before_training(tmp_path, monkeypatch):
    case, args, case_root, run, record, queue, result, logs = execution_fixture(tmp_path)
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("Source/data/reference changed")
    def wait(*values):
        values[-1]({"free_mib": 4000})
    monkeypatch.setattr(RUNNER, "load_helper", lambda name: SimpleNamespace(wait_for_memory=wait, GpuWaitExpired=RUNTIME.GpuWaitExpired))
    with pytest.raises(RuntimeError, match="Source/data/reference changed"):
        RUNNER.execute_candidate(case, args, case_root, run, record, queue, result, logs,
                                 SimpleNamespace(run=lambda *args: pytest.fail("No changed-source training")), float("inf"), check)


def test_complete_shared_queue_all_stages_and_cached_resume(tmp_path, monkeypatch):
    root, args, calls = ARTIFACTS.mock_queue_environment(tmp_path, monkeypatch)
    args[args.index("--device") + 1] = "cuda:0"
    args += ["--allow-shared-gpu", "--gpu-id", "7"]
    checks = []
    def wait(gpu_id, required, poll, deadline, callback):
        checks.append((gpu_id, required))
        return {"gpu_id": gpu_id, "free_mib": 8838, "processes": [{"pid": 2501102}]}
    helper = SimpleNamespace(wait_for_memory=wait, GpuWaitExpired=RUNTIME.GpuWaitExpired,
                             wrap_command=RUNTIME.wrap_command)
    monkeypatch.setattr(RUNNER, "load_helper", lambda name: helper)
    monkeypatch.setattr(RUNNER.life, "gpu_preflight", lambda *args: pytest.fail("Shared mode must not call exclusive preflight"))
    parent = RUNNER.life.ChildSupervisor
    class SharedSupervisor(parent):
        def run(self, command, log):
            if len(command) > 2 and command[2].endswith("que_shared_gpu_runtime.py"):
                assert self.environment["CUDA_VISIBLE_DEVICES"] == "7"
                parsed = RUNTIME.parse_arguments(command[3:])
                RUNNER.life.atomic_json(parsed.report, {"status": "completed"})
                command = [command[0], "-u", "scripts/train/train_temporal_baselines.py", *parsed.trainer_args]
            return super().run(command, log)
    monkeypatch.setattr(RUNNER.life, "ChildSupervisor", SharedSupervisor)
    assert RUNNER.main(args) == 0
    assert len(calls["training"]) == 18 and checks == [("7", 8192)] * 18
    result = root / "results/test_run"
    status = json.loads((result / "queue_status.json").read_text())
    assert status["status"] == "completed" and status["passed_cases"] == 18
    assert status["gpu_policy"]["allow_shared_gpu"] and status["gpu_id"] == "7"
    assert all(record["resource_attempts"][0]["resource_status"] == "completed" for record in status["cases"])
    assert RUNNER.main(args) == 0
    assert len(calls["training"]) == 18 and len(checks) == 18
    # Different allocator policy requires a distinct signed run.
    assert RUNNER.main(args + ["--shared-memory-limit-gib", "5"]) == 1
    assert len(calls["training"]) == 18


def test_status_distinguishes_waiting_from_training(tmp_path, capsys):
    result = tmp_path / "results"; result.mkdir()
    RUNNER.life.atomic_json(result / "queue_status.json", {
        "status": "waiting_gpu", "gpu_id": "7", "case_count": 95, "finished_cases": 28,
        "passed_cases": 28, "failed_cases": 0, "cases": [],
        "gpu_policy": {"allow_shared_gpu": True, "memory_limit_gib": 6, "headroom_gib": 2},
        "gpu_wait": {"free_mib": 4000, "required_mib": 8192},
    })
    assert RUNNER.print_status(result, tmp_path) == 0
    text = capsys.readouterr().out
    assert "共享 GPU 7" in text and "尚未启动下一次训练" in text and "4000" in text and "8192" in text
