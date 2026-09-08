from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("que_shared_gpu_runtime_tested", ROOT / "scripts/train/que_shared_gpu_runtime.py")
assert SPEC and SPEC.loader
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def test_query_allows_existing_processes_without_signalling_them(monkeypatch):
    outputs = iter([
        "GPU-abc, NVIDIA GeForce RTX 4090, 8838, 24564\n",
        "GPU-other, 100, Python\nGPU-abc, 2501102, VLLM::EngineCore\nGPU-abc, 1000642, /usr/bin/python3\n",
    ])
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs == {"capture_output": True, "text": True, "check": True, "timeout": 30}
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(RUNTIME.subprocess, "run", run)
    monkeypatch.setattr(RUNTIME.os, "kill", lambda *args: pytest.fail("Foreign process was signalled"))
    state = RUNTIME.query_gpu("7")
    assert state["gpu_id"] == "7" and state["free_mib"] == 8838
    assert state["total_mib"] == 24564
    assert [p["pid"] for p in state["processes"]] == [2501102, 1000642]
    assert len(commands) == 2 and all(c[0] == "nvidia-smi" for c in commands)
    assert "-i" in commands[0] and commands[0][2] == "7"


@pytest.mark.parametrize("output", ["", "GPU-abc, GPU, N/A, 24000\n", "GPU-abc, GPU, 25000, 24000\n",
                                     "GPU-abc, GPU, -1, 24000\n", "GPU-abc, GPU, 1, 0\n",
                                     "GPU-abc, GPU, 1, 24000\nGPU-def, GPU, 2, 24000\n"])
def test_query_invalid_memory_fails_closed(monkeypatch, output):
    monkeypatch.setattr(RUNTIME.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=output))
    with pytest.raises(RuntimeError):
        RUNTIME.query_gpu("7")


def test_query_no_processes_and_query_failure(monkeypatch):
    outputs = iter(["GPU-abc, GPU, 24000, 24564\n", ""])
    monkeypatch.setattr(RUNTIME.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=next(outputs)))
    assert RUNTIME.query_gpu("7")["processes"] == []

    def fail(*a, **k):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(RUNTIME.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        RUNTIME.query_gpu("7")


def test_wait_until_ready_budget_and_query_failure(monkeypatch):
    clock = [0.0]
    states = iter([{"free_mib": 7000}, {"free_mib": 8192}])
    waits = []
    monkeypatch.setattr(RUNTIME.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(RUNTIME.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(RUNTIME, "query_gpu", lambda _: next(states))
    assert RUNTIME.wait_for_memory("7", 8192, 30, 100, waits.append)["free_mib"] == 8192
    assert clock[0] == 30 and len(waits) == 1
    assert waits[0]["minimum_free_mib"] == 8192
    monkeypatch.setattr(RUNTIME, "query_gpu", lambda _: {"free_mib": 1000})
    with pytest.raises(RUNTIME.GpuWaitExpired):
        RUNTIME.wait_for_memory("7", 8192, 30, 45, waits.append)
    assert clock[0] == 45

    def bad(_):
        raise RuntimeError("nvidia-smi malformed")

    monkeypatch.setattr(RUNTIME, "query_gpu", bad)
    with pytest.raises(RuntimeError, match="malformed"):
        RUNTIME.wait_for_memory("7", 8192, 30, 100, waits.append)


def test_wait_interruption_propagates_and_sleep_is_bounded(monkeypatch):
    sleeps = []
    monkeypatch.setattr(RUNTIME.time, "monotonic", lambda: 0)
    monkeypatch.setattr(RUNTIME, "query_gpu", lambda _: {"free_mib": 1000})

    def interrupt(seconds):
        sleeps.append(seconds)
        raise KeyboardInterrupt()

    monkeypatch.setattr(RUNTIME.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        RUNTIME.wait_for_memory("7", 8192, 30, float("inf"), lambda _: None)
    assert sleeps == [30]
    with pytest.raises(ValueError):
        RUNTIME.wait_for_memory("7", 8192, 61, float("inf"), lambda _: None)


def direct_command():
    return [sys.executable, "-u", "scripts/train/train_temporal_baselines.py", "--device", "cuda:0",
            "--model", "msnet", "--seed", "20240604", "--batch-size", "8", "--minimum-free-gib", "6"]


def test_wrap_validates_real_parser_without_importing_torch(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    command = direct_command()
    wrapped = RUNTIME.wrap_command(command, 6, 2, tmp_path / "runtime.json")
    args = RUNTIME.parse_arguments(wrapped[3:])
    assert wrapped[:2] == command[:2]
    assert args.trainer_args == command[3:]
    assert args.memory_limit_gib == 6 and args.headroom_gib == 2
    assert args.report == tmp_path / "runtime.json"
    with pytest.raises(ValueError, match="unchanged"):
        RUNTIME.wrap_command([sys.executable, "-u", "another.py", "--device", "cuda:0"], 6, 2, tmp_path / "r.json")


@pytest.mark.parametrize("limit,headroom", [(float("nan"), 2), (float("inf"), 2), (0, 2), (-1, 2), (6, .5), (6, float("inf"))])
def test_invalid_runtime_policy_rejected_before_execution(tmp_path, limit, headroom):
    with pytest.raises(SystemExit) as exc:
        RUNTIME.wrap_command(direct_command(), limit, headroom, tmp_path / "r.json")
    assert exc.value.code == 2


@pytest.mark.parametrize("device", ["cpu", "auto", "cuda:1"])
def test_wrapper_rejects_nonlogical_zero_device(tmp_path, device):
    command = direct_command()
    command[4] = device
    with pytest.raises(SystemExit):
        RUNTIME.wrap_command(command, 6, 2, tmp_path / "r.json")


def fake_torch(monkeypatch, free_gib=8, total_gib=24):
    events = []

    class OutOfMemoryError(RuntimeError):
        pass

    cuda = SimpleNamespace(
        OutOfMemoryError=OutOfMemoryError,
        is_available=lambda: True,
        set_device=lambda index: events.append(("device", index)),
        mem_get_info=lambda index: (int(free_gib * 1024**3), int(total_gib * 1024**3)),
        set_per_process_memory_fraction=lambda fraction, device: events.append(("fraction", fraction, device)),
        max_memory_allocated=lambda index: 1024**3,
        max_memory_reserved=lambda index: 2 * 1024**3,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(RUNTIME.os, "kill", lambda *a: pytest.fail("Foreign process was signalled"))
    return events, OutOfMemoryError


def run_arguments(tmp_path):
    return RUNTIME.wrap_command(direct_command(), 6, 2, tmp_path / "runtime.json")[3:]


def test_runtime_caps_allocator_before_unchanged_trainer_and_records_peaks(tmp_path, monkeypatch):
    events, _ = fake_torch(monkeypatch)
    original_argv = sys.argv

    def train(path, run_name):
        assert events == [("device", 0), ("fraction", .25, 0)]
        assert path == str(RUNTIME.TRAINER) and run_name == "__main__"
        assert sys.argv == [str(RUNTIME.TRAINER), *direct_command()[3:]]
        events.append(("trained",))

    monkeypatch.setattr(RUNTIME.runpy, "run_path", train)
    assert RUNTIME.main(run_arguments(tmp_path)) == 0
    report = json.loads((tmp_path / "runtime.json").read_text())
    assert report["status"] == "completed" and report["exit_code"] == 0
    assert report["allocator_fraction"] == .25
    assert report["peak_allocated_gib"] == 1 and report["peak_reserved_gib"] == 2
    assert report["other_processes_signalled"] is False
    assert sys.argv is original_argv


@pytest.mark.parametrize("configured", [None, ":16:8"])
def test_runtime_preserves_trainer_cublas_setup_before_cuda(tmp_path, monkeypatch, configured):
    fake_torch(monkeypatch)
    if configured is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", configured)
    expected = configured or ":4096:8"
    cuda = sys.modules["torch"].cuda

    def set_device(index):
        assert RUNTIME.os.environ["CUBLAS_WORKSPACE_CONFIG"] == expected

    cuda.set_device = set_device
    monkeypatch.setattr(RUNTIME.runpy, "run_path", lambda *a, **k: None)
    assert RUNTIME.main(run_arguments(tmp_path)) == 0


def test_runtime_rechecks_memory_race_before_training(tmp_path, monkeypatch):
    events, _ = fake_torch(monkeypatch, free_gib=6.9)
    monkeypatch.setattr(RUNTIME.runpy, "run_path", lambda *a, **k: pytest.fail("Trainer started despite resource race"))
    assert RUNTIME.main(run_arguments(tmp_path)) == 75
    report = json.loads((tmp_path / "runtime.json").read_text())
    assert report["status"] == "resource_wait"
    assert events == [("device", 0)]


@pytest.mark.parametrize("kind", ["oom", "preflight", "ordinary", "system_exit"])
def test_runtime_preserves_error_types_and_never_marks_failure_success(tmp_path, monkeypatch, kind):
    _, oom = fake_torch(monkeypatch)

    def train(*a, **k):
        if kind == "oom":
            raise oom("CUDA out of memory")
        if kind == "preflight":
            raise RuntimeError("cuda:0 has 5.00 GiB free; 6.00 GiB is required.")
        if kind == "ordinary":
            raise RuntimeError("shape mismatch")
        raise SystemExit(2)

    monkeypatch.setattr(RUNTIME.runpy, "run_path", train)
    if kind == "ordinary":
        with pytest.raises(RuntimeError, match="shape mismatch"):
            RUNTIME.main(run_arguments(tmp_path))
    else:
        assert RUNTIME.main(run_arguments(tmp_path)) == (2 if kind == "system_exit" else 75)
    report = json.loads((tmp_path / "runtime.json").read_text())
    assert report["status"] == {"oom": "resource_oom", "preflight": "resource_wait", "ordinary": "failed", "system_exit": "failed"}[kind]


@pytest.mark.parametrize("visible", ["", "6,7", "-1", "GPU-a,GPU-b"])
def test_runtime_requires_one_visible_gpu_before_torch(tmp_path, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="exactly one"):
        RUNTIME.main(run_arguments(tmp_path))


def test_wrapper_cli_cpu_parse_only(tmp_path):
    command = [sys.executable, "-S", str(ROOT / "scripts/train/que_shared_gpu_runtime.py"),
               "--memory-limit-gib", "0", "--headroom-gib", "2", "--report", str(tmp_path / "r.json"),
               "--", "--device", "cuda:0"]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 2 and "memory-limit-gib" in result.stderr
    assert not (tmp_path / "r.json").exists()
