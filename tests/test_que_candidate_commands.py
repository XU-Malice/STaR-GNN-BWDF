"""Regression for the actual parser contract missed by direct fit/mocked tests."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    runner = _load("que_cli_test_runner", ROOT / "scripts/train/run_que_comprehensive_reconstruction.py")
    cli = _load("que_cli_test_helper", ROOT / "scripts/train/validate_que_candidate_commands.py")
    return runner, cli


def _record(case, rank):
    score = {"balanced_distance": rank, "q95": rank, "worst_ratio": rank}
    return {"case": case["case"], "model": case["model"], "settings": case,
            "technical_status": "PASS", "scores": {"pooled": score, "origin_mean": score}}


def _command(runner, case, tmp_path):
    args = argparse.Namespace(data_dir=tmp_path / "intentionally_absent_data", device="cuda:0")
    return runner.command_for(case, args, tmp_path / case["case"])


@pytest.mark.parametrize("shared", [False, True])
def test_real_parser_accepts_stage_a_and_adaptive_variations(modules, tmp_path, shared):
    runner, cli = modules
    base = runner.stage_a_cases()
    unique = {runner.setting_key(case): case for case in base}
    # Different actual rankings select different parents. Include historical
    # max-epoch/loss anchors, both scalers, optimizers and correction settings.
    for order in (base, list(reversed(base)), sorted(base, key=lambda c: (c["normalization"], not c["zero_init_correction"], -c["share_weight"]))):
        records = [_record(case, index) for index, case in enumerate(order)]
        b_cases, _ = runner.adaptive_cases(records, base, "B")
        b_records = [_record(case, index / 1000) for index, case in enumerate(b_cases)]
        c_cases, _ = runner.adaptive_cases(b_records + records, base + b_cases, "C")
        for case in b_cases + c_cases:
            unique[runner.setting_key(case)] = case
    # Explicit boundary options available outside the default matrix.
    for model in runner.MODELS:
        additions = ({"recurrent_layout": "daily_vectors"},) if model in ("gru", "lstm") else (
            {"cam_temporal_layout": "full_history_flat", "cam_attention_scaling": "none"},
            {"cam_temporal_layout": "conv2d_day_hour", "cam_channel_sizes": [32, 32, 1]},
        )
        for settings in additions:
            case = runner.make_case(model, **settings)
            unique[runner.setting_key(case)] = case
    cases = list(unique.values())
    commands = [_command(runner, case, tmp_path) for case in cases]
    if shared:
        args = argparse.Namespace(data_dir=tmp_path / "intentionally_absent_data", device="cuda:0",
                                  allow_shared_gpu=True, shared_memory_limit_gib=6.0,
                                  shared_headroom_gib=2.0)
        commands = [runner.command_for(case, args, tmp_path / case["case"]) for case in cases]
        # Also exercise the actual resource-wrapper parser for every generated
        # command. It forwards the unchanged trainer argument vector exactly.
        wrapper = runner.load_helper("que_shared_gpu_runtime")
        for index, command in enumerate(commands):
            actual = runner.launch_command(command, args, tmp_path / f"resource_{index}.json")
            assert wrapper.parse_arguments(actual[3:]).trainer_args == command[3:]
    before = set(tmp_path.iterdir())
    results = cli.validate_commands(ROOT / "scripts/train/train_temporal_baselines.py", commands)
    assert len(results) == len(cases) > 415
    assert set(tmp_path.iterdir()) == before  # No datasets, predictions or fits.
    assert {case["stage"] for case in cases} == {"A", "B", "C"}
    for case, result in zip(cases, results):
        assert result["status"] == "PASS"
        assert result["model"] == case["model"]
        assert result["training"]["batch_size"] == case["batch_size"]
        assert result["training"]["best_epoch_scale"] == case["best_epoch_scale"]
        assert result["training"]["learning_rate_scale"] == case["learning_rate_scale"]
        assert result["training"]["normalization"] == case["normalization"]
        assert result["training"]["loss"] == case["loss"]
        if case["model"] in ("gru", "lstm"):
            assert result["training"]["recurrent_layout"] == case["recurrent_layout"]
            assert result["training"].get("independent_weight_decay_override") == case["independent_weight_decay"]
        else:
            assert result["cam"]["channel_sizes"] == case["cam_channel_sizes"]
            assert result["cam"]["temporal_layout"] == case["cam_temporal_layout"]
            assert result["model_config"]["correction_layout"] == case["correction_layout"]


def test_old_comma_command_rejected_by_actual_argparse(modules, tmp_path):
    runner, cli = modules
    command = _command(runner, runner.make_case("msnet"), tmp_path)
    index = command.index("--cam-channel-sizes")
    command[index + 1:index + 4] = ["16,16,1"]
    old_argv = sys.argv
    # Python argparse versions differ in whether nargs or int conversion is
    # reported first. Both must reject the historic single comma token.
    with pytest.raises(ValueError, match="SystemExit: 2") as error:
        cli.validate_commands(ROOT / "scripts/train/train_temporal_baselines.py", [command])
    assert "argument --cam-channel-sizes:" in str(error.value)
    assert sys.argv is old_argv
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("settings,needle", [
    ({"max_epochs": 10, "best_epoch_scale": .5}, "max-epochs"),
    ({"cam_channel_sizes": [16, 16, 2]}, "1"),
    ({"zero_init_correction": True, "correction_mode": "direct"}, "residual"),
    ({"learning_rate_scale": 0}, "positive"),
])
def test_real_post_parse_configuration_checks_run(modules, tmp_path, settings, needle):
    runner, cli = modules
    command = _command(runner, runner.make_case("mscmnet_w", **settings), tmp_path)
    with pytest.raises(ValueError, match=needle):
        cli.validate_commands(ROOT / "scripts/train/train_temporal_baselines.py", [command])
    assert not list(tmp_path.iterdir())


def test_unexpected_script_rejected_without_execution(modules, tmp_path):
    runner, cli = modules
    command = _command(runner, runner.make_case("gru"), tmp_path)
    command[2] = str(tmp_path / "other.py")
    with pytest.raises(ValueError, match="unexpected trainer script"):
        cli.validate_commands(ROOT / "scripts/train/train_temporal_baselines.py", [command])
