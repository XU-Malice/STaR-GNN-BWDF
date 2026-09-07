"""Exercise the real trainer CLI and configuration before any GPU/data work.

This deliberately does not reimplement argparse. The trainer's unmodified main()
parses each exact launch command, loads its actual YAML and performs its option
checks. Device resolution is the stop boundary, before resource inspection,
dataset loading, output creation or fitting. Run sequentially in the queue.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib.util
import io
from pathlib import Path
import sys
from typing import Any


class _ConfigurationValidated(BaseException):
    """Private control flow: never mistaken for a trainer validation error."""


def validate_commands(training_script: Path, commands: list[list[str]]) -> list[dict[str, Any]]:
    """Validate exact ``python -u trainer.py ...`` commands without fitting.

    Raises ValueError for invalid arguments, invalid combined settings, or a
    trainer that no longer reaches the expected pre-data boundary. The caller
    must finish this for every candidate in a stage before launching that stage.
    """
    script = training_script.resolve(strict=True)
    module_name = "_que_real_cli_preflight_trainer"
    previous_module = sys.modules.get(module_name)
    old_argv = sys.argv
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import trainer for command preflight: {script}")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = trainer
    results = []
    try:
        spec.loader.exec_module(trainer)
        original_read = trainer.read_yaml
        loaded: list[dict[str, Any]] = []

        def read_yaml(path):
            value = original_read(path)
            loaded.append(value)
            return value

        def stop_before_device(*args, **kwargs):
            raise _ConfigurationValidated()

        def forbidden_work(*args, **kwargs):
            raise RuntimeError("Command preflight crossed the no-data/no-training boundary")

        trainer.read_yaml = read_yaml
        trainer.resolve_device = stop_before_device
        trainer.preflight_resources = forbidden_work
        trainer.load_paper_data = forbidden_work
        trainer.run_one_model = forbidden_work
        for index, command in enumerate(commands):
            if (len(command) < 4 or command[1] != "-u"
                    or not all(isinstance(value, str) for value in command)):
                raise ValueError(f"Candidate {index}: unexpected trainer command structure")
            command_script = Path(command[2])
            # Queue commands are relative to the repository, not the caller cwd.
            if not command_script.is_absolute():
                command_script = script.parents[2] / command_script
            if command_script.resolve() != script:
                raise ValueError(f"Candidate {index}: unexpected trainer script: {command[2]}")
            loaded.clear()
            sys.argv = [str(script), *command[3:]]
            output = io.StringIO()
            reached_boundary = False
            try:
                with redirect_stdout(output), redirect_stderr(output):
                    trainer.main()
            except _ConfigurationValidated:
                reached_boundary = True
            except (Exception, SystemExit) as exc:
                detail = output.getvalue().strip()
                raise ValueError(
                    f"Candidate {index} failed real trainer CLI/configuration preflight: "
                    f"{type(exc).__name__}: {exc}\n{detail}"
                ) from exc
            if not reached_boundary or not loaded:
                raise ValueError(f"Candidate {index}: trainer did not reach the pre-device boundary")
            config = loaded[0]
            # Read the model from the exact arguments; main() already validated it.
            model = command[command.index("--model") + 1]
            canonical = trainer.ALIASES.get(model, model)
            results.append({
                "status": "PASS", "scope": "actual_trainer_cli_and_configuration",
                "model": canonical, "training": copy.deepcopy(config["training"]),
                "cam": copy.deepcopy(config["cam"]),
                "model_config": copy.deepcopy(config["models"].get(canonical)),
            })
    finally:
        sys.argv = old_argv
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    return results
