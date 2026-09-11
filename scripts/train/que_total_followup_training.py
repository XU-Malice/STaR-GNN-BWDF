"""Finite TOTAL-only GRU/LSTM refinements in an external, frozen checkout.

The caller owns the old-queue terminal check and its original GPU file lock.
This helper never changes the trainer, seed, prediction arrays or paper targets.
Published test targets guide this retrospective search; improvement is useful
even when the fixed pooled NSE/RMSE targets cannot all hold simultaneously.
"""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable

SEED = 20240604
MODELS = ("gru", "lstm")
TOTAL_KEYS = tuple(itertools.product(("24h", "168h"), ("MAE", "MAPE", "RMSE", "NSE")))
MAX_PER_MODEL = 24
MEMORY_LIMIT_GIB = 6.0
HEADROOM_GIB = 2.0


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load frozen helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rank(record: dict[str, Any]) -> tuple[float, float, str]:
    values = {}
    for row in record.get("metrics", []):
        if row.get("series", "total") != "total" or row.get("mode", "pooled") != "pooled":
            continue
        key = (row["task"], row["metric"])
        if key not in TOTAL_KEYS or key in values:
            raise ValueError("TOTAL ranking requires eight distinct pooled metric cells")
        ratio = float(row["tolerance_ratio"])
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError("TOTAL tolerance ratios must be finite and nonnegative")
        values[key] = ratio
    if set(values) != set(TOTAL_KEYS):
        raise ValueError("TOTAL ranking requires all eight pooled metric cells")
    return max(values.values()), sum(values.values()) / 8, record["case"]


def _check_case(case: dict[str, Any], runner: Any) -> None:
    if case.get("model") not in MODELS or case.get("seed") != SEED:
        raise ValueError("Follow-up training requires GRU/LSTM and the frozen seed 20240604")
    # Reject incomplete settings rather than silently changing the experiment.
    runner.setting_key(case)
    for key in ("learning_rate_scale", "best_epoch_scale"):
        if not math.isfinite(float(case[key])) or float(case[key]) <= 0:
            raise ValueError(f"Invalid positive training parameter: {key}")
    if isinstance(case["batch_size"], bool) or not isinstance(case["batch_size"], int) or case["batch_size"] < 1:
        raise ValueError("Invalid batch size")
    if case["loss"] not in ("mse", "mae", "huber"):
        raise ValueError("Invalid training loss")
    if case["max_epochs"] is not None and (isinstance(case["max_epochs"], bool)
            or not isinstance(case["max_epochs"], int) or case["max_epochs"] < 1):
        raise ValueError("Invalid maximum epoch count")


def generate_followups(records: list[dict[str, Any]], runner: Any,
                       max_per_model: int = MAX_PER_MODEL) -> list[dict[str, Any]]:
    """Choose two TOTAL-minimax parents and return at most 24 cases per model.

    Every historical setting, including failed or unstable settings, is excluded.
    Proposals alternate between parents, use one training seed, and preserve all
    untouched preprocessing/model options. No DMA or origin-mean score is used.
    """
    if isinstance(max_per_model, bool) or not isinstance(max_per_model, int) or max_per_model < 0:
        raise ValueError("max_per_model must be a nonnegative integer")
    limit = min(max_per_model, MAX_PER_MODEL)
    seen: set[str] = set()
    eligible: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    for record in records:
        if record.get("model") not in MODELS:
            continue
        settings = record["settings"]
        _check_case(settings, runner)
        if settings["model"] != record["model"]:
            raise ValueError("Historical record and training model differ")
        seen.add(runner.setting_key(settings))
        status = str(record.get("technical_status", record.get("status", "PASS"))).lower()
        if not (status.startswith("pass") or status in ("completed", "validated")):
            continue
        if not record.get("metrics"):
            continue
        _rank(record)
        eligible[record["model"]].append(record)
    result = []
    for model in MODELS:
        parents, parent_keys = [], set()
        for record in sorted(eligible[model], key=_rank):
            key = runner.setting_key(record["settings"])
            if key not in parent_keys:
                parents.append(record)
                parent_keys.add(key)
            if len(parents) == 2:
                break
        proposals = []
        for parent in parents:
            base = parent["settings"]
            updates = []
            pairs = [(.5, .65), (.75, .8), (1.25, 1.2), (1.5, 1.5)]
            for lr_factor, epoch_factor in pairs:
                change = {"learning_rate_scale": round(base["learning_rate_scale"] * lr_factor, 12)}
                if base["max_epochs"] is None:
                    change["best_epoch_scale"] = round(base["best_epoch_scale"] * epoch_factor, 12)
                else:
                    change["max_epochs"] = max(1, round(base["max_epochs"] * epoch_factor))
                updates.append((change, {"learning_rate_factor": lr_factor, "epoch_factor": epoch_factor}))
            updates.extend(({"batch_size": batch}, {"batch_size": batch}) for batch in (2, 4, 16, 32))
            updates.extend(({"loss": loss}, {"loss": loss}) for loss in ("mse", "mae", "huber"))
            # Fill any remaining budget after historical/unchanged-setting dedup.
            for lr_factor, epoch_factor in zip((.5, .75, 1.25, 1.5), (1.5, 1.2, .8, .65)):
                change = {"learning_rate_scale": round(base["learning_rate_scale"] * lr_factor, 12)}
                if base["max_epochs"] is None:
                    change["best_epoch_scale"] = round(base["best_epoch_scale"] * epoch_factor, 12)
                else:
                    change["max_epochs"] = max(1, round(base["max_epochs"] * epoch_factor))
                updates.append((change, {"learning_rate_factor": lr_factor, "epoch_factor": epoch_factor}))
            proposals.append((parent, updates))
        count = 0
        for index in range(max((len(updates) for _, updates in proposals), default=0)):
            for parent, updates in proposals:
                if count >= limit or index >= len(updates):
                    continue
                settings = {key: copy.deepcopy(parent["settings"][key]) for key in runner.TRAINING_KEYS if key != "model"}
                update, refinement = updates[index]
                settings.update(update)
                case = runner.make_case(model, stage="F", **settings)
                _check_case(case, runner)
                key = runner.setting_key(case)
                if key in seen:
                    continue
                case.update(parent_case=parent["case"], parent_total_rank=list(_rank(parent)[:2]),
                            refinement=refinement, selection_mode="pooled_total8_worst_then_mean",
                            test_target_feedback=True, paper_reproduction_verified=False)
                seen.add(key)
                result.append(case)
                count += 1
    return result


def _command(case: dict[str, Any], runner: Any, tool_root: Path,
             data_dir: Path, case_root: Path) -> list[str]:
    args = SimpleNamespace(data_dir=data_dir, device="cuda:0", allow_shared_gpu=True,
                           shared_memory_limit_gib=MEMORY_LIMIT_GIB)
    command = runner.command_for(case, args, case_root)
    command[2] = str(tool_root / "scripts/train/train_temporal_baselines.py")
    # Bind configuration paths explicitly as well as pinning the working dir.
    command += ["--config", str(tool_root / "configs/model/mscmnet_baselines.yaml"),
                "--split-config", str(tool_root / "configs/data/paper_split.yaml")]
    return command


def preflight_followups(cases: list[dict[str, Any]], *, tool_root: Path,
                        data_dir: Path, output_root: Path, runner: Any = None,
                        command_validator: Any = None) -> list[dict[str, Any]]:
    """Exercise every exact trainer command on CPU before scheduling GPU work.

    The unchanged trainer has no --validate-only flag. Its existing validator
    invokes the real main/parser/configuration and stops before device/data work.
    """
    tool_root = Path(tool_root).resolve(strict=True)
    runner = runner or _load(tool_root / "scripts/train/run_que_comprehensive_reconstruction.py", "_que_followup_runner")
    commands = []
    for case in cases:
        _check_case(case, runner)
        commands.append(_command(case, runner, tool_root, Path(data_dir).resolve(),
                                 Path(output_root).resolve() / "cases" / case["case"]))
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        return (command_validator or runner.validate_commands)(
            tool_root / "scripts/train/train_temporal_baselines.py", commands)
    finally:
        sys.dont_write_bytecode = old


def _total_metrics(arrays: dict[str, Any], model: str, paper: dict[str, Any],
                   runner: Any) -> list[dict[str, Any]]:
    values = {}
    for row in runner.metric_rows(arrays, "pooled"):
        if row["series"] != "total":
            continue
        key = (row["task"], row["metric"])
        if key not in TOTAL_KEYS or key in values:
            raise ValueError("Invalid raw TOTAL metric table")
        value = float(row["value"])
        target = float(paper["tasks"][key[0]][runner.MODEL_NAMES[model]]["total"][key[1]])
        if not math.isfinite(value) or not math.isfinite(target) or (key[1] != "NSE" and target <= 0):
            raise ValueError("Nonfinite metric or invalid frozen paper target")
        difference = value - target
        relative = abs(difference) / abs(target) if target else None
        ratio = abs(difference) / .01 if key[1] == "NSE" else relative / .05
        values[key] = {**row, "paper_value": target, "difference": difference,
                       "absolute_relative_difference": relative, "tolerance_ratio": ratio,
                       "within_tolerance": ratio <= 1, "mode": "pooled"}
    if set(values) != set(TOTAL_KEYS):
        raise ValueError("Incomplete raw TOTAL metric table")
    return [values[key] for key in TOTAL_KEYS]


def execute_followup(case: dict[str, Any], *, tool_root: Path, data_dir: Path,
                     output_root: Path, paper_config: Path, evaluation: dict[str, Any],
                     campaign_manifest: dict[str, Any], gpu_id: str = "7",
                     runner: Any = None, supervisor: Any = None,
                     resource_helper: Any = None, source_check: Callable[[], None] | None = None,
                     command_validator: Any = None, max_resource_retries: int = 2,
                     poll_seconds: float = 30, deadline: float | None = None,
                     clock: Callable[[], float] = time.monotonic,
                     sleep: Callable[[float], None] = time.sleep,
                     retry_failed: bool = False) -> dict[str, Any]:
    """Run or revalidate one frozen candidate, preserving every prior artifact.

    The parent must hold the original shared-GPU lock until this returns. Only
    verified wrapper resource failures may retry automatically (at most twice).
    Failed/incomplete existing outputs require explicit retry_failed=True and
    are archived intact first. Completed outputs always require their receipt.
    """
    import yaml

    tool_root, data_dir = Path(tool_root).resolve(strict=True), Path(data_dir).resolve(strict=True)
    output_root, paper_config = Path(output_root).resolve(), Path(paper_config).resolve(strict=True)
    if str(gpu_id) != "7":
        raise ValueError("This external campaign is restricted to shared physical GPU 7")
    if isinstance(max_resource_retries, bool) or not isinstance(max_resource_retries, int) or not 0 <= max_resource_retries <= 2:
        raise ValueError("Resource retries must be an integer from zero to two")
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError("Resource polling must be positive and at most 60 seconds")
    deadline = clock() + 86400 if deadline is None else deadline
    if math.isnan(deadline):
        raise ValueError("Deadline cannot be NaN")
    for protected in (data_dir, *(tool_root / name for name in ("src", "scripts", "configs", "tests"))):
        if output_root == protected or protected in output_root.parents:
            raise ValueError("Follow-up output cannot be inside frozen source/config/data directories")
    runner = runner or _load(tool_root / "scripts/train/run_que_comprehensive_reconstruction.py", "_que_followup_runner")
    _check_case(case, runner)
    if case.get("stage") != "F" or case.get("case") != f"F_{case['model']}_{runner.setting_key(case)[:12]}":
        raise ValueError("Unsafe or mismatched follow-up case identity")
    if not re.fullmatch(r"[A-Za-z0-9_]+", case["case"]):
        raise ValueError("Unsafe follow-up case path")
    life = runner.life
    manifest = copy.deepcopy(campaign_manifest)
    if manifest.get("signature") != life.digest({key: value for key, value in manifest.items() if key != "signature"}):
        raise ValueError("Frozen campaign manifest self-signature mismatch")
    def check() -> None:
        if source_check is not None:
            source_check()
        elif life.fingerprints(tool_root, data_dir) != manifest["signatures"] or life.file_digest(paper_config) != manifest["paper_sha256"]:
            raise RuntimeError("Frozen source/data/paper changed; outputs preserved")
    check()
    sys.path.insert(0, str(tool_root / "src"))
    published = yaml.safe_load((tool_root / "configs/model/mscmnet_baselines.yaml").read_text())["models"]
    paper = yaml.safe_load(paper_config.read_text())
    request = {"signature": life.digest({"manifest": manifest["signature"], "settings": runner.setting_key(case)}),
               "case": copy.deepcopy(case), "model_config": runner.expected_model_config(published, case),
               "evaluation": copy.deepcopy(evaluation)}
    case_root = output_root / "cases" / case["case"]
    run = case_root / case["model"] / f"seed_{SEED}"
    log_root = output_root / "logs" / case["case"]
    record_path = output_root / "followup_records" / f"{case['case']}.json"
    # No writes through preexisting symlink output components.
    for path in (case_root, run, log_root, record_path, output_root / "resource_attempts" / case["case"]):
        for component in (path, *path.parents):
            if component.is_symlink():
                raise ValueError(f"Refusing symlink output component: {component}")
            if component == output_root:
                break
    record = {"case": case["case"], "model": case["model"], "stage": "F", "settings": copy.deepcopy(case),
              "run": str(run), "request": request, "seed": SEED, "selection_mode": "pooled_total8_worst_then_mean",
              "test_target_feedback": True, "paper_reproduction_verified": False,
              "working_directory": str(tool_root), "gpu_id": "7", "memory_limit_gib": MEMORY_LIMIT_GIB,
              "headroom_gib": HEADROOM_GIB, "thread_limit": 2,
              "resource_attempts": [], "elapsed_seconds": 0.0}
    previous = json.loads(record_path.read_text()) if record_path.is_file() else None
    def save() -> None:
        record["updated_utc"] = life.utc_now()
        life.atomic_json(record_path, record)
    def fail(reason: str, *, existing: bool = False) -> dict[str, Any]:
        record.update(technical_status="FAIL(existing)" if existing else "FAIL", validation=reason)
        save()
        return record
    def complete(existing: bool) -> dict[str, Any]:
        import numpy as np
        check()
        valid, reason = runner.validate_case(run, case, request)
        if not valid:
            return fail(reason, existing=existing)
        with np.load(run / "predictions_common46.npz", allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        rows = _total_metrics(arrays, case["model"], paper, runner)
        ratios = [row["tolerance_ratio"] for row in rows]
        valid, reason = runner.validate_case(run, case, request)
        if not valid:
            return fail(f"evidence_changed_during_metric_read:{reason}", existing=existing)
        record.update(technical_status="PASS(existing)" if existing else "PASS", validation=reason,
                      exit_code=0, completion_receipt=json.loads((run / "completion_receipt.json").read_text()),
                      raw_array_hashes={key: life.array_digest(value) for key, value in arrays.items()},
                      metrics=rows, worst_ratio=max(ratios), mean_ratio=sum(ratios) / 8,
                      matched_count=sum(ratio <= 1 for ratio in ratios), all8_matched=all(ratio <= 1 for ratio in ratios),
                      finished_utc=life.utc_now())
        check()
        save()
        return record
    def archive(label: str) -> str | None:
        if not case_root.exists():
            return None
        base = output_root / "resource_attempts" / case["case"]
        if base.is_symlink() or base.parent.is_symlink():
            raise ValueError("Refusing symlink resource archive")
        base.mkdir(parents=True, exist_ok=True)
        index = 1
        while (base / f"{label}_{index:03d}").exists():
            index += 1
        destination = base / f"{label}_{index:03d}"
        case_root.rename(destination)
        return str(destination)
    valid, reason = runner.validate_case(run, case, request)
    if valid:
        if previous and previous.get("request") == request:
            record.update({key: previous[key] for key in ("resource_attempts", "elapsed_seconds", "command", "launch_command") if key in previous})
        try:
            return complete(True)
        except Exception as exc:
            return fail(f"existing_validation_failed:{type(exc).__name__}:{exc}", existing=True)
    terminal_previous = previous and previous.get("technical_status") not in ("PAUSED", "waiting_gpu")
    if case_root.exists() or terminal_previous:
        if not retry_failed:
            if previous:
                record.update({key: previous[key] for key in ("resource_attempts", "elapsed_seconds", "command", "launch_command") if key in previous})
                record["previous_validation"] = previous.get("previous_validation", previous.get("validation"))
            return fail(f"existing_evidence_preserved:{reason}", existing=True)
        record["explicit_retry_archive"] = archive("explicit_retry")
        if previous:
            old = output_root / "resource_attempts" / case["case"] / f"record_{time.time_ns()}.json"
            life.atomic_json(old, previous)
            record["previous_record_archive"] = str(old)
    elif previous:
        if previous.get("request") != request:
            return fail("existing_request_changed", existing=True)
        record.update({key: previous[key] for key in ("resource_attempts", "elapsed_seconds") if key in previous})
    log_root.mkdir(parents=True, exist_ok=True)
    command = _command(case, runner, tool_root, data_dir, case_root)
    record.update(command=command, started_utc=life.utc_now(), technical_status="preflight")
    save()
    handlers = {}
    try:
        check()
        validation = preflight_followups([case], tool_root=tool_root, data_dir=data_dir,
                                        output_root=output_root, runner=runner, command_validator=command_validator)
        if len(validation) != 1 or validation[0].get("status") != "PASS":
            raise ValueError("Real trainer command preflight did not pass")
        life.atomic_json(log_root / "command_preflight.json", {"command": command, "validation": validation})
        helper = resource_helper or _load(tool_root / "scripts/train/que_shared_gpu_runtime.py", "_que_followup_resource")
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES="7", PYTHONPATH=str(tool_root / "src"),
                           PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            environment[key] = "2"
        supervisor = supervisor or life.ChildSupervisor(tool_root, environment)
        if hasattr(supervisor, "signal_handler"):
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.signal(signum, supervisor.signal_handler)
        attempts = record["resource_attempts"]
        while len(attempts) < max_resource_retries + 1:
            check()
            if clock() >= deadline:
                record.update(technical_status="PAUSED", validation="resource_deadline_expired")
                save()
                return record
            def waiting(snapshot: dict[str, Any]) -> None:
                check()
                record.update(technical_status="waiting_gpu", gpu_wait=snapshot)
                save()
            try:
                snapshot = helper.wait_for_memory("7", 8192, poll_seconds, deadline, waiting)
            except helper.GpuWaitExpired as exc:
                record.update(technical_status="PAUSED", validation=f"resource_deadline_expired:{exc}")
                save()
                return record
            check()
            number = len(attempts) + 1
            report_path = log_root / f"resource_attempt_{number:03d}.json"
            actual = helper.wrap_command(command, MEMORY_LIMIT_GIB, HEADROOM_GIB, report_path)
            record.update(technical_status="running", launch_command=actual)
            record.pop("gpu_wait", None)
            life.atomic_json(log_root / f"gpu_admission_{number:03d}.json", snapshot)
            save()
            tick = clock()
            rc = supervisor.run(actual, log_root / f"training_attempt_{number:03d}.log")
            duration = max(0.0, clock() - tick)
            detail = json.loads(report_path.read_text()) if report_path.is_file() else {"status": "missing_resource_report"}
            attempt = {"exit_code": rc, "elapsed_seconds": duration, "report": str(report_path),
                       "resource_status": detail.get("status"), "command": actual}
            if report_path.is_file():
                attempt["report_sha256"] = life.file_digest(report_path)
            attempts.append(attempt)
            record.update(exit_code=rc, elapsed_seconds=record["elapsed_seconds"] + duration,
                          resource_status=detail.get("status"))
            save()
            check()
            if rc == 75 and detail.get("status") in ("resource_wait", "resource_oom"):
                if len(attempts) >= max_resource_retries + 1:
                    return fail(f"shared_gpu_resource_retries_exhausted:{detail['status']}")
                attempt["preserved_output"] = archive(f"resource_attempt_{number:03d}")
                record.update(technical_status="waiting_gpu", validation="retrying_verified_resource_failure")
                save()
                remaining = deadline - clock()
                if remaining > 0:
                    sleep(min(poll_seconds, remaining))
                continue
            if rc != 0:
                return fail(f"trainer_exit_{rc}:{detail.get('status')}")
            if detail.get("status") != "completed":
                return fail("successful_trainer_missing_completed_resource_report")
            # These sidecars do not exist in unchanged trainer output. Never
            # replace an unexpected file already present in a fresh candidate.
            request_path, receipt_path = run / "request_signature.json", run / "completion_receipt.json"
            if request_path.exists() or receipt_path.exists():
                return fail("unexpected_existing_provenance_sidecar")
            if not run.is_dir():
                return fail("successful_trainer_missing_run_directory")
            life.atomic_json(request_path, request)
            valid, reason = runner.validate_case(run, case, request, require_receipt=False)
            if not valid:
                return fail(reason)
            status = json.loads((run / "status.json").read_text())
            life.atomic_json(receipt_path, {"request_sha256": life.digest(request), "files": runner.evidence_hashes(run, status)})
            return complete(False)
        return fail("shared_gpu_resource_retry_budget_already_exhausted")
    except (KeyboardInterrupt, life.InterruptedRun) as exc:
        fail(f"owned_followup_interrupted:{type(exc).__name__}:{exc}")
        raise
    except Exception as exc:
        return fail(f"followup_error:{type(exc).__name__}:{exc}")
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
