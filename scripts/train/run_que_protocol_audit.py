#!/usr/bin/env python3
"""Fixed, single-seed protocol audit; numerical closeness never changes the queue.

The 24 cases enumerate two unpublished preprocessing/optimizer choices for
each of six algorithms. They are reconstruction hypotheses, not a claim that
the original author's implementation or results have been recovered.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = ("gru", "lstm", "msnet", "mscmnet_m", "mscmnet_wm", "mscmnet_w")
SEED = 20240604
METRICS = ("MAE", "MAPE", "RMSE", "NSE")
DEFAULT_EXISTING = (
    "results/que_complete_reproduction_20260903",
    "results/que_targeted_reproduction_20260904",
    "results/que_selected_joint_baselines_20260901",
)
PREFLIGHT_TESTS = (
    "tests/test_reproduction_metrics.py",
    "tests/test_mscmnet_models.py",
    "tests/test_temporal_baseline_training.py",
    "tests/test_que_protocol_runner.py",
    "tests/test_stop_que_queue.py",
    "tests/test_que_data_protocol.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_digest(values: Any) -> str:
    import numpy as np
    array = np.asarray(values)
    result = hashlib.sha256()
    result.update(str(array.dtype).encode())
    result.update(str(array.shape).encode())
    result.update(np.ascontiguousarray(array).tobytes())
    return result.hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_cases() -> list[dict[str, Any]]:
    return [
        {
            "case": f"{model}_{normalization}_{optimizer}",
            "model": model,
            "normalization": normalization,
            "optimizer": optimizer,
            "seed": SEED,
            "batch_size": 8,
            "loss": "mse",
            "train_stride_hours": 24,
            "best_epoch_scale": 1.0,
            "learning_rate_scale": 1.0,
            "weight_decay": "published_per_model_or_dma",
            "correction_mode": "direct",
            "zero_init_correction": False,
            "fc2_share_supervision_weight": 0.0,
            "cam_attention_update": "replace",
            "cam_attention_scaling": "none",
            "cam_temporal_layout": "per_day_vectors",
            "protocol_status": "reconstruction_with_unpublished_choices",
        }
        for model, normalization, optimizer in itertools.product(
            MODELS, ("zscore", "minmax"), ("adam", "adamw")
        )
    ]


def strict_audit_passes(summary: dict[str, Any], expected_cases: int) -> bool:
    """Technical consistency only; paper-tolerance failures do not enter here."""
    return (
        summary.get("status") == "completed"
        and summary.get("valid_sources") == expected_cases
        and summary.get("invalid_sources") == 0
        and summary.get("stored_metric_roundoff_exceeded") == 0
        and summary.get("first_day_discrepancy_sources") == 0
        and set(summary.get("models_with_raw_predictions", [])) == set(MODELS)
        and summary.get("truth_groups_per_task") == {"24h": 1, "168h": 1}
        and summary.get("truth_origin_group_mismatch") is False
    )


def source_files(root: Path) -> list[Path]:
    """Include uncommitted source edits/new tests as well as tracked files."""
    paths = []
    for directory in ("src", "scripts", "configs", "tests"):
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    paths.extend(root / name for name in ("pyproject.toml", "uv.lock") if (root / name).is_file())
    return sorted(set(paths))


def fingerprints(root: Path, data_dir: Path) -> dict[str, Any]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Processed data directory missing: {data_dir}")
    data_paths = sorted(path for path in data_dir.rglob("*") if path.is_file())
    if not data_paths:
        raise ValueError(f"Processed data directory is empty: {data_dir}")
    source = {str(path.relative_to(root)): file_digest(path) for path in source_files(root)}
    data = {str(path.relative_to(data_dir)): file_digest(path) for path in data_paths}
    return {"source": source, "data": data, "source_sha256": digest(source), "data_sha256": digest(data)}


class InterruptedRun(Exception):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"Queue interrupted by signal {signum}; owned child groups stopped")


class ChildSupervisor:
    """Each child owns a new process group, so termination never targets others."""

    def __init__(self, root: Path, environment: dict[str, str]):
        self.root = root
        self.environment = environment
        self.child: subprocess.Popen[str] | None = None

    def terminate_owned(self) -> None:
        child = self.child
        if child is None:
            return
        # The group may still contain grandchildren after its leader exits.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)
        self.child = None

    def signal_handler(self, signum: int, _frame: Any) -> None:
        # Block repeated signals during bounded cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.terminate_owned()
        raise InterruptedRun(signum)

    def run(self, command: list[str], log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[{utc_now()}] command={json.dumps(command)}\n")
            stream.flush()
            self.child = subprocess.Popen(
                command, cwd=self.root, env=self.environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True,
            )
            try:
                assert self.child.stdout is not None
                for line in self.child.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                rc = self.child.wait()
                # Also clean children that a failed command might have orphaned.
                self.terminate_owned()
                return rc
            except BaseException:
                self.terminate_owned()
                raise


def gpu_preflight(gpu_id: str, minimum_free_mib: int) -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "-i", gpu_id, "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=30,
    )
    rows = list(csv.reader(query.stdout.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError(f"Ambiguous GPU result: {query.stdout!r}")
    gpu_uuid, name, free = (value.strip() for value in rows[0])
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=30,
    )
    foreign = [row for row in csv.reader(processes.stdout.strip().splitlines()) if row and row[0].strip() == gpu_uuid]
    if foreign:
        raise RuntimeError(f"GPU {gpu_id} is occupied; no process was killed: {foreign}")
    if int(free) < minimum_free_mib:
        raise RuntimeError(f"GPU {gpu_id} free memory {free} MiB < {minimum_free_mib} MiB")
    return {"gpu_uuid": gpu_uuid, "name": name, "free_mib": int(free), "time": utc_now()}


def command_for(case: dict[str, Any], args: argparse.Namespace, output_root: Path) -> list[str]:
    command = [
        sys.executable, "-u", "scripts/train/train_temporal_baselines.py",
        "--model", case["model"], "--seed", str(case["seed"]),
        "--normalization", case["normalization"], "--optimizer", case["optimizer"],
        "--batch-size", "8", "--loss", "mse", "--train-stride-hours", "24",
        "--learning-rate-scale", "1", "--best-epoch-scale", "1",
        "--data-dir", str(args.data_dir), "--output-root", str(output_root),
        "--device", args.device,
    ]
    if args.device == "cpu":
        command.append("--allow-cpu")
    if case["model"] not in ("gru", "lstm"):
        command += ["--cam-attention-update", "replace", "--cam-attention-scaling", "none", "--cam-temporal-layout", "per_day_vectors"]
    if case["model"].startswith("mscmnet_"):
        command += ["--correction-mode", "direct"]
    if case["model"] in ("mscmnet_wm", "mscmnet_w"):
        command += ["--fc2-share-supervision-weight", "0"]
    return command


def validate_case(run: Path, case: dict[str, Any], signature: str) -> tuple[bool, str]:
    """No execution flag alone makes a run reusable; inspect its actual files."""
    import numpy as np
    import yaml
    from dma_wdf.data.reproduction_metrics import canonical_forecast_origins

    try:
        request = json.loads((run / "request_signature.json").read_text())
        if request.get("signature") != signature:
            return False, "request_signature_mismatch"
        status = json.loads((run / "status.json").read_text())
        if status.get("status") != "completed" or status.get("model") != case["model"] or status.get("seed") != case["seed"]:
            return False, "incomplete_or_wrong_model_seed"
        if status.get("single_frozen_checkpoint_for_24h_and_168h") is not True:
            return False, "not_single_frozen_checkpoint"
        checkpoints = status.get("checkpoint_files", [])
        count = 10 if case["model"] in ("gru", "lstm") else 1
        if len(checkpoints) != count or len(set(checkpoints)) != count:
            return False, "wrong_checkpoint_count"
        for name in checkpoints:
            if Path(name).name != name or not (run / name).is_file() or (run / name).stat().st_size == 0:
                return False, "missing_or_invalid_checkpoint"
        config = yaml.safe_load((run / "resolved_config.yaml").read_text())
        if request.get("model_config") != config["model"]:
            return False, "published_model_config_mismatch"
        training = config["training"]
        for key in ("normalization", "optimizer", "batch_size", "loss", "learning_rate_scale", "best_epoch_scale"):
            if training.get(key) != case[key]:
                return False, f"resolved_config_mismatch:{key}"
        if config.get("seed") != case["seed"] or config.get("train_stride_hours") != 24 or config.get("max_epochs_override") is not None or config.get("max_train_batches") is not None:
            return False, "training_override_or_seed_mismatch"
        if case["model"] not in ("gru", "lstm"):
            for key, expected in (("attention_update", "replace"), ("attention_scaling", "none"), ("temporal_layout", "per_day_vectors")):
                if config["cam"].get(key) != expected:
                    return False, f"cam_mismatch:{key}"
        if case["model"].startswith("mscmnet_"):
            if config["model"].get("correction_mode") != "direct" or config["model"].get("zero_init_correction") is not False:
                return False, "correction_override"
            if config["model"].get("fc2", {}).get("share_supervision_weight", 0) != 0:
                return False, "auxiliary_share_loss"
        with np.load(run / "predictions_common46.npz", allow_pickle=False) as arrays:
            for horizon in (24, 168):
                for kind in ("true", "pred"):
                    values = arrays[f"y_{kind}_{horizon}h"]
                    if values.shape != (46, horizon, 10) or not np.isfinite(values).all():
                        return False, "invalid_prediction_shape_or_values"
                if array_digest(arrays[f"y_true_{horizon}h"]) != request["evaluation"]["truths"][f"{horizon}h"]["array_sha256"]:
                    return False, "truth_does_not_match_audited_data"
                if not np.allclose(arrays[f"y_pred_{horizon}h"][:, :24], arrays["y_pred_24h"], rtol=1e-5, atol=1e-5):
                    return False, "recursive_first_day_mismatch"
            if not np.array_equal(arrays["y_true_168h"][:, :24], arrays["y_true_24h"]):
                return False, "truth_first_day_mismatch"
            if arrays["dma_letters"].tolist() != list("ABCDEFGHIJ"):
                return False, "wrong_dma_order"
            starts = arrays["forecast_starts"]
            if starts.shape != (46,) or len(set(starts.tolist())) != 46:
                return False, "wrong_forecast_origins"
            try:
                same_origins = np.array_equal(
                    canonical_forecast_origins(starts),
                    canonical_forecast_origins(request["evaluation"]["forecast_starts"]),
                )
            except ValueError:
                return False, "origins_do_not_match_audited_data"
            if not same_origins:
                return False, "origins_do_not_match_audited_data"
        with (run / "metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        keys = [(row["task"], row["series"], row["metric"]) for row in rows]
        expected_keys = set(itertools.product(("24h", "168h"), [*"ABCDEFGHIJ", "total"], METRICS))
        if len(keys) != 88 or set(keys) != expected_keys or any(not math.isfinite(float(row["value"])) for row in rows):
            return False, "invalid_metric_keyset_or_values"
        with (run / "loss_curve.csv").open(newline="") as stream:
            loss_rows = list(csv.DictReader(stream))
        if not loss_rows or any(not math.isfinite(float(row["train_loss"])) for row in loss_rows):
            return False, "missing_or_nonfinite_loss_curve"
        if case["model"] in ("gru", "lstm"):
            if {row.get("dma") for row in loss_rows} != set("ABCDEFGHIJ"):
                return False, "loss_curve_missing_dma"
            for letter, epochs in zip("ABCDEFGHIJ", config["model"]["best_epochs"]):
                if [int(row["epoch"]) for row in loss_rows if row["dma"] == letter] != list(range(1, int(epochs) + 1)):
                    return False, "loss_curve_wrong_published_epochs"
        elif [int(row["epoch"]) for row in loss_rows] != list(range(1, int(config["model"]["best_epoch"]) + 1)):
            return False, "loss_curve_wrong_published_epochs"
        return True, "validated_artifacts"
    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
        return False, f"invalid_artifact:{type(exc).__name__}:{exc}"


def make_bundle(root: Path, result_root: Path, log_root: Path) -> Path:
    """Keep common-46 truth/predictions for independent re-evaluation."""
    bundle = root.parent / f"{result_root.name}_compact.tar.gz"
    temporary = bundle.with_name(bundle.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for directory in (result_root, log_root):
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or any(".backup-" in part for part in path.parts) or path.suffix == ".pt":
                    continue
                if path.suffix == ".npz" and path.name != "predictions_common46.npz":
                    continue
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        evidence_index = []
        provenance_path = result_root / "audit_existing/provenance.json"
        evidence = json.loads(provenance_path.read_text()) if provenance_path.is_file() else []
        allowed_root = (root / "results").resolve()
        for source in evidence:
            path = Path(source["source_path"]).resolve()
            if not path.is_relative_to(allowed_root):
                evidence_index.append({"source_id": source["source_id"], "status": "excluded_outside_project_results"})
                continue
            if path.is_relative_to(result_root.resolve()):
                continue
            source_id = source["source_id"]
            if not re.fullmatch(r"[a-f0-9]{16}", source_id):
                raise ValueError("Invalid evidence source identifier")
            source_files_to_save = {
                "predictions_common46.npz": source["npz_sha256"],
                "status.json": source["status_sha256"],
                "resolved_config.yaml": source["config_sha256"],
                "metrics.csv": None,
                "loss_curve.csv": None,
            }
            included = {}
            for name, expected_hash in source_files_to_save.items():
                candidate = (path.parent / name).resolve()
                if not candidate.is_relative_to(allowed_root):
                    raise ValueError("Evidence companion resolves outside project/results")
                if not candidate.is_file() and expected_hash is None:
                    continue
                actual_hash = file_digest(candidate)
                if expected_hash is not None and actual_hash != expected_hash:
                    raise ValueError(f"Audited evidence changed before packaging: {candidate}")
                archive.add(candidate, arcname=f"evidence/{source_id}/{name}", recursive=False)
                included[name] = actual_hash
            evidence_index.append({"source_id": source_id, "original_source_path": str(path), "files_sha256": included, "status": "included"})
        payload = (json.dumps(evidence_index, indent=2) + "\n").encode()
        info = tarfile.TarInfo("evidence/index.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    os.replace(temporary, bundle)
    checksum = file_digest(bundle)
    bundle.with_name(bundle.name + ".sha256").write_text(f"{checksum}  {bundle.name}\n")
    print(f"Result bundle: {bundle}\nSHA256: {checksum}", flush=True)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="que_protocol_audit_20260905")
    parser.add_argument("--gpu-id", default="6")
    parser.add_argument("--minimum-free-mib", type=int, default=8192)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/data_build")
    parser.add_argument("--results-root", type=Path, action="append", help="Existing results to re-evaluate; repeatable")
    parser.add_argument("--paper-config", type=Path, default=PROJECT_ROOT / "configs/evaluation/mscmnet_paper_metrics.yaml")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Print the fixed manifest; no imports, writes, lock, or GPU use")
    parser.add_argument("--audit-only", action="store_true", help="Re-evaluate saved predictions without training or GPU use")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.run_tag):
        parser.error("--run-tag must be a simple directory name")
    if not re.fullmatch(r"[0-9]+", args.gpu_id):
        parser.error("--gpu-id must be a physical numeric GPU index")
    cases = make_cases()
    if args.dry_run:
        print(json.dumps({"case_count": len(cases), "seed": SEED, "selection_policy": "fixed_all_cases_no_paper_target_feedback", "cases": cases}, indent=2))
        return 0
    root = PROJECT_ROOT
    args.data_dir = args.data_dir.resolve()
    args.paper_config = args.paper_config.resolve()
    result_root = root / "results" / args.run_tag
    log_root = root / "logs" / args.run_tag
    log_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "logs" / f"que_gpu_{args.gpu_id if args.device != 'cpu' else 'cpu'}.lock"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src") + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu_id if args.device != "cpu" and not args.audit_only else ""
    environment["PYTHONUNBUFFERED"] = "1"
    supervisor = ChildSupervisor(root, environment)
    old_handlers = {sig: signal.signal(sig, supervisor.signal_handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    queue = {"status": "running", "started_utc": utc_now(), "technical_success": False, "reproduction_claim": "not_established", "case_count": len(cases), "cases": []}
    exit_code = 0
    owns_lock = False
    lock = lock_path.open("a+")
    try:
        # Keep ownership until final status and bundle are safely written.
        with nullcontext(lock):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another project GPU queue owns {lock_path}; nothing stopped") from exc
            owns_lock = True
            existing_roots = args.results_root if args.results_root is not None else [root / path for path in DEFAULT_EXISTING]
            existing_roots = [path.resolve() for path in existing_roots]
            audit_base = [sys.executable, "-u", "scripts/reproduce/audit_que_saved_predictions.py", "--paper-config", str(args.paper_config)]
            data_audit = [sys.executable, "-u", "scripts/reproduce/audit_que_data_protocol.py", "--data-dir", str(args.data_dir), "--split-config", str(root / "configs/data/paper_split.yaml"), "--output-root", str(result_root / "audit_data_protocol")]
            print("Phase 0/3: CPU-only source-data protocol audit", flush=True)
            if supervisor.run(data_audit, log_root / "audit_data_protocol.log"):
                raise RuntimeError("Data protocol audit failed; no training started")
            data_report = json.loads((result_root / "audit_data_protocol/paper_data_statistics.json").read_text())
            evaluation = data_report["common_evaluation"]
            before = audit_base + ["--allow-invalid-evidence"] + [item for path in existing_roots for item in ("--results-root", str(path))] + ["--output-root", str(result_root / "audit_existing")]
            print("Phase 1/3: recompute existing saved predictions; no training", flush=True)
            if supervisor.run(before, log_root / "audit_existing.log"):
                raise RuntimeError("Existing-prediction audit failed; see audit_existing.log; no training started")
            if args.audit_only:
                queue.update(status="audit_completed", technical_success=True, cases=[])
            else:
                signatures = fingerprints(root, args.data_dir)
                import yaml
                published_models = yaml.safe_load((root / "configs/model/mscmnet_baselines.yaml").read_text())["models"]
                plan = {"cases": cases, "signatures": signatures, "paper_sha256": file_digest(args.paper_config), "data_dir": str(args.data_dir), "device": args.device, "evaluation": evaluation}
                plan["plan_sha256"] = digest(plan)
                manifest_path = result_root / "manifest.json"
                if manifest_path.exists() and json.loads(manifest_path.read_text()).get("plan_sha256") != plan["plan_sha256"]:
                    raise RuntimeError("Run-tag already contains a different source/data/plan fingerprint. Existing outputs preserved; use a new --run-tag.")
                atomic_json(manifest_path, plan)
                for name in signatures["source"]:
                    destination = result_root / "source_snapshot" / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(root / name, destination)
                print("Phase 2/3: server CPU preflight tests before any training", flush=True)
                missing = [name for name in PREFLIGHT_TESTS if not (root / name).is_file()]
                if missing:
                    raise FileNotFoundError(f"Preflight tests missing: {missing}")
                if supervisor.run([sys.executable, "-m", "pytest", "-q", *PREFLIGHT_TESTS], log_root / "preflight_tests.log"):
                    raise RuntimeError("Server preflight tests failed; no training started")
                for index, case in enumerate(cases, 1):
                    if fingerprints(root, args.data_dir) != signatures:
                        raise RuntimeError("Source/data changed during queue; stopped before next case. Existing outputs preserved.")
                    case_root = result_root / case["case"]
                    run = case_root / case["model"] / f"seed_{SEED}"
                    signature = digest({"plan_sha256": plan["plan_sha256"], "case": case})
                    valid, reason = validate_case(run, case, signature)
                    record = {"case": case["case"], "model": case["model"], "seed": SEED, "technical_status": "PASS(existing)" if valid else "running", "numeric_closeness": "reported_separately_after_queue", "protocol_status": case["protocol_status"], "validation": reason, "started_utc": utc_now()}
                    queue["cases"].append(record)
                    atomic_json(result_root / "queue_status.json", queue)
                    print(f"Case {index}/{len(cases)}: {case['case']} ({record['technical_status']})", flush=True)
                    if not valid:
                        if args.device != "cpu":
                            gpu = gpu_preflight(args.gpu_id, args.minimum_free_mib)
                            atomic_json(log_root / f"{case['case']}_gpu_preflight.json", gpu)
                        command = command_for(case, args, case_root)
                        if run.exists():
                            # Trainer's --overwrite archives the exact old run instead of deleting it.
                            command.append("--overwrite")
                        return_code = supervisor.run(command, log_root / f"{case['case']}.log")
                        if run.is_dir():
                            atomic_json(run / "request_signature.json", {"signature": signature, "plan_sha256": plan["plan_sha256"], "case": case, "model_config": published_models[case["model"]], "evaluation": evaluation})
                        valid, reason = validate_case(run, case, signature)
                        record.update(exit_code=return_code, technical_status="PASS" if return_code == 0 and valid else "FAIL", validation=reason)
                    record["finished_utc"] = utc_now()
                    atomic_json(result_root / "queue_status.json", queue)
                print("Phase 3/3: independently recompute every new result; no best-metric mixing", flush=True)
                new_audit = audit_base + ["--strict-first-day", "--results-root", str(result_root), "--output-root", str(result_root / "audit_new")]
                new_audit_rc = supervisor.run(new_audit, log_root / "audit_new.log")
                audit_receipt = json.loads((result_root / "audit_new/audit_summary.json").read_text())
                if not strict_audit_passes(audit_receipt, len(cases)):
                    new_audit_rc = 1
                queue["independent_metric_audit"] = audit_receipt
                after = audit_base + ["--allow-invalid-evidence"] + [item for path in [*existing_roots, result_root] for item in ("--results-root", str(path))] + ["--output-root", str(result_root / "audit_all")]
                combined_audit_rc = supervisor.run(after, log_root / "audit_all.log")
                audit_rc = new_audit_rc or combined_audit_rc
                failures = sum(not row["technical_status"].startswith("PASS") for row in queue["cases"])
                queue.update(status="completed" if not failures and audit_rc == 0 else "completed_with_failures", technical_success=not failures and audit_rc == 0, failed_cases=failures, audit_exit_code=audit_rc)
                queue["models"] = {
                    model: {"expected_cases": 4, "technical_passes": sum(row["model"] == model and row["technical_status"].startswith("PASS") for row in queue["cases"]), "numerical_comparison": "audit_new/paper_gaps.tsv", "reproduction_claim": "not_established"}
                    for model in MODELS
                }
                exit_code = 0 if queue["technical_success"] else 1
    except InterruptedRun as exc:
        queue.update(status="interrupted", error=str(exc))
        exit_code = 128 + exc.signum
    except Exception as exc:
        queue.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(queue["error"], file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        supervisor.terminate_owned()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        queue.update(finished_utc=utc_now(), exit_code=exit_code)
        queue["failed_cases"] = sum(row.get("technical_status") == "FAIL" for row in queue["cases"])
        if owns_lock:
            atomic_json(result_root / "queue_status.json", queue)
        if owns_lock and exit_code not in (130, 143):
            try:
                make_bundle(root, result_root, log_root)
            except Exception as exc:
                print(f"Bundle creation failed; individual results preserved: {exc}", file=sys.stderr)
                exit_code = 1
                queue.update(exit_code=1, technical_success=False, bundle_status="failed", bundle_error=str(exc))
                atomic_json(result_root / "queue_status.json", queue)
        lock.close()
        print(json.dumps({key: value for key, value in queue.items() if key != "cases"}, ensure_ascii=False, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
