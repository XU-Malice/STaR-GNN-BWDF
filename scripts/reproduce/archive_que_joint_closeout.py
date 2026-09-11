#!/usr/bin/env python3
"""Freeze four currently closest joint models and the complete experiment ledger.

This CPU-only helper does not stop jobs, modify source runs, deserialize model
weights, or claim original-paper reproduction. READY means an intact, verified
archive is available for the user's explicitly accepted joint-model closeout.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
JOINT_MODELS = ("msnet", "mscmnet_m", "mscmnet_wm", "mscmnet_w")


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


EVIDENCE = _module("que_joint_closeout_evidence", "que_total_watch_evidence.py")
FOCUS = _module("que_joint_closeout_focus", "run_que_total_focus.py")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _inventory(root: Path) -> dict[str, str]:
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError(f"Symlink archive root is not accepted: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink evidence is not accepted: {path}")
        if path.is_dir():
            continue
        EVIDENCE.regular(path)
        result[str(path.relative_to(root))] = EVIDENCE.file_digest(path)
    return result


def _copy_file(source: Path, destination: Path, expected: str | None = None) -> str:
    before = EVIDENCE.file_digest(source)
    if expected is not None and before != expected:
        raise ValueError(f"Source hash mismatch before archive copy: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if EVIDENCE.file_digest(source) != before or EVIDENCE.file_digest(destination) != before:
        raise ValueError(f"Evidence changed while archiving: {source}")
    return before


def _copy_run(source: Path, destination: Path) -> dict[str, str]:
    inventory = _inventory(source)
    status = EVIDENCE.read_json(source / "status.json")
    checkpoints = status.get("checkpoint_files", [])
    if len(checkpoints) != 1 or Path(checkpoints[0]).name != checkpoints[0]:
        raise ValueError("A joint-model archive requires one complete shared checkpoint")
    required = {"status.json", "resolved_config.yaml", "scaler_audit.json", "predictions_common46.npz",
                "metrics.csv", "loss_curve.csv", "request_signature.json", "completion_receipt.json", *checkpoints}
    if not required <= inventory.keys() or not (source / checkpoints[0]).stat().st_size:
        raise ValueError("Incomplete joint-model evidence or missing checkpoint")
    for name, expected in inventory.items():
        _copy_file(source / EVIDENCE.relative(name), destination / name, expected)
    if inventory != _inventory(source) or inventory != _inventory(destination):
        raise ValueError("Complete run file inventory changed while archiving")
    return inventory


def _write_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["case"], delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list))
                             else value for key, value in record.items()})


def experiment_ledger(context, queue: dict[str, Any], plans: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Include planned, running, reused, successful and failed parameter identities.

    A setting key can be skipped automatically in future searches only when it
    has an attempted record. Pending settings stay recorded but are not marked
    as previously trained. Requests and resolved configs retain effective values.
    """
    records = {}
    for record in queue.get("cases", []):
        name = record["case"]
        if name in records or name not in plans:
            raise ValueError("Duplicate or unplanned queue record in experiment ledger")
        if record.get("model") != plans[name]["model"] or record.get("settings") != plans[name]:
            raise ValueError("Queue record differs from frozen parameter plan")
        records[name] = record
    ledger, attempts = [], set()
    for name, case in sorted(plans.items()):
        record = records.get(name, {})
        status = str(record.get("technical_status", "pending"))
        attempted = bool(record) and (status not in {"pending", "planned", "not_started", "waiting_gpu"}
                                      or bool(record.get("resource_attempts")))
        key = context.runner.setting_key(case)
        request_path = context._run(case) / "request_signature.json"
        request = EVIDENCE.read_json(request_path) if request_path.exists() else None
        if request is not None and (request.get("case") != case or request.get("evaluation") != context.evaluation):
            raise ValueError(f"Historical request differs from frozen settings/evaluation: {name}")
        saved = destination / "history" / name
        request_hash = None
        if request is not None:
            # Requests are written atomically by the trainer and are immutable
            # for a particular setting identity, including resource retries.
            request_hash = _copy_file(request_path, saved / "request_signature.json")
        resolved = context._run(case) / "resolved_config.yaml"
        resolved_hash = _copy_file(resolved, saved / "resolved_config.yaml") if resolved.exists() else None
        status_path = context._run(case) / "status.json"
        training_status = EVIDENCE.read_json(status_path) if status_path.exists() else {}
        if status_path.exists():
            _json(saved / "status_at_capture.json", training_status)
        if attempted:
            attempts.add(key)
        ledger.append({"case": name, "model": case["model"], "stage": case["stage"], "seed": case["seed"],
                       "technical_status": status, "attempted": attempted, "exit_code": record.get("exit_code"),
                       "validation": record.get("validation"), "setting_key": key,
                       "settings_sha256": _digest(case), "settings": case,
                       "effective_model_config": request.get("model_config") if request else None,
                       "request_sha256": request_hash, "resolved_config_sha256": resolved_hash,
                       "source_training_git_commit": training_status.get("git_commit"),
                       "frozen_source_sha256": context.manifest["signatures"]["source_sha256"],
                       "evaluation_sha256": _digest(context.evaluation),
                       "elapsed_seconds": record.get("elapsed_seconds", training_status.get("elapsed_seconds")),
                       "record_at_capture": record})
    result = {"version": 1, "records": ledger, "case_count": len(ledger), "attempted_case_count": sum(r["attempted"] for r in ledger),
              "attempted_setting_keys": sorted(attempts), "all_planned_setting_keys": sorted({r["setting_key"] for r in ledger}),
              "pending_settings_are_not_claimed_as_trained": True,
              "deduplication_scope": "All frozen A/B/C plans and attempted records, including failures and reused results; setting key includes model and seed.",
              "evaluation": context.evaluation, "manifest_signature": context.manifest["signature"]}
    _json(destination / "experiment_ledger.json", result)
    _write_tsv(destination / "experiment_ledger.tsv", ledger)
    return result


def _archive_checked(payload: Path, archive: Path, inventory: dict[str, str]) -> str:
    with tarfile.open(archive, "w:gz") as tar:
        for name in sorted(inventory):
            tar.add(payload / name, arcname=name, recursive=False)
    _verify_archive(archive, inventory)
    return EVIDENCE.file_digest(archive)


def _verify_archive(archive: Path, inventory: dict[str, str]) -> None:
    observed = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            EVIDENCE.relative(member.name)
            if not member.isfile() or member.name in observed or member.name not in inventory:
                raise ValueError("Archive contains an unexpected or non-regular member")
            stream = tar.extractfile(member)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            observed[member.name] = digest.hexdigest()
    if observed != inventory:
        raise ValueError("Archived payload hash inventory mismatch")


def verify_ready(output_root: Path) -> dict[str, Any]:
    """Verify frozen payload and compressed archive before any later stop action."""
    output_root = Path(os.path.abspath(output_root))
    report = EVIDENCE.read_json(output_root / "completion_manifest.json")
    unsigned = {k: v for k, v in report.items() if k != "signature"}
    if report.get("status") != "READY" or report.get("signature") != _digest(unsigned):
        raise ValueError("Joint closeout completion manifest is invalid")
    if set(report.get("selected_models", {})) != set(JOINT_MODELS):
        raise ValueError("Closeout must contain all four joint models")
    if report.get("user_accepts_current_joint_models") is not True or report.get("original_paper_reproduction_claim") is not False:
        raise ValueError("Closeout acceptance must not claim full paper reproduction")
    if report.get("source_results") != report.get("source_result_root"):
        raise ValueError("Closeout source binding is inconsistent")
    inventory = report["payload_sha256"]
    for name in inventory:
        EVIDENCE.relative(name)
    actual = _inventory(output_root)
    expected_files = set(inventory) | {"completion_manifest.json", "joint_models_complete.tar.gz", "joint_models_complete.tar.gz.sha256"}
    if set(actual) != expected_files or any(actual[n] != h for n, h in inventory.items()):
        raise ValueError("Frozen joint-model payload hash inventory mismatch")
    archive = output_root / "joint_models_complete.tar.gz"
    if EVIDENCE.file_digest(archive) != report["archive_sha256"]:
        raise ValueError("Joint-model compressed archive hash mismatch")
    if (output_root / "joint_models_complete.tar.gz.sha256").read_text() != f"{report['archive_sha256']}  joint_models_complete.tar.gz\n":
        raise ValueError("Joint-model archive checksum sidecar mismatch")
    _verify_archive(archive, inventory)
    if EVIDENCE.read_json(output_root / "selected_models.json") != report["selected_models"]:
        raise ValueError("Selected model report differs from archived selection")
    for model, selected in report["selected_models"].items():
        if (len(selected["metrics"]) != 8 or selected["model"] != model or
                [(row["task"], row["metric"]) for row in selected["metrics"]] != list(FOCUS.KEYS)):
            raise ValueError("Selected model must retain eight metrics from one candidate")
        run = output_root / "best_models" / model
        for name, expected in selected["run_files_sha256"].items():
            if EVIDENCE.file_digest(run / EVIDENCE.relative(name)) != expected:
                raise ValueError("Selected complete run changed")
        checkpoint = EVIDENCE.read_json(run / "status.json")["checkpoint_files"]
        if len(checkpoint) != 1 or checkpoint[0] not in selected["run_files_sha256"]:
            raise ValueError("Selected complete run has no verified checkpoint")
    return report


def archive_joint_closeout(context, queue: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Publish one immutable complete closeout; originals remain read-only.

    The queue argument is captured once. A still-running queue may subsequently
    finish more cases without invalidating this explicitly timestamped snapshot.
    Existing READY snapshots are hash-checked and reused, never silently updated.
    """
    output_root = Path(os.path.abspath(output_root))
    context.check_fingerprints()
    if any(path.is_symlink() for path in (output_root, *output_root.parents)):
        raise ValueError("Closeout output must not use symlinks")
    if output_root == context.result_root or context.result_root in output_root.parents:
        raise ValueError("Closeout output must be outside the original result tree")
    if output_root.exists():
        report = verify_ready(output_root)
        if report["source_manifest_signature"] != context.manifest["signature"]:
            raise ValueError("Existing closeout belongs to a different frozen campaign")
        return report
    queue, plans = copy.deepcopy(queue), copy.deepcopy(context._plans())
    captured = datetime.now(timezone.utc).isoformat()
    if context._recompute_evaluation() != context.evaluation:
        raise ValueError("Evaluation differs from freshly audited source data")
    candidates, exclusions = FOCUS.collect_validated(context, queue)
    source_identity = _digest({"manifest": context.manifest["signature"], "plans": plans, "queue": queue,
                               "receipts": sorted((c["case"], c["evidence_digest"]) for c in candidates)})
    output_root.parent.mkdir(parents=True, exist_ok=True)
    audit_api = _module("que_joint_closeout_audit", "audit_que_total_objective.py")
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.building-", dir=output_root.parent) as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir()
        audit = audit_api.audit_candidates(candidates, context.paper_path, Path(temporary) / "audit")
        ranking = FOCUS.complete_records(audit["candidates_summary"], "pooled", context.paper)
        selected = {m: c for m, c in FOCUS.choose_models(ranking).items() if m in JOINT_MODELS}
        if set(selected) != set(JOINT_MODELS):
            missing = sorted(set(JOINT_MODELS) - set(selected))
            raise ValueError(f"Missing validated joint-model candidates: {missing}; closeout withheld")
        validated = {c["case"]: c for c in candidates}
        FOCUS.verify_candidate_evidence(candidates, context)
        for model, candidate in selected.items():
            copied = stage / "best_models" / model
            candidate["run_files_sha256"] = _copy_run(Path(candidate["run"]), copied)
            candidate["source_evidence_digest"] = validated[candidate["case"]]["evidence_digest"]
            request = EVIDENCE.read_json(copied / "request_signature.json")
            valid, reason = context.runner.validate_case(copied, candidate["settings"], request)
            if not valid or request != context._expected(candidate["settings"]):
                raise ValueError(f"Copied selected model did not validate: {model}: {reason}")
            context._verify_reuse(copied, candidate["settings"], request)
            candidate["archived_run_relative"] = f"best_models/{model}"
            candidate["source_run"] = candidate["run"]
            candidate["setting_key"] = context.runner.setting_key(candidate["settings"])
        ledger = experiment_ledger(context, queue, plans, stage)
        _json(stage / "queue_at_capture.json", queue)
        _json(stage / "frozen_plans.json", plans)
        _json(stage / "selected_models.json", selected)
        _json(stage / "selection_audit.json", {"reference": audit["reference"], "all_candidates": audit["candidates_summary"],
                    "exclusions": exclusions, "primary_metric_mode": "pooled", "diagnosis": audit["diagnosis"],
                    "pooled_total48_impossible_even_with_rounding": audit["pooled_total48_impossible_even_with_rounding"]})
        FOCUS.save_table(stage, selected)
        table = (stage / "current_table.md").read_text().splitlines()
        (stage / "current_table.md").write_text(
            "本表仅收存已接受的四个联合模型；GRU/LSTM 由后续专门搜索处理。\n\n" +
            "\n".join(line for line in table if not line.startswith(("|gru|", "|lstm|"))) + "\n")
        for name, expected in context.manifest["signatures"]["source"].items():
            _copy_file(context.project_root / EVIDENCE.relative(name), stage / "source_snapshot" / name, expected)
        for filename in ("archive_que_joint_closeout.py", "run_que_total_focus.py", "audit_que_total_objective.py", "que_total_watch_evidence.py"):
            _copy_file(Path(__file__).with_name(filename), stage / "closeout_tools" / filename)
        for name in ("manifest.json", "stage_b_manifest.json", "stage_c_manifest.json", "audit_data_protocol/paper_data_statistics.json"):
            original = context.result_root / name
            if original.exists():
                _copy_file(original, stage / "frozen_protocol" / name)
        docs = context.project_root / "docs"
        if docs.is_dir():
            for path in sorted(docs.glob("*QUE*.md")) + sorted(docs.glob("*MSCMNET*.md")):
                _copy_file(path, stage / "source_documentation" / path.name)
        _json(stage / "archive_scope.json", {"captured_utc": captured, "snapshot_identity": source_identity,
              "source_result_root": str(context.result_root), "source_manifest_signature": context.manifest["signature"],
              "user_accepts_current_joint_models": True, "original_paper_reproduction_claim": False,
              "selection_policy": "One intact candidate per model; all eight pooled TOTAL metrics jointly ranked by worst normalized gap, then mean gap.",
              "weights_deserialized": False, "source_runs_modified": False, "source_queue_stopped": False,
              "raw_dataset_included": False, "data_fingerprints": context.manifest["signatures"]["data"],
              "data_directory_required_for_retraining": str(context.data_dir),
              "later_source_queue_completions_are_outside_this_immutable_snapshot": True})
        context.check_fingerprints()
        FOCUS.verify_candidate_evidence([validated[c["case"]] for c in selected.values()], context)
        payload = _inventory(stage)
        archive_hash = _archive_checked(stage, stage / "joint_models_complete.tar.gz", payload)
        (stage / "joint_models_complete.tar.gz.sha256").write_text(f"{archive_hash}  joint_models_complete.tar.gz\n")
        report = {"version": 1, "status": "READY", "captured_utc": captured,
                  "completed_utc": datetime.now(timezone.utc).isoformat(), "snapshot_identity": source_identity,
                  "source_manifest_signature": context.manifest["signature"], "source_result_root": str(context.result_root),
                  "source_results": str(context.result_root), "project_root": str(context.project_root),
                  "report_path": str(output_root / "completion_manifest.json"),
                  "archive_path": str(output_root / "joint_models_complete.tar.gz"), "archive_sha256": archive_hash,
                  "selected_models": selected, "payload_sha256": payload,
                  "planned_cases": ledger["case_count"], "attempted_cases": ledger["attempted_case_count"],
                  "user_accepts_current_joint_models": True, "original_paper_reproduction_claim": False,
                  "all_selected_metric_cells_within_tolerance": all(r["within_tolerance"] for c in selected.values() for r in c["metrics"]),
                  "source_queue_stopped": False}
        report["signature"] = _digest(report)
        # The completion marker is written only after the payload and archive
        # have been independently verified; the whole directory publishes once.
        _json(stage / "completion_manifest.json", report)
        verify_ready(stage)
        if output_root.exists():
            raise ValueError("Closeout destination appeared during creation; no existing output overwritten")
        os.rename(stage, output_root)
    return verify_ready(output_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        report = verify_ready(args.output_root)
    else:
        context = EVIDENCE.Context(args.project_root, args.result_root)
        queue = EVIDENCE.read_json(args.result_root / "queue_status.json")
        report = archive_joint_closeout(context, queue, args.output_root)
    print(json.dumps({k: report[k] for k in ("status", "archive_path", "archive_sha256", "planned_cases", "attempted_cases")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
