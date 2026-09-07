"""Copy verified training results into a new queue without rewriting their origin.

The source manifest guard remains strict. Reuse is an explicit compatibility
operation: the source snapshot and completion receipts must be intact, and all
training/data/metric implementation files must be byte-identical. Only named
queue-control helpers and tests may differ. Original results are never edited.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any


# These files control scheduling/validation, never the numerical training path.
# New exceptions require an explicit code review; arbitrary scripts are not safe.
QUEUE_ONLY_FILES = frozenset({
    "scripts/train/run_que_comprehensive_reconstruction.py",
    "scripts/train/run_que_comprehensive_reconstruction_gpu6.sh",
    "scripts/train/reuse_que_completed_runs.py",
    "scripts/train/validate_que_candidate_commands.py",
})
REQUIRED_TRAINING_FILES = frozenset({
    "scripts/train/train_temporal_baselines.py",
    "configs/model/mscmnet_baselines.yaml",
    "configs/data/paper_split.yaml",
    "src/dma_wdf/models/mscmnet.py",
    "src/dma_wdf/data/reproduction_metrics.py",
    "pyproject.toml",
})
TERMINAL_SOURCE_STATES = frozenset({"failed", "completed", "completed_with_failures", "interrupted", "paused_time_budget"})
PROVENANCE_NAME = "reused_source_provenance.json"


def _plain_root(path: Path, *, must_exist: bool) -> Path:
    path = Path(os.path.abspath(path))
    # Reject even an in-tree link: its target can change after validation.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise RuntimeError(f"Reuse refuses symbolic link: {part}")
    if must_exist and not path.is_dir():
        raise RuntimeError(f"Reuse source directory is missing: {path}")
    return path


def _relative(name: str) -> Path:
    if not isinstance(name, str) or "\\" in name:
        raise RuntimeError("Unsafe reuse evidence path")
    parsed = PurePosixPath(name)
    if parsed.is_absolute() or not parsed.parts or any(p in ("", ".", "..") for p in parsed.parts) or str(parsed) != name:
        raise RuntimeError(f"Unsafe reuse evidence path: {name!r}")
    return Path(*parsed.parts)


def _regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required reuse evidence is missing: {path}") from exc
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"Reuse evidence must be a regular file: {path}")
    for ancestor in path.parents:
        if ancestor.is_symlink():
            raise RuntimeError(f"Reuse evidence traverses a symbolic link: {ancestor}")


def _plain_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"Reuse result is not a plain directory: {root}")
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise RuntimeError(f"Reuse refuses non-regular result entry: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    _regular_file(path)
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return result


def _check_manifest(manifest: dict[str, Any], life: Any, label: str) -> None:
    if manifest.get("signature") != life.digest({k: v for k, v in manifest.items() if k != "signature"}):
        raise RuntimeError(f"{label} manifest self-signature does not match")
    signatures = manifest.get("signatures", {})
    for kind in ("source", "data"):
        mapping = signatures.get(kind)
        if not isinstance(mapping, dict) or not mapping or signatures.get(f"{kind}_sha256") != life.digest(mapping):
            raise RuntimeError(f"{label} {kind} fingerprint is invalid")
        for name, value in mapping.items():
            _relative(name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise RuntimeError(f"{label} {kind} contains an invalid file digest")


def _is_queue_only(name: str) -> bool:
    return name in QUEUE_ONLY_FILES or name.startswith("tests/") or name.startswith("docs/") or name == "SOURCE_CHECKSUMS.sha256"


def _check_training_compatibility(old: dict[str, Any], current: dict[str, Any]) -> dict[str, str]:
    old_source = old["signatures"]["source"]
    current_source = current["signatures"]["source"]
    if not REQUIRED_TRAINING_FILES <= old_source.keys() or not REQUIRED_TRAINING_FILES <= current_source.keys():
        raise RuntimeError("Reuse manifest does not cover required training implementation files")
    changed = sorted(name for name in old_source.keys() | current_source.keys()
                     if old_source.get(name) != current_source.get(name) and not _is_queue_only(name))
    if changed:
        raise RuntimeError("Reuse training/source implementation changed: " + ", ".join(changed[:12]))
    if old["signatures"]["data"] != current["signatures"]["data"]:
        raise RuntimeError("Reuse source data fingerprints differ; no previous predictions imported")
    for key in ("seed", "paper_sha256", "selection_mode"):
        if old.get(key) != current.get(key):
            raise RuntimeError(f"Reuse protocol differs: {key}")
    return {name: value for name, value in old_source.items() if not _is_queue_only(name)}


def _case_path(root: Path, case: dict[str, Any], seed: int) -> Path:
    for value in (case.get("case"), case.get("model")):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            raise RuntimeError("Unsafe or missing reuse case/model name")
    return root / "cases" / case["case"] / case["model"] / f"seed_{seed}"


def prepare_reuse(source_root: Path, destination_root: Path, current_manifest: dict[str, Any],
                  current_evaluation: dict[str, Any], runner: Any) -> dict[str, Any]:
    """Validate the entire reuse set before the caller launches any GPU training.

    Checkpoints must be present on the server. A compact upload alone is not a
    reusable training directory because it intentionally excludes checkpoint PTs.
    The returned context is local in-memory state, not a manifest to serialize.
    """
    import yaml

    life = runner.life
    source = _plain_root(source_root, must_exist=True)
    destination = _plain_root(destination_root, must_exist=False)
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise RuntimeError("Reuse source and destination must be separate non-overlapping directories")
    old = _read_json(source / "manifest.json")
    _check_manifest(old, life, "Source")
    _check_manifest(current_manifest, life, "Current")
    training_sources = _check_training_compatibility(old, current_manifest)
    for name, expected_hash in old["signatures"]["source"].items():
        path = source / "source_snapshot" / _relative(name)
        _regular_file(path)
        if life.file_digest(path) != expected_hash:
            raise RuntimeError(f"Reuse source snapshot changed: {name}")
    queue = _read_json(source / "queue_status.json")
    if queue.get("status") not in TERMINAL_SOURCE_STATES:
        raise RuntimeError(f"Reuse source queue is not terminal: {queue.get('status')!r}")
    old_evaluation = _read_json(source / "audit_data_protocol/paper_data_statistics.json")["common_evaluation"]
    if old_evaluation != current_evaluation:
        raise RuntimeError("Reuse audited truth/origin definitions differ")
    config_path = source / "source_snapshot/configs/model/mscmnet_baselines.yaml"
    published = yaml.safe_load(config_path.read_text())["models"]
    old_cases = {c["case"]: c for c in old["base_cases"]}
    current_cases = {c["case"]: c for c in current_manifest["base_cases"]}
    if len(old_cases) != len(old["base_cases"]) or len(current_cases) != len(current_manifest["base_cases"]):
        raise RuntimeError("Reuse plan contains duplicate case identifiers")
    records = queue.get("cases", [])
    if not isinstance(records, list) or len({r.get("case") for r in records}) != len(records):
        raise RuntimeError("Reuse queue has invalid or duplicate case records")
    sources = {}
    for record in records:
        name = record.get("case")
        if not str(record.get("technical_status", "")).startswith("PASS") or name not in current_cases:
            continue
        case = old_cases.get(name)
        if case is None or case != current_cases[name] or record.get("settings") != case:
            raise RuntimeError(f"Reuse case settings disagree with the signed plans: {name}")
        if record.get("exit_code") != 0 or record.get("model") != case["model"] or case.get("seed") != runner.SEED:
            raise RuntimeError(f"Reuse case has inconsistent successful training status: {name}")
        run = _case_path(source, case, runner.SEED)
        _plain_tree(run)
        expected = {"signature": life.digest({"manifest": old["signature"], "settings": runner.setting_key(case)}),
                    "case": case, "model_config": runner.expected_model_config(published, case), "evaluation": old_evaluation}
        valid, reason = runner.validate_case(run, case, expected)
        if not valid:
            raise RuntimeError(f"Cannot reuse {name}: {reason}. Original files preserved; complete server checkpoints are required.")
        status = _read_json(run / "status.json")
        duration = float(record.get("elapsed_seconds", status.get("elapsed_seconds", 0.0)))
        if not math.isfinite(duration) or duration < 0:
            raise RuntimeError(f"Invalid original training duration: {name}")
        sources[name] = {"run": run, "case": case, "expected": expected,
                         "status": status, "record": record, "seconds": duration,
                         "request": _read_json(run / "request_signature.json"),
                         "receipt": _read_json(run / "completion_receipt.json")}
    return {"runner": runner, "source_root": source, "destination_root": destination,
            "source_manifest": old, "source_manifest_file_sha256": life.file_digest(source / "manifest.json"),
            "source_queue_status_sha256": life.file_digest(source / "queue_status.json"),
            "current_manifest": current_manifest, "current_evaluation": current_evaluation,
            "unchanged_training_sources": training_sources, "sources": sources,
            "available_count": len(sources), "training_seconds_saved": sum(s["seconds"] for s in sources.values())}


def import_case(context: dict[str, Any], case: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any] | None:
    """Atomically copy one prepared case and link its new receipt to old proof.

    Existing destination results are never overwritten. Revalidate a cached new
    result in the runner before invoking this function on a resumed queue.
    """
    entry = context["sources"].get(case["case"])
    if entry is None:
        return None
    runner = context["runner"]
    life = runner.life
    requested = {"signature": life.digest({"manifest": context["current_manifest"]["signature"], "settings": runner.setting_key(case)}),
                 "case": case, "model_config": entry["expected"]["model_config"], "evaluation": context["current_evaluation"]}
    if case != entry["case"] or expected != requested:
        raise RuntimeError(f"Reuse destination request differs from the prepared case: {case['case']}")
    original = entry["run"]
    _plain_tree(original)
    valid, reason = runner.validate_case(original, case, entry["expected"])
    if not valid or _read_json(original / "completion_receipt.json") != entry["receipt"]:
        raise RuntimeError(f"Reuse source changed after preflight: {case['case']}: {reason}")
    destination = _case_path(context["destination_root"], case, runner.SEED)
    _plain_root(destination, must_exist=False)
    if destination.exists():
        raise RuntimeError(f"Reuse destination already exists; preserved: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.reuse-", dir=destination.parent))
    metadata = {"source_root": str(context["source_root"]), "source_case": case["case"],
                "source_manifest_signature": context["source_manifest"]["signature"],
                "source_training_git_commit": entry["status"].get("git_commit"),
                "training_seconds_saved": entry["seconds"], "reused_utc": life.utc_now()}
    try:
        shutil.copytree(original, temporary, dirs_exist_ok=True, symlinks=False)
        _plain_tree(temporary)
        valid, reason = runner.validate_case(temporary, case, entry["expected"])
        if not valid:
            raise RuntimeError(f"Copied original evidence failed verification: {case['case']}: {reason}")
        proof = {"version": 1, "operation": "verified_copy_of_original_training", **metadata,
                 "source_run_relative_path": str(original.relative_to(context["source_root"])),
                 "source_manifest": context["source_manifest"],
                 "source_manifest_file_sha256": context["source_manifest_file_sha256"],
                 "source_queue_status_sha256": context["source_queue_status_sha256"],
                 "source_request": entry["request"], "source_completion_receipt": entry["receipt"],
                 "unchanged_training_sources": context["unchanged_training_sources"],
                 "destination_manifest_signature": context["current_manifest"]["signature"],
                 "training_performed_during_import": False,
                 "original_status_config_predictions_and_checkpoints_unchanged": True}
        life.atomic_json(temporary / PROVENANCE_NAME, proof)
        life.atomic_json(temporary / "request_signature.json", expected)
        files = runner.evidence_hashes(temporary, entry["status"])
        if PROVENANCE_NAME not in files:
            raise RuntimeError("Runner receipt must include reused-source provenance before reuse is permitted")
        life.atomic_json(temporary / "completion_receipt.json", {"request_sha256": life.digest(expected), "files": files})
        valid, reason = runner.validate_case(temporary, case, expected)
        if not valid:
            raise RuntimeError(f"New reuse receipt failed verification: {case['case']}: {reason}")
        if destination.exists():
            raise RuntimeError(f"Reuse destination appeared during import; preserved: {destination}")
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return metadata
