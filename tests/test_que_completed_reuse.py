"""Verified cross-manifest reuse must preserve the original training evidence."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARTIFACTS = load_module("que_reuse_test_artifacts", ROOT / "tests/test_que_comprehensive_runner.py")
RUNNER = ARTIFACTS.RUNNER
REUSE = load_module("que_completed_reuse_tests", ROOT / "scripts/train/reuse_que_completed_runs.py")


def signed(manifest):
    manifest.pop("signature", None)
    for kind in ("source", "data"):
        manifest["signatures"][f"{kind}_sha256"] = RUNNER.life.digest(manifest["signatures"][kind])
    manifest["signature"] = RUNNER.life.digest(manifest)
    return manifest


def tree_hashes(root):
    return {str(p.relative_to(root)): RUNNER.life.file_digest(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def evidence(tmp_path):
    source, destination = tmp_path / "old_run", tmp_path / "new_run"
    case = RUNNER.make_case("gru", best_epoch_scale=.5)
    snapshot = source / "source_snapshot"
    source_files = {}
    for name in REUSE.REQUIRED_TRAINING_FILES | {"scripts/train/run_que_comprehensive_reconstruction.py", "tests/old_check.py"}:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "configs/model/mscmnet_baselines.yaml":
            path.write_text(yaml.safe_dump({"models": {"gru": ARTIFACTS.model_fixture(case)}}))
        else:
            path.write_text(f"original source fixture: {name}\n")
        source_files[name] = RUNNER.life.file_digest(path)
    manifest = signed({"version": 1, "base_cases": [case], "maximum_cases": 415,
                       "signatures": {"source": source_files, "data": {"demand.parquet": "d" * 64}},
                       "paper_sha256": "e" * 64, "seed": RUNNER.SEED, "selection_mode": "pooled", "device": "cpu"})
    RUNNER.life.atomic_json(source / "manifest.json", manifest)
    evaluation = ARTIFACTS.evaluation_fixture()
    RUNNER.life.atomic_json(source / "audit_data_protocol/paper_data_statistics.json", {"common_evaluation": evaluation})
    request = {"signature": RUNNER.life.digest({"manifest": manifest["signature"], "settings": RUNNER.setting_key(case)}),
               "case": case, "model_config": ARTIFACTS.model_fixture(case), "evaluation": evaluation}
    run = source / "cases" / case["case"] / case["model"] / f"seed_{RUNNER.SEED}"
    ARTIFACTS.build_artifacts(run, case, request)
    status = json.loads((run / "status.json").read_text())
    status.update(git_commit="a42c2848b0886688ee1a13eeb989ecc4fc4c7df4", elapsed_seconds=123.5)
    RUNNER.life.atomic_json(run / "status.json", status)
    RUNNER.life.atomic_json(run / "completion_receipt.json", {"request_sha256": RUNNER.life.digest(request), "files": RUNNER.evidence_hashes(run, status)})
    queue = {"status": "failed", "cases": [{"case": case["case"], "settings": case, "model": "gru", "technical_status": "PASS", "exit_code": 0, "elapsed_seconds": 124.0}]}
    RUNNER.life.atomic_json(source / "queue_status.json", queue)
    current = copy.deepcopy(manifest)
    current["signatures"]["source"]["scripts/train/run_que_comprehensive_reconstruction.py"] = "f" * 64
    current["signatures"]["source"]["scripts/train/reuse_que_completed_runs.py"] = "a" * 64
    signed(current)
    new_request = {**request, "signature": RUNNER.life.digest({"manifest": current["signature"], "settings": RUNNER.setting_key(case)})}
    return {"source": source, "destination": destination, "case": case, "old_manifest": manifest,
            "current_manifest": current, "evaluation": evaluation, "run": run, "request": request,
            "new_request": new_request, "queue": queue}


def prepare(evidence):
    return REUSE.prepare_reuse(evidence["source"], evidence["destination"], evidence["current_manifest"], evidence["evaluation"], RUNNER)


def test_verified_copy_preserves_original_weights_arrays_configuration_and_commit(evidence):
    before = tree_hashes(evidence["source"])
    context = prepare(evidence)
    assert context["available_count"] == 1
    assert context["training_seconds_saved"] == 124
    metadata = REUSE.import_case(context, evidence["case"], evidence["new_request"])
    run = evidence["destination"] / evidence["run"].relative_to(evidence["source"])
    assert tree_hashes(evidence["source"]) == before
    assert metadata["source_training_git_commit"] == "a42c2848b0886688ee1a13eeb989ecc4fc4c7df4"
    assert metadata["training_seconds_saved"] == 124
    original_receipt = json.loads((evidence["run"] / "completion_receipt.json").read_text())
    for name, digest in original_receipt["files"].items():
        assert RUNNER.life.file_digest(run / name) == digest
    provenance = json.loads((run / REUSE.PROVENANCE_NAME).read_text())
    assert provenance["training_performed_during_import"] is False
    assert provenance["source_manifest"] == evidence["old_manifest"]
    assert provenance["source_request"] == evidence["request"]
    assert provenance["source_completion_receipt"] == original_receipt
    assert RUNNER.validate_case(evidence["run"], evidence["case"], evidence["request"])[0]
    assert RUNNER.validate_case(run, evidence["case"], evidence["new_request"])[0]
    receipt = json.loads((run / "completion_receipt.json").read_text())
    assert receipt["files"][REUSE.PROVENANCE_NAME] == RUNNER.life.file_digest(run / REUSE.PROVENANCE_NAME)
    # The provenance is part of the receipt, not an unaudited side note.
    (run / REUSE.PROVENANCE_NAME).write_text("{}")
    assert not RUNNER.validate_case(run, evidence["case"], evidence["new_request"])[0]


def test_duplicate_import_refuses_to_overwrite_any_destination(evidence):
    context = prepare(evidence)
    REUSE.import_case(context, evidence["case"], evidence["new_request"])
    before = tree_hashes(evidence["destination"])
    with pytest.raises(RuntimeError, match="already exists"):
        REUSE.import_case(context, evidence["case"], evidence["new_request"])
    assert tree_hashes(evidence["destination"]) == before
    assert not list(evidence["destination"].rglob("*.reuse-*"))


def test_missing_old_case_returns_none_without_creating_output(evidence):
    context = prepare(evidence)
    assert REUSE.import_case(context, RUNNER.make_case("msnet"), {}) is None
    assert not evidence["destination"].exists()


@pytest.mark.parametrize("filename", ["checkpoint_0.pt", "predictions_common46.npz", "completion_receipt.json", "request_signature.json", "resolved_config.yaml", "status.json", "scaler_audit.json"])
def test_corrupt_original_evidence_rejected_before_any_import(evidence, filename):
    (evidence["run"] / filename).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Cannot reuse"):
        prepare(evidence)
    assert not evidence["destination"].exists()


def test_compact_bundle_without_checkpoints_is_not_reusable(evidence):
    (evidence["run"] / "checkpoint_0.pt").unlink()
    with pytest.raises(RuntimeError, match="server checkpoints are required"):
        prepare(evidence)


@pytest.mark.parametrize("filename", ["scripts/train/train_temporal_baselines.py", "src/dma_wdf/models/mscmnet.py", "src/dma_wdf/data/reproduction_metrics.py", "configs/model/mscmnet_baselines.yaml", "pyproject.toml", "scripts/reproduce/unreviewed_helper.py"])
def test_changed_or_new_training_implementation_is_never_reused(evidence, filename):
    current = evidence["current_manifest"]
    current["signatures"]["source"][filename] = "b" * 64
    signed(current)
    with pytest.raises(RuntimeError, match="training/source implementation changed"):
        prepare(evidence)


def test_missing_critical_file_in_source_coverage_is_rejected(evidence):
    current = evidence["current_manifest"]
    current["signatures"]["source"].pop("pyproject.toml")
    signed(current)
    with pytest.raises(RuntimeError, match="required training"):
        prepare(evidence)


def test_old_manifest_self_signature_and_nested_fingerprints_are_checked(evidence):
    old = evidence["old_manifest"]
    old["maximum_cases"] += 1
    RUNNER.life.atomic_json(evidence["source"] / "manifest.json", old)
    with pytest.raises(RuntimeError, match="self-signature"):
        prepare(evidence)
    old.pop("signature")
    old["signatures"]["source_sha256"] = "0" * 64
    old["signature"] = RUNNER.life.digest(old)
    RUNNER.life.atomic_json(evidence["source"] / "manifest.json", old)
    with pytest.raises(RuntimeError, match="source fingerprint"):
        prepare(evidence)


def test_snapshot_file_does_not_match_its_signed_hash(evidence):
    (evidence["source"] / "source_snapshot/tests/old_check.py").write_text("changed even a non-training file")
    with pytest.raises(RuntimeError, match="source snapshot changed"):
        prepare(evidence)


def test_data_fingerprints_cannot_be_relaxed(evidence):
    current = evidence["current_manifest"]
    current["signatures"]["data"]["demand.parquet"] = "1" * 64
    signed(current)
    with pytest.raises(RuntimeError, match="data fingerprints differ"):
        prepare(evidence)


def test_audited_truth_or_origin_change_is_rejected(evidence):
    evidence["evaluation"] = copy.deepcopy(evidence["evaluation"])
    evidence["evaluation"]["truths"]["24h"]["array_sha256"] = "2" * 64
    with pytest.raises(RuntimeError, match="truth/origin definitions differ"):
        prepare(evidence)


def test_running_source_queue_cannot_be_imported(evidence):
    queue = evidence["queue"]
    queue["status"] = "running"
    RUNNER.life.atomic_json(evidence["source"] / "queue_status.json", queue)
    with pytest.raises(RuntimeError, match="not terminal"):
        prepare(evidence)


def test_case_identifier_does_not_override_changed_settings(evidence):
    current = evidence["current_manifest"]
    current["base_cases"][0]["batch_size"] *= 2
    signed(current)
    with pytest.raises(RuntimeError, match="settings disagree"):
        prepare(evidence)


def test_request_mismatch_and_changes_after_preflight_are_detected(evidence):
    context = prepare(evidence)
    changed = {**evidence["new_request"], "signature": "not-the-new-manifest"}
    with pytest.raises(RuntimeError, match="destination request differs"):
        REUSE.import_case(context, evidence["case"], changed)
    (evidence["run"] / "checkpoint_0.pt").write_bytes(b"modified-after-preflight")
    with pytest.raises(RuntimeError, match="changed after preflight"):
        REUSE.import_case(context, evidence["case"], evidence["new_request"])
    assert not evidence["destination"].exists()


@pytest.mark.parametrize("kind", ["source_root", "source_checkpoint", "destination_parent"])
def test_symbolic_links_are_never_followed(evidence, tmp_path, kind):
    if kind == "source_root":
        alias = tmp_path / "source_link"
        alias.symlink_to(evidence["source"], target_is_directory=True)
        evidence["source"] = alias
    elif kind == "source_checkpoint":
        path = evidence["run"] / "checkpoint_0.pt"
        target = tmp_path / "outside.pt"
        path.rename(target)
        path.symlink_to(target)
    else:
        alias = tmp_path / "destination_link"
        alias.symlink_to(tmp_path, target_is_directory=True)
        evidence["destination"] = alias / "new_destination"
    with pytest.raises(RuntimeError, match="symbolic link|non-regular"):
        prepare(evidence)


@pytest.mark.parametrize("kind", ["same", "inside_source", "contains_source"])
def test_source_and_destination_must_be_disjoint(evidence, kind):
    evidence["destination"] = {"same": evidence["source"], "inside_source": evidence["source"] / "nested", "contains_source": evidence["source"].parent}[kind]
    with pytest.raises(RuntimeError, match="non-overlapping"):
        prepare(evidence)


def test_path_traversal_in_even_a_signed_manifest_is_rejected(evidence):
    old = evidence["old_manifest"]
    old["signatures"]["source"]["../outside.py"] = "3" * 64
    signed(old)
    RUNNER.life.atomic_json(evidence["source"] / "manifest.json", old)
    with pytest.raises(RuntimeError, match="Unsafe reuse evidence path"):
        prepare(evidence)
