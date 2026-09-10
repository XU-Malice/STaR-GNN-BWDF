"""Exercise the real export/live-source boundary, including generated egg metadata.

These tests deliberately do not replace verify_training_sources or the original
Context.check_fingerprints guard.  Metadata is deployment provenance, while all
actual numerical sources and the frozen original tree remain checked.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
METADATA = "src/star_gnn_bwdf.egg-info"
KNOWN_METADATA = (
    "PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt",
    "requires.txt", "top_level.txt", "namespace_packages.txt", "not-zip-safe", "zip-safe",
)
CORE = (
    "src/dma_wdf/models/model.py", "configs/model/model.yaml",
    "scripts/train/train_temporal_baselines.py", "scripts/train/que_shared_gpu_runtime.py",
)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


FOCUS = load("source_compat_focus", "scripts/reproduce/run_que_total_focus.py")
WATCH = load("source_compat_evidence", "scripts/reproduce/que_total_watch_evidence.py")


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def freeze_minimal(project):
    """Construct a minimal context but retain its real full fingerprint method."""
    context = WATCH.Context.__new__(WATCH.Context)
    context.project_root = project
    context.result_root = project / "results/original"
    context.data_dir = project / "data/processed"
    put(context.data_dir, "demand.bin", "frozen demand fixture")
    context.paper_path = put(project, "configs/evaluation/mscmnet_paper_metrics.yaml", "tasks: {}\n")
    context.manifest = {
        "signatures": WATCH._fingerprints(project, context.data_dir),
        "paper_sha256": WATCH.file_digest(context.paper_path),
    }
    FOCUS.write(context.result_root / "manifest.json", context.manifest)
    context._manifest_file_hash = WATCH.file_digest(context.result_root / "manifest.json")
    context.check_fingerprints()
    return context


def exported_tree(project, destination):
    for directory in ("src", "configs"):
        shutil.copytree(project / directory, destination / directory)
    for name in CORE[2:]:
        if (project / name).exists():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / name, path)
    return destination


@pytest.fixture
def pair(tmp_path):
    original = tmp_path / "live"
    for index, name in enumerate(CORE):
        put(original, name, f"# unchanged numerical input {index}\n")
    put(original, f"{METADATA}/PKG-INFO", "editable-install metadata from original environment\n")
    context = freeze_minimal(original)
    exported = exported_tree(original, tmp_path / "export")
    shutil.rmtree(exported / METADATA)
    return context, exported


def test_missing_generated_pkg_info_in_pure_export_is_allowed(pair, tmp_path):
    context, exported = pair
    report = tmp_path / "audit/source_compatibility.json"
    checked = FOCUS.verify_training_sources(context, exported, audit_path=report)
    assert set(CORE).issubset(checked)
    assert f"{METADATA}/PKG-INFO" not in checked
    serialized = json.dumps(json.loads(report.read_text()))
    assert "PKG-INFO" in serialized
    assert context.manifest["signatures"]["source"][f"{METADATA}/PKG-INFO"] in serialized
    context.check_fingerprints()


def test_different_generated_metadata_preserves_original_frozen_evidence(pair, tmp_path):
    context, exported = pair
    original_manifest = copy.deepcopy(context.manifest)
    path = put(exported, f"{METADATA}/PKG-INFO", "metadata from a different build environment\n")
    report = tmp_path / "compatibility.json"
    FOCUS.verify_training_sources(context, exported, audit_path=report)
    serialized = json.dumps(json.loads(report.read_text()))
    assert hashlib.sha256(path.read_bytes()).hexdigest() in serialized
    assert context.manifest == original_manifest
    context.check_fingerprints()


def test_all_known_immediate_metadata_files_can_differ(tmp_path):
    project = tmp_path / "live"
    for name in CORE:
        put(project, name, "# same numerical bytes\n")
    for name in KNOWN_METADATA:
        put(project, f"{METADATA}/{name}", f"original {name}\n")
    context = freeze_minimal(project)
    exported = exported_tree(project, tmp_path / "export")
    for name in KNOWN_METADATA:
        put(exported, f"{METADATA}/{name}", f"new deployment {name}\n")
    checked = FOCUS.verify_training_sources(context, exported)
    assert not set(checked).intersection(f"{METADATA}/{name}" for name in KNOWN_METADATA)
    context.check_fingerprints()


@pytest.mark.parametrize("name", CORE)
def test_changed_numerical_source_or_entrypoint_is_rejected(pair, name):
    context, exported = pair
    (exported / name).write_text("# actual changed behavior\n")
    with pytest.raises(ValueError):
        FOCUS.verify_training_sources(context, exported)


@pytest.mark.parametrize("name", ("src", "configs", CORE[-1]))
def test_missing_numerical_directory_or_entrypoint_is_rejected(pair, name):
    context, exported = pair
    path = exported / name
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    with pytest.raises(ValueError):
        FOCUS.verify_training_sources(context, exported)


@pytest.mark.parametrize("name", ("src/dma_wdf/new_model.py", "configs/model/new_model.yaml"))
def test_new_numerical_file_cannot_bypass_frozen_inventory(pair, name):
    context, exported = pair
    put(exported, name, "# previously unaudited numerical input\n")
    with pytest.raises(ValueError):
        FOCUS.verify_training_sources(context, exported)


@pytest.mark.parametrize("name", ("model.py", "nested/PKG-INFO"))
def test_egg_info_directory_does_not_exempt_arbitrary_files(pair, name):
    context, exported = pair
    put(exported, f"{METADATA}/{name}", "not an exempt immediate metadata file\n")
    with pytest.raises(ValueError):
        FOCUS.verify_training_sources(context, exported)


def test_exported_numerical_symlink_is_rejected_even_with_identical_bytes(pair):
    context, exported = pair
    path = exported / CORE[0]
    path.unlink()
    path.symlink_to(context.project_root / CORE[0])
    with pytest.raises(ValueError):
        FOCUS.verify_training_sources(context, exported)


def test_mutated_live_metadata_still_fails_original_full_fingerprint_guard(pair):
    context, exported = pair
    FOCUS.verify_training_sources(context, exported)
    (context.project_root / METADATA / "PKG-INFO").write_text("changed after original evidence was frozen\n")
    with pytest.raises(ValueError, match="fingerprint"):
        FOCUS.verify_training_sources(context, exported)


def test_metadata_alone_is_not_a_valid_numerical_source_inventory(tmp_path):
    project = tmp_path / "live"
    put(project, f"{METADATA}/PKG-INFO", "metadata only\n")
    context = freeze_minimal(project)
    exported = exported_tree(project, tmp_path / "export")
    with pytest.raises(ValueError):
        FOCUS.verify_training_sources(context, exported)


def test_bytecode_cache_exclusion_matches_original_source_fingerprints(pair):
    context, exported = pair
    put(exported, "src/dma_wdf/__pycache__/model.cpython-311.pyc", "generated cache")
    put(exported, "src/dma_wdf/model.pyo", "generated optimized cache")
    checked = FOCUS.verify_training_sources(context, exported)
    assert not any("__pycache__" in name or name.endswith((".pyc", ".pyo")) for name in checked)


def test_real_source_compatibility_then_completed_candidates_and_full_archive(tmp_path, monkeypatch):
    fixtures = load("source_compat_full_fixture", "tests/test_que_total_focus.py")
    archive = load("source_compat_real_archive", "scripts/reproduce/archive_que_joint_closeout.py")
    e = fixtures.evidence.__wrapped__(tmp_path, monkeypatch)
    context = e["context"]
    put(e["project"], CORE[-1], "# unchanged shared runtime fixture\n")
    put(e["project"], f"{METADATA}/PKG-INFO", "editable installation metadata\n")
    context.manifest["signatures"] = WATCH._fingerprints(e["project"], e["data"])
    fixtures.FIXTURES.sign(context.manifest)
    FOCUS.write(e["result"] / "manifest.json", context.manifest)
    context._manifest_file_hash = WATCH.file_digest(e["result"] / "manifest.json")
    for case in e["cases"]:
        run = context._run(case)
        expected = context._expected(case)
        FOCUS.write(run / "request_signature.json", expected)
        receipt = {"request_sha256": WATCH.digest(expected),
                   "files": context.runner.evidence_hashes(run, FOCUS.read(run / "status.json"))}
        FOCUS.write(run / "completion_receipt.json", receipt)
    exported = exported_tree(e["project"], tmp_path / "pure_git_export")
    shutil.rmtree(exported / METADATA)
    checked = FOCUS.verify_training_sources(context, exported, audit_path=tmp_path / "source_audit.json")
    assert "src/dma_wdf/models/mscmnet.py" in checked
    candidates, exclusions = FOCUS.collect_validated(context, e["queue"])
    assert len(candidates) == 6 and exclusions == []
    output = tmp_path / "joint_closeout"
    report = archive.archive_joint_closeout(context, e["queue"], output)
    assert report["status"] == "READY"
    assert archive.verify_ready(output) == report
    assert Path(report["archive_path"]).is_file()
    assert len(list(output.rglob("checkpoint_*.pt"))) == 4
    assert set(report["selected_models"]) == set(FOCUS.MODELS[2:])
    context.check_fingerprints()
