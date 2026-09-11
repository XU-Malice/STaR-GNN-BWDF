"""Real raw-prediction selection, complete archive, and retryable closeout checks."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


ARCHIVE = load("test_joint_closeout", "scripts/reproduce/archive_que_joint_closeout.py")
FIXTURE = load("test_joint_closeout_fixture", "tests/test_que_total_focus.py")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    return FIXTURE.evidence.__wrapped__(tmp_path, monkeypatch)


def test_complete_four_model_archive_is_self_contained_and_verified(evidence, tmp_path):
    e, destination = evidence, tmp_path / "closed"
    report = ARCHIVE.archive_joint_closeout(e["context"], e["queue"], destination)
    assert report["status"] == "READY"
    assert report["source_results"] == str(e["result"])
    assert report["project_root"] == str(e["project"])
    assert set(report["selected_models"]) == set(ARCHIVE.JOINT_MODELS)
    assert report["planned_cases"] == report["attempted_cases"] == 6
    assert report["original_paper_reproduction_claim"] is False
    with tarfile.open(report["archive_path"], "r:gz") as tar:
        members = tar.getnames()
    assert sum(name.endswith(".pt") for name in members) == 4
    assert "experiment_ledger.json" in members
    assert any(name.startswith("source_snapshot/src/") for name in members)
    for model, chosen in report["selected_models"].items():
        assert len(chosen["metrics"]) == 8
        assert {r["task"] for r in chosen["metrics"]} == {"24h", "168h"}
        src, copied = Path(chosen["run"]), destination / chosen["archived_run_relative"]
        assert ARCHIVE._inventory(src) == ARCHIVE._inventory(copied)
        assert chosen["case"] == e["context"]._plans()[chosen["case"]]["case"]
    assert ARCHIVE.verify_ready(destination) == report
    # Artifact validation is independent of later source-file loss/change.
    original = Path(report["selected_models"]["msnet"]["run"])
    checkpoint_name = json.loads((original / "status.json").read_text())["checkpoint_files"][0]
    (original / checkpoint_name).write_bytes(b"unrelated later source edit")
    assert ARCHIVE.verify_ready(destination) == report


@pytest.mark.parametrize("problem", ["missing", "hash_mismatch"])
def test_incomplete_or_corrupt_weights_cannot_publish_ready(evidence, tmp_path, problem):
    e = evidence
    case = next(c for c in e["cases"] if c["model"] == "msnet")
    run = e["context"]._run(case)
    checkpoint = run / json.loads((run / "status.json").read_text())["checkpoint_files"][0]
    if problem == "missing":
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(b"wrong bytes")
    destination = tmp_path / "not_ready"
    with pytest.raises(ValueError, match="Missing validated joint-model"):
        ARCHIVE.archive_joint_closeout(e["context"], e["queue"], destination)
    assert not destination.exists()


def test_extra_symlink_in_selected_run_refuses_complete_copy(evidence, tmp_path):
    e = evidence
    case = next(c for c in e["cases"] if c["model"] == "msnet")
    (e["context"]._run(case) / "unexpected_link").symlink_to(tmp_path)
    destination = tmp_path / "not_ready"
    with pytest.raises(ValueError, match="Symlink"):
        ARCHIVE.archive_joint_closeout(e["context"], e["queue"], destination)
    assert not destination.exists()


@pytest.mark.parametrize("member", ["checkpoint", "joint_models_complete.tar.gz"])
def test_ready_reverification_rejects_payload_or_tar_corruption(evidence, tmp_path, member):
    e, destination = evidence, tmp_path / "closed"
    ARCHIVE.archive_joint_closeout(e["context"], e["queue"], destination)
    if member == "checkpoint":
        status = json.loads((destination / "best_models/msnet/status.json").read_text())
        member = "best_models/msnet/" + status["checkpoint_files"][0]
    (destination / member).write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="hash"):
        ARCHIVE.verify_ready(destination)
    with pytest.raises(ValueError, match="hash"):
        ARCHIVE.archive_joint_closeout(e["context"], e["queue"], destination)


def test_history_records_failed_reused_and_pending_without_duplicate_keys(evidence, tmp_path):
    e = evidence
    queue = copy.deepcopy(e["queue"])
    queue["cases"][0].update(technical_status="FAIL", exit_code=1, validation="resource_failure")
    queue["cases"][1].update(technical_status="PASS(reused)")
    plans = copy.deepcopy(e["context"]._plans())
    pending = FIXTURE.FIXTURES.RUNNER.make_case("gru", "C", learning_rate_scale=.75)
    plans[pending["case"]] = pending
    # A stage-distinct planned alias demonstrates setting-key deduplication;
    # real source plans are validated separately by Context._plans().
    alias = copy.deepcopy(queue["cases"][0]["settings"])
    alias.update(stage="C", case="C_" + alias["case"][2:])
    plans[alias["case"]] = alias
    clone = copy.deepcopy(queue["cases"][0])
    clone.update(case=alias["case"], settings=alias)
    queue["cases"].append(clone)
    destination = tmp_path / "ledger"
    destination.mkdir()
    report = ARCHIVE.experiment_ledger(e["context"], queue, plans, destination)
    assert report["case_count"] == 8
    assert report["attempted_case_count"] == 7
    assert len(report["attempted_setting_keys"]) == 6
    records = {r["case"]: r for r in report["records"]}
    assert records[pending["case"]]["attempted"] is False
    assert records[queue["cases"][0]["case"]]["technical_status"] == "FAIL"
    assert records[queue["cases"][1]["case"]]["technical_status"] == "PASS(reused)"
    assert (destination / "experiment_ledger.tsv").is_file()
    assert report["pending_settings_are_not_claimed_as_trained"] is True


def test_captured_queue_and_snapshot_stay_immutable_on_resume(evidence, tmp_path):
    e, destination = evidence, tmp_path / "closed"
    report = ARCHIVE.archive_joint_closeout(e["context"], e["queue"], destination)
    original_inventory = ARCHIVE._inventory(destination)
    e["queue"].update(status="completed", updated_utc="later")
    assert ARCHIVE.archive_joint_closeout(e["context"], e["queue"], destination) == report
    assert ARCHIVE._inventory(destination) == original_inventory
    assert json.loads((destination / "queue_at_capture.json").read_text())["status"] == "running"


def test_tar_verification_failure_never_publishes_completion(evidence, tmp_path, monkeypatch):
    e, destination = evidence, tmp_path / "not_ready"
    def refuse(*args):
        raise ValueError("synthetic archive hash mismatch")
    monkeypatch.setattr(ARCHIVE, "_verify_archive", refuse)
    with pytest.raises(ValueError, match="hash mismatch"):
        ARCHIVE.archive_joint_closeout(e["context"], e["queue"], destination)
    assert not destination.exists()


def test_selection_uses_eight_cells_of_one_complete_candidate():
    metrics_a = [{"tolerance_ratio": .1} for _ in range(8)]
    metrics_b = [{"tolerance_ratio": 0.} for _ in range(8)]
    metrics_b[7] = {"tolerance_ratio": 1.1}
    a = {"model": "msnet", "case": "a", "metrics": metrics_a}
    b = {"model": "msnet", "case": "b", "metrics": metrics_b}
    selected = ARCHIVE.FOCUS.choose_models([a, b])["msnet"]
    assert selected is a
    assert selected["metrics"] is metrics_a


def test_changed_source_evaluation_blocks_archive(evidence, tmp_path, monkeypatch):
    monkeypatch.setattr(evidence["context"], "_recompute_evaluation", lambda: {})
    with pytest.raises(ValueError, match="Evaluation differs"):
        ARCHIVE.archive_joint_closeout(evidence["context"], evidence["queue"], tmp_path / "closed")


def test_waiting_retry_is_preserved_as_attempted_in_history(evidence, tmp_path):
    e = evidence
    queue = copy.deepcopy(e["queue"])
    queue["cases"][0].update(technical_status="waiting_gpu", resource_attempts=[{"exit_code": 75}])
    queue["cases"][1].update(technical_status="waiting_gpu", resource_attempts=[])
    destination = tmp_path / "ledger"
    destination.mkdir()
    ledger = ARCHIVE.experiment_ledger(e["context"], queue, e["context"]._plans(), destination)
    records = {record["case"]: record for record in ledger["records"]}
    assert records[queue["cases"][0]["case"]]["attempted"] is True
    assert records[queue["cases"][1]["case"]]["attempted"] is False
