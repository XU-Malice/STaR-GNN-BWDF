from __future__ import annotations

import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


FOCUS = load("test_focus_runner", "scripts/reproduce/run_que_total_focus.py")
FIXTURES = load("test_focus_source_fixtures", "tests/test_que_total_watch_evidence.py")
AUDIT = load("test_focus_real_audit", "scripts/reproduce/audit_que_total_objective.py")
FOLLOW = load("test_focus_followups", "scripts/train/que_total_followup_training.py")
SEARCH = load("test_focus_real_search", "scripts/reproduce/search_que_total_recurrent.py")
@pytest.fixture
def evidence(tmp_path, monkeypatch):
    e = FIXTURES.evidence.__wrapped__(tmp_path, monkeypatch)
    # Original watcher fixture intentionally used impossible per-DMA NSE values
    # to prove those values never enter ranking. Use valid, distant targets here.
    for task in e["paper"]["tasks"].values():
        for model in task.values():
            for letter in "ABCDEFGHIJ":
                model[letter]["NSE"] = .1
    c = e["context"]
    c.paper_path.write_text(yaml.safe_dump(e["paper"]))
    c.manifest["paper_sha256"] = FIXTURES.WATCH.file_digest(c.paper_path)
    c.manifest["signatures"] = FIXTURES.WATCH._fingerprints(e["project"], e["data"])
    FIXTURES.sign(c.manifest)
    FOCUS.write(e["result"] / "manifest.json", c.manifest)
    c._manifest_file_hash = FIXTURES.WATCH.file_digest(e["result"] / "manifest.json")
    for case in e["cases"]:
        run = c._run(case)
        request = c._expected(case)
        FOCUS.write(run / "request_signature.json", request)
        if case["model"] in ("gru", "lstm"):
            scaler = FOCUS.read(run / "scaler_audit.json")
            scaler["per_dma"] = {letter: {"parameters": {"mean": [0.], "std": [1.]}} for letter in "ABCDEFGHIJ"}
            FOCUS.write(run / "scaler_audit.json", scaler)
        receipt = {"request_sha256": FIXTURES.WATCH.digest(request),
                   "files": c.runner.evidence_hashes(run, FOCUS.read(run / "status.json"))}
        FOCUS.write(run / "completion_receipt.json", receipt)
    return e


def test_old_abc_results_are_verified_in_place(evidence):
    e = evidence
    sources, exclusions = FOCUS.collect_validated(e["context"], e["queue"])
    assert len(sources) == 6 and not exclusions
    assert all(Path(c["run"]).is_relative_to(e["result"]) for c in sources)
    assert all(c["evidence_digest"] for c in sources)


def test_corrupted_completed_artifact_is_excluded(evidence):
    e = evidence
    run = e["context"]._run(e["cases"][0])
    status = FOCUS.read(run / "status.json")
    (run / status["checkpoint_files"][0]).write_bytes(b"different weight bytes")
    sources, exclusions = FOCUS.collect_validated(e["context"], e["queue"])
    assert len(sources) == 5
    assert len(exclusions) == 1 and "completed_evidence_changed" in exclusions[0]["reason"]


def test_source_change_refuses_continuation(evidence):
    e = evidence
    (e["project"] / "scripts/train/run_que_comprehensive_reconstruction.py").write_text("modified")
    with pytest.raises(ValueError, match="fingerprint"):
        FOCUS.collect_validated(e["context"], e["queue"])


def test_receipt_and_weight_drift_after_initial_validation_rejected(evidence):
    e = evidence
    candidates, _ = FOCUS.collect_validated(e["context"], e["queue"])
    run = Path(candidates[0]["run"])
    (run / "checkpoint_gru_dma_A.pt").write_bytes(b"changed after validation")
    with pytest.raises(ValueError, match="changed after collection"):
        FOCUS.verify_candidate_evidence(candidates, e["context"])


def test_gpu_work_needs_terminal_status_and_released_shared_lock(tmp_path):
    (tmp_path / "logs").mkdir()
    assert FOCUS.terminal_and_lock({"status": "running"}, tmp_path, "7") is None
    assert FOCUS.terminal_and_lock({"status": "recurrent_assembly"}, tmp_path, "7") is None
    with (tmp_path / "logs/que_gpu_7.lock").open("a+") as old:
        fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert FOCUS.terminal_and_lock({"status": "completed"}, tmp_path, "7") is None
    new = FOCUS.terminal_and_lock({"status": "completed_with_failures"}, tmp_path, "7")
    assert new is not None
    new.close()


def test_real_audit_ranking_connects_to_followup_generator(evidence, tmp_path):
    e = evidence
    candidates, _ = FOCUS.collect_validated(e["context"], e["queue"])
    report = AUDIT.audit_candidates(candidates, e["context"].paper_path, tmp_path / "audit")
    ranking = FOCUS.read(Path(report["candidates_summary_file"]))
    records = FOCUS.complete_records(ranking, "pooled", e["paper"])
    assert len(records) == 6
    assert all(len(c["metrics"]) == 8 for c in records)
    # Synthetic fixture configurations use different dimensions, but generation
    # still passes through the real full setting identity and finite plan.
    cases = FOLLOW.generate_followups(records, FIXTURES.RUNNER, max_per_model=24)
    assert cases and len(cases) <= 48
    assert {c["model"] for c in cases} == {"gru", "lstm"}
    assert {c["seed"] for c in cases} == {20240604}
    assert not {FIXTURES.RUNNER.setting_key(c) for c in cases}.intersection(
        FIXTURES.RUNNER.setting_key(c["settings"]) for c in records)


def test_analysis_publishes_separate_whole_mode_tables(evidence, tmp_path):
    e = evidence
    candidates, _ = FOCUS.collect_validated(e["context"], e["queue"])
    output = tmp_path / "output"
    output.mkdir()
    args = SimpleNamespace(output_root=output, search_starts=2, search_sweeps=1, pair_trials=0)
    state = {}
    FOCUS.analyse(candidates, e["context"], args, state, AUDIT, SEARCH)
    assert state["matched_total_cells"] == 48
    assert (output / "total_comparison.tsv").exists()
    assert (output / "total_comparison_diagnostic_origin_mean.tsv").exists()
    selection = FOCUS.read(output / "selection.json")
    assert selection["primary_mode"] == "pooled"
    assert selection["original_paper_method_recovered"] is False
    assert set(selection["selected_by_mode"]) == {"pooled", "origin_mean"}
    assert len(list((output / "cohorts").rglob("checkpoint_*.pt"))) == 40


def test_status_requires_no_gpu_or_source_checkout(tmp_path, capsys):
    FOCUS.write(tmp_path / "campaign_status.json", {"status": "waiting_original_queue", "feasibility_blocked": True})
    assert FOCUS.main(["--status", "--output-root", str(tmp_path)]) == 0
    assert "不相容" in capsys.readouterr().out


def test_install_script_syntax_and_live_tree_is_not_merged():
    path = ROOT / "scripts/reproduce/install_que_total_focus.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "git merge" not in text and "git checkout" not in text and "pip install" not in text


def test_complete_ranking_rejects_partial_metric_table(evidence):
    with pytest.raises(ValueError, match="eight"):
        FOCUS.complete_records([{"mode": "pooled", "metrics": [1]}], "pooled", evidence["paper"])


def test_closeout_precedes_stop_and_freezes_all_four_cases(tmp_path):
    events = []
    output = tmp_path / "output"
    source = tmp_path / "source"
    FOCUS.write(source / "queue_status.json", {"status": "running"})
    report = {"status": "READY", "report_path": str(output / "joint_closeout/snapshot/READY.json"),
        "archive_path": str(output / "joint_closeout/snapshot/models.tar.gz"), "archive_sha256": "abc",
        "selected_models": {m: {"case": "selected_" + m} for m in FOCUS.MODELS[2:]}}
    def archive(*args):
        events.append("archive")
        return report
    def verify(path):
        events.append("verify")
        return report
    def stop(**kwargs):
        assert events == ["archive", "verify"]
        assert kwargs["execute"] is True
        events.append("stop")
        return {"status": "stop_requested"}
    args = SimpleNamespace(output_root=output, source_results=source, project_root=tmp_path)
    result = FOCUS.prepare_joint_handoff(None, args, {},
        SimpleNamespace(archive_joint_closeout=archive, verify_ready=verify),
        SimpleNamespace(stop_after_closeout=stop))
    assert result == report
    assert events == ["archive", "verify", "stop"]
    candidates = [{"model": m, "case": "selected_" + m} for m in FOCUS.MODELS]
    candidates += [{"model": "msnet", "case": "later_unarchived_msnet"}]
    assert FOCUS.restrict_to_closed_joint_models(candidates, report) == candidates[:6]


def test_corrupt_closeout_never_requests_stop(tmp_path):
    output = tmp_path / "output"
    FOCUS.write(output / "joint_closeout_reference.json", {"report_path": str(output / "snapshot/READY.json"),
                                                         "archive_sha256": "expected"})
    args = SimpleNamespace(output_root=output, source_results=tmp_path, project_root=tmp_path)
    calls = []
    archive = SimpleNamespace(verify_ready=lambda path: {"archive_sha256": "changed"})
    with pytest.raises(ValueError, match="differs"):
        FOCUS.prepare_joint_handoff(None, args, {}, archive,
            SimpleNamespace(stop_after_closeout=lambda **kw: calls.append(kw)))
    assert not calls


def test_missing_frozen_joint_model_blocks_continuation():
    report = {"selected_models": {m: {"case": m} for m in FOCUS.MODELS[2:]}}
    with pytest.raises(ValueError, match="no longer valid"):
        FOCUS.restrict_to_closed_joint_models([{"model": "gru", "case": "gru"}], report)


@pytest.mark.parametrize("source_status,watch,expected", [("running", False, "cpu_snapshot_completed"),
                                                          ("completed", True, "completed_search")])
def test_campaign_real_cpu_flow_and_terminal_zero_training_bundle(evidence, tmp_path, monkeypatch, source_status, watch, expected):
    e = evidence
    queue = {**e["queue"], "status": source_status}
    FOCUS.write(e["result"] / "queue_status.json", queue)
    real_module, real_read = FOCUS.module, FOCUS.read
    def modules(name, path):
        if name == "que_focus_evidence":
            return SimpleNamespace(Context=lambda *args: e["context"])
        return real_module(name, path)
    monkeypatch.setattr(FOCUS, "module", modules)
    monkeypatch.setattr(FOCUS, "read", lambda path: {"commit": "test-export"} if Path(path) == ROOT / "deployment.json" else real_read(path))
    # Source fixture imports were verified through its existing test seam;
    # separate tests cover fingerprint rejection and the real artifact boundary.
    monkeypatch.setattr(FOCUS, "verify_training_sources", lambda *args: ["test-fixture-core"])
    output = tmp_path / "campaign"
    args = ["--project-root", str(e["project"]), "--source-results", str(e["result"]),
            "--output-root", str(output), "--max-followups-per-model", "0",
            "--search-starts", "1", "--search-sweeps", "1", "--pair-trials", "0"]
    if watch:
        args.append("--watch")
    assert FOCUS.main(args) == 0
    state = real_read(output / "campaign_status.json")
    assert state["status"] == expected and state["matched_total_cells"] == 48
    if watch:
        assert Path(state["archive"]).is_file()
        assert real_read(output / "followup_plan.json")["cases"] == []
    assert real_read(e["result"] / "queue_status.json") == queue


def test_full_joint_archive_handoff_and_recurrent_cpu_search(evidence, tmp_path, monkeypatch):
    e = evidence
    queue = {**e["queue"], "status": "running", "current_stage": "C"}
    FOCUS.write(e["result"] / "queue_status.json", queue)
    real_module, real_read = FOCUS.module, FOCUS.read
    calls = []
    def stop(**kw):
        # Real four-model payload, weights and compressed archive must exist
        # before the process-control boundary can be reached.
        report = kw["validate_archive"](kw["archive_root"])
        assert report["status"] == "READY"
        assert set(report["selected_models"]) == set(FOCUS.MODELS[2:])
        calls.append(report)
        FOCUS.write(e["result"] / "queue_status.json", {**queue, "status": "interrupted"})
        return {"status": "stopped"}
    def modules(name, path):
        if name == "que_focus_evidence":
            return SimpleNamespace(Context=lambda *args: e["context"])
        if name == "que_joint_handoff_stop":
            return SimpleNamespace(stop_after_closeout=stop)
        return real_module(name, path)
    monkeypatch.setattr(FOCUS, "module", modules)
    monkeypatch.setattr(FOCUS, "read", lambda path: {"commit": "test-export"}
        if Path(path) == ROOT / "deployment.json" else real_read(path))
    monkeypatch.setattr(FOCUS, "verify_training_sources", lambda *args: ["fixture-core"])
    output = tmp_path / "focused"
    assert FOCUS.main(["--project-root", str(e["project"]), "--source-results", str(e["result"]),
        "--output-root", str(output), "--watch", "--close-joint-first", "--max-followups-per-model", "0",
        "--search-starts", "1", "--search-sweeps", "1", "--pair-trials", "0"]) == 0
    assert len(calls) == 1
    state = real_read(output / "campaign_status.json")
    assert state["status"] == "completed_search"
    assert Path(state["joint_archive"]).is_file()
    assert len(list((output / "joint_closeout/best_models").rglob("checkpoint_*.pt"))) == 4
    assert real_read(output / "followup_plan.json")["cases"] == []
    assert (output / "total_comparison.tsv").is_file()
    with tarfile.open(state["archive"], "r:gz") as bundle:
        assert not any(name.endswith((".pt", "joint_models_complete.tar.gz")) for name in bundle.getnames())
