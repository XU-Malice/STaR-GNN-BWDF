"""Total-only stopping must never accept partial or mutable training evidence."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WATCH = load("test_total_watch_evidence", ROOT / "scripts/reproduce/que_total_watch_evidence.py")
ART = load("test_total_watch_artifacts", ROOT / "tests/test_que_comprehensive_runner.py")
RUNNER = ART.RUNNER
REUSE = load("test_total_watch_reuse", ROOT / "scripts/train/reuse_que_completed_runs.py")
ASSEMBLY = load("test_total_watch_native_assembly", ROOT / "scripts/reproduce/assemble_que_recurrent_candidates.py")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sign(manifest):
    manifest.pop("signature", None)
    manifest["signature"] = WATCH.digest(manifest)
    return manifest


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    project, result = tmp_path / "project", tmp_path / "project/results/queue"
    data = project / "data/processed/data_build"
    data.mkdir(parents=True)
    (data / "demand.parquet").write_bytes(b"frozen source-data fixture")
    for name in REUSE.REQUIRED_TRAINING_FILES | {"scripts/train/run_que_comprehensive_reconstruction.py"}:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source, imported through test-only audited runner seam\n")
    algorithm = project / "scripts/reproduce/assemble_que_recurrent_candidates.py"
    algorithm.parent.mkdir(parents=True)
    algorithm.write_bytes((ROOT / "scripts/reproduce/assemble_que_recurrent_candidates.py").read_bytes())
    cases = [RUNNER.make_case(m) for m in WATCH.MODELS]
    published = {m: ART.model_fixture(c) for m, c in zip(WATCH.MODELS, cases)}
    for m in ("gru", "lstm"):
        published[m].update(family="independent_recurrent", cell_type=m.upper())
    config = project / "configs/model/mscmnet_baselines.yaml"
    config.write_text(yaml.safe_dump({"models": published}))
    paper = {"tasks": {}}
    for task in ("24h", "168h"):
        paper["tasks"][task] = {}
        rows = [r for r in RUNNER.metric_rows(ART.arrays_fixture(), "pooled") if r["task"] == task]
        for name in WATCH.MODEL_NAMES.values():
            paper["tasks"][task][name] = {}
            for row in rows:
                paper["tasks"][task][name].setdefault(row["series"], {})[row["metric"]] = row["value"] if row["series"] == "total" else 1e6
    paper_path = project / "configs/evaluation/mscmnet_paper_metrics.yaml"
    paper_path.parent.mkdir(parents=True)
    paper_path.write_text(yaml.safe_dump(paper))
    manifest = sign({"version": 1, "base_cases": cases, "maximum_cases": 415, "signatures": WATCH._fingerprints(project, data),
                     "paper_sha256": WATCH.file_digest(paper_path), "seed": WATCH.SEED, "selection_mode": "pooled", "device": "cuda:0", "data_dir": str(data)})
    write(result / "manifest.json", manifest)
    evaluation = ART.evaluation_fixture()
    write(result / "audit_data_protocol/paper_data_statistics.json", {"common_evaluation": evaluation})
    queue = {"status": "running", "cases": []}
    for case in cases:
        request = {"signature": WATCH.digest({"manifest": manifest["signature"], "settings": RUNNER.setting_key(case)}),
                   "case": case, "model_config": RUNNER.expected_model_config(published, case), "evaluation": evaluation}
        run = result / "cases" / case["case"] / case["model"] / f"seed_{WATCH.SEED}"
        ART.build_artifacts(run, case, request)
        source_config = yaml.safe_load((run / "resolved_config.yaml").read_text())
        source_config["protocol"] = {"dma_letters": list("ABCDEFGHIJ"), "day_hours": 24}
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(source_config))
        status = WATCH.read_json(run / "status.json")
        if case["model"] in ("gru", "lstm"):
            names = [f"checkpoint_{case['model']}_dma_{letter}.pt" for letter in "ABCDEFGHIJ"]
            for old, new in zip(status["checkpoint_files"], names):
                (run / old).rename(run / new)
            status["checkpoint_files"] = names
            write(run / "status.json", status)
        write(run / "completion_receipt.json", {"request_sha256": WATCH.digest(request), "files": RUNNER.evidence_hashes(run, status)})
        RUNNER.score_case(run, case, paper)
        queue["cases"].append(ART.make_record(case))
        queue["cases"][-1]["exit_code"] = 0
    # Only module loading and physical source-data decoding are substituted.
    # Real artifact validation, hashes, metrics, source guards and export execute.
    monkeypatch.setattr(WATCH, "_load_module", lambda name, path: REUSE if "reuse" in name else RUNNER)
    monkeypatch.setattr(WATCH.Context, "_recompute_evaluation", lambda self: copy.deepcopy(evaluation))
    context = WATCH.Context(project, result)
    return {"context": context, "project": project, "result": result, "data": data, "manifest": manifest,
            "queue": queue, "cases": cases, "paper": paper, "evaluation": evaluation, "published": published}


def first_run(e):
    return e["context"]._run(e["cases"][0])


def test_real_evidence_context_integrates_with_observer_before_stop(evidence):
    e = evidence
    main = load("test_full_total_watch_main", ROOT / "scripts/reproduce/watch_que_total_match.py")
    write(e["result"] / "queue_status.json", e["queue"])
    args = main.parse_args(["--project-root", str(e["project"]), "--run-tag", "queue", "--watch", "--stop-on-success"])
    class Handle:
        stopped = False
        def describe(self): return {"pid": 12345, "test_bound_handle": True}
        def exited(self): return self.stopped
        def close(self): pass
        def request_stop(self):
            state = json.loads((args.output_root / "watch_status.json").read_text())
            assert state["selection"]["verified_all_matched"] is True
            assert state["matched_total_cells"] == 48
            assert Path(state["archive"]["path"]).is_file()
            frozen = json.loads((Path(state["selected_root"]) / "freeze_manifest.json").read_text())
            assert frozen["numerical_total_match_verified"] is True
            assert sum(name.endswith(".pt") for name in frozen["files"]) == 24
            self.stopped = True
            return True
    handle = Handle()
    assert main.run(args, context_factory=lambda *a, **k: e["context"], binder=lambda *a: handle) == 0
    assert handle.stopped


def mutate_rows(path, mutator, delimiter=","):
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fields, rows = reader.fieldnames, list(reader)
    mutator(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_total_only_scan_ignores_dma_closeness_and_existing_score_flags(evidence):
    e, c = evidence, evidence["context"]
    for record in e["queue"]["cases"]:
        record["scores"] = {"pooled": {"total8_passed": 0, "dma80_passed": 0}}
    report = c.scan(e["queue"])
    assert report["all_matched"] and report["matched_models"] == list(WATCH.MODELS)
    assert report["verification"] == "tentative" and not report["verified_all_matched"]
    assert all(x["matched_count"] == 8 for x in report["models"].values())
    assert "Stage-D" in report["scope"]


def test_total_minimax_finds_all_eight_match_when_mean_best_fails(evidence):
    c = evidence["context"]
    case = evidence["cases"][0]
    keys = WATCH.TOTAL_KEYS
    def candidate(ratios):
        rows = []
        for (task, metric), ratio in zip(keys, ratios):
            target = c.paper["tasks"][task]["GRU"]["total"][metric]
            diff = ratio * (.01 if metric == "NSE" else target * .05)
            rows.append({"task": task, "series": "total", "metric": metric, "value": target + diff})
        return c._candidate(case, c._total_rows("gru", rows))
    a, b = candidate([1.1] + [0] * 7), candidate([.99] * 8)
    assert a["mean_ratio"] < b["mean_ratio"]
    best = min([a, b], key=lambda x: (x["worst_ratio"], x["mean_ratio"], x["case"]))
    assert best is b and best["all8_matched"]


@pytest.mark.parametrize("problem", ["nan", "missing", "duplicate", "wrong_paper", "extra", "stored_disagrees", "missing_receipt"])
def test_scan_excludes_invalid_total_summaries(evidence, problem):
    run = first_run(evidence)
    def change(rows):
        idx = next(i for i, row in enumerate(rows) if row["series"] == "total" and row["mode"] == "pooled")
        if problem == "nan": rows[idx]["value"] = "nan"
        elif problem == "missing": rows.pop(idx)
        elif problem == "duplicate": rows.append(dict(rows[idx]))
        elif problem == "wrong_paper": rows[idx]["paper_value"] = "123456"
        elif problem == "extra": rows.append({**rows[idx], "metric": "MSE"})
        elif problem == "stored_disagrees": rows[idx]["value"] = "123456"
    if problem == "missing_receipt":
        (run / "completion_receipt.json").unlink()
    else:
        mutate_rows(run / "paper_gaps.tsv", change, "\t")
    report = evidence["context"].scan(evidence["queue"])
    assert report["models"]["gru"] is None and not report["all_matched"]
    assert len(report["exclusions"]) == 1


@pytest.mark.parametrize("status,exit_code", [("running", 0), ("FAIL", 1), ("PASS", 1)])
def test_partial_failed_or_wrong_exit_candidates_never_match(evidence, status, exit_code):
    evidence["queue"]["cases"][0].update(technical_status=status, exit_code=exit_code)
    assert evidence["context"].scan(evidence["queue"])["models"]["gru"] is None


def test_queue_cannot_substitute_unplanned_settings(evidence):
    evidence["queue"]["cases"][0]["settings"] = {**evidence["cases"][0], "max_epochs": 99}
    report = evidence["context"].scan(evidence["queue"])
    assert report["models"]["gru"] is None
    assert "frozen plan" in report["exclusions"][0]["reason"]


def test_deep_verification_recomputes_all_six_and_atomic_freeze_includes_weights(evidence):
    c = evidence["context"]
    report = c.verify_selected(c.scan(evidence["queue"]))
    assert report["verified_all_matched"]
    assert report["verification"] == "raw_predictions_and_receipts_verified"
    destination = evidence["project"] / "results/total_watch/frozen"
    result = c.export_to(destination, report)
    assert result == destination
    manifest = WATCH.read_json(result / "freeze_manifest.json")
    assert manifest["weights_included"] and manifest["criteria"]["dma_closeness_required"] is False
    assert len(list((result / "selected_results").rglob("*.pt"))) == 24
    assert all(WATCH.file_digest(result / n) == h for n, h in manifest["files"].items())
    for model, chosen in report["models"].items():
        assert chosen["case"] in chosen["run"]
        assert len(chosen["metrics"]) == 8
    with pytest.raises(ValueError, match="exists"):
        c.export_to(destination, report)


@pytest.mark.parametrize("change,expected", [("checkpoint", "completed_evidence_changed"), ("request", "request_signature_mismatch"),
                                              ("frozen_weights", "not_single_frozen_checkpoint"), ("raw_truth", "truth_does_not_match"),
                                              ("first_day", "invalid_artifact"), ("loss", "nonfinite_loss"), ("source", "fingerprints changed")])
def test_deep_validation_rejects_mutated_receipts_raw_arrays_and_source(evidence, change, expected):
    c, run = evidence["context"], first_run(evidence)
    report = c.scan(evidence["queue"])
    if change == "checkpoint":
        (run / WATCH.read_json(run / "status.json")["checkpoint_files"][0]).write_bytes(b"changed")
    elif change == "request":
        value = WATCH.read_json(run / "request_signature.json")
        value["case"]["learning_rate_scale"] = 999
        write(run / "request_signature.json", value)
    elif change == "frozen_weights":
        value = WATCH.read_json(run / "status.json")
        value["single_frozen_checkpoint_for_24h_and_168h"] = False
        write(run / "status.json", value)
    elif change in ("raw_truth", "first_day"):
        arrays = ART.arrays_fixture()
        if change == "raw_truth":
            arrays["y_true_168h"] = arrays["y_true_168h"] + .1
            arrays["y_true_24h"] = arrays["y_true_168h"][:, :24]
        else:
            arrays["y_pred_24h"] = arrays["y_pred_24h"] + 1
        np.savez_compressed(run / "predictions_common46.npz", **arrays)
    elif change == "loss":
        mutate_rows(run / "loss_curve.csv", lambda rows: rows[0].update(train_loss="nan"))
    else:
        (evidence["project"] / "src/dma_wdf/models/mscmnet.py").write_text("changed source")
    with pytest.raises(ValueError, match=expected):
        c.verify_selected(report)


def test_source_fingerprint_guard_runs_before_repository_import(evidence, monkeypatch):
    (evidence["project"] / "scripts/new_file.py").write_text("new file")
    imported = []
    monkeypatch.setattr(WATCH, "_load_module", lambda *a: imported.append(a))
    with pytest.raises(ValueError, match="fingerprints changed"):
        WATCH.Context(evidence["project"], evidence["result"])
    assert imported == []


def test_deep_gate_reaudits_raw_source_evaluation(evidence, monkeypatch):
    c = evidence["context"]
    monkeypatch.setattr(c, "_recompute_evaluation", lambda: {"bad": "evaluation"})
    with pytest.raises(ValueError, match="freshly audited"):
        c.verify_selected(c.scan(evidence["queue"]))


def test_unverified_or_edited_report_cannot_be_exported(evidence):
    c = evidence["context"]
    report = c.scan(evidence["queue"])
    with pytest.raises(ValueError, match="fully verified"):
        c.export_to(evidence["project"] / "results/frozen", report)
    verified = c.verify_selected(report)
    verified["models"]["gru"]["metrics"][0]["value"] = 0
    with pytest.raises(ValueError, match="fully verified"):
        c.export_to(evidence["project"] / "results/frozen", verified)


def test_evidence_change_between_verification_and_export_prevents_freeze(evidence):
    c = evidence["context"]
    verified = c.verify_selected(c.scan(evidence["queue"]))
    (first_run(evidence) / "checkpoint_gru_dma_A.pt").write_bytes(b"changed after verification")
    target = evidence["project"] / "results/frozen"
    with pytest.raises(ValueError, match="changed before freeze"):
        c.export_to(target, verified)
    assert not target.exists()
    assert not list(target.parent.glob(".frozen.freeze-*"))


def test_reused_original_evidence_is_verified(evidence):
    e, c, run = evidence, evidence["context"], first_run(evidence)
    old = copy.deepcopy(e["manifest"])
    old["signatures"]["source"]["scripts/train/run_que_comprehensive_reconstruction.py"] = "a" * 64
    old["signatures"]["source_sha256"] = WATCH.digest(old["signatures"]["source"])
    sign(old)
    old_request = {**c._expected(e["cases"][0]), "signature": WATCH.digest({"manifest": old["signature"], "settings": RUNNER.setting_key(e["cases"][0])})}
    original_receipt = WATCH.read_json(run / "completion_receipt.json")
    original_receipt["request_sha256"] = WATCH.digest(old_request)
    provenance = {"source_manifest": old, "source_request": old_request, "source_completion_receipt": original_receipt,
                  "unchanged_training_sources": REUSE._check_training_compatibility(old, e["manifest"]), "training_performed_during_import": False,
                  "destination_manifest_signature": e["manifest"]["signature"]}
    write(run / "reused_source_provenance.json", provenance)
    receipt = WATCH.read_json(run / "completion_receipt.json")
    receipt["files"]["reused_source_provenance.json"] = WATCH.file_digest(run / "reused_source_provenance.json")
    write(run / "completion_receipt.json", receipt)
    e["queue"]["cases"][0]["technical_status"] = "PASS(reused)"
    assert c.verify_selected(c.scan(e["queue"]))["verified_all_matched"]
    provenance["source_completion_receipt"]["files"]["checkpoint_gru_dma_A.pt"] = "b" * 64
    write(run / "reused_source_provenance.json", provenance)
    receipt["files"]["reused_source_provenance.json"] = WATCH.file_digest(run / "reused_source_provenance.json")
    write(run / "completion_receipt.json", receipt)
    with pytest.raises(ValueError, match="Original training evidence changed"):
        c.verify_selected(c.scan(e["queue"]))


def add_native_d(e, model="gru"):
    c = e["context"]
    case = next(case for case in e["cases"] if case["model"] == model)
    run = c.result_root / "recurrent_assembled" / model / "pooled"
    manifest = ASSEMBLY.assemble_recurrent_candidates(candidate_dirs=[c._run(case)], paper_config=c.paper_path, output_root=run)
    report = c.scan(e["queue"])
    assert report["candidate_counts"][model] == 2
    assert not report["exclusions"]
    with (run / "metrics.tsv").open() as f:
        rows = c._total_rows(model, list(csv.DictReader(f, delimiter="\t")))
    report["models"][model] = c._assembly_candidate(run, model, manifest, rows)
    return run, manifest, report


def refresh_d_manifest(run, manifest, request=False):
    if request:
        manifest["request_sha256"] = ASSEMBLY._json_hash(manifest["request"])
    manifest["artifact_sha256"] = {name: WATCH.file_digest(run / name) for name in manifest["artifact_sha256"]}
    write(run / "manifest.json", manifest)


def test_native_d_completed_cohort_is_included_and_verified_without_dma_closeness(evidence):
    run, manifest, report = add_native_d(evidence)
    assert report["models"]["gru"]["all8_matched"]
    # All DMA reference values are intentionally far away; only total gates count.
    assert manifest["score"]["all88_passed"] is False
    verified = evidence["context"].verify_selected(report)
    candidate = verified["models"]["gru"]
    assert candidate["kind"] == "assembled_recurrent"
    assert candidate["matched_count"] == 8
    assert len(candidate["verified_source_evidence"]) == 1
    target = evidence["project"] / "results/d_frozen"
    evidence["context"].export_to(target, verified)
    assert len(list((target / "selected_results/gru/checkpoints").glob("*.pt"))) == 10
    assert len(list((target / "selected_results/gru/source_configs").glob("*.yaml"))) == 10
    assert (target / "selected_results/gru/manifest.json").exists()
    assert len(list((target / "selected_source_evidence").rglob("completion_receipt.json"))) == 1
    assert (target / "selected_results/gru/predictions_common46.npz").read_bytes() == (run / "predictions_common46.npz").read_bytes()


@pytest.mark.parametrize("change", ["wrong_mode", "wrong_seed", "missing_network", "duplicate_dma", "outside_source", "bad_request_hash", "bad_artifact_hash"])
def test_native_d_scan_rejects_bad_identity_mapping_and_hashes(evidence, change):
    run, manifest, _ = add_native_d(evidence)
    if change == "wrong_mode": manifest["mode"] = "origin_mean"
    elif change == "wrong_seed": manifest["seed"] += 1
    elif change == "missing_network": manifest["selected_networks"].pop()
    elif change == "duplicate_dma": manifest["selected_networks"][1]["dma"] = "A"
    elif change == "outside_source":
        manifest["request"]["candidates"][0]["path"] = "/tmp/outside/gru/seed_20240604"
        manifest["request_sha256"] = ASSEMBLY._json_hash(manifest["request"])
    elif change == "bad_request_hash": manifest["request_sha256"] = "0" * 64
    else: manifest["artifact_sha256"]["metrics.tsv"] = "0" * 64
    write(run / "manifest.json", manifest)
    report = evidence["context"].scan(evidence["queue"])
    assert report["candidate_counts"]["gru"] == 1
    assert len(report["exclusions"]) == 1 and report["exclusions"][0]["case"] == "D_gru_pooled"


def test_native_d_incomplete_or_failed_source_is_excluded(evidence):
    add_native_d(evidence)
    evidence["queue"]["cases"][0].update(technical_status="FAIL", exit_code=1)
    report = evidence["context"].scan(evidence["queue"])
    assert report["models"]["gru"] is None
    assert "PASS" in report["exclusions"][0]["reason"]


def test_native_d_deep_verification_checks_source_receipts(evidence):
    run, manifest, report = add_native_d(evidence)
    source = first_run(evidence)
    (source / "checkpoint_gru_dma_A.pt").write_bytes(b"changed source weights")
    with pytest.raises(ValueError, match="completed_evidence_changed"):
        evidence["context"].verify_selected(report)


def test_native_d_checkpoint_cannot_be_substituted_even_with_updated_artifact_hashes(evidence):
    run, manifest, report = add_native_d(evidence)
    item = manifest["selected_networks"][0]
    (run / item["checkpoint"]).write_bytes(b"different whole network")
    item["checkpoint_sha256"] = WATCH.file_digest(run / item["checkpoint"])
    refresh_d_manifest(run, manifest)
    report["models"]["gru"] = evidence["context"]._assembly_candidate(run, "gru", manifest, report["models"]["gru"]["metrics"])
    with pytest.raises(ValueError, match="differs from intact source"):
        evidence["context"].verify_selected(report)


def test_native_d_predictions_must_equal_exact_selected_columns_both_horizons(evidence):
    run, manifest, report = add_native_d(evidence)
    with np.load(run / "predictions_common46.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["y_pred_168h"][:, :, 0] += np.float32(1e-5)
    arrays["y_pred_24h"] = arrays["y_pred_168h"][:, :24].copy()
    np.savez_compressed(run / "predictions_common46.npz", **arrays)
    refresh_d_manifest(run, manifest)
    with pytest.raises(ValueError, match="exact selected source columns"):
        evidence["context"].verify_selected(report)


def test_native_d_nonfinite_raw_predictions_are_rejected_even_with_updated_hash(evidence):
    run, manifest, report = add_native_d(evidence)
    with np.load(run / "predictions_common46.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["y_pred_168h"][0, 0, 0] = np.nan
    arrays["y_pred_24h"] = arrays["y_pred_168h"][:, :24].copy()
    np.savez_compressed(run / "predictions_common46.npz", **arrays)
    refresh_d_manifest(run, manifest)
    with pytest.raises(ValueError, match="non-finite"):
        evidence["context"].verify_selected(report)


def test_native_d_different_weights_between_horizons_declaration_rejected(evidence):
    run, manifest, _ = add_native_d(evidence)
    manifest["selected_networks"][0]["same_source_for_all_horizons_and_metrics"] = False
    write(run / "manifest.json", manifest)
    report = evidence["context"].scan(evidence["queue"])
    assert report["candidate_counts"]["gru"] == 1
    assert "mapping mismatch" in report["exclusions"][0]["reason"]
