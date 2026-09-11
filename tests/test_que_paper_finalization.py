"""Real-file retention and narrowly bounded, resumable checkpoint cleanup."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_que_paper_finalizer", ROOT / "scripts/reproduce/finalize_que_paper_baselines.py"
)
FINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINAL)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(path):
    return {p.relative_to(path).as_posix(): sha(p) for p in path.rglob("*") if p.is_file()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def metrics(truth, prediction):
    """Independent formulas, including the unusual sum-of-DMA-MAE definition."""
    error = prediction - truth
    totals = truth.sum(axis=2)
    residual = error.sum(axis=2)
    return [
        float(np.abs(error).mean(axis=(0, 1)).sum()),
        float(np.abs(residual / totals).mean()),
        float(np.sqrt(np.square(residual).mean())),
        float(1 - np.square(residual).sum() / np.square(totals - totals.mean()).sum()),
    ]


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    project = tmp_path / "project"
    history = project / "results/campaign"
    history.mkdir(parents=True)
    (project / "logs").mkdir()
    selection = json.loads((ROOT / "configs/evaluation/que_selected_total8_20260911.json").read_text())
    selection["numerical_source_sha256"] = {}
    selection["output_relative"] = "results/paper_final"
    selection["provenance"] = {"selection_uses_published_test_targets": True,
                               "independent_test_validation": False}
    origin = np.arange(46, dtype=float)[:, None, None]
    hour = np.arange(168, dtype=float)[None, :, None]
    dma = np.arange(10, dtype=float)[None, None, :]
    truth = 50.0 + origin * .2 + hour * .05 + dma * 2.0
    starts = np.array([(datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
                       for i in range(46)])
    chosen = []
    for number, (model, record) in enumerate(selection["models"].items()):
        run = history / "selected" / model
        run.mkdir(parents=True)
        prediction = truth + .12 * (number + 1) + .01 * np.sin(hour)
        np.savez_compressed(run / "predictions_common46.npz", y_true_24h=truth[:, :24],
                            y_true_168h=truth, y_pred_24h=prediction[:, :24],
                            y_pred_168h=prediction, forecast_starts=starts,
                            dma_letters=np.array(list("ABCDEFGHIJ")))
        checkpoints = ([f"checkpoints/checkpoint_{model}_dma_{d}.pt" for d in "ABCDEFGHIJ"]
                       if model in ("gru", "lstm") else [f"checkpoint_{model}.pt"])
        for relative in checkpoints:
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"opaque saved weights for {model}: {relative}".encode())
            chosen.append(path)
        (run / "resolved_config.yaml").write_text(f"model: {model}\nseed: 20240604\n")
        write_json(run / "status.json", {"status": "completed", "model": model,
                                         "checkpoint_files": checkpoints})
        write_json(run / "scaler_audit.json", {"normalization": "zscore"})
        manifest = history / "manifests" / f"{model}.json"
        write_json(manifest, {"model": model, "artifact_sha256": inventory(run)})
        record.update(source_run_relative=run.relative_to(project).as_posix(),
                      original_run_relative=run.relative_to(project).as_posix(),
                      source_manifest_relative=manifest.relative_to(project).as_posix(),
                      source_manifest_sha256=sha(manifest), artifact_sha256=inventory(run),
                      checkpoints=checkpoints, settings={"fixture": True},
                      expected_metrics=metrics(truth[:, :24], prediction[:, :24]) + metrics(truth, prediction))
    unused = []
    for index in range(3):
        run = history / "unused" / f"case_{index}"
        run.mkdir(parents=True)
        weights = run / "checkpoint_gru_dma_A.pt"
        weights.write_bytes(f"discardable candidate {index}".encode())
        unused.append(weights)
        (run / "resolved_config.yaml").write_text(f"learning_rate: 0.00{index + 1}\n")
        write_json(run / "status.json", {"status": "completed", "metric": index + .25})
        np.savez_compressed(run / "predictions_common46.npz", prediction=np.arange(12))
    outside = project / "results/unrelated/checkpoint_other.pt"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"not this reproduction project campaign")
    monkeypatch.setattr(FINAL, "HISTORY_ROOTS", ("results/campaign",))
    monkeypatch.setattr(FINAL, "assert_idle", lambda project: None)
    return {"project": project, "history": history, "selection": selection,
            "destination": project / selection["output_relative"], "selected": chosen,
            "unused": unused, "outside": outside}


def test_dry_run_saves_all_twenty_four_models_and_history_without_deleting(evidence):
    e = evidence
    before = inventory(e["history"])
    result = FINAL.finalize(e["project"], e["selection"], e["destination"])
    assert result["status"] == "READY"
    assert result["deleted_files"] == 0
    assert result["planned_files"] == 3
    assert inventory(e["history"]) == before
    assert len(list((e["destination"] / "models").rglob("*.pt"))) == 24
    for model, record in e["selection"]["models"].items():
        copied = e["destination"] / "models" / model
        for relative, expected in record["artifact_sha256"].items():
            assert sha(copied / relative) == expected
    with tarfile.open(e["destination"] / "best_models_complete.tar.gz", "r:gz") as archive:
        assert len([x for x in archive.getnames() if x.endswith(".pt")]) == 24
    with tarfile.open(e["destination"] / "history_records.tar.gz", "r:gz") as archive:
        names = archive.getnames()
        assert any("unused/case_0/resolved_config.yaml" in x for x in names)
        assert any("unused/case_0/status.json" in x for x in names)
        assert not any(x.endswith(".pt") for x in names)
    assert FINAL.verify_ready(e["destination"], e["selection"])["status"] == "READY"


def test_execute_deletes_only_planned_unused_weights_and_preserves_all_result_files(evidence):
    e = evidence
    all_before = inventory(e["history"])
    result = FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert result["status"] == "CLEANED"
    assert result["deleted_files"] == 3
    assert all(p.is_file() for p in e["selected"])
    assert all(not p.exists() for p in e["unused"])
    assert e["outside"].read_bytes() == b"not this reproduction project campaign"
    removed = {p.relative_to(e["history"]).as_posix() for p in e["unused"]}
    assert inventory(e["history"]) == {k: v for k, v in all_before.items() if k not in removed}
    FINAL.verify_ready(e["destination"], e["selection"])


@pytest.mark.parametrize("corruption", ["missing", "changed"])
def test_invalid_selected_checkpoint_never_removes_any_unused_checkpoint(evidence, corruption):
    e = evidence
    checkpoint = e["selected"][0]
    if corruption == "missing":
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(b"corrupt selected network")
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert all(p.exists() for p in e["unused"])
    assert not (e["destination"] / "retention_manifest.json").exists()


def test_recorded_metrics_are_recomputed_before_retaining_or_cleaning(evidence):
    e = evidence
    e["selection"]["models"]["gru"]["expected_metrics"][0] += 10
    with pytest.raises((ValueError, RuntimeError)):
        FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert all(p.exists() for p in e["unused"])


@pytest.mark.parametrize("kind", ["weight", "parent"])
def test_symlink_in_cleanup_scope_is_refused_without_following_it(evidence, tmp_path, kind):
    e = evidence
    external = tmp_path / "external"
    external.mkdir()
    target = external / "checkpoint_external.pt"
    target.write_bytes(b"external must remain untouched")
    if kind == "weight":
        (e["history"] / "unused/link.pt").symlink_to(target)
    else:
        (e["history"] / "external_parent").symlink_to(external, target_is_directory=True)
    with pytest.raises((ValueError, RuntimeError)):
        FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert target.read_bytes() == b"external must remain untouched"
    assert all(p.exists() for p in e["unused"])


@pytest.mark.parametrize("unsafe", ["../escape", "/tmp/outside_selection"])
def test_selected_source_cannot_escape_project(evidence, unsafe):
    e = evidence
    e["selection"]["models"]["gru"]["source_run_relative"] = unsafe
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert all(p.exists() for p in e["unused"])


def test_active_training_refusal_happens_before_deletion(evidence, monkeypatch):
    e = evidence
    def occupied(project):
        raise RuntimeError("owned training process is active")
    monkeypatch.setattr(FINAL, "assert_idle", occupied)
    with pytest.raises(RuntimeError, match="active"):
        FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert all(p.exists() for p in e["unused"])


def test_history_result_changed_during_archiving_blocks_deletion(evidence, monkeypatch):
    e = evidence
    original = FINAL.create_tar
    changed = False
    def race(archive, items):
        nonlocal changed
        original(archive, items)
        if not changed:
            (e["unused"][0].parent / "status.json").write_text('{"modified_after_capture": true}\n')
            changed = True
    monkeypatch.setattr(FINAL, "create_tar", race)
    with pytest.raises((ValueError, RuntimeError)):
        FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert changed
    assert all(p.exists() for p in e["unused"])


@pytest.mark.parametrize("which", ["checkpoint", "best_models_complete.tar.gz", "history_records.tar.gz"])
def test_corrupt_ready_payload_refuses_cleanup(evidence, which):
    e = evidence
    FINAL.finalize(e["project"], e["selection"], e["destination"])
    path = (next((e["destination"] / "models").rglob("*.pt")) if which == "checkpoint"
            else e["destination"] / which)
    path.write_bytes(b"corruption after READY")
    with pytest.raises((ValueError, RuntimeError, tarfile.TarError)):
        FINAL.verify_ready(e["destination"], e["selection"])
    with pytest.raises((ValueError, RuntimeError, tarfile.TarError)):
        FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert all(p.exists() for p in e["unused"])


def test_repeated_execution_does_not_expand_the_original_deletion_plan(evidence):
    e = evidence
    first = FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    plan_before = (e["destination"] / "cleanup_plan.json").read_bytes()
    new_weight = e["history"] / "unused/new_after_closeout/checkpoint_msnet.pt"
    new_weight.parent.mkdir()
    new_weight.write_bytes(b"not part of original plan")
    second = FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert first["deleted_files"] == second["deleted_files"] == 3
    assert new_weight.read_bytes() == b"not part of original plan"
    assert (e["destination"] / "cleanup_plan.json").read_bytes() == plan_before
    assert all(p.exists() for p in e["selected"])


def test_partial_cleanup_resumes_exact_plan_and_records_without_deleting_new_files(evidence, monkeypatch):
    e = evidence
    original = FINAL.remove_planned
    calls = 0
    def interrupt(project, record):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interrupted cleanup")
        return original(project, record)
    monkeypatch.setattr(FINAL, "remove_planned", interrupt)
    with pytest.raises(RuntimeError, match="interrupted cleanup"):
        FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert sum(not p.exists() for p in e["unused"]) == 1
    plan_before = (e["destination"] / "cleanup_plan.json").read_bytes()
    monkeypatch.setattr(FINAL, "remove_planned", original)
    result = FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert result["status"] == "CLEANED"
    assert result["deleted_files"] == 3
    assert (e["destination"] / "cleanup_plan.json").read_bytes() == plan_before
    assert all(p.exists() for p in e["selected"])


def test_unplanned_disappearance_after_dry_run_refuses_remaining_deletions(evidence):
    e = evidence
    FINAL.finalize(e["project"], e["selection"], e["destination"])
    e["unused"][0].unlink()
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert all(p.exists() for p in e["unused"][1:])


def test_replaced_unused_checkpoint_with_same_bytes_is_not_deleted(evidence):
    e = evidence
    FINAL.finalize(e["project"], e["selection"], e["destination"])
    weight = e["unused"][0]
    replacement = weight.with_suffix(".tmp")
    replacement.write_bytes(weight.read_bytes())
    replacement.replace(weight)
    with pytest.raises((ValueError, RuntimeError)):
        FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert all(p.exists() for p in e["unused"])


def test_history_changed_after_dry_run_blocks_all_deletions(evidence):
    e = evidence
    FINAL.finalize(e["project"], e["selection"], e["destination"])
    (e["unused"][0].parent / "status.json").write_text('{"changed_after_ready": true}\n')
    with pytest.raises((ValueError, RuntimeError)):
        FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert all(p.exists() for p in e["unused"])


@pytest.mark.parametrize("name", ["logs/que_gpu_7.lock", "results/que_recurrent_focus_20260910/campaign.lock"])
def test_existing_campaign_lock_refuses_cleanup_with_real_flock(evidence, name):
    e = evidence
    path = e["project"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="holds"):
            FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert all(p.exists() for p in e["unused"])


def test_non_temporal_or_unknown_checkpoints_are_preserved_inside_history_scope(evidence):
    e = evidence
    extra = [e["history"] / name for name in ["checkpoint_dcrnn.pt", "checkpoint_star_gnn.pt", "unrelated.pt"]]
    for path in extra:
        path.write_bytes(b"not an enumerated temporal baseline filename")
    result = FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    assert result["deleted_files"] == 3
    assert all(path.exists() for path in extra)


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "duplicate"])
def test_tar_verifier_rejects_unsafe_members(tmp_path, kind):
    path = tmp_path / "unsafe.tar.gz"
    name = {"traversal": "../escape", "absolute": "/tmp/escape",
            "symlink": "link", "duplicate": "twice"}[kind]
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "../../escape"
            archive.addfile(member)
        else:
            member.size = 3
            archive.addfile(member, io.BytesIO(b"abc"))
            if kind == "duplicate":
                archive.addfile(member, io.BytesIO(b"abc"))
    with pytest.raises(ValueError):
        FINAL.verify_tar(path, {name: hashlib.sha256(b"abc").hexdigest()})
