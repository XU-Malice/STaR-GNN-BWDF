from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dma_wdf.data.reproduction_metrics import compute_reproduction_metrics

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("que_total_search", ROOT / "scripts/reproduce/search_que_total_recurrent.py")
assert SPEC and SPEC.loader
SEARCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEARCH)


def make_source(tmp_path: Path, name: str, *, candidate: int = 0, model: str = "gru", seed: int = 20240604) -> Path:
    run = tmp_path / name / model / f"seed_{seed}"
    run.mkdir(parents=True)
    rng = np.random.default_rng(177)
    truth = rng.uniform(20, 40, (46, 168, 10)).astype(np.float32)
    errors = np.where(np.arange(10) % 2 == candidate, 1., 4.).astype(np.float32)
    prediction = truth + errors
    origins = np.char.add((np.datetime64("2023-01-13") + np.arange(46)).astype(str), "T00:00:00+01:00")
    np.savez_compressed(run / "predictions_common46.npz", y_true_24h=truth[:, :24], y_true_168h=truth,
                        y_pred_24h=prediction[:, :24], y_pred_168h=prediction,
                        forecast_starts=origins, dma_letters=np.array(list("ABCDEFGHIJ")))
    checkpoints = [f"checkpoint_{model}_dma_{letter}.pt" for letter in "ABCDEFGHIJ"]
    for checkpoint in checkpoints:
        # These bytes are deliberately not valid torch checkpoints; searching
        # must copy intact files without importing torch or deserializing them.
        (run / checkpoint).write_bytes(f"opaque-checkpoint-and-embedded-scaler:{name}:{checkpoint}".encode())
    status = {"status": "completed", "model": model, "seed": seed, "checkpoint_files": checkpoints,
              "single_frozen_checkpoint_for_24h_and_168h": True}
    config = {"seed": seed, "protocol": {"dma_letters": list("ABCDEFGHIJ"), "day_hours": 24},
              "model": {"family": "independent_recurrent", "cell_type": model.upper(),
                        "best_epochs": list(range(10, 20)), "hidden_sizes": [[32]] * 10},
              "training": {"normalization": "zscore", "batch_size": 8 + candidate}}
    scaler = {"test_values_used_for_fit": False, "normalization": "zscore",
              "per_dma": {letter: {"parameters": {"mean": [float(j)], "std": [1.]}} for j, letter in enumerate("ABCDEFGHIJ")}}
    (run / "status.json").write_text(json.dumps(status))
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (run / "scaler_audit.json").write_text(json.dumps(scaler))
    return run


def modify_npz(run: Path, edit) -> None:
    path = run / "predictions_common46.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    edit(arrays)
    np.savez_compressed(path, **arrays)


def paper_for(tmp_path: Path, source: Path, *, mode: str = "pooled", prediction=None, name="paper.yaml", dma_targets=False) -> Path:
    paper: dict = {"tasks": {}}
    with np.load(source / "predictions_common46.npz", allow_pickle=False) as archive:
        for task in SEARCH.TASKS:
            truth = archive[f"y_true_{task}"]
            pred = truth + np.float32(1) if prediction is None else prediction[:, :truth.shape[1]]
            rows = compute_reproduction_metrics(truth, pred, mode=mode)
            metrics = {}
            for row in rows:
                if row["series"] == "total" or (dma_targets and row["series"] in SEARCH.LETTERS):
                    metrics.setdefault(row["series"], {})[row["metric"]] = row["value"]
            paper["tasks"][task] = {"GRU": metrics, "LSTM": metrics}
    path = tmp_path / name
    path.write_text(yaml.safe_dump(paper))
    return path


@pytest.mark.parametrize("mode", ["pooled", "origin_mean"])
def test_selects_whole_checkpoint_scaler_and_exact_both_horizon_columns(tmp_path: Path, mode: str) -> None:
    sources = [make_source(tmp_path, "first", candidate=0), make_source(tmp_path, "second", candidate=1)]
    hashes_before = {str(p): SEARCH._sha(p) for source in sources for p in source.iterdir()}
    paper = paper_for(tmp_path, sources[0], mode=mode)
    out = tmp_path / "cohort"
    manifest = SEARCH.search_and_export(sources[::-1], paper, out, mode, starts=4, sweeps=3, pair_trials=20)
    assert manifest["score"]["all_total8_passed"] is True
    assert manifest["score"]["worst_ratio"] == pytest.approx(0, abs=1e-10)
    assert manifest["dma_targets_used_in_objective"] is False
    assert manifest["global_optimum_proven"] is False
    assert manifest["original_method_recovered"] is False
    assert manifest["held_out_validation"] is False
    assert len(manifest["total_comparison"]) == 8
    assert len((out / "total_comparison.tsv").read_text().splitlines()) == 9
    assert len((out / "metrics.tsv").read_text().splitlines()) == 91
    with np.load(out / "predictions_common46.npz", allow_pickle=False) as saved:
        for j, selected in enumerate(manifest["selected_networks"]):
            source = sources[j % 2]
            assert selected["source_path"] == str(source)
            assert selected["per_dma_model_config"]["best_epochs"] == j + 10
            assert (out / selected["checkpoint"]).read_bytes() == (source / Path(selected["checkpoint"]).name).read_bytes()
            assert (out / selected["source_scaler_audit"]).read_bytes() == (source / "scaler_audit.json").read_bytes()
            with np.load(source / "predictions_common46.npz", allow_pickle=False) as original:
                for task in SEARCH.TASKS:
                    assert np.array_equal(saved[f"y_pred_{task}"][:, :, j], original[f"y_pred_{task}"][:, :, j])
    assert hashes_before == {str(p): SEARCH._sha(p) for source in sources for p in source.iterdir()}
    mtimes = {str(p): p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    assert SEARCH.search_and_export(sources, paper, out, mode, 4, 3, 20) == manifest
    assert mtimes == {str(p): p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}


def test_dma_targets_never_influence_objective_or_tiebreak_and_runs_are_deterministic(tmp_path: Path) -> None:
    sources = [make_source(tmp_path, "first", candidate=0), make_source(tmp_path, "second", candidate=1)]
    plain = paper_for(tmp_path, sources[0])
    perturbed = yaml.safe_load(plain.read_text())
    for task in SEARCH.TASKS:
        for letter in SEARCH.LETTERS:
            perturbed["tasks"][task]["GRU"][letter] = {metric: (1e9 if letter < "F" else -1e9) for metric in SEARCH.METRICS}
    dma_paper = tmp_path / "dma_perturbed.yaml"
    dma_paper.write_text(yaml.safe_dump(perturbed))
    first = SEARCH.search_and_export(sources, plain, tmp_path / "one", starts=4, sweeps=2, pair_trials=11, search_seed=42)
    second = SEARCH.search_and_export(sources[::-1], dma_paper, tmp_path / "two", starts=4, sweeps=2, pair_trials=11, search_seed=42)
    assert first["selected_networks"] == second["selected_networks"]
    assert first["score"] == second["score"]
    assert first["budget"] == second["budget"]
    assert (tmp_path / "one/search_trace.json").read_bytes() == (tmp_path / "two/search_trace.json").read_bytes()
    assert (tmp_path / "one/predictions_common46.npz").read_bytes() == (tmp_path / "two/predictions_common46.npz").read_bytes()


@pytest.mark.parametrize("mode", ["pooled", "origin_mean"])
def test_numpy_delta_totals_match_exact_metric_helper_without_dma_metric_evaluations(tmp_path: Path, mode: str, monkeypatch) -> None:
    sources = [make_source(tmp_path, "first", candidate=0), make_source(tmp_path, "second", candidate=1)]
    for k, source in enumerate(sources):
        def vary(arrays, k=k):
            error = (np.arange(46)[:, None, None] + 1) * np.array([1, -1] * 5)[None, None, :] * (.01 + .02 * k)
            arrays["y_pred_168h"] = arrays["y_true_168h"] + error
            arrays["y_pred_24h"] = arrays["y_pred_168h"][:, :24].copy()
        modify_npz(source, vary)
    paper = paper_for(tmp_path, sources[0], mode=mode)
    candidates = SEARCH._LEGACY._load_candidates(sources)
    targets = SEARCH._total_targets(paper, "gru")
    def forbidden(*args, **kwargs):
        raise AssertionError("Search alternatives must not recompute full DMA metrics")
    monkeypatch.setattr(SEARCH, "compute_reproduction_metrics", forbidden)
    search = SEARCH._TotalSearch(candidates, targets, mode)
    baseline = (0,) * 10
    replacement = (1, 1) + (0,) * 8
    delta = search.replace(search.totals(baseline), baseline, ((0, 1), (1, 1)))
    key, score = search.evaluate(replacement, delta)
    assert key == (score["worst_ratio"], score["mean_ratio"], replacement)
    assembled = SEARCH._assemble(candidates, replacement)
    for task in SEARCH.TASKS:
        exact = {row["metric"]: row["value"] for row in compute_reproduction_metrics(assembled[f"y_true_{task}"], assembled[f"y_pred_{task}"], mode=mode) if row["series"] == "total"}
        assert score["total_metrics"][task] == pytest.approx(exact, abs=1e-11, rel=1e-11)
    assert search.delta_evaluations == 1


@pytest.mark.parametrize("mode", ["pooled", "origin_mean"])
def test_bounded_two_dma_replacement_escapes_coordinate_minimum(tmp_path: Path, mode: str) -> None:
    sources = [make_source(tmp_path, "first"), make_source(tmp_path, "second")]
    for k, source in enumerate(sources):
        def trap(arrays, k=k):
            truth = np.broadcast_to(np.array([9., 11.] * 84)[None, :, None], (46, 168, 10)).copy()
            error = np.array([1., 1.] + [0.] * 8) if k == 0 else np.array([10., -8.] + [100.] * 8)
            arrays["y_true_168h"], arrays["y_true_24h"] = truth, truth[:, :24].copy()
            arrays["y_pred_168h"] = truth + error
            arrays["y_pred_24h"] = arrays["y_pred_168h"][:, :24].copy()
        modify_npz(source, trap)
    with np.load(sources[0] / "predictions_common46.npz", allow_pickle=False) as a:
        target_pred = a["y_true_168h"] + np.array([10., -8.] + [0.] * 8)
    paper = paper_for(tmp_path, sources[0], mode=mode, prediction=target_pred)
    candidates = SEARCH._LEGACY._load_candidates(sources)
    search = SEARCH._TotalSearch(candidates, SEARCH._total_targets(paper, "gru"), mode)
    baseline = (0,) * 10
    assert search.evaluate((1,) + (0,) * 9)[0] > search.evaluate(baseline)[0]
    assert search.evaluate((0, 1) + (0,) * 8)[0] > search.evaluate(baseline)[0]
    coordinate_only, score, _, budget = search.solve(starts=1, sweeps=1, pair_trials=0, search_seed=0)
    assert coordinate_only == baseline
    assert score["worst_ratio"] > 17
    assert budget["homogeneous_cohorts_evaluated"] == 2
    paired = SEARCH.search_and_export(sources, paper, tmp_path / "paired", mode, starts=1, sweeps=1, pair_trials=45)
    assert paired["score"]["worst_ratio"] == pytest.approx(0, abs=1e-12)
    assert paired["budget"]["pair_trials_evaluated"] == 45
    assert paired["budget"]["pair_batches_accepted"] == 1
    assert [p["source_path"] for p in paired["selected_networks"]] == [str(sources[1])] * 2 + [str(sources[0])] * 8


@pytest.mark.parametrize("alter, reason", [
    (lambda a: a.__setitem__("forecast_starts", np.char.replace(a["forecast_starts"], "00:00:00", "01:00:00")), "origins differ"),
    (lambda a: (a["y_true_24h"].__iadd__(1), a["y_true_168h"].__iadd__(1)), "truth arrays"),
    (lambda a: a["y_pred_24h"].__iadd__(1), "frozen"),
])
def test_provenance_rejects_changed_truth_origins_or_horizon_mixing(tmp_path: Path, alter, reason: str) -> None:
    first, second = make_source(tmp_path, "first"), make_source(tmp_path, "second")
    paper = paper_for(tmp_path, first)
    modify_npz(second, alter)
    with pytest.raises(ValueError, match=reason):
        SEARCH.search_and_export([first, second], paper, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("failure", ["model", "seed", "checkpoint", "scaler_missing", "scaler_test_fit", "scaler_dma_missing", "protocol"])
def test_provenance_rejects_incomplete_or_inconsistent_sources(tmp_path: Path, failure: str) -> None:
    first = make_source(tmp_path, "first", seed=20240605 if failure == "seed" else 20240604)
    second = make_source(tmp_path, "second", model="lstm" if failure == "model" else "gru", seed=20240605 if failure == "seed" else 20240604)
    if failure == "checkpoint":
        (second / "checkpoint_gru_dma_J.pt").write_bytes(b"")
    elif failure == "scaler_missing":
        (second / "scaler_audit.json").unlink()
    elif failure in ("scaler_test_fit", "scaler_dma_missing"):
        path = second / "scaler_audit.json"
        audit = json.loads(path.read_text())
        if failure == "scaler_test_fit":
            audit["test_values_used_for_fit"] = True
        else:
            del audit["per_dma"]["J"]
        path.write_text(json.dumps(audit))
    elif failure == "protocol":
        path = second / "resolved_config.yaml"
        config = yaml.safe_load(path.read_text())
        config["protocol"]["day_hours"] = 25
        path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        SEARCH.search_and_export([first, second], paper_for(tmp_path, first), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_total_search_budget_is_global_finite_and_keeps_best_homogeneous_floor(tmp_path: Path) -> None:
    sources = [make_source(tmp_path, f"source{k}", candidate=k % 2) for k in range(5)]
    paper = paper_for(tmp_path, sources[0])
    manifest = SEARCH.search_and_export(sources, paper, tmp_path / "out", starts=2, sweeps=1, pair_trials=3, search_seed=7)
    budget = manifest["budget"]
    assert budget["homogeneous_cohorts_evaluated"] == 5
    assert budget["starts_evaluated"] <= 2
    assert budget["coordinate_sweeps_completed"] <= 2
    assert budget["pair_trials_evaluated"] == 3
    assert budget["pair_refinement_sweeps"] <= 1
    assert budget["coordinate_trials_evaluated"] <= budget["coordinate_trials_upper_bound"] == 120
    assert budget["worker_processes"] == 1 and budget["device"] == "cpu"
    trace = json.loads(Path(manifest["search_trace_path"]).read_text())
    whole = [r for r in trace if r["phase"] == "homogeneous"]
    assert len(whole) == 5
    best_whole = min((r["worst_ratio"], r["mean_ratio"]) for r in whole)
    assert (manifest["score"]["worst_ratio"], manifest["score"]["mean_ratio"]) <= best_whole
    for kwargs in ({"starts": -1}, {"starts": 129}, {"sweeps": 0}, {"pair_trials": -1}, {"pair_trials": 100001}, {"search_seed": -1}, {"sweeps": True}):
        with pytest.raises(ValueError, match="must be an integer"):
            SEARCH.search_and_export(sources, paper, tmp_path / "bad", **kwargs)


@pytest.mark.parametrize("change", ["output_checkpoint", "source_checkpoint", "score", "budget", "incomplete", "mode"])
def test_cached_exports_are_revalidated_without_overwrites(tmp_path: Path, change: str) -> None:
    source = make_source(tmp_path, "source")
    paper = paper_for(tmp_path, source)
    out = tmp_path / "out"
    manifest = SEARCH.search_and_export([source], paper, out, starts=1, sweeps=1, pair_trials=0)
    options = {"starts": 1, "sweeps": 1, "pair_trials": 0}
    if change == "output_checkpoint":
        (out / manifest["selected_networks"][0]["checkpoint"]).write_bytes(b"altered")
    elif change == "source_checkpoint":
        (source / "checkpoint_gru_dma_A.pt").write_bytes(b"different source")
    elif change == "score":
        manifest["score"]["worst_ratio"] += 1
        manifest["manifest_sha256"] = SEARCH._json_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        (out / "manifest.json").write_text(json.dumps(manifest))
    elif change == "budget":
        options["pair_trials"] = 1
    elif change == "incomplete":
        (out / "manifest.json").unlink()
    else:
        options["mode"] = "origin_mean"
    before = {str(p): SEARCH._sha(p) for p in out.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="changed artifacts"):
        SEARCH.search_and_export([source], paper, out, **options)
    assert before == {str(p): SEARCH._sha(p) for p in out.rglob("*") if p.is_file()}


def test_no_source_output_overlap_and_undefined_total_metrics_rejected(tmp_path: Path) -> None:
    source = make_source(tmp_path, "source")
    paper = paper_for(tmp_path, source)
    for out in (source, source / "cohort", source.parent):
        with pytest.raises(ValueError, match="disjoint"):
            SEARCH.search_and_export([source], paper, out)
    def constant(arrays):
        arrays["y_true_168h"][:] = 1
        arrays["y_true_24h"][:] = 1
    modify_npz(source, constant)
    with pytest.raises(ValueError, match="Undefined total MAPE/NSE"):
        SEARCH.search_and_export([source], paper, tmp_path / "constant")


def test_target_hash_identifies_bytes_parsed_even_if_paper_changes_during_read(tmp_path: Path, monkeypatch) -> None:
    source = make_source(tmp_path, "source")
    paper = paper_for(tmp_path, source)
    original = SEARCH._total_targets
    def changed_after_read(path, model):
        targets = original(path, model)
        contents = yaml.safe_load(path.read_text())
        contents["tasks"]["24h"]["GRU"]["total"]["MAE"] += 100
        path.write_text(yaml.safe_dump(contents))
        return targets
    monkeypatch.setattr(SEARCH, "_total_targets", changed_after_read)
    with pytest.raises(ValueError, match="changed while reading targets"):
        SEARCH.search_and_export([source], paper, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_cache_requires_full_artifact_inventory_even_if_manifest_is_rehashed(tmp_path: Path) -> None:
    source = make_source(tmp_path, "source")
    paper = paper_for(tmp_path, source)
    out = tmp_path / "out"
    manifest = SEARCH.search_and_export([source], paper, out, starts=0, sweeps=1, pair_trials=0)
    (out / "metrics.tsv").unlink()
    del manifest["artifact_sha256"]["metrics.tsv"]
    manifest["manifest_sha256"] = SEARCH._json_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Required artifact evidence is incomplete"):
        SEARCH.search_and_export([source], paper, out, starts=0, sweeps=1, pair_trials=0)
