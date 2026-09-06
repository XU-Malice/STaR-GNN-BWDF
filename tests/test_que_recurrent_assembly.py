from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dma_wdf.data.reproduction_metrics import compute_reproduction_metrics

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("que_recurrent_assembly", ROOT / "scripts/reproduce/assemble_que_recurrent_candidates.py")
assert SPEC and SPEC.loader
ASSEMBLY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSEMBLY)


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
        (run / checkpoint).write_bytes(f"fixture-not-deserialized:{name}:{checkpoint}".encode())
    status = {"status": "completed", "model": model, "seed": seed, "checkpoint_files": checkpoints,
              "single_frozen_checkpoint_for_24h_and_168h": True}
    config = {"seed": seed, "protocol": {"dma_letters": list("ABCDEFGHIJ"), "day_hours": 24},
              "model": {"family": "independent_recurrent", "cell_type": model.upper(),
                        "best_epochs": list(range(10, 20)), "hidden_sizes": [[32]] * 10},
              "training": {"normalization": "zscore", "batch_size": 8 + candidate}}
    (run / "status.json").write_text(json.dumps(status))
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    return run


def paper_for_mixture(tmp_path: Path, source: Path, mode: str = "pooled") -> Path:
    paper: dict = {"tasks": {}}
    with np.load(source / "predictions_common46.npz", allow_pickle=False) as archive:
        for task in ("24h", "168h"):
            truth = archive[f"y_true_{task}"]
            rows = compute_reproduction_metrics(truth, truth + np.float32(1), mode=mode)
            metrics: dict = {}
            for row in rows:
                if row["series"] != "physical_total":
                    metrics.setdefault(row["series"], {})[row["metric"]] = row["value"]
            paper["tasks"][task] = {"GRU": metrics, "LSTM": metrics}
    path = tmp_path / f"paper_{mode}.yaml"
    path.write_text(yaml.safe_dump(paper))
    return path


def modify_npz(run: Path, edit) -> None:
    path = run / "predictions_common46.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    edit(arrays)
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize("mode", ["pooled", "origin_mean"])
def test_selects_intact_dma_networks_and_both_horizons_together(tmp_path: Path, mode: str) -> None:
    first = make_source(tmp_path, "first", candidate=0)
    second = make_source(tmp_path, "second", candidate=1)
    sources = (first, second)
    before = {str(p): ASSEMBLY._sha(p) for source in sources for p in source.iterdir()}
    paper = paper_for_mixture(tmp_path, first, mode)
    output = tmp_path / f"assembled_{mode}"
    manifest = ASSEMBLY.assemble_recurrent_candidates(candidate_dirs=[second, first], paper_config=paper,
                                                     output_root=output, mode=mode)
    assert manifest["score"]["all88_passed"] is True
    assert manifest["score"]["balanced_distance"] == pytest.approx(0, abs=1e-12)
    assert manifest["paper_reproduction_verified"] is False
    assert manifest["global_optimum_proven"] is False
    assert len(manifest["selected_networks"]) == 10
    assert len((output / "paper_gaps.tsv").read_text().splitlines()) == 89
    with np.load(output / "predictions_common46.npz", allow_pickle=False) as assembled:
        for j, selected in enumerate(manifest["selected_networks"]):
            source = sources[j % 2]
            assert selected["source_path"] == str(source)
            assert selected["per_dma_model_config"]["best_epochs"] == j + 10
            checkpoint = output / selected["checkpoint"]
            assert checkpoint.read_bytes() == (source / checkpoint.name).read_bytes()
            with np.load(source / "predictions_common46.npz", allow_pickle=False) as original:
                for task in ("24h", "168h"):
                    assert np.array_equal(assembled[f"y_pred_{task}"][:, :, j], original[f"y_pred_{task}"][:, :, j])
    assert before == {str(p): ASSEMBLY._sha(p) for source in sources for p in source.iterdir()}
    # Exact requests may resume without rewriting any source or cohort artifact.
    cached = ASSEMBLY.assemble_recurrent_candidates(candidate_dirs=[first, second], paper_config=paper,
                                                   output_root=output, mode=mode)
    assert cached == manifest
    checkpoint.write_bytes(b"corrupted output")
    with pytest.raises(ValueError, match="changed artifacts"):
        ASSEMBLY.assemble_recurrent_candidates(candidate_dirs=[first, second], paper_config=paper,
                                              output_root=output, mode=mode)


@pytest.mark.parametrize("alter, reason", [
    (lambda a: a.__setitem__("forecast_starts", np.char.replace(a["forecast_starts"], "00:00:00", "01:00:00")), "origins differ"),
    (lambda a: (a["y_true_24h"].__iadd__(1), a["y_true_168h"].__iadd__(1)), "truth arrays"),
    (lambda a: a["y_pred_24h"].__iadd__(1), "frozen"),
])
def test_refuses_mismatched_origins_truth_or_horizon_sources(tmp_path: Path, alter, reason: str) -> None:
    first, second = make_source(tmp_path, "first"), make_source(tmp_path, "second")
    modify_npz(second, alter)
    with pytest.raises(ValueError, match=reason):
        ASSEMBLY.assemble_recurrent_candidates(candidate_dirs=[first, second],
            paper_config=paper_for_mixture(tmp_path, first), output_root=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_accepts_same_instant_with_different_iso_separator(tmp_path: Path) -> None:
    first, second = make_source(tmp_path, "first"), make_source(tmp_path, "second")
    modify_npz(second, lambda a: a.__setitem__("forecast_starts", np.char.replace(a["forecast_starts"], "T", " ")))
    sources = ASSEMBLY._load_candidates([first, second])
    assert sources[0]["invariants"]["origins_sha256"] == sources[1]["invariants"]["origins_sha256"]


def test_coordinate_search_recomputes_coupled_total_not_only_individual_dma_scores(tmp_path: Path) -> None:
    first, second = make_source(tmp_path, "first"), make_source(tmp_path, "second")
    for run, offset in ((first, 1), (second, -1)):
        def change(arrays, offset=offset):
            for task in ("24h", "168h"):
                arrays[f"y_pred_{task}"] = arrays[f"y_true_{task}"] + np.float32(offset)
        modify_npz(run, change)
    paper = {"tasks": {}}
    with np.load(first / "predictions_common46.npz", allow_pickle=False) as archive:
        for task in ("24h", "168h"):
            truth = archive[f"y_true_{task}"]
            prediction = truth + np.array([1, -1] * 5, dtype=np.float32)
            rows = compute_reproduction_metrics(truth, prediction)
            targets = {}
            for row in rows:
                if row["series"] != "physical_total":
                    targets.setdefault(row["series"], {})[row["metric"]] = row["value"]
            paper["tasks"][task] = {"GRU": targets}
    paper_path = tmp_path / "coupled_targets.yaml"
    paper_path.write_text(yaml.safe_dump(paper))
    result = ASSEMBLY.assemble_recurrent_candidates(candidate_dirs=[first, second], paper_config=paper_path,
                                                   output_root=tmp_path / "cohort")
    # Individual absolute errors tie. Only recomputing the summed-demand metrics
    # can discover that five intact networks from each source match the totals.
    choices = [row["source_path"] for row in result["selected_networks"]]
    assert choices.count(str(first)) == choices.count(str(second)) == 5
    assert result["score"]["all88_passed"] is True
    assert result["score"]["total_metrics"]["24h"]["RMSE"] < 1e-5


@pytest.mark.parametrize("other_model, other_seed", [("lstm", 20240604), ("gru", 20240605)])
def test_refuses_mixed_model_or_seed(tmp_path: Path, other_model: str, other_seed: int) -> None:
    first = make_source(tmp_path, "first")
    second = make_source(tmp_path, "second", model=other_model, seed=other_seed)
    with pytest.raises(ValueError, match="same recurrent model and family seed"):
        ASSEMBLY._load_candidates([first, second])


def test_refuses_joint_model_missing_checkpoint_or_misdeclared_family(tmp_path: Path) -> None:
    joint = make_source(tmp_path, "joint", model="msnet")
    with pytest.raises(ValueError, match="joint columns are forbidden"):
        ASSEMBLY._load_candidates([joint])
    source = make_source(tmp_path, "recurrent")
    checkpoint = source / "checkpoint_gru_dma_A.pt"
    checkpoint.write_bytes(b"")
    with pytest.raises(ValueError, match="empty independent checkpoint"):
        ASSEMBLY._load_candidates([source])
    checkpoint.write_bytes(b"valid fixture")
    config = yaml.safe_load((source / "resolved_config.yaml").read_text())
    config["model"]["family"] = "joint"
    (source / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="independent recurrent family"):
        ASSEMBLY._load_candidates([source])


def test_source_output_collision_and_changed_source_cache_are_refused(tmp_path: Path) -> None:
    source = make_source(tmp_path, "source")
    paper = paper_for_mixture(tmp_path, source)
    for output in (source, source / "assembled", source.parent):
        with pytest.raises(ValueError, match="disjoint"):
            ASSEMBLY.assemble_recurrent_candidates(candidate_dirs=[source], paper_config=paper, output_root=output)
    output = tmp_path / "out"
    ASSEMBLY.assemble_recurrent_candidates(candidate_dirs=[source], paper_config=paper, output_root=output)
    (source / "checkpoint_gru_dma_A.pt").write_bytes(b"new checkpoint")
    with pytest.raises(ValueError, match="differs from this request"):
        ASSEMBLY.assemble_recurrent_candidates(candidate_dirs=[source], paper_config=paper, output_root=output)
