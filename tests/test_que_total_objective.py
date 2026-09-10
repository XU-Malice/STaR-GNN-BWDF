"""Fixed-truth, whole-candidate TOTAL reporting from real small NumPy bundles."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("que_total_objective_under_test", ROOT / "scripts/reproduce/audit_que_total_objective.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def bundle(error_scale: float = 1.) -> dict[str, np.ndarray]:
    origin = np.arange(46)[:, None, None]
    hour = np.arange(168)[None, :, None]
    dma = np.arange(10)[None, None, :]
    true = 50. + origin * .3 + np.sin(hour / 3.) * (2 + dma * .2) + dma
    # Unequal origin errors distinguish pooled RMSE/NSE from origin means.
    pred = true + error_scale * (.05 + origin * .02) * (1 + dma * .1) + hour * 0.
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {"y_true_24h": true[:, :24].copy(), "y_pred_24h": pred[:, :24].copy(),
            "y_true_168h": true, "y_pred_168h": pred,
            "forecast_starts": np.array([(start + timedelta(days=index)).isoformat() for index in range(46)]),
            "dma_letters": np.array(list("ABCDEFGHIJ"))}


def write_candidate(tmp_path: Path, model: str, case: str, arrays: dict[str, np.ndarray]) -> dict:
    run = tmp_path / "sources" / model / case
    run.mkdir(parents=True)
    np.savez(run / "predictions_common46.npz", **arrays)
    return {"model": model, "case": case, "run": run, "settings": {"fixture": case}}


def write_paper(tmp_path: Path, arrays: dict[str, np.ndarray], mode: str = "pooled", *, second_horizon=None) -> Path:
    tasks = {}
    for task in ("24h", "168h"):
        source = second_horizon if task == "168h" and second_horizon is not None else arrays
        table = {}
        for row in AUDIT.compute_reproduction_metrics(source[f"y_true_{task}"], source[f"y_pred_{task}"], mode=mode):
            if row["series"] in AUDIT.SERIES:
                table.setdefault(row["series"], {})[row["metric"]] = row["value"]
        tasks[task] = {display: table for display in AUDIT.MODEL_DISPLAY.values()}
    path = tmp_path / "paper.yaml"
    path.write_text(yaml.safe_dump({"tasks": tasks}))
    return path


def tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_recomputes_all88_cells_selects_one_complete_run_and_preserves_sources(tmp_path):
    arrays = bundle()
    candidates = [write_candidate(tmp_path, model, "reference", arrays) for model in AUDIT.MODEL_DISPLAY]
    candidates.append(write_candidate(tmp_path, "gru", "worse", bundle(3.)))
    # Stored metrics are deliberately wrong: the helper must use raw arrays.
    (candidates[0]["run"] / "metrics.csv").write_text("task,metric,value\n24h,RMSE,999\n")
    source_hashes = {str(path): AUDIT._file_sha256(path) for path in (tmp_path / "sources").rglob("*") if path.is_file()}
    out = tmp_path / "audit"
    summary = AUDIT.audit_candidates(candidates, write_paper(tmp_path, arrays), out)
    assert summary["modes"]["pooled"]["total48_all_within_tolerance"]
    assert summary["modes"]["pooled"]["metrics528_passed"] == 528
    assert summary["selected_per_mode"]["pooled"]["gru"]["case"] == "reference"
    assert summary["fixed_common_truth_origins_validated"]
    assert summary["reference"]["test_sequences"] == 46
    assert len(tsv(out / "metrics_all_candidates.tsv")) == len(candidates) * 88 * 2
    assert len(tsv(out / "feasible_bounds.tsv")) == 6 * 2 * 2
    for mode in AUDIT.METRIC_MODES:
        table = tsv(out / f"total_comparison_{mode}.tsv")
        assert len(table) == 48 and {row["mode"] for row in table} == {mode}
        for model in AUDIT.MODEL_DISPLAY:
            rows = [row for row in table if row["model"] == model]
            assert len(rows) == 8 and len({row["run"] for row in rows}) == 1
    assert json.loads((out / "protocol_summary.json").read_text()) == summary
    assert json.loads((out / "candidates_summary.json").read_text()) == summary["candidates_summary"]
    for name, expected_hash in summary["report_files_sha256"].items():
        assert AUDIT._file_sha256(out / name) == expected_hash
    assert source_hashes == {str(path): AUDIT._file_sha256(path) for path in (tmp_path / "sources").rglob("*") if path.is_file()}
    assert not list(out.glob(".audit-total-*"))


def test_closer_origin_mean_table_is_never_promoted_to_primary(tmp_path):
    arrays = bundle()
    candidates = [write_candidate(tmp_path, model, "same", arrays) for model in AUDIT.MODEL_DISPLAY]
    summary = AUDIT.audit_candidates(candidates, write_paper(tmp_path, arrays, mode="origin_mean"), tmp_path / "audit")
    assert summary["modes"]["origin_mean"]["total48_all_within_tolerance"]
    assert not summary["modes"]["pooled"]["total48_all_within_tolerance"]
    assert summary["primary_metric_mode"] == "pooled"
    assert summary["primary_metric_changed"] is False
    assert summary["origin_mean_automatically_promoted"] is False
    assert summary["origin_mean_is_confirmed_publisher_convention"] is False
    assert summary["original_paper_recovery_claim"] is False
    assert summary["modes"]["origin_mean"]["role"] == "diagnostic_unconfirmed_hypothesis"


def test_complementary_horizons_cannot_form_a_selected_candidate(tmp_path):
    first, second = bundle(.2), bundle(2.)
    # First candidate wins 24h; second wins 168h. Their complete bundles each
    # maintain their own frozen first day, while later days have opposite errors.
    first["y_pred_168h"][:, 24:] = first["y_true_168h"][:, 24:] + 5.
    second["y_pred_168h"][:, 24:] = second["y_true_168h"][:, 24:] + .05
    candidates = [write_candidate(tmp_path, "gru", "first", first), write_candidate(tmp_path, "gru", "second", second)]
    summary = AUDIT.audit_candidates(candidates, write_paper(tmp_path, first, second_horizon=second), tmp_path / "audit")
    scored = [item for item in summary["candidates_summary"] if item["mode"] == "pooled"]
    assert all(item["total8_passed"] < 8 for item in scored)
    assert any(item["metrics"][:4] == summary["selected_per_mode"]["pooled"]["gru"]["metrics"][:4] for item in scored)
    expected = min(scored, key=lambda item: (item["worst_ratio"], item["mean_ratio"], item["case"], item["run"]))
    selected = summary["selected_per_mode"]["pooled"]["gru"]
    assert selected["case"] == expected["case"]
    rows = [row for row in tsv(tmp_path / "audit/total_comparison_pooled.tsv") if row["model"] == "gru"]
    assert {row["run"] for row in rows} == {selected["run"]}
    assert not summary["modes"]["pooled"]["complete_six_model_table"]
    assert len(tsv(tmp_path / "audit/total_comparison_pooled.tsv")) == 48


def test_minimax_selects_all_eight_match_when_lower_mean_candidate_fails(tmp_path):
    first, second = bundle(), bundle(1.02)
    candidates = [write_candidate(tmp_path, "gru", "lower_mean", first),
                  write_candidate(tmp_path, "gru", "all_eight", second)]
    paper_path = write_paper(tmp_path, first)
    paper = yaml.safe_load(paper_path.read_text())
    # Only one paper cell moves: the first candidate still matches seven
    # cells exactly but misses this cell by >5%; second fits all eight.
    paper["tasks"]["24h"]["GRU"]["total"]["MAE"] *= 1.06
    paper_path.write_text(yaml.safe_dump(paper))
    summary = AUDIT.audit_candidates(candidates, paper_path, tmp_path / "audit")
    scored = {item["case"]: item for item in summary["candidates_summary"] if item["mode"] == "pooled"}
    assert scored["lower_mean"]["mean_ratio"] < scored["all_eight"]["mean_ratio"]
    assert scored["lower_mean"]["worst_ratio"] > 1.
    assert scored["all_eight"]["worst_ratio"] < 1.
    assert summary["selected_per_mode"]["pooled"]["gru"]["case"] == "all_eight"


@pytest.mark.parametrize("mismatch", ["truth", "origins", "first_day", "missing_field", "nonfinite", "dma_order"])
def test_invalid_common_evidence_cannot_overwrite_existing_reports(tmp_path, mismatch):
    arrays, other = bundle(), bundle()
    if mismatch == "truth":
        other["y_true_168h"][0, 100, 0] += 1.
    elif mismatch == "origins":
        other["forecast_starts"] = np.array([(datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat() for index in range(46)])
    elif mismatch == "first_day":
        other["y_pred_24h"][0, 0, 0] += 1.
    elif mismatch == "missing_field":
        del other["y_pred_168h"]
    elif mismatch == "nonfinite":
        other["y_pred_168h"][0, 100, 0] = np.nan
    elif mismatch == "dma_order":
        other["dma_letters"] = other["dma_letters"][::-1]
    candidates = [write_candidate(tmp_path, "gru", "a", arrays), write_candidate(tmp_path, "lstm", "b", other)]
    out = tmp_path / "audit"
    out.mkdir()
    (out / "protocol_summary.json").write_text('{"previous": true}\n')
    with pytest.raises(ValueError):
        AUDIT.audit_candidates(candidates, write_paper(tmp_path, arrays), out)
    assert (out / "protocol_summary.json").read_text() == '{"previous": true}\n'
    assert list(out.iterdir()) == [out / "protocol_summary.json"]


def test_equivalent_timezone_origin_strings_match_and_keep_raw_hashes(tmp_path):
    arrays, other = bundle(), bundle()
    offset = timezone(timedelta(hours=2))
    other["forecast_starts"] = np.array([datetime.fromisoformat(value).astimezone(offset).isoformat(sep=" ") for value in arrays["forecast_starts"]])
    candidates = [write_candidate(tmp_path, "gru", "a", arrays), write_candidate(tmp_path, "lstm", "b", other)]
    summary = AUDIT.audit_candidates(candidates, write_paper(tmp_path, arrays), tmp_path / "audit")
    provenance = json.loads((tmp_path / "audit/candidate_provenance.json").read_text())
    assert provenance[0]["origins_sha256"] == provenance[1]["origins_sha256"] == summary["reference"]["origins_sha256"]
    assert provenance[0]["array_hashes"]["forecast_starts"] != provenance[1]["array_hashes"]["forecast_starts"]


def test_current_variance_proves_gru_lstm_24h_pooled_total48_impossible(tmp_path):
    arrays = bundle()
    value = arrays["y_true_24h"].sum(axis=2)
    fixed = 500. + (value - value.mean()) * np.sqrt(1585.4421529337 / value.var())
    arrays["y_true_168h"][:, :24] = fixed[:, :, None] / 10.
    arrays["y_true_24h"] = arrays["y_true_168h"][:, :24].copy()
    arrays["y_pred_168h"] = arrays["y_true_168h"] + .5
    arrays["y_pred_24h"] = arrays["y_pred_168h"][:, :24].copy()
    summary = AUDIT.audit_candidates([write_candidate(tmp_path, "gru", "fixed", arrays)],
                                     ROOT / "configs/evaluation/mscmnet_paper_metrics.yaml", tmp_path / "audit")
    assert summary["reference"]["truth_population_variance_total"]["24h"] == pytest.approx(1585.4421529337)
    assert summary["pooled_total48_impossible_exact_paper_values"]
    assert summary["pooled_total48_impossible_even_with_rounding"]
    for rounding in ("0.0", "0.0005"):
        blocked = {(row["model"], row["task"]): row for row in summary["pooled_infeasible_pairs_by_rounding"][rounding]}
        for model in ("gru", "lstm"):
            assert blocked[model, "24h"]["minimax_pair_lower_bound"] > 1.
    assert any("All 48 TOTAL cells are mathematically impossible" in sentence for sentence in summary["diagnosis"])


def test_output_cannot_be_inside_a_prediction_source(tmp_path):
    arrays = bundle()
    candidate = write_candidate(tmp_path, "gru", "source", arrays)
    with pytest.raises(ValueError, match="separate"):
        AUDIT.audit_candidates([candidate], write_paper(tmp_path, arrays), candidate["run"] / "audit")
