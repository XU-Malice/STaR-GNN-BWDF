from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dma_wdf.data.reproduction_metrics import (
    METRIC_MODES,
    array_sha256,
    canonical_forecast_origins,
    compute_reproduction_metrics,
    rmse_nse_feasibility,
    validate_prediction_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("saved_prediction_audit", ROOT/"scripts/reproduce/audit_que_saved_predictions.py")
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def bundle(n: int = 46) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(21)
    truth = rng.normal(20, 2, (n, 168, 10)).astype(np.float32)
    prediction = truth+rng.normal(0, 1, truth.shape).astype(np.float32)
    starts = np.char.add((np.datetime64("2023-01-13")+np.arange(n)).astype(str), "T00:00:00+01:00")
    return {"y_true_24h": truth[:, :24].copy(), "y_pred_24h": prediction[:, :24].copy(),
            "y_true_168h": truth, "y_pred_168h": prediction, "forecast_starts": starts,
            "dma_letters": np.array(list("ABCDEFGHIJ"))}


def values(rows: list[dict], series: str = "total") -> dict:
    return {row["metric"]: row["value"] for row in rows if row["series"] == series}


def test_bundle_checks_finite_shapes_origins_and_first_day() -> None:
    arrays = bundle()
    result = validate_prediction_bundle(arrays)
    assert result["first_day_consistent"] and result["test_sequences"] == 46
    assert len(result["array_hashes"]) == 6
    bad = {**arrays, "y_pred_24h": arrays["y_pred_24h"].copy()}
    bad["y_pred_24h"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="frozen"):
        validate_prediction_bundle(bad)
    assert validate_prediction_bundle(bad, require_first_day_consistency=False)["first_day_consistent"] is False
    bad["y_pred_24h"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_prediction_bundle(bad)
    with pytest.raises(ValueError, match="DMA order"):
        validate_prediction_bundle({**arrays, "dma_letters": arrays["dma_letters"][::-1]})
    with pytest.raises(ValueError, match="increasing"):
        validate_prediction_bundle({**arrays, "forecast_starts": arrays["forecast_starts"][::-1]})
    with pytest.raises(ValueError, match="shape"):
        validate_prediction_bundle({**arrays, "y_true_24h": arrays["y_true_24h"][0]})


def test_origin_comparison_uses_instants_and_preserves_raw_provenance() -> None:
    arrays = bundle(2)
    iso = arrays["forecast_starts"]
    spaces = np.char.replace(iso, "T", " ")
    utc = np.asarray(["2023-01-12T23:00:00Z", "2023-01-13T23:00:00+00:00"])
    results = [validate_prediction_bundle({**arrays, "forecast_starts": starts}, expected_sequences=2)
               for starts in (iso, spaces, utc)]
    assert len({result["origins_sha256"] for result in results}) == 1
    assert len({result["array_hashes"]["forecast_starts"] for result in results}) == 3
    for result, starts in zip(results, (iso, spaces, utc)):
        assert result["forecast_starts"] == starts.tolist()
        assert result["array_hashes"]["forecast_starts"] == array_sha256(starts)
    assert np.array_equal(canonical_forecast_origins(iso), canonical_forecast_origins(utc))


@pytest.mark.parametrize("starts,reason", [
    (["2023-01-13"], "explicit timezone"),
    (["2023-01-13T00:00:00"], "explicit timezone"),
    (["NaT"], "Invalid"),
    (["2023-01-13T00:00:00+01:00", "2023-01-12T23:00:00Z"], "unique instants"),
    (["2023-01-13T00:00:00Z", "2023-01-13T00:30:00+01:00"], "increasing"),
    (["2023-01-13T00:00:00.0000001Z"], "Sub-microsecond"),
])
def test_origin_comparison_rejects_invalid_or_ambiguous_instants(starts, reason) -> None:
    with pytest.raises(ValueError, match=reason):
        canonical_forecast_origins(starts)


def test_saved_prediction_audit_groups_equivalent_recurrent_and_joint_origins(tmp_path: Path) -> None:
    first = make_source(tmp_path, "recurrent_style")
    second = make_source(tmp_path, "joint_style")
    with np.load(second, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["forecast_starts"] = np.char.replace(arrays["forecast_starts"], "T", " ")
    np.savez_compressed(second, **arrays)
    assert AUDIT.file_sha256(first) != AUDIT.file_sha256(second)
    summary = AUDIT.audit_saved_predictions(results_roots=[tmp_path],
        paper_config=ROOT/"configs/evaluation/mscmnet_paper_metrics.yaml", output_root=tmp_path/"audit")
    assert summary["valid_sources"] == 2 and summary["invalid_sources"] == 0
    assert summary["truth_groups_per_task"] == {"24h": 1, "168h": 1}
    assert summary["truth_origin_group_mismatch"] is False


def test_pooled_identity_and_separate_total_mae() -> None:
    true = np.array([[[10., 20.], [12., 22.]], [[14., 24.], [16., 26.]]])
    pred = true+np.array([1., -1.])
    rows = compute_reproduction_metrics(true, pred, ["A", "B"])
    total = values(rows)
    assert total["MAE"] == 2
    assert values(rows, "physical_total")["MAE"] == 0
    assert total["RMSE"] == 0 and total["NSE"] == 1
    pred = true+1
    total = values(compute_reproduction_metrics(true, pred, ["A", "B"]))
    assert total["RMSE"]**2/(1-total["NSE"]) == pytest.approx(np.var(true.sum(2)))


def test_origin_mean_is_complete_nonlinear_alternative_not_pooled() -> None:
    truth = np.array([[[1.], [3.]], [[11.], [13.]]])
    prediction = truth+np.array([[[1.]], [[3.]]])
    pooled = values(compute_reproduction_metrics(truth, prediction, ["A"], mode="pooled"))
    macro = values(compute_reproduction_metrics(truth, prediction, ["A"], mode="origin_mean"))
    assert pooled["RMSE"] == pytest.approx(np.sqrt(5))
    assert pooled["NSE"] == pytest.approx(1-5/26)
    assert macro["RMSE"] == 2 and macro["NSE"] == -4
    assert pooled["MAE"] == macro["MAE"] and pooled["MAPE"] == macro["MAPE"]
    assert len(compute_reproduction_metrics(bundle()["y_true_24h"], bundle()["y_pred_24h"])) == 45


def test_no_silent_masking_or_undefined_origin_dropping() -> None:
    truth = np.array([[[2.], [2.]], [[1.], [3.]]])
    rows = compute_reproduction_metrics(truth, truth+1, ["A"], mode="origin_mean")
    assert np.isnan(values(rows)["NSE"])
    truth[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="cannot be masked"):
        compute_reproduction_metrics(truth, truth, ["A"])
    with pytest.raises(ValueError, match="matching"):
        compute_reproduction_metrics(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="mode"):
        compute_reproduction_metrics(bundle()["y_true_24h"], bundle()["y_pred_24h"], mode="best")


@pytest.mark.parametrize("target_rmse,target_nse,minimum_increase", [(10.194, .916, .062543), (9.711, .920, .084826)])
def test_gru_lstm_24h_tolerance_is_mathematically_infeasible(target_rmse, target_nse, minimum_increase) -> None:
    truth = np.array([-1., 1.])*np.sqrt(1585.4421432386016)
    exact = rmse_nse_feasibility(truth, target_rmse, target_nse)
    rounded = rmse_nse_feasibility(truth, target_rmse, target_nse, rounding_half_unit=.0005)
    assert exact["pair_feasible"] is False and rounded["pair_feasible"] is False
    assert exact["minimum_relative_rmse_increase_for_nse"] == pytest.approx(minimum_increase, abs=1e-6)
    assert exact["implied_nse_low"] > exact["allowed_nse_high"]


def test_feasibility_positive_is_necessary_only_and_constant_truth_is_undefined() -> None:
    truth = np.array([-1., 1.])*np.sqrt(1585.4421432386016)
    assert rmse_nse_feasibility(truth, 7.924, .957)["pair_feasible"] is True
    assert rmse_nse_feasibility(np.ones(4), 1., .9)["reason"] == "NSE_UNDEFINED_CONSTANT_TRUTH"
    with pytest.raises(ValueError):
        rmse_nse_feasibility(truth, 1., .9, error_relative_tolerance=-1)
    assert array_sha256(truth) != array_sha256(truth.astype(np.float32))


def make_source(tmp_path: Path, name: str = "case") -> Path:
    run = tmp_path/name/"msnet"/"seed_20240604"
    run.mkdir(parents=True)
    np.savez_compressed(run/"predictions_common46.npz", **bundle())
    (run/"status.json").write_text(json.dumps({"status": "completed", "model": "msnet", "seed": 20240604,
                                             "git_commit": "fixture", "checkpoint_files": ["checkpoint_msnet.pt"]}))
    (run/"resolved_config.yaml").write_text(yaml.safe_dump({"model": {"display_name": "MSNet"}}))
    return run/"predictions_common46.npz"


def test_saved_prediction_audit_complete_both_views_and_explicit_selection(tmp_path: Path) -> None:
    first = make_source(tmp_path)
    second = make_source(tmp_path, "case2")
    checksum = AUDIT.file_sha256(first)
    kwargs = {"results_roots": [tmp_path], "paper_config": ROOT/"configs/evaluation/mscmnet_paper_metrics.yaml",
              "output_root": tmp_path/"audit"}
    report = AUDIT.audit_saved_predictions(**kwargs)
    assert report["valid_sources"] == 2 and report["invalid_sources"] == 0
    assert report["source_selection"]["msnet"]["status"] == "UNSELECTED_MULTIPLE_SOURCES"
    assert report["metric_modes"] == list(METRIC_MODES)
    assert report["truth_groups_per_task"] == {"24h": 1, "168h": 1}
    assert AUDIT.file_sha256(first) == checksum
    report = AUDIT.audit_saved_predictions(**kwargs, source_selection={"msnet": str(second)})
    assert report["source_selection"]["msnet"]["source_path"] == str(second)
    assert report["numeric_counts"]["pooled"]["msnet"]["complete_sources"] == 2
    assert report["numeric_counts"]["origin_mean"]["msnet"]["complete_sources"] == 2
    for name in ("feasibility.tsv", "paper_gaps.tsv", "metrics_recomputed.tsv", "provenance.json", "closeness_by_source.tsv"):
        assert (tmp_path/"audit"/name).is_file()
    with pytest.raises(ValueError, match="not a valid"):
        AUDIT.audit_saved_predictions(**kwargs, source_selection={"gru": str(second)})


def test_saved_prediction_audit_reports_absence_and_corruption(tmp_path: Path) -> None:
    paper = ROOT/"configs/evaluation/mscmnet_paper_metrics.yaml"
    report = AUDIT.audit_saved_predictions(results_roots=[tmp_path/"missing"], paper_config=paper, output_root=tmp_path/"audit")
    assert report["status"] == "no_predictions" and report["invalid_sources"] == 0
    source = make_source(tmp_path)
    source.write_bytes(b"not-an-npz")
    report = AUDIT.audit_saved_predictions(results_roots=[tmp_path], paper_config=paper, output_root=tmp_path/"audit")
    assert report["status"] == "invalid_evidence" and report["invalid_sources"] == 1
    assert report["valid_sources"] == 0
    with pytest.raises(ValueError, match="separate"):
        AUDIT.audit_saved_predictions(results_roots=[tmp_path], paper_config=paper, output_root=source.parent)


def test_archived_sources_are_excluded_and_legacy_first_day_warnings_preserved(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    make_source(tmp_path, "old.backup-20260905")
    arrays = bundle()
    arrays["y_pred_24h"][0, 0, 0] += .01
    np.savez_compressed(source, **arrays)
    kwargs = {"results_roots": [tmp_path], "paper_config": ROOT/"configs/evaluation/mscmnet_paper_metrics.yaml",
              "output_root": tmp_path/"audit"}
    summary = AUDIT.audit_saved_predictions(**kwargs)
    assert summary["valid_sources"] == 1 and summary["first_day_discrepancy_sources"] == 1
    assert summary["invalid_sources"] == 0
    strict = AUDIT.audit_saved_predictions(**kwargs, strict_first_day=True)
    assert strict["valid_sources"] == 0 and strict["invalid_sources"] == 1


def test_closeness_cannot_combine_complementary_sources_or_modes() -> None:
    rows = []
    for source in ("first", "second"):
        for mode in METRIC_MODES:
            # Each source/mode gets a different failing total cell. Their union
            # contains successes for all cells but no actual complete run passes.
            failing = ("24h" if source == "first" else "168h",
                       "MAE" if mode == "pooled" else "RMSE")
            for task in ("24h", "168h"):
                for series in list("ABCDEFGHIJ")+["total"]:
                    for metric in ("MAE", "MAPE", "RMSE", "NSE"):
                        value = 2.0 if series == "total" and (task, metric) == failing else 1.0
                        row = {"source_id": source, "source_path": f"/{source}/predictions_common46.npz",
                               "model": "msnet", "seed": 20240604, "mode": mode,
                               "task": task, "series": series, "metric": metric, "value": value}
                        rows.append(AUDIT._gap(row, 1.0, .05, .01))
    compact = AUDIT.closeness_by_source(rows)
    assert len(compact) == 4
    assert all(row["complete88_cells"] for row in compact)
    assert all(row["total8_passed"] == 7 and row["dma80_passed"] == 80 for row in compact)
    assert all(row["metrics88_passed"] == 87 and not row["all88_passed"] for row in compact)
    assert all(not row["total8_all_within_tolerance"] for row in compact)
    assert all(row["max_total6_error_relative_gap"] == 1 for row in compact)
    assert all(row["total_worst_tolerance_ratio"] == 20 for row in compact)


def test_closeness_rejects_duplicate_cells_and_requires_complete_grid() -> None:
    row = {"source_id": "one", "source_path": "/one/predictions_common46.npz", "model": "msnet",
           "seed": 20240604, "mode": "pooled", "task": "24h", "series": "total",
           "metric": "MAE", "value": 1.0}
    gap = AUDIT._gap(row, 1.0, .05, .01)
    result = AUDIT.closeness_by_source([gap])[0]
    assert result["total8_passed"] == 1 and not result["complete88_cells"] and not result["all88_passed"]
    with pytest.raises(ValueError, match="Duplicate"):
        AUDIT.closeness_by_source([gap, gap])
