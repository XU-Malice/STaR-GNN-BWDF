#!/usr/bin/env python
"""Read-only, fixed-truth TOTAL objective reports for validated Que candidates.

Training provenance must be validated by the caller. This module reads only
NumPy prediction evidence, recomputes every paper cell, and describes candidate
selection against the published table. Such selection is retrospective test
matching, not independent validation or recovery of the original paper method.
The primary convention remains pooled; origin_mean is a separate hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dma_wdf.data.reproduction_metrics import (  # noqa: E402
    METRIC_MODES,
    METRICS,
    array_sha256,
    compute_reproduction_metrics,
    rmse_nse_feasibility,
    validate_prediction_bundle,
)

MODEL_DISPLAY = {
    "gru": "GRU", "lstm": "LSTM", "msnet": "MSNet",
    "mscmnet_m": "MSCMNet_M", "mscmnet_wm": "MSCMNet_WM", "mscmnet_w": "MSCMNet_W",
}
TOTAL_KEYS = tuple((task, metric) for task in ("24h", "168h") for metric in METRICS)
SERIES = tuple("ABCDEFGHIJ") + ("total",)
ERROR_RELATIVE_TOLERANCE = .05
NSE_ABSOLUTE_TOLERANCE = .01


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"], delimiter="\t")
        writer.writeheader()
        writer.writerows(_json_safe(row) for row in rows)


def rmse_nse_minimax_lower_bound(
    truth_variance: float, target_rmse: float, target_nse: float, *, rounding_half_unit: float = 0.,
) -> float:
    """Necessary lower bound on the worst normalized pooled RMSE/NSE gap.

    This is an analytic interval intersection over a fixed variance, not a
    prediction transformation. A bound <= 1 does not establish a TOTAL match.
    With rounding it is a conservative bound over the rounding intervals.
    """
    if not np.isfinite([truth_variance, target_rmse, target_nse, rounding_half_unit]).all():
        raise ValueError("The minimax bound requires finite inputs.")
    if truth_variance <= 1e-12:
        return float("inf")
    if target_rmse < 0 or target_nse > 1 or rounding_half_unit < 0:
        raise ValueError("Invalid RMSE/NSE bound inputs.")

    def intersects(ratio: float) -> bool:
        r_low = max(0., max(0., target_rmse - rounding_half_unit) * (1 - ERROR_RELATIVE_TOLERANCE * ratio))
        r_high = (target_rmse + rounding_half_unit) * (1 + ERROR_RELATIVE_TOLERANCE * ratio)
        n_low = target_nse - rounding_half_unit - NSE_ABSOLUTE_TOLERANCE * ratio
        n_high = min(1., target_nse + rounding_half_unit + NSE_ABSOLUTE_TOLERANCE * ratio)
        return max(r_low, math.sqrt(max(0., truth_variance * (1 - n_high)))) <= min(
            r_high, math.sqrt(max(0., truth_variance * (1 - n_low)))
        )

    if intersects(0.):
        return 0.
    low, high = 0., 1.
    while not intersects(high):
        high *= 2
        if not math.isfinite(high):
            return float("inf")
    for _ in range(80):
        mid = (low + high) / 2
        if intersects(mid):
            high = mid
        else:
            low = mid
    return high


def _gap(value: float, target: float, metric: str) -> dict[str, Any]:
    difference = abs(value - target)
    tolerance = NSE_ABSOLUTE_TOLERANCE if metric == "NSE" else ERROR_RELATIVE_TOLERANCE * abs(target)
    ratio = difference / tolerance if tolerance else (0. if difference == 0 else float("inf"))
    if not math.isfinite(value):
        ratio = float("inf")
    return {
        "paper_value": target, "absolute_difference": difference,
        "absolute_relative_difference": difference / abs(target) if target else (0. if difference == 0 else float("inf")),
        "normalized_tolerance_ratio": ratio,
        "within_numeric_tolerance": bool(math.isfinite(value) and ratio <= 1.),
    }


def _selection_key(candidate: dict[str, Any]) -> tuple[float, float, str, str]:
    return candidate["worst_ratio"], candidate["mean_ratio"], candidate["case"], candidate["run"]


def audit_candidates(candidates: list[dict[str, Any]], paper_config: Path, output_root: Path) -> dict[str, Any]:
    """Audit supplied, provenance-validated runs without modifying source evidence.

    Each candidate has ``model``, ``case``, ``run`` and optional ``settings``.
    Exactly 46 ordered common origins and identical truth values are mandatory.
    All eight TOTAL cells for a selected model come from the same complete run
    and convention. Each convention has its own whole six-model table, including
    explicit missing rows if a model has no candidate. Reports publish through
    atomic file replacement, with a hashed summary committed last.
    """
    if not candidates:
        raise ValueError("At least one validated prediction candidate is required to establish fixed truth.")
    paper_config, output_root = Path(paper_config).resolve(), Path(output_root).resolve()
    paper_hash = _file_sha256(paper_config)
    paper = yaml.safe_load(paper_config.read_text(encoding="utf-8"))["tasks"]
    for task in ("24h", "168h"):
        for display in MODEL_DISPLAY.values():
            for series in SERIES:
                for metric in METRICS:
                    target = float(paper[task][display][series][metric])
                    if not math.isfinite(target) or (metric != "NSE" and target < 0) or (metric == "NSE" and target > 1):
                        raise ValueError(f"Invalid paper target: {task}/{display}/{series}/{metric}")

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        model, case, run = str(item["model"]), str(item["case"]), Path(item["run"]).resolve()
        if model not in MODEL_DISPLAY or not case:
            raise ValueError(f"Unknown model or empty candidate case: {model!r}/{case!r}")
        if (model, str(run)) in seen:
            raise ValueError(f"Duplicate candidate run for {model}: {run}")
        seen.add((model, str(run)))
        if output_root == run or run in output_root.parents:
            raise ValueError("Audit output must be separate from every source run directory.")
        normalized.append({"model": model, "case": case, "run": str(run), "settings": _json_safe(item.get("settings", {}))})
    if (output_root / "predictions_common46.npz").exists() or (output_root / "status.json").exists():
        raise ValueError("Audit output cannot replace a source run directory.")

    reference: dict[str, np.ndarray] | None = None
    reference_info: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for candidate in sorted(normalized, key=lambda item: (item["model"], item["case"], item["run"])):
        path = Path(candidate["run"]) / "predictions_common46.npz"
        before_hash = _file_sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        invariants = validate_prediction_bundle(arrays, expected_sequences=46, require_first_day_consistency=True)
        if _file_sha256(path) != before_hash:
            raise ValueError(f"Prediction source changed during reading: {path}")
        if reference is None:
            reference = {task: arrays[f"y_true_{task}"].copy() for task in ("24h", "168h")}
            reference_info = {
                "run": candidate["run"], "test_sequences": 46,
                "origins_sha256": invariants["origins_sha256"],
                "forecast_starts": invariants["forecast_starts"],
                "dma_letters": invariants["dma_letters"],
                "truth_sha256": {task: array_sha256(value) for task, value in reference.items()},
                "truth_float64_sha256": {task: array_sha256(np.asarray(value, dtype=np.float64)) for task, value in reference.items()},
                "truth_population_variance_total": {task: float(np.var(np.asarray(value, dtype=np.float64).sum(axis=2))) for task, value in reference.items()},
                "truth_comparison": "Exact array value equality across every candidate and model; no tolerance or scaling.",
                "origins_comparison": "Identical ordered timezone-aware instants after canonical UTC representation.",
            }
        else:
            if invariants["origins_sha256"] != reference_info["origins_sha256"]:
                raise ValueError(f"Common forecast origins differ for candidate {candidate['case']}: {path}")
            for task, true in reference.items():
                if not np.array_equal(arrays[f"y_true_{task}"], true):
                    raise ValueError(f"Fixed common truth differs for {task} candidate {candidate['case']}: {path}")
        provenance.append({**candidate, "predictions_sha256": before_hash, **invariants,
                           "training_provenance_validation": "Delegated to caller; this helper does not deserialize checkpoints."})
        for mode in METRIC_MODES:
            rows: list[dict[str, Any]] = []
            for task in ("24h", "168h"):
                for values in compute_reproduction_metrics(arrays[f"y_true_{task}"], arrays[f"y_pred_{task}"], mode=mode):
                    if values["series"] not in SERIES:
                        continue
                    target = float(paper[task][MODEL_DISPLAY[candidate["model"]]][values["series"]][values["metric"]])
                    rows.append({"model": candidate["model"], "case": candidate["case"], "run": candidate["run"],
                                 "task": task, **values, **_gap(float(values["value"]), target, values["metric"])})
            keys = {(row["task"], row["series"], row["metric"]) for row in rows}
            expected = {(task, series, metric) for task in ("24h", "168h") for series in SERIES for metric in METRICS}
            if len(rows) != 88 or keys != expected:
                raise ValueError("Recomputed candidate must contain all 88 unique paper metric cells.")
            totals = {(row["task"], row["metric"]): row for row in rows if row["series"] == "total"}
            ratios = [totals[key]["normalized_tolerance_ratio"] for key in TOTAL_KEYS]
            dma_passed = sum(row["within_numeric_tolerance"] for row in rows if row["series"] != "total")
            total_passed = sum(row["within_numeric_tolerance"] for row in totals.values())
            worst, mean = max(ratios), sum(ratios) / len(ratios)
            scored.append({
                **candidate, "mode": mode, "metrics": [totals[key]["value"] for key in TOTAL_KEYS],
                "normalized_tolerance_ratios": ratios, "worst_ratio": worst, "mean_ratio": mean,
                "total_worst_tolerance_ratio": worst, "total_mean_tolerance_ratio": mean,
                "total8_passed": total_passed, "dma80_passed": dma_passed, "metrics88_passed": total_passed + dma_passed,
                "complete88_cells": True, "finite88_cells": all(math.isfinite(row["value"]) for row in rows),
                "total8_all_within_tolerance": total_passed == 8, "all88_passed": total_passed + dma_passed == 88,
            })
            all_rows.extend(rows)

    assert reference is not None
    bounds = []
    for task, true in reference.items():
        total_true = np.asarray(true, dtype=np.float64).sum(axis=2)
        for model, display in MODEL_DISPLAY.items():
            target = paper[task][display]["total"]
            for rounding in (0., .0005):
                result = rmse_nse_feasibility(total_true, float(target["RMSE"]), float(target["NSE"]),
                                              error_relative_tolerance=ERROR_RELATIVE_TOLERANCE,
                                              nse_absolute_tolerance=NSE_ABSOLUTE_TOLERANCE, rounding_half_unit=rounding)
                bounds.append({"model": model, "task": task, "series": "total", "truth_sha256": reference_info["truth_sha256"][task],
                               "origins_sha256": reference_info["origins_sha256"], **result,
                               "minimax_pair_lower_bound": rmse_nse_minimax_lower_bound(result["truth_variance"], float(target["RMSE"]),
                                                                                       float(target["NSE"]), rounding_half_unit=rounding)})

    selected_per_mode: dict[str, dict[str, Any]] = {}
    modes: dict[str, dict[str, Any]] = {}
    tables: dict[str, list[dict[str, Any]]] = {}
    for mode in METRIC_MODES:
        selected_per_mode[mode], tables[mode] = {}, []
        for model, display in MODEL_DISPLAY.items():
            choices = sorted((item for item in scored if item["model"] == model and item["mode"] == mode), key=_selection_key)
            for rank, item in enumerate(choices, 1):
                item["rank_within_model_mode"] = rank
                item["selected"] = rank == 1
            if choices:
                chosen = choices[0]
                selected_per_mode[mode][model] = {"status": "SELECTED_COMPLETE_CANDIDATE", **chosen}
                totals = [row for row in all_rows if row["model"] == model and row["mode"] == mode
                          and row["run"] == chosen["run"] and row["series"] == "total"]
            else:
                selected_per_mode[mode][model] = {"status": "MISSING_CANDIDATE", "model": model, "mode": mode, "total8_passed": 0,
                                                 "dma80_passed": 0, "metrics88_passed": 0, "complete88_cells": False}
                totals = [{"model": model, "mode": mode, "case": None, "run": None, "task": task, "series": "total", "metric": metric,
                           "value": None, "paper_value": float(paper[task][display]["total"][metric]), "within_numeric_tolerance": False}
                          for task, metric in TOTAL_KEYS]
            tables[mode].extend({**row, "selection_status": selected_per_mode[mode][model]["status"],
                                 "convention_role": "PRIMARY_POOLED_UNCHANGED" if mode == "pooled" else "DIAGNOSTIC_UNCONFIRMED_HYPOTHESIS"}
                                for row in totals)
        selected = list(selected_per_mode[mode].values())
        complete = all(item["complete88_cells"] for item in selected)
        matched = sum(item["total8_passed"] for item in selected)
        modes[mode] = {"role": "primary" if mode == "pooled" else "diagnostic_unconfirmed_hypothesis",
                       "complete_six_model_table": complete, "total48_passed": matched,
                       "total48_all_within_tolerance": complete and matched == 48,
                       "dma480_passed": sum(item["dma80_passed"] for item in selected),
                       "metrics528_passed": sum(item["metrics88_passed"] for item in selected),
                       "comparison_file": str(output_root / f"total_comparison_{mode}.tsv")}

    blocked = {str(rounding): [{"model": row["model"], "task": row["task"], "reason": row["reason"],
                               "minimax_pair_lower_bound": row["minimax_pair_lower_bound"]}
                              for row in bounds if row["rounding_half_unit"] == rounding and not row["pair_feasible"]]
               for rounding in (0., .0005)}
    impossible = bool(blocked[str(.0005)])
    diagnosis = [
        "Pooled is the unchanged primary metric convention. origin_mean is an unconfirmed diagnostic hypothesis and is never promoted by closeness.",
        "Each selected model uses one complete candidate for both horizons and all eight TOTAL cells. Candidate selection is retrospective paper-table matching.",
    ]
    if impossible:
        names = ", ".join(f"{MODEL_DISPLAY[item['model']]} {item['task']}" for item in blocked[str(.0005)])
        diagnosis.append(f"All 48 TOTAL cells are mathematically impossible within 5% relative error and 0.01 absolute NSE under pooled fixed truth, even allowing ±0.0005 paper rounding: disjoint RMSE/NSE constraints for {names}.")
    else:
        diagnosis.append("Pooled RMSE/NSE constraints do not rule out all 48 TOTAL cells; this necessary condition does not establish a model, MAE/MAPE match, or paper reproduction.")
    diagnosis.append("The pooled identity NSE = 1 - RMSE² / Var(truth) does not apply to origin_mean metrics. No original-paper recovery claim is made.")
    summary: dict[str, Any] = {
        "status": "completed", "primary_metric_mode": "pooled", "primary_metric_changed": False,
        "metric_modes": list(METRIC_MODES), "origin_mean_is_confirmed_publisher_convention": False,
        "origin_mean_automatically_promoted": False, "original_paper_recovery_claim": False,
        "source_predictions_modified": False, "truth_modified": False, "checkpoints_deserialized": False,
        "candidate_training_provenance": "Caller-validated; this report independently validates arrays and fixed common truth/origins.",
        "selection_policy": "Within each model and whole-table mode, minimize worst TOTAL8 normalized tolerance gap, then mean gap, then case/run for deterministic ties. No mixing candidates, horizons, or metric modes within a model.",
        "error_relative_tolerance": ERROR_RELATIVE_TOLERANCE, "nse_absolute_tolerance": NSE_ABSOLUTE_TOLERANCE,
        "total_metric_order": [{"task": task, "metric": metric} for task, metric in TOTAL_KEYS],
        "candidate_count": len(normalized), "paper_config": str(paper_config), "paper_config_sha256": paper_hash,
        "reference": reference_info, "fixed_common_truth_origins_validated": True,
        "selected_per_mode": selected_per_mode, "modes": modes,
        "candidates_summary": scored, "candidates_summary_file": str(output_root / "candidates_summary.json"),
        "pooled_infeasible_pairs_by_rounding": blocked,
        "pooled_total48_impossible_exact_paper_values": bool(blocked[str(0.)]),
        "pooled_total48_impossible_even_with_rounding": impossible,
        "diagnosis": diagnosis,
    }
    if _file_sha256(paper_config) != paper_hash:
        raise ValueError("Paper reference changed during the audit.")
    for item in provenance:
        if _file_sha256(Path(item["run"]) / "predictions_common46.npz") != item["predictions_sha256"]:
            raise ValueError(f"Prediction source changed during the audit: {item['run']}")

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".audit-total-", dir=output_root) as temporary:
        staging = Path(temporary)
        _write_tsv(staging / "feasible_bounds.tsv", bounds)
        _write_tsv(staging / "metrics_all_candidates.tsv", all_rows)
        for mode, rows in tables.items():
            _write_tsv(staging / f"total_comparison_{mode}.tsv", rows)
        _write_json(staging / "candidates_summary.json", scored)
        _write_json(staging / "candidate_provenance.json", provenance)
        summary["report_files_sha256"] = {path.name: _file_sha256(path) for path in sorted(staging.iterdir())}
        _write_json(staging / "protocol_summary.json", summary)
        # Each file is replaced atomically; the summary is the final commit
        # marker, with hashes that expose an interrupted multi-file publication.
        for name in summary["report_files_sha256"]:
            os.replace(staging / name, output_root / name)
        os.replace(staging / "protocol_summary.json", output_root / "protocol_summary.json")
    return _json_safe(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="JSON list of caller-validated model/case/run/settings candidates.")
    parser.add_argument("--paper-config", type=Path, default=ROOT / "configs/evaluation/mscmnet_paper_metrics.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_candidates(json.loads(args.candidates.read_text(encoding="utf-8")), args.paper_config, args.output_root)
    print(json.dumps({"status": summary["status"], "primary_metric_mode": summary["primary_metric_mode"],
                      "modes": summary["modes"], "diagnosis": summary["diagnosis"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
