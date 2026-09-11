#!/usr/bin/env python
"""Recompute both complete metric conventions from saved prediction evidence.

This never trains a model, changes a prediction/checkpoint, or selects a metric
convention by closeness to the published test table. Numerical infeasibility is
an audit result, not an execution failure. Corrupt completed artifacts fail.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from zipfile import BadZipFile
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dma_wdf.data.reproduction_metrics import (  # noqa: E402
    METRIC_MODES,
    array_sha256,
    compute_reproduction_metrics,
    rmse_nse_feasibility,
    validate_prediction_bundle,
)

MODEL_DISPLAY = {
    "gru": "GRU", "lstm": "LSTM", "msnet": "MSNet",
    "mscmnet_m": "MSCMNet_M", "mscmnet_wm": "MSCMNet_WM", "mscmnet_w": "MSCMNet_W",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _gap(row: dict[str, Any], paper_value: float, relative: float, absolute_nse: float) -> dict[str, Any]:
    actual = float(row["value"])
    difference = actual-paper_value
    relative_gap = abs(difference)/max(abs(paper_value), 1e-12)
    accepted = abs(difference) <= absolute_nse if row["metric"] == "NSE" else relative_gap <= relative
    return {
        **row, "paper_value": paper_value, "signed_difference": difference,
        "absolute_difference": abs(difference), "absolute_relative_difference": relative_gap,
        "within_numeric_tolerance": bool(math.isfinite(actual) and accepted),
    }


def closeness_by_source(
    gaps: list[dict[str, Any]], *, error_relative_tolerance: float = .05,
    nse_absolute_tolerance: float = .01,
) -> list[dict[str, Any]]:
    """One compact row per exact run and complete metric convention.

    This is retrospective numerical description, not source selection. In
    particular complementary successes across sources/modes cannot combine.
    ``total8_passed``/``dma80_passed`` are counts; ``all88_passed`` is a boolean.
    """
    groups: dict[tuple[str, str, Any, str], list[dict[str, Any]]] = {}
    for row in gaps:
        key = (row["source_id"], row["model"], row["seed"], row["mode"])
        groups.setdefault(key, []).append(row)
    expected_total = {(task, "total", metric) for task in ("24h", "168h")
                      for metric in ("MAE", "MAPE", "RMSE", "NSE")}
    expected_dma = {(task, letter, metric) for task in ("24h", "168h")
                    for letter in "ABCDEFGHIJ" for metric in ("MAE", "MAPE", "RMSE", "NSE")}

    def finite_gap(value: Any) -> float:
        converted = float(value)
        return converted if math.isfinite(converted) else float("inf")

    def tolerance_ratio(gap: float, tolerance: float) -> float:
        return gap/tolerance if tolerance > 0 else (0.0 if gap == 0 else float("inf"))

    result = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        rows = groups[key]
        cells = {(row["task"], row["series"], row["metric"]) for row in rows}
        if len(cells) != len(rows):
            raise ValueError("Duplicate metric cells within one source and mode cannot be counted.")
        if len({row["source_path"] for row in rows}) != 1:
            raise ValueError("A source identifier cannot merge different source paths.")
        total = [row for row in rows if row["series"] == "total"]
        dma = [row for row in rows if row["series"] in "ABCDEFGHIJ"]
        total_passed = sum(bool(row["within_numeric_tolerance"]) for row in total)
        dma_passed = sum(bool(row["within_numeric_tolerance"]) for row in dma)
        errors = [finite_gap(row["absolute_relative_difference"]) for row in total if row["metric"] != "NSE"]
        efficiencies = [finite_gap(row["absolute_difference"]) for row in total if row["metric"] == "NSE"]
        complete_total = expected_total.issubset(cells)
        complete = cells == expected_total | expected_dma
        result.append({
            "source_id": key[0], "source_path": rows[0]["source_path"], "model": key[1],
            "seed": key[2], "mode": key[3], "git_commit": rows[0].get("git_commit"),
            "complete88_cells": complete, "total8_passed": total_passed,
            "dma80_passed": dma_passed, "metrics88_passed": total_passed+dma_passed,
            "total8_all_within_tolerance": complete_total and total_passed == 8,
            "all88_passed": complete and total_passed+dma_passed == 88,
            "max_total6_error_relative_gap": max(errors) if errors else float("nan"),
            "max_total2_nse_absolute_gap": max(efficiencies) if efficiencies else float("nan"),
            "total_worst_tolerance_ratio": max(
                [tolerance_ratio(value, error_relative_tolerance) for value in errors]
                + [tolerance_ratio(value, nse_absolute_tolerance) for value in efficiencies]
            ) if total else float("nan"),
            "error_relative_tolerance": error_relative_tolerance,
            "nse_absolute_tolerance": nse_absolute_tolerance,
            "interpretation": "RETROSPECTIVE_NUMERICAL_DESCRIPTION_NOT_VERIFIED_REPRODUCTION",
        })
    return result


def audit_saved_predictions(
    *, results_roots: list[Path], paper_config: Path, output_root: Path,
    source_selection: dict[str, str] | None = None,
    error_relative_tolerance: float = .05, nse_absolute_tolerance: float = .01,
    strict_first_day: bool = False,
) -> dict[str, Any]:
    if not 0 <= error_relative_tolerance < 1 or not math.isfinite(error_relative_tolerance):
        raise ValueError("Relative tolerance must be finite and in [0, 1).")
    if nse_absolute_tolerance < 0 or not math.isfinite(nse_absolute_tolerance):
        raise ValueError("NSE tolerance must be finite and nonnegative.")
    paper = yaml.safe_load(paper_config.read_text(encoding="utf-8"))["tasks"]
    output_root = output_root.resolve()
    # Never put audit files in a source run directory where they could overwrite evidence.
    if (output_root / "predictions_common46.npz").exists() or (output_root / "status.json").exists():
        raise ValueError("Audit output must be separate from every original run directory.")
    output_root.mkdir(parents=True, exist_ok=True)
    source_status: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for root in results_roots:
        root = root.resolve()
        if not root.exists():
            source_status.append({"path": str(root), "status": "MISSING_ROOT"})
            continue
        if root.is_file():
            if root.name != "predictions_common46.npz":
                raise ValueError(f"Explicit evidence file must be predictions_common46.npz: {root}")
            paths.add(root)
        else:
            paths.update(path.resolve() for path in root.rglob("predictions_common46.npz"))
    metric_rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    discrepancies: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    truth_groups: dict[tuple[str, str], dict[str, Any]] = {}
    valid_by_model: dict[str, list[str]] = {}
    for path in sorted(paths):
        if any(".backup-" in part or part in {"source_snapshot", ".pytest_cache"} for part in path.parts):
            source_status.append({"path": str(path), "status": "SKIP_ARCHIVED_OR_SNAPSHOT"})
            continue
        run = path.parent
        try:
            status_path = run / "status.json"
            if not status_path.exists():
                source_status.append({"path": str(path), "status": "SKIP_NO_STATUS"})
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "completed":
                source_status.append({"path": str(path), "status": "SKIP_NOT_COMPLETED"})
                continue
            model = status.get("model")
            if model not in MODEL_DISPLAY:
                source_status.append({"path": str(path), "status": "SKIP_OTHER_MODEL", "model": model})
                continue
            config_path = run / "resolved_config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            invariants = validate_prediction_bundle(arrays, require_first_day_consistency=strict_first_day)
            source_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
            common = {"source_id": source_id, "source_path": str(path), "model": model,
                      "seed": status.get("seed"), "git_commit": status.get("git_commit")}
            source_metrics: list[dict[str, Any]] = []
            for task in ("24h", "168h"):
                truth = arrays[f"y_true_{task}"]
                truth_hash = array_sha256(truth)
                key = (task, truth_hash)
                group = truth_groups.setdefault(key, {"truth": truth, "source_ids": [], "origins_hashes": set()})
                group["source_ids"].append(source_id)
                group["origins_hashes"].add(invariants["origins_sha256"])
                for mode in METRIC_MODES:
                    rows = compute_reproduction_metrics(truth, arrays[f"y_pred_{task}"], mode=mode)
                    for values in rows:
                        row = {**common, "task": task, **values, "truth_sha256": truth_hash}
                        source_metrics.append(row)
                        target = paper[task][MODEL_DISPLAY[model]].get(row["series"], {}).get(row["metric"])
                        if target is not None:
                            gaps.append(_gap(row, float(target), error_relative_tolerance, nse_absolute_tolerance))
            old_path = run / "metrics.csv"
            if old_path.exists():
                with old_path.open(newline="", encoding="utf-8") as stream:
                    old_rows = list(csv.DictReader(stream))
                old = {(row["task"], row["series"], row["metric"]): float(row["value"]) for row in old_rows}
                for row in source_metrics:
                    key = (row["task"], row["series"], row["metric"])
                    if row["mode"] == "pooled" and key in old:
                        delta = float(row["value"])-old[key]
                        discrepancies.append({**row, "stored_value": old[key], "recomputed_minus_stored": delta,
                                              "within_float32_sum_roundoff": math.isfinite(delta) and abs(delta) <= 5e-5})
            metric_rows.extend(source_metrics)
            provenance.append({
                **common, **invariants, "npz_sha256": file_sha256(path),
                "status_sha256": file_sha256(status_path), "config_sha256": file_sha256(config_path),
                "source_status": status, "resolved_config": config,
                "checkpoint_files_present": {name: (run/name).is_file() for name in status.get("checkpoint_files", [])},
                "checkpoint_identity_verified": False,
                "checkpoint_note": "No checkpoint deserialization; first-day equality alone does not establish training provenance.",
            })
            valid_by_model.setdefault(model, []).append(str(path))
            source_status.append({"path": str(path), "source_id": source_id, "model": model,
                                  "status": "VALID" if invariants["first_day_consistent"] else "VALID_ARRAYS_FIRST_DAY_DISCREPANCY"})
        except (ValueError, KeyError, TypeError, OSError, EOFError, BadZipFile, yaml.YAMLError) as exc:
            source_status.append({"path": str(path), "status": "INVALID_COMPLETED_EVIDENCE", "error": str(exc)})
    valid_ids = {item["source_id"] for item in provenance}
    metric_rows = [row for row in metric_rows if row["source_id"] in valid_ids]
    gaps = [row for row in gaps if row["source_id"] in valid_ids]
    discrepancies = [row for row in discrepancies if row["source_id"] in valid_ids]
    for group in truth_groups.values():
        group["source_ids"] = [source_id for source_id in group["source_ids"] if source_id in valid_ids]
    truth_groups = {key: group for key, group in truth_groups.items() if group["source_ids"]}
    feasibility: list[dict[str, Any]] = []
    for (task, truth_hash), group in sorted(truth_groups.items()):
        truth = np.asarray(group["truth"], dtype=np.float64)
        for model, display in MODEL_DISPLAY.items():
            for index, letter in enumerate(list("ABCDEFGHIJ") + ["total"]):
                observed = truth[:, :, index] if letter != "total" else truth.sum(axis=2)
                target = paper[task][display][letter]
                for rounding in (0.0, .0005):
                    feasibility.append({
                        "truth_sha256": truth_hash, "task": task, "target_model": model, "series": letter,
                        "source_ids": ",".join(group["source_ids"]),
                        **rmse_nse_feasibility(observed, float(target["RMSE"]), float(target["NSE"]),
                                              error_relative_tolerance=error_relative_tolerance,
                                              nse_absolute_tolerance=nse_absolute_tolerance,
                                              rounding_half_unit=rounding),
                    })
    selection: dict[str, dict[str, Any]] = {}
    requested = source_selection or {}
    if set(requested).difference(MODEL_DISPLAY):
        raise ValueError("Source-selection keys must use one of the six model identifiers.")
    for model in MODEL_DISPLAY:
        candidates = valid_by_model.get(model, [])
        if model in requested:
            chosen = str(Path(requested[model]).resolve())
            if chosen not in candidates:
                raise ValueError(f"Selected source is not a valid audited {model} artifact: {chosen}")
            selection[model] = {"status": "EXPLICIT_SOURCE", "source_path": chosen}
        elif len(candidates) == 1:
            selection[model] = {"status": "SOLE_SOURCE", "source_path": candidates[0]}
        else:
            selection[model] = {"status": "MISSING" if not candidates else "UNSELECTED_MULTIPLE_SOURCES",
                                "candidate_count": len(candidates)}
    invalid = sum(row["status"] == "INVALID_COMPLETED_EVIDENCE" for row in source_status)
    blocked = sorted({row["target_model"] for row in feasibility
                      if row["series"] == "total" and row["rounding_half_unit"] == .0005 and not row["pair_feasible"]})
    closeness = closeness_by_source(gaps, error_relative_tolerance=error_relative_tolerance,
                                   nse_absolute_tolerance=nse_absolute_tolerance)
    numeric_counts = {}
    for mode in METRIC_MODES:
        numeric_counts[mode] = {}
        for model in MODEL_DISPLAY:
            rows = [row for row in closeness if row["mode"] == mode and row["model"] == model]
            numeric_counts[mode][model] = {
                "complete_sources": sum(row["complete88_cells"] for row in rows),
                "total8_within_tolerance_sources": sum(row["total8_all_within_tolerance"] for row in rows),
                "all88_within_tolerance_sources": sum(row["all88_passed"] for row in rows),
            }
    summary = {
        "status": "invalid_evidence" if invalid else ("completed" if provenance else "no_predictions"),
        "valid_sources": len(provenance), "invalid_sources": invalid,
        "first_day_discrepancy_sources": sum(not row["first_day_consistent"] for row in provenance),
        "models_with_raw_predictions": sorted(valid_by_model), "source_selection": selection,
        "paper_config": str(paper_config.resolve()), "paper_config_sha256": file_sha256(paper_config),
        "metric_modes": list(METRIC_MODES),
        "numeric_counts": numeric_counts,
        "compact_comparison_file": "closeness_by_source.tsv",
        "origin_mean_provenance": "UNCONFIRMED_HYPOTHESIS_NOT_A_VERIFIED_PUBLISHER_CONVENTION",
        "selection_policy": "No source, prediction, or individual metric selected by paper-test closeness.",
        "pooled_total_blocked_models_even_with_rounding": blocked,
        "truth_groups_per_task": {task: sum(key[0] == task for key in truth_groups) for task in ("24h", "168h")},
        "truth_origin_group_mismatch": any(len(group["origins_hashes"]) != 1 for group in truth_groups.values()),
        "stored_metric_roundoff_exceeded": sum(not row["within_float32_sum_roundoff"] for row in discrepancies),
        "interpretation": "Feasible only means a necessary RMSE/NSE constraint is not violated; it never proves complete reproduction.",
    }
    for name, rows in (("metrics_recomputed.tsv", metric_rows), ("paper_gaps.tsv", gaps),
                       ("feasibility.tsv", feasibility), ("source_status.tsv", source_status),
                       ("stored_metric_discrepancies.tsv", discrepancies),
                       ("closeness_by_source.tsv", closeness)):
        _write_tsv(output_root/name, rows)
    for name, value in (("provenance.json", provenance), ("audit_summary.json", summary)):
        (output_root/name).write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, action="append", required=True)
    parser.add_argument("--paper-config", type=Path, default=ROOT/"configs/evaluation/mscmnet_paper_metrics.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-selection", type=Path)
    parser.add_argument("--error-relative-tolerance", type=float, default=.05)
    parser.add_argument("--nse-absolute-tolerance", type=float, default=.01)
    parser.add_argument("--strict-first-day", action="store_true",
                        help="Reject first-day differences beyond declared float32 tolerance (default: record diagnostic warning).")
    parser.add_argument("--allow-invalid-evidence", action="store_true",
                        help="Historical audit only: report/exclude corrupt completed sources but still exit zero.")
    args = parser.parse_args()
    selection = json.loads(args.source_selection.read_text(encoding="utf-8")) if args.source_selection else None
    summary = audit_saved_predictions(results_roots=args.results_root, paper_config=args.paper_config,
                                     output_root=args.output_root, source_selection=selection,
                                     error_relative_tolerance=args.error_relative_tolerance,
                                     nse_absolute_tolerance=args.nse_absolute_tolerance,
                                     strict_first_day=args.strict_first_day)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["invalid_sources"] and not args.allow_invalid_evidence else 0


if __name__ == "__main__":
    raise SystemExit(main())
