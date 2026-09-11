#!/usr/bin/env python3
"""Bounded CPU search for a test-target-guided GRU/LSTM total-only cohort.

Each DMA retains one whole source checkpoint, embedded scaler, and both horizons.
This is retrospective numerical reconstruction, not original-method recovery or
held-out validation. The caller must first validate complete training provenance.
No checkpoint is deserialized and no inference, fitting, or calibration is done.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dma_wdf.data.reproduction_metrics import (  # noqa: E402
    METRIC_MODES, METRICS, array_sha256, compute_reproduction_metrics,
    validate_prediction_bundle,
)

# Keep legacy Stage-D source and behavior unchanged. Its loader verifies intact
# independent checkpoints, frozen horizons, common truth/origins and family seed.
_SPEC = importlib.util.spec_from_file_location("_que_total_legacy_loader", Path(__file__).with_name("assemble_que_recurrent_candidates.py"))
assert _SPEC and _SPEC.loader
_LEGACY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LEGACY)
_sha, _json_hash = _LEGACY._sha, _LEGACY._json_hash
_write_json, _write_tsv = _LEGACY._write_json, _LEGACY._write_tsv
LETTERS, TASKS = tuple("ABCDEFGHIJ"), ("24h", "168h")
SEED = 20240604
RELATIVE_TOLERANCE, NSE_TOLERANCE = .05, .01


def _total_targets(paper_config: Path, model: str) -> dict[tuple[str, str], float]:
    paper = yaml.safe_load(paper_config.read_text(encoding="utf-8"))
    targets = {(task, metric): float(paper["tasks"][task][model.upper()]["total"][metric])
               for task in TASKS for metric in METRICS}
    if not all(math.isfinite(value) for value in targets.values()):
        raise ValueError("All eight published total targets must be finite.")
    return targets


def _validate_budget(starts: int, sweeps: int, pair_trials: int, search_seed: int) -> None:
    for name, value, lower, upper in (("starts", starts, 0, 128), ("sweeps", sweeps, 1, 64),
                                      ("pair_trials", pair_trials, 0, 100000),
                                      ("search_seed", search_seed, 0, 2**32 - 1)):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")


def _validate_scalers(candidates: list[dict[str, Any]]) -> None:
    for candidate in candidates:
        if candidate["seed"] != SEED:
            raise ValueError(f"Total recurrent search requires fixed family seed {SEED}.")
        path = candidate["path"] / "scaler_audit.json"
        if not path.is_file():
            raise ValueError(f"Missing source scaler audit: {path}")
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("test_values_used_for_fit") is not False or set(audit.get("per_dma", {})) != set(LETTERS):
            raise ValueError("Scaler provenance must declare train-only fitting for all ten DMAs.")
        candidate["scaler_audit"] = audit


class _TotalSearch:
    """O(candidate count * prediction size) input storage, O(prediction size) work.

    Alternative evaluation updates only two-dimensional summed predictions. Only
    each source DMA's MAE is cached, because paper total MAE is its sum; DMA target
    values and DMA target gaps are never read, scored, or used in a tie-break.
    """

    def __init__(self, candidates: list[dict[str, Any]], targets: dict[tuple[str, str], float], mode: str):
        if mode not in METRIC_MODES:
            raise ValueError("Select one supported metric convention for the entire experiment.")
        self.candidates, self.mode = candidates, mode
        self.count = len(candidates)
        self.axes = None if mode == "pooled" else 1
        self.target = np.array([[targets[(task, metric)] for metric in METRICS] for task in TASKS])
        self.denominator = np.maximum(np.abs(self.target), 1e-12) * RELATIVE_TOLERANCE
        self.denominator[:, METRICS.index("NSE")] = NSE_TOLERANCE
        self.truth = {task: np.asarray(candidates[0]["arrays"][f"y_true_{task}"], dtype=np.float64) for task in TASKS}
        self.true_sum = {task: values.sum(axis=2) for task, values in self.truth.items()}
        self.variance = {task: np.var(values, axis=self.axes) for task, values in self.true_sum.items()}
        for task in TASKS:
            if np.any(np.abs(self.true_sum[task]) < 1e-12) or np.any(self.variance[task] <= 1e-12):
                raise ValueError("Undefined total MAPE/NSE cannot be masked during selection.")
        self.mae = np.empty((self.count, 10, 2), dtype=np.float64)
        for k, candidate in enumerate(candidates):
            for t, task in enumerate(TASKS):
                for j in range(10):
                    error = self.truth[task][:, :, j] - self.column(task, k, j)
                    self.mae[k, j, t] = float(np.mean(np.mean(np.abs(error), axis=self.axes)))
        self.evaluations = 0
        self.delta_evaluations = 0
        self.exact_sum_evaluations = 0

    def column(self, task: str, candidate: int, dma: int) -> np.ndarray:
        return np.asarray(self.candidates[candidate]["arrays"][f"y_pred_{task}"][:, :, dma], dtype=np.float64)

    def totals(self, choice: tuple[int, ...]) -> dict[str, np.ndarray]:
        # Rebuild on accepted moves to prevent accumulated delta-roundoff drift.
        self.exact_sum_evaluations += 1
        return {task: np.stack([self.column(task, k, j) for j, k in enumerate(choice)], axis=2).sum(axis=2)
                for task in TASKS}

    def replace(self, totals: dict[str, np.ndarray], choice: tuple[int, ...], updates: tuple[tuple[int, int], ...]) -> dict[str, np.ndarray]:
        result = {}
        for task in TASKS:
            summed = totals[task].copy()
            for dma, candidate in updates:
                summed += self.column(task, candidate, dma) - self.column(task, choice[dma], dma)
            result[task] = summed
        self.delta_evaluations += 1
        return result

    def evaluate(self, choice: tuple[int, ...], totals: dict[str, np.ndarray] | None = None) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if totals is None:
            totals = self.totals(choice)
        self.evaluations += 1
        values = np.empty((2, 4), dtype=np.float64)
        for t, task in enumerate(TASKS):
            error = self.true_sum[task] - totals[task]
            mse = np.mean(error**2, axis=self.axes)
            values[t] = (float(sum(self.mae[k, j, t] for j, k in enumerate(choice))),
                         float(np.mean(np.mean(np.abs(error / self.true_sum[task]), axis=self.axes))),
                         float(np.mean(np.sqrt(mse))),
                         float(np.mean(1.0 - mse / self.variance[task])))
        if not np.isfinite(values).all():
            raise ValueError("Undefined total metric cannot be dropped during selection.")
        ratios = np.abs(values - self.target) / self.denominator
        score = {"worst_ratio": float(ratios.max()), "mean_ratio": float(ratios.mean()),
                 "total8_passed": int(np.sum(ratios <= 1)), "all_total8_passed": bool(np.all(ratios <= 1)),
                 "total_metrics": {task: dict(zip(METRICS, map(float, values[t]))) for t, task in enumerate(TASKS)}}
        return (score["worst_ratio"], score["mean_ratio"], choice), score

    def coordinate_sweep(self, choice: tuple[int, ...], totals: dict[str, np.ndarray], key: tuple[Any, ...]) -> tuple[tuple[int, ...], dict[str, np.ndarray], tuple[Any, ...], int]:
        trials = 0
        for dma in range(10):
            selected, selected_key = choice, key
            for candidate in range(self.count):
                if candidate == choice[dma]:
                    continue
                alternative = choice[:dma] + (candidate,) + choice[dma + 1:]
                candidate_totals = self.replace(totals, choice, ((dma, candidate),))
                alternative_key, _ = self.evaluate(alternative, candidate_totals)
                trials += 1
                if alternative_key < selected_key:
                    # Exact rebasing before a move can win avoids delta-roundoff
                    # causing false strict improvements or violating choice ties.
                    alternative_key, _ = self.evaluate(alternative)
                    if alternative_key < selected_key:
                        selected, selected_key = alternative, alternative_key
            if selected != choice:
                choice, key = selected, selected_key
                totals = self.totals(choice)
        return choice, totals, key, trials

    def solve(self, starts: int, sweeps: int, pair_trials: int, search_seed: int) -> tuple[tuple[int, ...], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        _validate_budget(starts, sweeps, pair_trials, search_seed)
        rng = np.random.default_rng(search_seed)
        trace: list[dict[str, Any]] = []
        homogeneous = []
        for candidate in range(self.count):
            choice = tuple([candidate] * 10)
            key, score = self.evaluate(choice)
            homogeneous.append((key, choice))
            trace.append({"phase": "homogeneous", "candidate": candidate, "choice": list(choice), **score})
        homogeneous.sort(key=lambda item: item[0])
        best_key, best_choice = homogeneous[0]
        # Every intact cohort is evaluated, but only a bounded subset is refined.
        # Refine up to four best homogeneous seeds, reserving slots for mixtures.
        whole_count = min(self.count, starts, max(1, min(4, starts // 2)))
        initial = [item[1] for item in homogeneous[:whole_count]]
        homogeneous_choices = {item[1] for item in homogeneous}
        seen = set(initial)
        mixed_goal = min(max(0, starts - whole_count), self.count**10 - self.count)
        for _ in range(max(1, 20 * mixed_goal)):
            if len(initial) - whole_count >= mixed_goal:
                break
            choice = tuple(map(int, rng.integers(0, self.count, size=10)))
            if choice not in seen and choice not in homogeneous_choices:
                initial.append(choice)
                seen.add(choice)
        # At most count+mixed_goal entries suffice to fill missing distinct mixed
        # seeds after skipping the count possible homogeneous tuples.
        for choice in itertools.islice(itertools.product(range(self.count), repeat=10), self.count + mixed_goal):
            if len(initial) - whole_count >= mixed_goal:
                break
            if choice not in seen and choice not in homogeneous_choices:
                initial.append(choice)
                seen.add(choice)
        coordinate_trials = 0
        completed_sweeps = 0
        for start_id, choice in enumerate(initial):
            totals = self.totals(choice)
            key, score = self.evaluate(choice, totals)
            trace.append({"phase": "start", "start": start_id, "kind": "homogeneous" if start_id < whole_count else "seeded_mixed",
                          "sweep": 0, "choice": list(choice), **score})
            for sweep in range(1, sweeps + 1):
                before = choice
                choice, totals, key, trials = self.coordinate_sweep(choice, totals, key)
                coordinate_trials += trials
                completed_sweeps += 1
                _, score = self.evaluate(choice, totals)
                trace.append({"phase": "coordinate", "start": start_id, "sweep": sweep, "choice": list(choice), **score})
                if before == choice:
                    break
            if key < best_key:
                best_choice, best_key = choice, key
        choice, totals, key = best_choice, self.totals(best_choice), best_key
        pairs = list(itertools.combinations(range(10), 2))
        pair_count = accepted_pairs = pair_refinement_sweeps = 0
        if self.count > 1 and pair_trials:
            domain = len(pairs) * (self.count - 1)**2
            offset = int(rng.integers(domain))
            stride = int(rng.integers(1, max(2, domain)))
            while math.gcd(stride, domain) != 1:
                stride = (stride + 1) % domain or 1
            for epoch in range(1, min(sweeps, pair_trials) + 1):
                rounds_left = min(sweeps, pair_trials) - epoch + 1
                count = math.ceil((pair_trials - pair_count) / rounds_left)
                selected, selected_key = choice, key
                for _ in range(count):
                    # A modular permutation avoids an O(C^2) proposal list and
                    # covers the entire fixed-incumbent pair domain before reuse.
                    proposal = (offset + pair_count * stride) % domain
                    pair_count += 1
                    pair_index, replacements = divmod(proposal, (self.count - 1)**2)
                    first, second = pairs[pair_index]
                    left, right = divmod(replacements, self.count - 1)
                    left += left >= choice[first]
                    right += right >= choice[second]
                    mutable = list(choice)
                    mutable[first], mutable[second] = left, right
                    alternative = tuple(mutable)
                    alternative_key, _ = self.evaluate(alternative, self.replace(totals, choice, ((first, left), (second, right))))
                    if alternative_key < selected_key:
                        alternative_key, _ = self.evaluate(alternative)
                        if alternative_key < selected_key:
                            selected, selected_key = alternative, alternative_key
                accepted = selected != choice
                if accepted:
                    accepted_pairs += 1
                    choice, key, totals = selected, selected_key, self.totals(selected)
                    choice, totals, key, trials = self.coordinate_sweep(choice, totals, key)
                    coordinate_trials += trials
                    pair_refinement_sweeps += 1
                _, score = self.evaluate(choice, totals)
                trace.append({"phase": "pair", "epoch": epoch, "pair_trials_cumulative": pair_count,
                              "accepted": accepted, "choice": list(choice), **score})
        _, score = self.evaluate(choice)
        budget = {"refinement_starts_limit": starts, "homogeneous_cohorts_evaluated": self.count,
                  "homogeneous_refinement_starts": whole_count, "mixed_starts": len(initial) - whole_count,
                  "starts_evaluated": len(initial),
                  "coordinate_sweeps_per_start_limit": sweeps, "coordinate_sweeps_completed": completed_sweeps,
                  "pair_trials_limit": pair_trials, "pair_trials_evaluated": pair_count,
                  "pair_batches_accepted": accepted_pairs, "pair_refinement_sweeps": pair_refinement_sweeps,
                  "coordinate_trials_evaluated": coordinate_trials,
                  "coordinate_trials_upper_bound": (len(initial) + 1) * sweeps * 10 * max(0, self.count - 1),
                  "total_metric_evaluations": self.evaluations, "delta_evaluations": self.delta_evaluations,
                  "exact_sum_evaluations": self.exact_sum_evaluations,
                  "loaded_input_array_bytes": sum(a.nbytes for c in self.candidates for a in c["arrays"].values()),
                  "worker_processes": 1, "device": "cpu", "search_seed": search_seed,
                  "memory_note": "Input arrays retained once; O(candidate count * 20) MAE cache; constant-size total prediction work arrays; no trial-history prediction cache."}
        return choice, score, trace, budget


def _assemble(candidates: list[dict[str, Any]], choice: tuple[int, ...]) -> dict[str, np.ndarray]:
    reference = candidates[0]["arrays"]
    arrays = {key: reference[key].copy() for key in ("dma_letters", "forecast_starts", "y_true_24h", "y_true_168h")}
    for task in TASKS:
        arrays[f"y_pred_{task}"] = np.stack([candidates[k]["arrays"][f"y_pred_{task}"][:, :, j] for j, k in enumerate(choice)], axis=2)
    validate_prediction_bundle(arrays, require_first_day_consistency=True)
    return arrays


def _exact_metrics(arrays: dict[str, np.ndarray], targets: dict[tuple[str, str], float], model: str, mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metrics, comparison = [], []
    values = {task: {} for task in TASKS}
    for task in TASKS:
        for row in compute_reproduction_metrics(arrays[f"y_true_{task}"], arrays[f"y_pred_{task}"], mode=mode):
            record = {"model": model, "seed": SEED, "task": task, **row}
            metrics.append(record)
            if row["series"] != "total":
                continue
            actual, target = float(row["value"]), targets[(task, row["metric"])]
            if not math.isfinite(actual):
                raise ValueError("Exact reevaluation produced an undefined total metric.")
            ratio = _LEGACY._ratio(actual, target, row["metric"], RELATIVE_TOLERANCE, NSE_TOLERANCE)
            values[task][row["metric"]] = actual
            comparison.append({**record, "paper_value": target, "signed_difference": actual - target,
                               "absolute_difference": abs(actual - target),
                               "absolute_relative_difference": abs(actual - target) / max(abs(target), 1e-12),
                               "tolerance_ratio": ratio, "within_numeric_tolerance": ratio <= 1})
    ratios = np.array([row["tolerance_ratio"] for row in comparison])
    score = {"worst_ratio": float(ratios.max()), "mean_ratio": float(ratios.mean()),
             "total8_passed": int(np.sum(ratios <= 1)), "all_total8_passed": bool(np.all(ratios <= 1)),
             "total_metrics": values}
    return metrics, comparison, score


def _check_source_hashes(candidates: list[dict[str, Any]], paper_config: Path, request: dict[str, Any]) -> None:
    for candidate in candidates:
        for name, digest in candidate["hashes"].items():
            if _sha(candidate["path"] / name) != digest:
                raise ValueError(f"Source provenance changed during search: {candidate['path'] / name}")
    if _sha(paper_config) != request["paper_sha256"]:
        raise ValueError("Published target evidence changed during search.")
    for path, key in ((Path(__file__), "algorithm_sha256"), (Path(_SPEC.origin), "legacy_loader_sha256"),
                      (ROOT / "src/dma_wdf/data/reproduction_metrics.py", "metric_implementation_sha256")):
        if _sha(path) != request[key]:
            raise ValueError("Search/loader/metric source changed during search.")


def _cached(output_root: Path, candidates: list[dict[str, Any]], targets: dict[tuple[str, str], float], request: dict[str, Any]) -> dict[str, Any]:
    try:
        manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
        unsigned = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if manifest.get("manifest_sha256") != _json_hash(unsigned) or manifest.get("request") != request or manifest.get("request_sha256") != _json_hash(request):
            raise ValueError("Manifest or request identity changed")
        model = candidates[0]["model"]
        required_metadata = {"status": "completed", "model": model, "seed": SEED, "mode": request["mode"],
                             "output_root": str(output_root), "dma_targets_used_in_objective": False,
                             "paper_reproduction_verified": False, "original_method_recovered": False,
                             "held_out_validation": False, "global_optimum_proven": False}
        if any(manifest.get(key) != value for key, value in required_metadata.items()):
            raise ValueError("Cached cohort identity or interpretation changed")
        artifacts = manifest["artifact_sha256"]
        required_artifacts = {"predictions_common46.npz", "metrics.tsv", "paper_gaps.tsv", "total_comparison.tsv", "search_trace.json"}
        required_artifacts |= {f"checkpoints/checkpoint_{model}_dma_{letter}.pt" for letter in LETTERS}
        required_artifacts |= {f"source_configs/dma_{letter}.yaml" for letter in LETTERS}
        required_artifacts |= {f"source_scalers/dma_{letter}.json" for letter in LETTERS}
        if set(artifacts) != required_artifacts:
            raise ValueError("Required artifact evidence is incomplete")
        actual_names = {str(path.relative_to(output_root)) for path in output_root.rglob("*") if path.is_file() and path.name != "manifest.json"}
        if set(artifacts) != actual_names:
            raise ValueError("Artifact inventory changed")
        for name, digest in artifacts.items():
            path = output_root / name
            if path.is_symlink() or output_root not in path.resolve().parents or _sha(path) != digest:
                raise ValueError("Artifact content changed")
        selected = manifest["selected_networks"]
        if len(selected) != 10:
            raise ValueError("Selected cohort is incomplete")
        path_index = {str(c["path"]): k for k, c in enumerate(candidates)}
        choice = tuple(path_index[row["source_path"]] for row in selected)
        expected = _assemble(candidates, choice)
        with np.load(output_root / "predictions_common46.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(expected) or any(not np.array_equal(saved[key], value) for key, value in expected.items()):
                raise ValueError("Saved predictions differ from intact source columns")
            arrays = {key: saved[key] for key in saved.files}
        for j, (entry, k) in enumerate(zip(selected, choice)):
            candidate, letter = candidates[k], LETTERS[j]
            checkpoint = f"checkpoint_{candidate['model']}_dma_{letter}.pt"
            required = {"dma": letter, "column_index": j, "family_seed": SEED,
                        "checkpoint": f"checkpoints/{checkpoint}", "checkpoint_sha256": candidate["hashes"][checkpoint],
                        "source_config": f"source_configs/dma_{letter}.yaml", "source_config_sha256": candidate["hashes"]["resolved_config.yaml"],
                        "source_scaler_audit": f"source_scalers/dma_{letter}.json", "source_scaler_audit_sha256": candidate["hashes"]["scaler_audit.json"],
                        "source_npz_sha256": candidate["hashes"]["predictions_common46.npz"],
                        "per_dma_model_config": {key: value[j] if isinstance(value, list) and len(value) == 10 else value for key, value in candidate["config"]["model"].items()},
                        "training_config": candidate["config"]["training"],
                        "same_source_for_all_horizons_and_metrics": True}
            if any(entry.get(key) != value for key, value in required.items()):
                raise ValueError("Selected source provenance changed")
            for file_key, hash_key in (("checkpoint", "checkpoint_sha256"), ("source_config", "source_config_sha256"), ("source_scaler_audit", "source_scaler_audit_sha256")):
                if artifacts.get(entry[file_key]) != entry[hash_key]:
                    raise ValueError("Source-copy hash mismatch")
        _, comparison, score = _exact_metrics(arrays, targets, candidates[0]["model"], request["mode"])
        if manifest["score"] != score or manifest["total_comparison"] != comparison or manifest["prediction_invariants"] != validate_prediction_bundle(arrays):
            raise ValueError("Exact saved-prediction reevaluation changed")
        return manifest
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise ValueError(f"Existing total-search output differs from this request or has changed artifacts; use a new output directory: {exc}") from exc


def search_and_export(candidate_dirs, paper_config, output_root, mode="pooled", starts=12, sweeps=8, pair_trials=300, search_seed=0) -> dict[str, Any]:
    """Search already provenance-validated complete runs of one recurrent family.

    All homogeneous cohorts are evaluated; ``starts`` caps coordinate-refined
    seeds (up to four best homogeneous, then seeded mixed); ``pair_trials`` is a
    global proposal limit. Search state contains one source index per DMA,
    used for all 24h/168h outputs. Ranking is (max total tolerance ratio, mean
    total tolerance ratio, deterministic source-index choice); no DMA target gap
    participates. A matching immutable cached export is independently rechecked.
    """
    _validate_budget(starts, sweeps, pair_trials, search_seed)
    if mode not in METRIC_MODES:
        raise ValueError("Select one supported metric convention for the entire experiment.")
    candidates = _LEGACY._load_candidates(list(candidate_dirs))
    _validate_scalers(candidates)
    paper_config, output_root = Path(paper_config).resolve(), Path(output_root).resolve()
    if any(output_root == c["path"] or output_root in c["path"].parents or c["path"] in output_root.parents for c in candidates):
        raise ValueError("Total-search output must be disjoint from every source candidate directory.")
    model = candidates[0]["model"]
    paper_sha256 = _sha(paper_config)
    targets = _total_targets(paper_config, model)
    if _sha(paper_config) != paper_sha256:
        raise ValueError("Published target evidence changed while reading targets.")
    request = {"version": 1, "algorithm": "TOTAL_ONLY_MULTISTART_COORDINATE_AND_PAIRED_DMA_SEARCH",
               "algorithm_sha256": _sha(Path(__file__)), "legacy_loader_sha256": _sha(Path(_SPEC.origin)),
               "metric_implementation_sha256": _sha(ROOT / "src/dma_wdf/data/reproduction_metrics.py"),
               "mode": mode, "model": model, "seed": SEED, "starts": starts, "sweeps": sweeps,
               "pair_trials": pair_trials, "search_seed": search_seed,
               "objective": ["max_total8_tolerance_ratio", "mean_total8_tolerance_ratio", "deterministic_choice"],
               "error_relative_tolerance": RELATIVE_TOLERANCE, "nse_absolute_tolerance": NSE_TOLERANCE,
               "paper_sha256": paper_sha256,
               "candidates": [{"path": str(c["path"]), "files": c["hashes"]} for c in candidates]}
    if output_root.exists():
        cached = _cached(output_root, candidates, targets, request)
        _check_source_hashes(candidates, paper_config, request)
        return cached
    search = _TotalSearch(candidates, targets, mode)
    choice, approximate_score, trace, budget = search.solve(starts, sweeps, pair_trials, search_seed)
    arrays = _assemble(candidates, choice)
    invariant = validate_prediction_bundle(arrays, require_first_day_consistency=True)
    metrics, comparison, score = _exact_metrics(arrays, targets, model, mode)
    if any(not math.isclose(score["total_metrics"][task][metric], approximate_score["total_metrics"][task][metric], rel_tol=1e-11, abs_tol=1e-11)
           for task in TASKS for metric in METRICS):
        raise ValueError("Optimized total metrics disagree with exact reproduction metric reevaluation.")
    trace.append({"phase": "exact_final_reevaluation", "choice": list(choice), **score})
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".que_total_recurrent_", dir=output_root.parent) as temporary:
        stage = Path(temporary) / "cohort"
        for name in ("checkpoints", "source_configs", "source_scalers"):
            (stage / name).mkdir(parents=True)
        selected = []
        for j, k in enumerate(choice):
            candidate, letter = candidates[k], LETTERS[j]
            checkpoint = f"checkpoint_{model}_dma_{letter}.pt"
            copies = ((checkpoint, f"checkpoints/{checkpoint}"), ("resolved_config.yaml", f"source_configs/dma_{letter}.yaml"),
                      ("scaler_audit.json", f"source_scalers/dma_{letter}.json"))
            for source_name, destination in copies:
                shutil.copyfile(candidate["path"] / source_name, stage / destination)
                if _sha(stage / destination) != candidate["hashes"][source_name]:
                    raise ValueError("Source provenance changed while copying checkpoint/config/scaler evidence.")
            individual = {key: value[j] if isinstance(value, list) and len(value) == 10 else value for key, value in candidate["config"]["model"].items()}
            selected.append({"dma": letter, "column_index": j, "source_path": str(candidate["path"]),
                             "checkpoint": copies[0][1], "checkpoint_sha256": candidate["hashes"][checkpoint],
                             "source_config": copies[1][1], "source_config_sha256": candidate["hashes"]["resolved_config.yaml"],
                             "source_scaler_audit": copies[2][1], "source_scaler_audit_sha256": candidate["hashes"]["scaler_audit.json"],
                             "source_npz_sha256": candidate["hashes"]["predictions_common46.npz"],
                             "family_seed": SEED, "per_dma_model_config": individual, "training_config": candidate["config"]["training"],
                             "same_source_for_all_horizons_and_metrics": True})
        np.savez_compressed(stage / "predictions_common46.npz", **arrays)
        _write_tsv(stage / "metrics.tsv", metrics)
        _write_tsv(stage / "paper_gaps.tsv", comparison)
        _write_tsv(stage / "total_comparison.tsv", comparison)
        _write_json(stage / "search_trace.json", trace)
        artifacts = {str(path.relative_to(stage)): _sha(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        manifest = {"status": "completed", "model": model, "seed": SEED, "mode": mode,
                    "selection": "PUBLISHED_TEST_TOTAL_TARGET_GUIDED_INDEPENDENT_NETWORK_COHORT",
                    "interpretation": "TEST_TARGET_GUIDED_NUMERICAL_RECONSTRUCTION_NOT_ORIGINAL_METHOD_RECOVERY",
                    "paper_reproduction_verified": False, "original_method_recovered": False, "held_out_validation": False,
                    "global_optimum_proven": False, "dma_targets_used_in_objective": False,
                    "checkpoint_note": "Exact source checkpoint bytes, including embedded scaler, copied without deserialization or independent re-inference.",
                    "metric_note": "One convention for both horizons; paper total MAE sums DMA MAEs; all eight total targets jointly ranked.",
                    "limitations": ["Bounded heuristic search has no global optimum guarantee.",
                                    "Selection uses published test targets, so numeric tolerance does not establish original-method recovery or held-out generalization.",
                                    "Source training provenance must be fully validated by the caller; byte and prediction-column checks are not independent checkpoint re-inference."],
                    "request": request, "request_sha256": _json_hash(request), "selected_networks": selected,
                    "score": score, "total_comparison": comparison, "budget": budget,
                    "prediction_invariants": invariant, "artifact_sha256": artifacts,
                    "output_root": str(output_root), "metrics_path": str(output_root / "metrics.tsv"),
                    "paper_gaps_path": str(output_root / "paper_gaps.tsv"),
                    "total_comparison_path": str(output_root / "total_comparison.tsv"),
                    "search_trace_path": str(output_root / "search_trace.json")}
        manifest["manifest_sha256"] = _json_hash(manifest)
        _write_json(stage / "manifest.json", manifest)
        _check_source_hashes(candidates, paper_config, request)
        # Exclusive mkdir claims a NEW directory, including against concurrent
        # exporters. Publish completion manifest last; never replace old outputs.
        output_root.mkdir()
        for artifact in sorted(stage.iterdir()):
            if artifact.name != "manifest.json":
                artifact.rename(output_root / artifact.name)
        (stage / "manifest.json").rename(output_root / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True)
    parser.add_argument("--paper-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=METRIC_MODES, default="pooled")
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--pair-trials", type=int, default=300)
    parser.add_argument("--search-seed", type=int, default=0)
    args = parser.parse_args()
    result = search_and_export(args.candidate_dir, args.paper_config, args.output_root, args.mode,
                               args.starts, args.sweeps, args.pair_trials, args.search_seed)
    print(json.dumps({key: result[key] for key in ("status", "model", "mode", "score", "budget", "output_root")}, indent=2))


if __name__ == "__main__":
    main()
