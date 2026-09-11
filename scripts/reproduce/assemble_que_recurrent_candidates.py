#!/usr/bin/env python3
"""Select a transparent cohort of ten independently trained GRU/LSTM networks.

Every DMA contributes one intact network and its entire 168 h rollout; its 24 h
prediction comes from the same source. Selection uses the published test table,
so this is retrospective numerical reconstruction, not held-out validation.
Joint-model columns, prediction averaging/calibration, and seed selection are
never allowed. Metric conventions are separate complete experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

LETTERS = tuple("ABCDEFGHIJ")
TASKS = ("24h", "168h")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _load_candidates(candidate_dirs: list[Path]) -> list[dict[str, Any]]:
    if not candidate_dirs:
        raise ValueError("At least one explicitly validated candidate directory is required.")
    paths = sorted({Path(value).resolve() for value in candidate_dirs}, key=str)
    candidates: list[dict[str, Any]] = []
    for path in paths:
        names = ["status.json", "resolved_config.yaml", "predictions_common46.npz"]
        hashes = {name: _sha(path / name) for name in names}
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        config = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
        model = status.get("model")
        if model not in ("gru", "lstm"):
            raise ValueError("Only independent GRU/LSTM networks may be assembled; joint columns are forbidden.")
        if status.get("status") != "completed":
            raise ValueError(f"Candidate is not completed: {path}")
        seed = status.get("seed")
        if type(seed) is not int or type(config.get("seed")) is not int or config.get("seed") != seed:
            raise ValueError("Candidate status/config must identify the same integer family seed.")
        if status.get("single_frozen_checkpoint_for_24h_and_168h") is not True:
            raise ValueError("Candidate must declare a single frozen checkpoint per DMA for both horizons.")
        model_config = config.get("model", {})
        if model_config.get("family") != "independent_recurrent" or model_config.get("cell_type") != model.upper():
            raise ValueError("Resolved configuration does not identify the independent recurrent family.")
        if not isinstance(config.get("protocol"), dict) or not isinstance(config.get("training"), dict):
            raise ValueError("Candidate must retain resolved data protocol and training configuration.")
        expected_checkpoints = [f"checkpoint_{model}_dma_{letter}.pt" for letter in LETTERS]
        if sorted(status.get("checkpoint_files", [])) != sorted(expected_checkpoints):
            raise ValueError("Candidate must declare exactly ten independent DMA checkpoint files.")
        for name in expected_checkpoints:
            checkpoint = path / name
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                raise ValueError(f"Missing or empty independent checkpoint: {checkpoint}")
            hashes[name] = _sha(checkpoint)
        for optional in ("scaler_audit.json",):
            if (path / optional).is_file():
                hashes[optional] = _sha(path / optional)
        with np.load(path / "predictions_common46.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        invariants = validate_prediction_bundle(arrays, require_first_day_consistency=True)
        if candidates:
            first = candidates[0]
            if model != first["model"] or seed != first["seed"]:
                raise ValueError("Every candidate must use the same recurrent model and family seed.")
            if config["protocol"] != first["config"]["protocol"]:
                raise ValueError("Candidates have different resolved data protocols.")
            if invariants["origins_sha256"] != first["invariants"]["origins_sha256"]:
                raise ValueError("Candidate forecast origins differ from the common audited origins.")
            for task in TASKS:
                if array_sha256(arrays[f"y_true_{task}"]) != first["invariants"]["array_hashes"][f"y_true_{task}"]:
                    raise ValueError("Candidate truth arrays/dtypes must be identical; test labels cannot change.")
        candidates.append({"path": path, "model": model, "seed": seed, "config": config,
                           "status": status, "arrays": arrays, "hashes": hashes,
                           "invariants": invariants})
    return candidates


def _targets(paper_config: Path, model: str) -> dict[tuple[str, str, str], float]:
    paper = yaml.safe_load(paper_config.read_text(encoding="utf-8"))["tasks"]
    targets = {(task, series, metric): float(paper[task][model.upper()][series][metric])
               for task in TASKS for series in (*LETTERS, "total") for metric in METRICS}
    if len(targets) != 88 or not all(math.isfinite(value) for value in targets.values()):
        raise ValueError("A complete finite 88-cell model target table is required.")
    return targets


def _ratio(actual: float, target: float, metric: str, relative: float, nse: float) -> float:
    denominator = nse if metric == "NSE" else max(abs(target), 1e-12) * relative
    return abs(actual - target) / denominator


class _Search:
    """Cache independent DMA metrics; recompute coupled totals at each step."""

    def __init__(self, candidates: list[dict[str, Any]], targets: dict[tuple[str, str, str], float],
                 mode: str, relative: float, nse: float):
        self.candidates, self.targets, self.mode = candidates, targets, mode
        self.relative, self.nse = relative, nse
        self.true_sum = {task: candidates[0]["arrays"][f"y_true_{task}"].astype(np.float64).sum(2) for task in TASKS}
        self.pred = {task: np.stack([c["arrays"][f"y_pred_{task}"].astype(np.float64) for c in candidates]) for task in TASKS}
        self.values = np.empty((len(candidates), 10, 2, 4), dtype=np.float64)
        self.ratios = np.empty_like(self.values)
        for index, candidate in enumerate(candidates):
            for t, task in enumerate(TASKS):
                rows = compute_reproduction_metrics(candidate["arrays"][f"y_true_{task}"], self.pred[task][index], mode=mode)
                for row in rows:
                    if row["series"] not in LETTERS:
                        continue
                    j, m = LETTERS.index(row["series"]), METRICS.index(row["metric"])
                    value = float(row["value"])
                    if not math.isfinite(value):
                        raise ValueError("Undefined DMA metric cannot be silently dropped during cohort selection.")
                    self.values[index, j, t, m] = value
                    self.ratios[index, j, t, m] = _ratio(value, targets[(task, row["series"], row["metric"])], row["metric"], relative, nse)

    def evaluate(self, choice: tuple[int, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        dma_ratios = self.ratios[np.asarray(choice), np.arange(10)]
        dma_values = self.values[np.asarray(choice), np.arange(10)]
        total_ratios, total_values = [], {}
        for t, task in enumerate(TASKS):
            summed = np.stack([self.pred[task][selected, :, :, j] for j, selected in enumerate(choice)], axis=2).sum(2)
            rows = compute_reproduction_metrics(self.true_sum[task][:, :, None], summed[:, :, None], ("aggregate",), mode=self.mode)
            values = {row["metric"]: float(row["value"]) for row in rows if row["series"] == "total"}
            # The published total MAE sums DMA MAEs; it is not summed-flow MAE.
            values["MAE"] = float(dma_values[:, t, METRICS.index("MAE")].sum())
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("Undefined total metric cannot be silently dropped during cohort selection.")
            total_values[task] = values
            total_ratios.extend(_ratio(values[metric], self.targets[(task, "total", metric)], metric, self.relative, self.nse) for metric in METRICS)
        all_ratios = np.concatenate([np.asarray(total_ratios), dma_ratios.ravel()])
        score = {"balanced_distance": float(.5 * np.mean(total_ratios) + .5 * np.mean(dma_ratios)),
                 "worst_ratio": float(np.max(all_ratios)), "q95": float(np.quantile(all_ratios, .95)),
                 "total8_passed": int(np.sum(np.asarray(total_ratios) <= 1)),
                 "dma80_passed": int(np.sum(dma_ratios <= 1)), "all88_passed": bool(np.all(all_ratios <= 1)),
                 "total_metrics": total_values}
        key = (score["balanced_distance"], score["worst_ratio"], score["q95"], choice)
        return key, score

    def solve(self, max_sweeps: int) -> tuple[tuple[int, ...], dict[str, Any], list[dict[str, Any]]]:
        homogeneous = [(tuple([i] * 10), self.evaluate(tuple([i] * 10))[0]) for i in range(len(self.candidates))]
        best_whole = min(homogeneous, key=lambda item: item[1])[0]
        per_dma = tuple(min(range(len(self.candidates)), key=lambda k: (
            float(self.ratios[k, j].mean()), float(self.ratios[k, j].max()), k)) for j in range(10))
        starts = list(dict.fromkeys((best_whole, per_dma)))
        best_choice, best_key = best_whole, self.evaluate(best_whole)[0]
        trace: list[dict[str, Any]] = []
        for start_number, initial in enumerate(starts):
            choice = initial
            key, score = self.evaluate(choice)
            trace.append({"start": start_number, "sweep": 0, "choice": list(choice), **score})
            for sweep in range(1, max_sweeps + 1):
                previous = choice
                for j in range(10):
                    current_best, current_key = choice, key
                    for candidate in range(len(self.candidates)):
                        alternative = choice[:j] + (candidate,) + choice[j + 1:]
                        alternative_key, _ = self.evaluate(alternative)
                        if alternative_key < current_key:
                            current_best, current_key = alternative, alternative_key
                    choice, key = current_best, current_key
                _, score = self.evaluate(choice)
                trace.append({"start": start_number, "sweep": sweep, "choice": list(choice), **score})
                if choice == previous:
                    break
            if key < best_key:
                best_choice, best_key = choice, key
        return best_choice, self.evaluate(best_choice)[1], trace


def assemble_recurrent_candidates(
    *, candidate_dirs: list[Path], paper_config: Path, output_root: Path,
    mode: str = "pooled", max_sweeps: int = 5,
    error_relative_tolerance: float = .05, nse_absolute_tolerance: float = .01,
) -> dict[str, Any]:
    """Assemble explicit, already-runner-validated candidates of one model/seed.

    Source checkpoint bytes are copied and hashed, never deserialized. Thus the
    manifest establishes file provenance, not an independent re-inference proof.
    The caller must validate source training provenance before passing paths.
    """
    if mode not in METRIC_MODES or not 1 <= max_sweeps <= 5:
        raise ValueError("Use one complete supported metric mode and 1–5 coordinate sweeps.")
    if not 0 < error_relative_tolerance < 1 or not 0 < nse_absolute_tolerance < 1:
        raise ValueError("Both numeric tolerances must be finite and strictly between zero and one.")
    candidates = _load_candidates(candidate_dirs)
    paper_config, output_root = Path(paper_config).resolve(), Path(output_root).resolve()
    if any(output_root == c["path"] or output_root in c["path"].parents or c["path"] in output_root.parents for c in candidates):
        raise ValueError("Assembly output must be disjoint from all source candidate directories.")
    model, seed = candidates[0]["model"], candidates[0]["seed"]
    targets = _targets(paper_config, model)
    request = {"version": 1, "algorithm_sha256": _sha(Path(__file__)),
               "mode": mode, "model": model, "seed": seed, "max_sweeps": max_sweeps,
               "error_relative_tolerance": error_relative_tolerance, "nse_absolute_tolerance": nse_absolute_tolerance,
               "paper_sha256": _sha(paper_config),
               "candidates": [{"path": str(c["path"]), "files": c["hashes"]} for c in candidates]}
    request_hash = _json_hash(request)
    if output_root.exists():
        manifest_path = output_root / "manifest.json"
        if manifest_path.is_file():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = old.get("artifact_sha256", {})
            if old.get("request_sha256") == request_hash and artifacts and all(
                (output_root / name).is_file() and _sha(output_root / name) == digest for name, digest in artifacts.items()
            ):
                return old
        raise ValueError("Existing assembly differs from this request or has changed artifacts; use a new output directory.")
    search = _Search(candidates, targets, mode, error_relative_tolerance, nse_absolute_tolerance)
    choice, score, trace = search.solve(max_sweeps)
    reference = candidates[0]["arrays"]
    assembled = {"dma_letters": reference["dma_letters"].copy(), "forecast_starts": reference["forecast_starts"].copy()}
    for task in TASKS:
        assembled[f"y_true_{task}"] = reference[f"y_true_{task}"].copy()
        assembled[f"y_pred_{task}"] = np.stack([candidates[k]["arrays"][f"y_pred_{task}"][:, :, j]
                                                for j, k in enumerate(choice)], axis=2)
    invariant = validate_prediction_bundle(assembled, require_first_day_consistency=True)
    metrics, gaps = [], []
    for task in TASKS:
        for values in compute_reproduction_metrics(assembled[f"y_true_{task}"], assembled[f"y_pred_{task}"], mode=mode):
            row = {"model": model, "seed": seed, "task": task, **values}
            metrics.append(row)
            target = targets.get((task, row["series"], row["metric"]))
            if target is None:
                continue
            difference = float(row["value"]) - target
            ratio = _ratio(float(row["value"]), target, row["metric"], error_relative_tolerance, nse_absolute_tolerance)
            gaps.append({**row, "paper_value": target, "signed_difference": difference,
                         "absolute_difference": abs(difference), "absolute_relative_difference": abs(difference) / max(abs(target), 1e-12),
                         "tolerance_ratio": ratio, "within_numeric_tolerance": ratio <= 1})
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recurrent_assembly_", dir=output_root.parent) as temporary:
        stage = Path(temporary) / "cohort"
        stage.mkdir()
        (stage / "checkpoints").mkdir()
        (stage / "source_configs").mkdir()
        selected = []
        for j, k in enumerate(choice):
            c, letter = candidates[k], LETTERS[j]
            name = f"checkpoint_{model}_dma_{letter}.pt"
            destination = stage / "checkpoints" / name
            shutil.copyfile(c["path"] / name, destination)
            if _sha(destination) != c["hashes"][name]:
                raise ValueError("Source checkpoint changed while copying the assembly.")
            config_name = f"source_configs/dma_{letter}.yaml"
            shutil.copyfile(c["path"] / "resolved_config.yaml", stage / config_name)
            individual = {key: (value[j] if isinstance(value, list) and len(value) == 10 else value)
                          for key, value in c["config"]["model"].items()}
            selected.append({"dma": letter, "column_index": j, "source_path": str(c["path"]),
                             "checkpoint": f"checkpoints/{name}", "checkpoint_sha256": c["hashes"][name],
                             "source_config": config_name, "source_config_sha256": c["hashes"]["resolved_config.yaml"],
                             "source_npz_sha256": c["hashes"]["predictions_common46.npz"],
                             "family_seed": seed, "per_dma_model_config": individual,
                             "training_config": c["config"]["training"],
                             "same_source_for_all_horizons_and_metrics": True})
        # Refuse a provenance race in any source, including unselected candidates.
        for c in candidates:
            for name, digest in c["hashes"].items():
                if _sha(c["path"] / name) != digest:
                    raise ValueError(f"Candidate evidence changed during assembly: {c['path'] / name}")
        if _sha(paper_config) != request["paper_sha256"]:
            raise ValueError("Paper targets changed during assembly.")
        np.savez_compressed(stage / "predictions_common46.npz", **assembled)
        _write_tsv(stage / "metrics.tsv", metrics)
        _write_tsv(stage / "paper_gaps.tsv", gaps)
        _write_json(stage / "search_trace.json", trace)
        artifact_hashes = {str(path.relative_to(stage)): _sha(path) for path in sorted(stage.rglob("*")) if path.is_file()}
        manifest = {"status": "completed", "model": model, "seed": seed, "mode": mode,
                    "selection": "PUBLISHED_TEST_TABLE_GUIDED_INDEPENDENT_NETWORK_COHORT",
                    "interpretation": "RETROSPECTIVE_NUMERICAL_RECONSTRUCTION_NOT_HELD_OUT_VALIDATION",
                    "paper_reproduction_verified": False, "global_optimum_proven": False,
                    "checkpoint_note": "Exact source bytes copied without deserialization or independent re-inference.",
                    "metric_note": "One complete convention for all 88 cells; no per-cell convention mixing.",
                    "request_sha256": request_hash, "request": request, "selected_networks": selected,
                    "score": score, "prediction_invariants": invariant, "artifact_sha256": artifact_hashes,
                    "output_root": str(output_root), "metrics_path": str(output_root / "metrics.tsv"),
                    "paper_gaps_path": str(output_root / "paper_gaps.tsv")}
        _write_json(stage / "manifest.json", manifest)
        stage.rename(output_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True,
                        help="One explicitly runner-validated completed recurrent run; may repeat.")
    parser.add_argument("--paper-config", type=Path, default=ROOT / "configs/evaluation/mscmnet_paper_metrics.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=METRIC_MODES, default="pooled")
    parser.add_argument("--max-sweeps", type=int, default=5)
    args = parser.parse_args()
    result = assemble_recurrent_candidates(candidate_dirs=args.candidate_dir, paper_config=args.paper_config,
        output_root=args.output_root, mode=args.mode, max_sweeps=args.max_sweeps)
    print(json.dumps({key: result[key] for key in ("status", "model", "mode", "score", "output_root")}, indent=2))


if __name__ == "__main__":
    main()
