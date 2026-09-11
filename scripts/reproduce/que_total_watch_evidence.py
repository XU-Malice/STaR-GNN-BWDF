#!/usr/bin/env python3
"""Read-only total-eight selection and verified evidence freezing for a live queue.

This helper is deployed outside the running repository. It never trains, edits
queue files, changes metric definitions, unpickles checkpoints, or signals jobs.
An apparent match is tentative until source, data, receipts and raw predictions
have been independently checked. DMA closeness never enters ranking or gates.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any

MODELS = ("gru", "lstm", "msnet", "mscmnet_m", "mscmnet_wm", "mscmnet_w")
MODEL_NAMES = dict(zip(MODELS, ("GRU", "LSTM", "MSNet", "MSCMNet_M", "MSCMNet_WM", "MSCMNet_W")))
METRICS = ("MAE", "MAPE", "RMSE", "NSE")
TOTAL_KEYS = tuple((task, metric) for task in ("24h", "168h") for metric in METRICS)
SEED = 20240604


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def relative(name: str) -> Path:
    value = PurePosixPath(name)
    if not isinstance(name, str) or "\\" in name or value.is_absolute() or not value.parts or ".." in value.parts or str(value) != name:
        raise ValueError(f"Unsafe relative evidence path: {name!r}")
    return Path(*value.parts)


def regular(path: Path) -> None:
    if not stat.S_ISREG(path.lstat().st_mode) or any(p.is_symlink() for p in path.parents):
        raise ValueError(f"Evidence must be a regular file without symlink ancestors: {path}")


def file_digest(path: Path) -> str:
    regular(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    regular(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fingerprints(project: Path, data_dir: Path) -> dict[str, Any]:
    paths = []
    for directory in ("src", "scripts", "configs", "tests"):
        paths += [p for p in (project / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")]
    paths += [project / n for n in ("pyproject.toml", "uv.lock") if (project / n).is_file()]
    source = {str(p.relative_to(project)): file_digest(p) for p in sorted(set(paths))}
    data = {str(p.relative_to(data_dir)): file_digest(p) for p in sorted(data_dir.rglob("*")) if p.is_file()}
    if not source or not data:
        raise ValueError("Empty source or processed-data fingerprint")
    return {"source": source, "data": data, "source_sha256": digest(source), "data_sha256": digest(data)}


class Context:
    def __init__(self, project_root: Path, result_root: Path, error_relative_tolerance: float = .05,
                 nse_absolute_tolerance: float = .01):
        self.project_root = Path(os.path.abspath(project_root))
        self.result_root = Path(os.path.abspath(result_root))
        self.error_relative_tolerance = float(error_relative_tolerance)
        self.nse_absolute_tolerance = float(nse_absolute_tolerance)
        if any(not math.isfinite(x) or x <= 0 for x in (self.error_relative_tolerance, self.nse_absolute_tolerance)):
            raise ValueError("Tolerances must be finite and positive")
        self.manifest = read_json(self.result_root / "manifest.json")
        unsigned = {k: v for k, v in self.manifest.items() if k != "signature"}
        if self.manifest.get("signature") != digest(unsigned):
            raise ValueError("Frozen manifest self-signature mismatch")
        if self.manifest.get("seed") != SEED or self.manifest.get("selection_mode") != "pooled":
            raise ValueError("Only the frozen single-seed pooled protocol is supported")
        self.data_dir = Path(self.manifest["data_dir"])
        self.paper_path = self.project_root / "configs/evaluation/mscmnet_paper_metrics.yaml"
        self._manifest_file_hash = file_digest(self.result_root / "manifest.json")
        # Check the bytes of ALL source files before importing any repo module.
        self.check_fingerprints()
        sys.path.insert(0, str(self.project_root / "src"))
        self.runner = _load_module("que_total_watch_original_runner", self.project_root / "scripts/train/run_que_comprehensive_reconstruction.py")
        import yaml
        self.paper = yaml.safe_load(self.paper_path.read_text())
        self.published = yaml.safe_load((self.project_root / "configs/model/mscmnet_baselines.yaml").read_text())["models"]
        self.evaluation = read_json(self.result_root / "audit_data_protocol/paper_data_statistics.json")["common_evaluation"]
        self._evaluation_file_hash = file_digest(self.result_root / "audit_data_protocol/paper_data_statistics.json")
        self._verified_digest = None
        self._pass_cases: dict[str, dict[str, Any]] = {}

    def check_fingerprints(self) -> None:
        if file_digest(self.result_root / "manifest.json") != self._manifest_file_hash:
            raise ValueError("Frozen manifest changed")
        if _fingerprints(self.project_root, self.data_dir) != self.manifest["signatures"]:
            raise ValueError("Frozen source/data fingerprints changed; automatic action withheld")
        if file_digest(self.paper_path) != self.manifest["paper_sha256"]:
            raise ValueError("Frozen paper reference changed")

    def _plans(self) -> dict[str, dict[str, Any]]:
        cases = list(self.manifest["base_cases"])
        for stage in ("b", "c"):
            path = self.result_root / f"stage_{stage}_manifest.json"
            if not path.exists():
                continue
            value = read_json(path)
            if value.get("parent_manifest_signature") != self.manifest["signature"] or value.get("selection_sha256") != digest({k: v for k, v in value.items() if k != "selection_sha256"}):
                raise ValueError("Frozen adaptive manifest mismatch")
            cases.extend(value["cases"])
        result = {}
        for case in cases:
            if case["model"] not in MODELS or case["seed"] != SEED or case["stage"] not in ("A", "B", "C"):
                raise ValueError("Invalid frozen candidate")
            if case["case"] != f"{case['stage']}_{case['model']}_{self.runner.setting_key(case)[:12]}" or case["case"] in result:
                raise ValueError("Frozen candidate identity mismatch or duplicate")
            result[case["case"]] = case
        return result

    def _run(self, case: dict[str, Any]) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_]+", case["case"]) or case["model"] not in MODELS:
            raise ValueError("Unsafe case identity")
        return self.result_root / "cases" / case["case"] / case["model"] / f"seed_{SEED}"

    def _total_rows(self, model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        values = {}
        for row in rows:
            if row.get("series") != "total":
                continue
            key = (row["task"], row["metric"])
            if key not in TOTAL_KEYS or key in values:
                raise ValueError("Missing, extra or duplicate total metric keys")
            value = float(row["value"])
            target = float(self.paper["tasks"][key[0]][MODEL_NAMES[model]]["total"][key[1]])
            if not math.isfinite(value) or not math.isfinite(target) or (key[1] != "NSE" and target <= 0):
                raise ValueError("Nonfinite metric or invalid reference target")
            if "paper_value" in row and (not math.isfinite(float(row["paper_value"])) or float(row["paper_value"]) != target):
                raise ValueError("Stored paper reference differs from frozen reference")
            difference = value - target
            relative_difference = abs(difference) / abs(target) if target else None
            ratio = abs(difference) / self.nse_absolute_tolerance if key[1] == "NSE" else relative_difference / self.error_relative_tolerance
            values[key] = {"task": key[0], "series": "total", "metric": key[1], "value": value, "paper_value": target,
                           "difference": difference, "absolute_relative_difference": relative_difference,
                           "tolerance_ratio": ratio, "within_tolerance": ratio <= 1}
        if set(values) != set(TOTAL_KEYS):
            raise ValueError("Total-eight metric table incomplete")
        return [values[key] for key in TOTAL_KEYS]

    def _candidate(self, case: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
        ratios = [r["tolerance_ratio"] for r in rows]
        return {"kind": "complete_training", "case": case["case"], "model": case["model"], "run": str(self._run(case)), "settings": copy.deepcopy(case),
                "metrics": rows, "matched_count": sum(r <= 1 for r in ratios), "all8_matched": all(r <= 1 for r in ratios),
                "worst_ratio": max(ratios), "mean_ratio": sum(ratios) / 8,
                "max_error_relative_difference": max(r["absolute_relative_difference"] for r in rows if r["metric"] != "NSE"),
                "max_nse_absolute_difference": max(abs(r["difference"]) for r in rows if r["metric"] == "NSE")}

    def _assembly_manifest(self, run: Path, model: str) -> dict[str, Any]:
        """Check the completed native Stage-D identity and artifact hashes."""
        if model not in ("gru", "lstm") or run != self.result_root / "recurrent_assembled" / model / "pooled":
            raise ValueError("Only native pooled independent recurrent cohorts are accepted")
        manifest = read_json(run / "manifest.json")
        if any(manifest.get(k) != v for k, v in {"status": "completed", "model": model, "seed": SEED, "mode": "pooled",
                                                "selection": "PUBLISHED_TEST_TABLE_GUIDED_INDEPENDENT_NETWORK_COHORT",
                                                "output_root": str(run), "metrics_path": str(run / "metrics.tsv"),
                                                "paper_gaps_path": str(run / "paper_gaps.tsv")}.items()):
            raise ValueError("Stage-D completed identity/protocol mismatch")
        request = manifest["request"]
        request_hash = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if manifest.get("request_sha256") != request_hash:
            raise ValueError("Stage-D request self-hash mismatch")
        algorithm = "scripts/reproduce/assemble_que_recurrent_candidates.py"
        expected = {"version": 1, "algorithm_sha256": self.manifest["signatures"]["source"][algorithm], "model": model, "seed": SEED,
                    "mode": "pooled", "max_sweeps": 5, "error_relative_tolerance": .05, "nse_absolute_tolerance": .01,
                    "paper_sha256": self.manifest["paper_sha256"], "candidates": request.get("candidates")}
        if request != expected:
            raise ValueError("Stage-D request differs from the frozen native assembly protocol")
        entries = request["candidates"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("Stage-D has no source candidates")
        eligible = {str(self._run(c)): c for c in self._pass_cases.values() if c["model"] == model}
        paths = [entry["path"] for entry in entries]
        if paths != sorted(set(paths)) or any(path not in eligible for path in paths):
            raise ValueError("Stage-D source candidate is not a unique frozen-plan PASS result in this queue")
        required = {"predictions_common46.npz", "metrics.tsv", "paper_gaps.tsv", "search_trace.json"}
        required |= {f"checkpoints/checkpoint_{model}_dma_{letter}.pt" for letter in "ABCDEFGHIJ"}
        required |= {f"source_configs/dma_{letter}.yaml" for letter in "ABCDEFGHIJ"}
        artifacts = manifest.get("artifact_sha256", {})
        if set(artifacts) != required or any(file_digest(run / relative(n)) != h for n, h in artifacts.items()):
            raise ValueError("Stage-D artifact keyset or digest mismatch")
        selected = manifest.get("selected_networks", [])
        if len(selected) != 10:
            raise ValueError("Stage-D requires exactly ten intact DMA networks")
        for j, (letter, item) in enumerate(zip("ABCDEFGHIJ", selected)):
            if any(item.get(k) != v for k, v in {"dma": letter, "column_index": j, "family_seed": SEED,
                        "checkpoint": f"checkpoints/checkpoint_{model}_dma_{letter}.pt", "source_config": f"source_configs/dma_{letter}.yaml",
                        "same_source_for_all_horizons_and_metrics": True}.items()) or item.get("source_path") not in paths:
                raise ValueError("Stage-D DMA/source/checkpoint mapping mismatch")
            if item.get("checkpoint_sha256") != artifacts[item["checkpoint"]] or item.get("source_config_sha256") != artifacts[item["source_config"]]:
                raise ValueError("Stage-D selected-network artifact digest mismatch")
        return manifest

    def _assembly_candidate(self, run: Path, model: str, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
        settings = {"case": f"D_{model}_pooled", "model": model, "seed": SEED, "mode": "pooled",
                    "assembly_request_sha256": manifest["request_sha256"], "selected_networks": copy.deepcopy(manifest["selected_networks"])}
        candidate = self._candidate(settings, rows)
        candidate.update(kind="assembled_recurrent", run=str(run))
        return candidate

    def _verify_assembly(self, prior: dict[str, Any], model: str, plans: dict[str, dict[str, Any]]) -> dict[str, Any]:
        import numpy as np
        import yaml
        from dma_wdf.data.reproduction_metrics import validate_prediction_bundle, canonical_forecast_origins, compute_reproduction_metrics
        run = self.result_root / "recurrent_assembled" / model / "pooled"
        manifest = self._assembly_manifest(run, model)
        initial = self._assembly_candidate(run, model, manifest, prior["metrics"])
        if prior["case"] != initial["case"] or prior["run"] != str(run) or prior["settings"] != initial["settings"]:
            raise ValueError("Selected Stage-D cohort identity changed")
        manifest_hash = file_digest(run / "manifest.json")
        selected_paths = {item["source_path"] for item in manifest["selected_networks"]}
        selected_sources = {}
        verified_source_evidence = []
        # The assembly request includes every search source; validate them all,
        # including inputs not ultimately selected, against frozen training plans.
        for entry in manifest["request"]["candidates"]:
            source = Path(entry["path"])
            case_name = source.parent.parent.name
            case = plans[case_name]
            if source != self._run(case) or case != self._pass_cases.get(case_name) or case["model"] != model:
                raise ValueError("Stage-D source escaped the frozen PASS candidates")
            expected = self._expected(case)
            receipt_before = read_json(source / "completion_receipt.json")
            for name in [*receipt_before.get("files", {}), "request_signature.json", "completion_receipt.json"]:
                regular(source / relative(name))
            valid, reason = self.runner.validate_case(source, case, expected)
            if not valid:
                raise ValueError(f"Stage-D source {case_name}: {reason}")
            self._verify_reuse(source, case, expected)
            status = read_json(source / "status.json")
            required = {"status.json", "resolved_config.yaml", "predictions_common46.npz", "scaler_audit.json"}
            required |= {f"checkpoint_{model}_dma_{letter}.pt" for letter in "ABCDEFGHIJ"}
            if set(entry.get("files", {})) != required or any(file_digest(source / relative(n)) != h for n, h in entry["files"].items()):
                raise ValueError("Stage-D request source hashes changed")
            if str(source) in selected_paths:
                with np.load(source / "predictions_common46.npz", allow_pickle=False) as loaded:
                    arrays = {name: loaded[name] for name in loaded.files}
                config = yaml.safe_load((source / "resolved_config.yaml").read_text())
                if config["model"].get("family") != "independent_recurrent" or config["model"].get("cell_type") != model.upper():
                    raise ValueError("Stage-D source is not an independent recurrent family")
                selected_sources[str(source)] = {"arrays": arrays, "config": config, "files": entry["files"]}
                receipt = read_json(source / "completion_receipt.json")
                files = {**receipt["files"], "request_signature.json": file_digest(source / "request_signature.json"),
                         "completion_receipt.json": file_digest(source / "completion_receipt.json")}
                verified_source_evidence.append({"case": case_name, "run": str(source), "files": files})
        with np.load(run / "predictions_common46.npz", allow_pickle=False) as loaded:
            assembled = {name: loaded[name] for name in loaded.files}
        validate_prediction_bundle(assembled)
        if not np.array_equal(canonical_forecast_origins(assembled["forecast_starts"]), canonical_forecast_origins(self.evaluation["forecast_starts"])):
            raise ValueError("Stage-D forecast origins differ from the common source-data audit")
        for task in ("24h", "168h"):
            if self.runner.life.array_digest(assembled[f"y_true_{task}"]) != self.evaluation["truths"][task]["array_sha256"]:
                raise ValueError("Stage-D truth differs from common source data")
        for j, item in enumerate(manifest["selected_networks"]):
            source = selected_sources[item["source_path"]]
            files, config, arrays = source["files"], source["config"], source["arrays"]
            checkpoint_name = Path(item["checkpoint"]).name
            if item["checkpoint_sha256"] != files[checkpoint_name] or item["source_config_sha256"] != files["resolved_config.yaml"] or item["source_npz_sha256"] != files["predictions_common46.npz"]:
                raise ValueError("Stage-D selected checkpoint/configuration/prediction differs from intact source")
            individual = {key: value[j] if isinstance(value, list) and len(value) == 10 else value for key, value in config["model"].items()}
            if item["per_dma_model_config"] != individual or item["training_config"] != config["training"]:
                raise ValueError("Stage-D selected training configuration differs from source")
            for task in ("24h", "168h"):
                if not np.array_equal(assembled[f"y_pred_{task}"][:, :, j], arrays[f"y_pred_{task}"][:, :, j]):
                    raise ValueError("Stage-D predictions are not the exact selected source columns for both horizons")
        rows = self._total_rows(model, self.runner.metric_rows(assembled, "pooled"))
        current = self._assembly_candidate(run, model, manifest, rows)
        if not current["all8_matched"]:
            raise ValueError("Stage-D raw total metrics do not meet all eight thresholds")
        if len(prior["metrics"]) != 8 or any(a["task"] != b["task"] or a["metric"] != b["metric"] or abs(a["value"] - b["value"]) > 5e-5 for a, b in zip(prior["metrics"], rows)):
            raise ValueError("Stage-D selection scores changed during verification")
        # Verify the entire stored metric table for integrity, with no DMA target
        # thresholds. Numeric DMA accuracy plays no part in accepting a cohort.
        with (run / "metrics.tsv").open(newline="") as f:
            stored = list(csv.DictReader(f, delimiter="\t"))
        stored_map = {(r["task"], r["series"], r["metric"]): float(r["value"]) for r in stored}
        recomputed = {(task, r["series"], r["metric"]): r["value"] for task in ("24h", "168h")
                      for r in compute_reproduction_metrics(assembled[f"y_true_{task}"], assembled[f"y_pred_{task}"], mode="pooled")}
        if len(stored) != len(recomputed) or set(stored_map) != set(recomputed) or any(not math.isfinite(v) or abs(v - recomputed[k]) > 5e-5 for k, v in stored_map.items()):
            raise ValueError("Stage-D stored metrics disagree with raw assembled predictions")
        # Check once more after the CPU audit to exclude a replaced assembly.
        if file_digest(run / "manifest.json") != manifest_hash or self._assembly_manifest(run, model) != manifest:
            raise ValueError("Stage-D assembly changed during verification")
        current["verified_files"] = {**manifest["artifact_sha256"], "manifest.json": manifest_hash}
        current["verified_source_evidence"] = verified_source_evidence
        return current

    def scan(self, queue: dict[str, Any]) -> dict[str, Any]:
        plans, choices, exclusions = self._plans(), {m: [] for m in MODELS}, []
        seen = set()
        self._pass_cases = {}
        for record in queue.get("cases", []):
            if not str(record.get("technical_status", "")).startswith("PASS"):
                continue
            name = record.get("case")
            try:
                if name in seen:
                    raise ValueError("Duplicate queue candidate")
                seen.add(name)
                case = plans[name]
                if record.get("model") != case["model"] or record.get("settings") != case or record.get("exit_code") != 0:
                    raise ValueError("Queue candidate differs from frozen plan")
                self._pass_cases[name] = case
                run = self._run(case)
                for filename in ("paper_gaps.tsv", "metrics.csv", "completion_receipt.json"):
                    regular(run / filename)
                with (run / "paper_gaps.tsv").open(newline="") as f:
                    gaps = self._total_rows(case["model"], [r for r in csv.DictReader(f, delimiter="\t") if r.get("mode") == "pooled"])
                with (run / "metrics.csv").open(newline="") as f:
                    stored = self._total_rows(case["model"], list(csv.DictReader(f)))
                if any(abs(a["value"] - b["value"]) > 5e-5 for a, b in zip(gaps, stored)):
                    raise ValueError("Paper gaps disagree with stored raw-metric summary")
                # Scores are recomputed; old within_tolerance flags are ignored.
                choices[case["model"]].append(self._candidate(case, stored))
            except (KeyError, TypeError, ValueError, OSError) as exc:
                exclusions.append({"case": name, "reason": f"{type(exc).__name__}: {exc}"})
        for model in ("gru", "lstm"):
            run = self.result_root / "recurrent_assembled" / model / "pooled"
            if not (run / "manifest.json").exists():
                continue
            try:
                manifest = self._assembly_manifest(run, model)
                with (run / "paper_gaps.tsv").open(newline="") as f:
                    gaps = self._total_rows(model, list(csv.DictReader(f, delimiter="\t")))
                with (run / "metrics.tsv").open(newline="") as f:
                    rows = self._total_rows(model, list(csv.DictReader(f, delimiter="\t")))
                if any(abs(a["value"] - b["value"]) > 5e-5 for a, b in zip(gaps, rows)):
                    raise ValueError("Assembly summaries disagree")
                choices[model].append(self._assembly_candidate(run, model, manifest, rows))
            except (KeyError, TypeError, ValueError, OSError) as exc:
                exclusions.append({"case": f"D_{model}_pooled", "reason": f"{type(exc).__name__}: {exc}"})
        best = {m: min(v, key=lambda c: (c["worst_ratio"], c["mean_ratio"], c["case"])) if v else None for m, v in choices.items()}
        matched = [m for m, c in best.items() if c and c["all8_matched"]]
        return {"version": 1, "mode": "pooled", "seed": SEED, "models": best, "matched_models": matched,
                "all_matched": len(matched) == len(MODELS), "verification": "tentative", "verified_all_matched": False,
                "candidate_counts": {m: len(v) for m, v in choices.items()}, "exclusions": exclusions,
                "criteria": {"error_relative_tolerance": self.error_relative_tolerance, "nse_absolute_tolerance": self.nse_absolute_tolerance,
                             "total_metrics_required": 48, "dma_closeness_required": False},
                "scope": "all_complete_training_candidates_and_completed_pooled_Stage-D_independent_network_cohorts",
                "test_target_feedback": True, "paper_reproduction_verified": False}

    def _recompute_evaluation(self) -> dict[str, Any]:
        module = _load_module("que_total_watch_source_data_audit", self.project_root / "scripts/reproduce/audit_que_data_protocol.py")
        frames, _, bounds = module.load_paper_data(data_dir=self.data_dir, split_config_path=self.project_root / "configs/data/paper_split.yaml", require_audit=True)
        return module.audit_frames(frames, bounds)[0]["common_evaluation"]

    def _expected(self, case: dict[str, Any]) -> dict[str, Any]:
        return {"signature": digest({"manifest": self.manifest["signature"], "settings": self.runner.setting_key(case)}),
                "case": case, "model_config": self.runner.expected_model_config(self.published, case), "evaluation": self.evaluation}

    def _verify_reuse(self, run: Path, case: dict[str, Any], expected: dict[str, Any]) -> None:
        path = run / "reused_source_provenance.json"
        if not path.exists():
            return
        value = read_json(path)
        helper = _load_module("que_total_watch_original_reuse", self.project_root / "scripts/train/reuse_que_completed_runs.py")
        original = value["source_manifest"]
        helper._check_manifest(original, self.runner.life, "Reused")
        unchanged = helper._check_training_compatibility(original, self.manifest)
        if value.get("unchanged_training_sources") != unchanged or value.get("training_performed_during_import") is not False or value.get("destination_manifest_signature") != self.manifest["signature"]:
            raise ValueError("Reused-source provenance mismatch")
        old_expected = {**expected, "signature": digest({"manifest": original["signature"], "settings": self.runner.setting_key(case)})}
        if value.get("source_request") != old_expected:
            raise ValueError("Reused source request mismatch")
        receipt = value["source_completion_receipt"]
        if receipt.get("request_sha256") != digest(old_expected):
            raise ValueError("Reused source receipt request mismatch")
        names = set(self.runner.evidence_hashes(run, read_json(run / "status.json"))) - {"reused_source_provenance.json"}
        if set(receipt.get("files", {})) != names or any(file_digest(run / relative(n)) != h for n, h in receipt["files"].items()):
            raise ValueError("Original training evidence changed during reuse")

    def verify_selected(self, report: dict[str, Any]) -> dict[str, Any]:
        self._verified_digest = None
        self.check_fingerprints()
        if file_digest(self.result_root / "audit_data_protocol/paper_data_statistics.json") != self._evaluation_file_hash or self._recompute_evaluation() != self.evaluation:
            raise ValueError("Evaluation does not agree with freshly audited source data")
        plans = self._plans()
        if set(report.get("models", {})) != set(MODELS) or any(report["models"][m] is None for m in MODELS):
            raise ValueError("Six complete model candidates are required")
        import numpy as np
        verified = copy.deepcopy(report)
        for model in MODELS:
            prior = report["models"][model]
            if prior.get("kind") == "assembled_recurrent":
                verified["models"][model] = self._verify_assembly(prior, model, plans)
                continue
            case = plans[prior["case"]]
            if model != case["model"] or prior["settings"] != case or prior["run"] != str(self._run(case)):
                raise ValueError("Selected candidate differs from frozen identity")
            run, expected = self._run(case), self._expected(case)
            regular(run / "completion_receipt.json")
            receipt = read_json(run / "completion_receipt.json")
            for name in [*receipt.get("files", {}), "request_signature.json", "completion_receipt.json"]:
                regular(run / relative(name))
            valid, reason = self.runner.validate_case(run, case, expected)
            if not valid:
                raise ValueError(f"{case['case']}: {reason}")
            self._verify_reuse(run, case, expected)
            with np.load(run / "predictions_common46.npz", allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
            rows = self._total_rows(model, self.runner.metric_rows(arrays, "pooled"))
            current = self._candidate(case, rows)
            if not current["all8_matched"]:
                raise ValueError(f"{case['case']}: raw total metrics do not meet all eight thresholds")
            if len(prior["metrics"]) != 8 or any(a["task"] != b["task"] or a["metric"] != b["metric"] or abs(a["value"] - b["value"]) > 5e-5 for a, b in zip(prior["metrics"], rows)):
                raise ValueError("Selection scores changed during verification")
            current["verified_files"] = {**receipt["files"], "request_signature.json": file_digest(run / "request_signature.json"),
                                         "completion_receipt.json": file_digest(run / "completion_receipt.json")}
            verified["models"][model] = current
        self.check_fingerprints()
        verified.update(all_matched=True, matched_models=list(MODELS), verification="raw_predictions_and_receipts_verified", verified_all_matched=True)
        self._verified_digest = digest(verified)
        return verified

    def export_to(self, target_dir: Path, report: dict[str, Any]) -> Path:
        if self._verified_digest is None or digest(report) != self._verified_digest or report.get("verified_all_matched") is not True:
            raise ValueError("Only the untouched fully verified report can be frozen")
        self.check_fingerprints()
        target = Path(os.path.abspath(target_dir))
        if target == self.project_root or target.is_relative_to(self.result_root) or any(target.is_relative_to(self.project_root / x) for x in ("src", "scripts", "configs", "tests", "data")):
            raise ValueError("Freeze destination must be outside queue/source/data trees")
        if target.exists() or any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError("Freeze destination already exists or traverses a symlink")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.freeze-", dir=target.parent))
        hashes = {}
        def copy_file(source: Path, name: str, expected: str) -> None:
            destination = temporary / relative(name)
            if file_digest(source) != expected:
                raise ValueError(f"Evidence changed before freeze: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if file_digest(destination) != expected:
                raise ValueError(f"Frozen evidence digest mismatch: {name}")
            hashes[name] = expected
        try:
            for model, candidate in report["models"].items():
                for name, expected in candidate["verified_files"].items():
                    copy_file(Path(candidate["run"]) / relative(name), f"selected_results/{model}/{name}", expected)
                for source in candidate.get("verified_source_evidence", []):
                    for name, expected in source["files"].items():
                        copy_file(Path(source["run"]) / relative(name), f"selected_source_evidence/{source['case']}/{name}", expected)
            for name, expected in self.manifest["signatures"]["source"].items():
                copy_file(self.project_root / relative(name), f"source_snapshot/{name}", expected)
            copy_file(self.result_root / "manifest.json", "reference/queue_manifest.json", self._manifest_file_hash)
            copy_file(self.paper_path, "reference/paper_metrics.yaml", self.manifest["paper_sha256"])
            copy_file(self.result_root / "audit_data_protocol/paper_data_statistics.json", "reference/paper_data_statistics.json", self._evaluation_file_hash)
            selection = temporary / "selection_report.json"
            selection.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            hashes["selection_report.json"] = file_digest(selection)
            archive_manifest = {"version": 1, "numerical_total_match_verified": True, "models": list(MODELS), "mode": "pooled", "seed": SEED,
                                "criteria": report["criteria"], "test_target_feedback": True, "paper_reproduction_verified": False,
                                "weights_included": True, "files": hashes, "source_data_included": False,
                                "source_data_fingerprints": self.manifest["signatures"]["data"],
                                "recipe": "Complete-training models use selected_results/<model>/resolved_config.yaml and frozen checkpoints for both horizons. Stage-D recurrent cohorts use selected_results/<model>/manifest.json selected_networks (one whole network per DMA, same source both horizons), checkpoints/ and source_configs/; selected_source_evidence retains original training receipts/configuration/predictions/weights. source_snapshot contains exact training/metric implementation. Source data must match reference/queue_manifest.json."}
            (temporary / "freeze_manifest.json").write_text(json.dumps(archive_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            self.check_fingerprints()
            if target.exists():
                raise ValueError("Freeze destination appeared concurrently")
            os.rename(temporary, target)
            return target
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
