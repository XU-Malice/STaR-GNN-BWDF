#!/usr/bin/env python3
"""Bounded, single-seed reconstruction search with complete-result provenance.

The adaptive phase explicitly uses published test-table distances and is therefore
retrospective reconstruction search, never an unbiased generalization estimate.
The primary pooled convention is fixed before execution; origin_mean is reported
separately and is never mixed across cells or selected as a closer convention.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tarfile
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Both imports remain CPU-only and work with python -S --dry-run.
_spec = importlib.util.spec_from_file_location("que_lifecycle", Path(__file__).with_name("run_que_protocol_audit.py"))
assert _spec and _spec.loader
life = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(life)
MODELS = life.MODELS
SEED = life.SEED
MODEL_NAMES = dict(zip(MODELS, ("GRU", "LSTM", "MSNet", "MSCMNet_M", "MSCMNet_WM", "MSCMNet_W")))
MAX_CASES = 415
PRIMARY_MODE = "pooled"
LAYOUTS = ("per_day_vectors", "full_history_flat", "per_day_flat", "conv2d_day_hour")
TRAINING_KEYS = ("model", "seed", "normalization", "optimizer", "batch_size", "recurrent_layout", "scaler_fit_scope", "demand_scaling", "cam_temporal_layout", "cam_attention_scaling", "cam_channel_sizes", "correction_layout", "best_epoch_scale", "learning_rate_scale", "max_epochs", "loss", "correction_mode", "zero_init_correction", "share_weight", "joint_weight_decay", "independent_weight_decay")


def setting_key(case: dict[str, Any]) -> str:
    return life.digest({key: case[key] for key in TRAINING_KEYS})


def make_case(model: str, stage: str = "A", **settings: Any) -> dict[str, Any]:
    case = {"model": model, "seed": SEED, "normalization": "zscore", "optimizer": "adamw", "batch_size": 8,
            "recurrent_layout": "hourly", "scaler_fit_scope": "windows", "demand_scaling": "per_dma",
            "cam_temporal_layout": "per_day_vectors", "cam_attention_scaling": "sqrt_dim", "cam_channel_sizes": [16, 16, 1], "correction_layout": "global_flat", "best_epoch_scale": 1.0, "learning_rate_scale": 1.0, "max_epochs": None, "loss": "mse",
            "correction_mode": "direct", "zero_init_correction": False, "share_weight": 0.0, "joint_weight_decay": None, "independent_weight_decay": None}
    case.update(settings)
    case["stage"] = stage
    case["case"] = f"{stage}_{model}_{setting_key(case)[:12]}"
    changed_published = case["best_epoch_scale"] != 1 or case["learning_rate_scale"] != 1 or case["max_epochs"] is not None or case["loss"] != "mse" or case["joint_weight_decay"] is not None or case["independent_weight_decay"] is not None
    extended = case["correction_mode"] != "direct" or case["share_weight"] != 0 or case["zero_init_correction"]
    case["interpretation"] = "EXPLORATORY_PARAMETERS_OR_ARCHITECTURE" if changed_published or extended else "UNPUBLISHED_IMPLEMENTATION_HYPOTHESIS"
    return case


def stage_a_cases() -> list[dict[str, Any]]:
    base = [make_case(model, normalization=norm, optimizer=opt, batch_size=batch)
            for model, norm, opt, batch in itertools.product(MODELS, ("zscore", "minmax"), ("adam", "adamw"), (4, 8, 16))]
    base += [make_case(model, normalization=norm, batch_size=batch, correction_mode="residual", zero_init_correction=True)
             for model, norm, batch in itertools.product(("mscmnet_m", "mscmnet_w"), ("zscore", "minmax"), (1, 4, 8))]
    base += [make_case("mscmnet_wm", normalization=norm, batch_size=batch, share_weight=.05)
             for norm, batch in itertools.product(("zscore", "minmax"), (4, 8))]
    historical = [
        make_case("gru", batch_size=16, best_epoch_scale=.35),
        make_case("gru", max_epochs=100),
        make_case("lstm", max_epochs=100),
        make_case("lstm", optimizer="adam", independent_weight_decay=0.0),
        make_case("msnet", loss="huber"),
        make_case("mscmnet_m", batch_size=4, loss="huber", correction_mode="residual", zero_init_correction=True),
        make_case("mscmnet_w", batch_size=1, max_epochs=50, correction_mode="residual", zero_init_correction=True),
    ]
    for case in historical:
        case["historical_anchor"] = True
    return historical + base


def rank_key(record: dict[str, Any], mode: str = PRIMARY_MODE) -> tuple[float, float, float, str]:
    score = record["scores"][mode]
    return (score["balanced_distance"], score["q95"], score["worst_ratio"], record["case"])


def choose_diverse(records: list[dict[str, Any]], model: str, count: int = 2) -> list[dict[str, Any]]:
    eligible = [r for r in records if r["model"] == model and r.get("technical_status", "").startswith("PASS") and r.get("scores")]
    rankings = [sorted(eligible, key=lambda r: rank_key(r, mode)) for mode in (PRIMARY_MODE, "origin_mean")]
    candidates, seen = [], set()
    for pair in zip(*rankings):
        for record in pair:
            if record["case"] not in seen:
                seen.add(record["case"])
                candidates.append(record)
    # Cover materially different interpretations before filling by exact rank.
    selected, signatures = [], set()
    for record in candidates:
        case = record["settings"]
        key = (case["normalization"], case["optimizer"]) if model in ("gru", "lstm") else (case["normalization"], case["correction_mode"], case["share_weight"])
        if key not in signatures:
            selected.append(record)
            signatures.add(key)
        if len(selected) == count:
            return selected
    for record in candidates:
        if record not in selected:
            selected.append(record)
        if len(selected) == count:
            break
    return selected


def adaptive_cases(records: list[dict[str, Any]], existing: list[dict[str, Any]], stage: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage not in ("B", "C"):
        raise ValueError("Adaptive stage must be B or C")
    seen = {setting_key(case) for case in existing}
    result, selection = [], {}
    def add(parent: dict[str, Any], **updates: Any) -> None:
        settings = {key: parent["settings"][key] for key in TRAINING_KEYS if key != "model"}
        settings.update(updates)
        case = make_case(parent["model"], stage=stage, **settings)
        key = setting_key(case)
        if key not in seen:
            case["parent_case"] = parent["case"]
            seen.add(key)
            result.append(case)
    for model in MODELS:
        parents = choose_diverse(records, model)
        selection[model] = [{"case": p["case"], "ranks": {mode: rank_key(p, mode) for mode in (PRIMARY_MODE, "origin_mean")}, "settings": p["settings"]} for p in parents]
        if stage == "B":
            for parent in parents:
                for epoch_scale, lr_scale in itertools.product((.35, .5, 1.0, 2.0, 4.0), (.3, 1.0, 3.0)):
                    if (epoch_scale, lr_scale) != (1.0, 1.0):
                        add(parent, best_epoch_scale=epoch_scale, learning_rate_scale=lr_scale, max_epochs=None)
            continue
        for parent in parents:
            if model in ("gru", "lstm"):
                add(parent, best_epoch_scale=.1, max_epochs=None)
                for decay in (0.0, .001):
                    add(parent, independent_weight_decay=decay)
            for loss in ("mae", "huber"):
                add(parent, loss=loss)
            for batch in (1, 2, 32):
                add(parent, batch_size=batch)
            if model not in ("gru", "lstm"):
                for scope, demand in (("train_rows", "per_dma"), ("windows", "shared"), ("train_rows", "shared")):
                    add(parent, scaler_fit_scope=scope, demand_scaling=demand)
                for decay in (0.0, .001):
                    add(parent, joint_weight_decay=decay)
                opposite = "hourwise_shared" if parent["settings"]["correction_layout"] == "global_flat" else "global_flat"
                add(parent, correction_layout=opposite)
            if model in ("mscmnet_m", "mscmnet_w"):
                for epochs in (50, 100):
                    add(parent, max_epochs=epochs, best_epoch_scale=1.0)
            if model in ("mscmnet_wm", "mscmnet_w"):
                for weight in (0.0, .01, .05, .1):
                    add(parent, share_weight=weight)
        if model not in ("gru", "lstm") and parents:
            for width in ([1, 1, 1], [8, 8, 1], [32, 32, 1]):
                add(parents[0], cam_channel_sizes=width)
    if len(existing) + len(result) > MAX_CASES:
        raise AssertionError("Finite experiment budget exceeded")
    return result, selection


# Kept as a convenient import for tests and offline plan review.
def stage_b_cases(records: list[dict[str, Any]], existing: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return adaptive_cases(records, existing, "B")


def command_for(case: dict[str, Any], args: argparse.Namespace, output_root: Path) -> list[str]:
    command = [sys.executable, "-u", "scripts/train/train_temporal_baselines.py", "--model", case["model"], "--seed", str(SEED),
               "--normalization", case["normalization"], "--optimizer", case["optimizer"], "--batch-size", str(case["batch_size"]),
               "--loss", case["loss"], "--train-stride-hours", "24", "--learning-rate-scale", str(case["learning_rate_scale"]), "--best-epoch-scale", str(case["best_epoch_scale"]),
               "--scaler-fit-scope", case["scaler_fit_scope"], "--demand-scaling", case["demand_scaling"],
               "--data-dir", str(args.data_dir), "--output-root", str(output_root), "--device", args.device]
    if args.device == "cpu":
        command.append("--allow-cpu")
    if case["model"] in ("gru", "lstm"):
        command += ["--recurrent-layout", case["recurrent_layout"]]
    else:
        command += ["--cam-attention-update", "replace", "--cam-attention-scaling", case["cam_attention_scaling"],
                    "--cam-temporal-layout", case["cam_temporal_layout"], "--cam-channel-sizes", ",".join(map(str, case["cam_channel_sizes"])), "--correction-layout", case["correction_layout"]]
    if case["model"].startswith("mscmnet_"):
        command += ["--correction-mode", case["correction_mode"]]
        if case["zero_init_correction"]:
            command.append("--zero-init-correction")
    if case["model"] in ("mscmnet_w", "mscmnet_wm"):
        command += ["--fc2-share-supervision-weight", str(case["share_weight"])]
    if case["max_epochs"] is not None:
        command += ["--max-epochs", str(case["max_epochs"])]
    if case["joint_weight_decay"] is not None:
        command += ["--joint-weight-decay", str(case["joint_weight_decay"])]
    if case["independent_weight_decay"] is not None:
        command += ["--recurrent-weight-decay", str(case["independent_weight_decay"])]
    return command


def expected_model_config(published: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(published[case["model"]]))
    if case["model"] not in ("gru", "lstm"):
        result["correction_layout"] = case["correction_layout"]
    if case["model"].startswith("mscmnet_"):
        result.update(correction_mode=case["correction_mode"], zero_init_correction=case["zero_init_correction"])
    if case["model"] in ("mscmnet_w", "mscmnet_wm"):
        result["fc2"]["share_supervision_weight"] = case["share_weight"]
    return result


def actual_epochs(published: int, case: dict[str, Any]) -> int:
    return int(case["max_epochs"]) if case["max_epochs"] is not None else max(1, round(published * case["best_epoch_scale"]))


def evidence_hashes(run: Path, status: dict[str, Any]) -> dict[str, str]:
    names = ["status.json", "resolved_config.yaml", "predictions_common46.npz", "metrics.csv", "loss_curve.csv", "scaler_audit.json", *status["checkpoint_files"]]
    if any(Path(name).name != name for name in names):
        raise ValueError("Unsafe evidence filename")
    return {name: life.file_digest(run / name) for name in names}


def validate_case(run: Path, case: dict[str, Any], expected: dict[str, Any], *, require_receipt: bool = True) -> tuple[bool, str]:
    import numpy as np
    import yaml
    from dma_wdf.data.reproduction_metrics import validate_prediction_bundle, canonical_forecast_origins
    try:
        request = json.loads((run / "request_signature.json").read_text())
        if request != expected:
            return False, "request_signature_mismatch"
        status = json.loads((run / "status.json").read_text())
        if status.get("status") != "completed" or status.get("model") != case["model"] or status.get("seed") != SEED:
            return False, "incomplete_or_wrong_model_seed"
        if status.get("single_frozen_checkpoint_for_24h_and_168h") is not True:
            return False, "not_single_frozen_checkpoint"
        count = 10 if case["model"] in ("gru", "lstm") else 1
        checkpoints = status.get("checkpoint_files", [])
        if len(checkpoints) != count or len(set(checkpoints)) != count:
            return False, "wrong_checkpoint_count"
        for name in checkpoints:
            if Path(name).name != name or not (run / name).is_file() or not (run / name).stat().st_size:
                return False, "missing_or_invalid_checkpoint"
        config = yaml.safe_load((run / "resolved_config.yaml").read_text())
        if config["model"] != expected["model_config"]:
            return False, "published_model_config_mismatch"
        training = config["training"]
        checks = {key: case[key] for key in ("normalization", "optimizer", "batch_size", "scaler_fit_scope", "demand_scaling")}
        checks.update({key: case[key] for key in ("loss", "best_epoch_scale", "learning_rate_scale")})
        if training.get("joint_weight_decay_override") != case["joint_weight_decay"]:
            return False, "joint_weight_decay_mismatch"
        if training.get("independent_weight_decay_override") != case["independent_weight_decay"]:
            return False, "unexpected_independent_decay_override"
        if case["model"] in ("gru", "lstm"):
            checks["recurrent_layout"] = case["recurrent_layout"]
        for key, value in checks.items():
            if training.get(key) != value:
                return False, f"resolved_config_mismatch:{key}"
        if config.get("seed") != SEED or config.get("train_stride_hours") != 24 or config.get("max_epochs_override") != case["max_epochs"] or config.get("max_train_batches") is not None:
            return False, "training_override_or_seed_mismatch"
        if case["model"] not in ("gru", "lstm"):
            for key, value in {"attention_update": "replace", "attention_scaling": case["cam_attention_scaling"], "temporal_layout": case["cam_temporal_layout"], "channel_sizes": case["cam_channel_sizes"]}.items():
                if config["cam"].get(key) != value:
                    return False, f"cam_mismatch:{key}"
        scaler = json.loads((run / "scaler_audit.json").read_text())
        for key in ("scaler_fit_scope", "demand_scaling", "normalization"):
            if scaler.get(key) != case[key]:
                return False, f"scaler_audit_mismatch:{key}"
        if scaler.get("test_values_used_for_fit") is not False:
            return False, "scaler_train_only_provenance_missing"
        # State dictionaries must exist, be finite, and have positive scales.
        def scaler_states(value: Any) -> int:
            found = 0
            if isinstance(value, dict):
                for scale_key in ("std", "value_range", "scale"):
                    if scale_key in value:
                        scale = np.asarray(value[scale_key], dtype=float)
                        if not np.isfinite(scale).all() or (scale <= 0).any():
                            raise ValueError("Invalid fitted scaler scale")
                        found += int(scale.size > 0)
                for entry in value.values():
                    found += scaler_states(entry)
            elif isinstance(value, list):
                for entry in value:
                    found += scaler_states(entry)
            elif isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite scaler audit")
            return found
        if not scaler_states(scaler):
            return False, "missing_fitted_scaler_states"
        with np.load(run / "predictions_common46.npz", allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        validate_prediction_bundle(arrays)
        if not np.array_equal(canonical_forecast_origins(arrays["forecast_starts"]), canonical_forecast_origins(expected["evaluation"]["forecast_starts"])):
            return False, "origins_do_not_match_audited_data"
        for horizon in (24, 168):
            if life.array_digest(arrays[f"y_true_{horizon}h"]) != expected["evaluation"]["truths"][f"{horizon}h"]["array_sha256"]:
                return False, "truth_does_not_match_audited_data"
        with (run / "loss_curve.csv").open(newline="") as stream:
            losses = list(csv.DictReader(stream))
        if not losses or any(not math.isfinite(float(row["train_loss"])) for row in losses):
            return False, "missing_or_nonfinite_loss_curve"
        if case["model"] in ("gru", "lstm"):
            if {r.get("dma") for r in losses} != set("ABCDEFGHIJ"):
                return False, "loss_curve_missing_dma"
            for letter, epochs in zip("ABCDEFGHIJ", config["model"]["best_epochs"]):
                if [int(r["epoch"]) for r in losses if r["dma"] == letter] != list(range(1, actual_epochs(int(epochs), case) + 1)):
                    return False, "loss_curve_wrong_published_epochs"
        elif [int(r["epoch"]) for r in losses] != list(range(1, actual_epochs(int(config["model"]["best_epoch"]), case) + 1)):
            return False, "loss_curve_wrong_published_epochs"
        verify_stored_metrics(run, arrays)
        if require_receipt:
            receipt = json.loads((run / "completion_receipt.json").read_text())
            if receipt.get("request_sha256") != life.digest(expected) or receipt.get("files") != evidence_hashes(run, status):
                return False, "completed_evidence_changed"
        return True, "validated_artifacts"
    except Exception as exc:
        return False, f"invalid_artifact:{type(exc).__name__}:{exc}"


def metric_rows(arrays: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    from dma_wdf.data.reproduction_metrics import compute_reproduction_metrics
    result = []
    for task in ("24h", "168h"):
        rows = compute_reproduction_metrics(arrays[f"y_true_{task}"], arrays[f"y_pred_{task}"], mode=mode)
        result.extend({**row, "task": task} for row in rows if row["series"] != "physical_total")
    return result


def verify_stored_metrics(run: Path, arrays: dict[str, Any]) -> None:
    with (run / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = {(r["task"], r["series"], r["metric"]): float(r["value"]) for r in rows}
    expected = {(r["task"], r["series"], r["metric"]): r["value"] for r in metric_rows(arrays, PRIMARY_MODE)}
    if len(rows) != 88 or set(values) != set(expected):
        raise ValueError("Metric keyset differs from complete 88-cell table")
    for key, value in values.items():
        if not math.isfinite(value) or not math.isfinite(expected[key]) or abs(value - expected[key]) > 5e-5:
            raise ValueError(f"Stored metric differs from raw prediction: {key}")


def score_case(run: Path, case: dict[str, Any], paper: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    with np.load(run / "predictions_common46.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    result, all_gaps = {}, []
    for mode in (PRIMARY_MODE, "origin_mean"):
        total, dma, gaps = [], [], []
        for row in metric_rows(arrays, mode):
            target = float(paper["tasks"][row["task"]][MODEL_NAMES[case["model"]]][row["series"]][row["metric"]])
            difference = float(row["value"]) - target
            absolute_relative = abs(difference) / abs(target) if target else (0.0 if difference == 0 else float("inf"))
            ratio = abs(difference) / .01 if row["metric"] == "NSE" else absolute_relative / .05
            if not math.isfinite(ratio):
                raise ValueError(f"Undefined metric in complete table: {mode} {row}")
            (total if row["series"] == "total" else dma).append(ratio)
            gaps.append({**row, "mode": mode, "paper_value": target, "difference": difference, "absolute_relative_difference": absolute_relative, "tolerance_ratio": ratio, "within_tolerance": ratio <= 1})
        if len(total) != 8 or len(dma) != 80:
            raise ValueError("Incomplete scoring table")
        result[mode] = {"balanced_distance": .5 * float(np.mean(total)) + .5 * float(np.mean(dma)), "total_mean_ratio": float(np.mean(total)),
                        "dma_mean_ratio": float(np.mean(dma)), "q95": float(np.quantile(total + dma, .95)), "worst_ratio": max(total + dma),
                        "total8_passed": sum(q <= 1 for q in total), "dma80_passed": sum(q <= 1 for q in dma), "all88_within_tolerance": max(total + dma) <= 1}
        all_gaps.extend(gaps)
    life.atomic_json(run / "reconstruction_scores.json", {"scores": result, "selection_mode": PRIMARY_MODE, "test_target_feedback": True, "reproduction_verified": False})
    write_tsv(run / "paper_gaps.tsv", all_gaps)
    return result


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def refresh_status(queue: dict[str, Any], result_root: Path) -> None:
    records = queue["cases"]
    queue["passed_cases"] = sum(r.get("technical_status", "").startswith("PASS") for r in records)
    queue["failed_cases"] = sum(r.get("technical_status") == "FAIL" for r in records)
    queue["finished_cases"] = queue["passed_cases"] + queue["failed_cases"]
    queue["updated_utc"] = life.utc_now()
    life.atomic_json(result_root / "queue_status.json", queue)
    rows = []
    for record in records:
        row = {key: record.get(key) for key in ("case", "model", "stage", "technical_status", "exit_code", "validation", "elapsed_seconds")}
        row.update({key: record["settings"][key] for key in TRAINING_KEYS if key not in ("model", "cam_channel_sizes")})
        row["cam_channel_sizes"] = ",".join(map(str, record["settings"]["cam_channel_sizes"]))
        row.update(record.get("scores", {}).get(PRIMARY_MODE, {}))
        rows.append(row)
    write_tsv(result_root / "case_status.tsv", rows)
    best = {}
    for model in MODELS:
        eligible = [r for r in records if r["model"] == model and r.get("technical_status", "").startswith("PASS") and r.get("scores")]
        best[model] = {}
        for mode in (PRIMARY_MODE, "origin_mean"):
            recipes = {}
            for criterion, key in (("best_balanced_88", lambda r: rank_key(r, mode)),
                                   ("best_total_8", lambda r: (r["scores"][mode]["total_mean_ratio"], *rank_key(r, mode))),
                                   ("best_minimax_88", lambda r: (r["scores"][mode]["worst_ratio"], *rank_key(r, mode)))):
                ranked = sorted(eligible, key=key)
                recipes[criterion] = [{"case": r["case"], "settings": r["settings"], "scores": r["scores"][mode], "interpretation": r["settings"]["interpretation"]} for r in ranked[:3]]
            best[model][mode] = recipes
    life.atomic_json(result_root / "closest_complete_configurations.json", {"primary_mode": PRIMARY_MODE, "test_target_feedback": True, "paper_reproduction_verified": False, "models": best})


def print_status(result_root: Path, log_root: Path) -> int:
    path = result_root / "queue_status.json"
    if not path.is_file():
        print(f"尚未生成状态：{path}")
        return 0
    try:
        queue = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"读取状态失败：{exc}")
        return 1
    total = queue.get("case_count", 95)
    done = queue.get("finished_cases", 0)
    print(f"队列状态：{queue.get('status')}；阶段：{queue.get('current_stage', '预检')}")
    print(f"已结束：{done}/{total} 个已生成任务；成功：{queue.get('passed_cases', 0)}；失败：{queue.get('failed_cases', 0)}")
    print(f"已生成任务剩余（含正在运行）：{max(0, total-done)}；整个有限计划上限：{queue.get('maximum_case_count', MAX_CASES)}")
    if not queue.get("stage_c_selected"):
        print("后续候选尚待自动生成，当前分母不是最终任务数。")
    active = queue.get("active_case")
    print(f"当前任务：{active or '无'}；状态更新时间：{queue.get('updated_utc')}")
    times = [float(r["elapsed_seconds"]) for r in queue.get("cases", []) if r.get("elapsed_seconds") and r.get("technical_status", "").startswith("PASS")]
    if len(times) >= 3 and total > done:
        estimate = sum(times) / len(times) * (total - done) / 3600
        print(f"按已完成任务均耗时粗估，已生成剩余任务约{estimate:.1f}小时；不同模型/轮次耗时有较大差异。")
    if queue.get("error"):
        print("错误：", queue["error"])
    if active and re.fullmatch(r"[A-Za-z0-9_]+", active):
        log = log_root / f"{active}.log"
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 8192))
                lines = stream.read().decode("utf-8", errors="replace").splitlines()[-6:]
            print("最后日志：")
            print("\n".join(lines))
    if queue.get("status") in ("completed", "completed_with_failures"):
        print("训练队列已结束；数值接近程度见 closest_complete_configurations.json，技术成功不等于论文复现。")
    return 0


def run_recurrent_assembly(queue: dict[str, Any], result_root: Path, paper_config: Path) -> None:
    script = PROJECT_ROOT / "scripts/reproduce/assemble_que_recurrent_candidates.py"
    spec = importlib.util.spec_from_file_location("que_recurrent_assembly", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    summaries = {}
    for model in ("gru", "lstm"):
        candidates = [result_root / "cases" / r["case"] / model / f"seed_{SEED}" for r in queue["cases"]
                      if r["model"] == model and r.get("technical_status", "").startswith("PASS") and r.get("scores")]
        summaries[model] = {}
        for mode in (PRIMARY_MODE, "origin_mean"):
            print(f"阶段D：CPU选择 {model} 每DMA的完整网络配置，统一口径={mode}；两个时域共用同一网络。", flush=True)
            output = result_root / "recurrent_assembled" / model / mode
            report = module.assemble_recurrent_candidates(candidate_dirs=candidates, paper_config=paper_config, output_root=output, mode=mode)
            summaries[model][mode] = {"output_root": str(output), "status": report.get("status", "completed"), "score": report.get("score"), "selection_type": "per_DMA_config_selection", "reproduction_verified": False}
    queue["recurrent_assembly"] = summaries


def make_bundle(root: Path, result_root: Path, log_root: Path) -> Path:
    bundle = root.parent / f"{result_root.name}_compact.tar.gz"
    temporary = bundle.with_name(bundle.name + ".tmp")
    size = 0
    with tarfile.open(temporary, "w:gz") as archive:
        for directory in (result_root, log_root):
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                    continue
                if any(".backup-" in part or "__pycache__" == part for part in path.parts) or path.suffix in (".pt", ".pyc", ".tmp"):
                    continue
                if path.suffix == ".npz" and path.name != "predictions_common46.npz":
                    continue
                size += path.stat().st_size
                if size > 8 * 1024**3:
                    raise RuntimeError("Compact evidence exceeds 8 GiB; originals preserved")
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    os.replace(temporary, bundle)
    checksum = life.file_digest(bundle)
    bundle.with_name(bundle.name + ".sha256").write_text(f"{checksum}  {bundle.name}\n")
    print(f"结果包：{bundle}\nSHA256：{checksum}", flush=True)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="que_comprehensive_reconstruction_20260906")
    parser.add_argument("--gpu-id", default="6")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--minimum-free-mib", type=int, default=8192)
    parser.add_argument("--time-budget-hours", "--budget-hours", dest="time_budget_hours", type=float, default=96.0, help="Per invocation hours; 0 runs the complete finite queue without a wall-clock cutoff")
    parser.add_argument("--status", action="store_true", help="Read existing progress only; no training, writes or GPU access")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed/data_build")
    parser.add_argument("--paper-config", type=Path, default=PROJECT_ROOT / "configs/evaluation/mscmnet_paper_metrics.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.run_tag) or not re.fullmatch(r"[0-9]+", args.gpu_id):
        parser.error("run-tag must be a simple directory name and gpu-id a numeric physical index")
    if not math.isfinite(args.time_budget_hours) or args.time_budget_hours < 0 or args.minimum_free_mib < 1:
        parser.error("time budget must be nonnegative and minimum free GPU memory positive")
    base_cases = stage_a_cases()
    if args.status:
        return print_status(PROJECT_ROOT / "results" / args.run_tag, PROJECT_ROOT / "logs" / args.run_tag)
    if args.dry_run:
        print(json.dumps({"stage_a_count": len(base_cases), "maximum_case_count": MAX_CASES, "seed": SEED, "stage_b_upper_bound": 168, "stage_c_upper_bound": 152,
                          "selection_mode": PRIMARY_MODE, "selection_policy": "retrospective_union_of_two_whole_table_rankings_with_structural_diversity", "published_epochs_lr_wd_fixed": False, "stage_a_historical_anchors": 7, "stage_a_reference_grid": 72,
                          "time_budget_hours": args.time_budget_hours, "cases": base_cases}, indent=2))
        return 0
    root = PROJECT_ROOT
    sys.path.insert(0, str(root / "src"))
    import yaml
    args.data_dir = args.data_dir.resolve()
    args.paper_config = args.paper_config.resolve()
    result_root, log_root = root / "results" / args.run_tag, root / "logs" / args.run_tag
    log_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    lock = (root / "logs" / f"que_gpu_{args.gpu_id if args.device != 'cpu' else 'cpu'}.lock").open("a+")
    owns_lock = False
    owns_run = False
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id if args.device != "cpu" else ""
    env["PYTHONUNBUFFERED"] = "1"
    supervisor = life.ChildSupervisor(root, env)
    handlers = {sig: signal.signal(sig, supervisor.signal_handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    queue = {"status": "preflight", "started_utc": life.utc_now(), "case_count": len(base_cases), "maximum_case_count": MAX_CASES,
             "stage_b_selected": False, "seed": SEED, "primary_mode": PRIMARY_MODE, "test_target_feedback": True,
             "paper_reproduction_verified": False, "technical_success": False, "cases": []}
    exit_code, started = 0, time.monotonic()
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another queue owns this project/GPU lock; nothing stopped") from exc
        owns_lock = True
        signatures = life.fingerprints(root, args.data_dir)
        published = yaml.safe_load((root / "configs/model/mscmnet_baselines.yaml").read_text())["models"]
        paper = yaml.safe_load(args.paper_config.read_text())
        manifest = {"version": 1, "base_cases": base_cases, "maximum_cases": MAX_CASES, "signatures": signatures, "paper_sha256": life.file_digest(args.paper_config),
                    "device": args.device, "data_dir": str(args.data_dir), "selection_mode": PRIMARY_MODE, "seed": SEED}
        manifest["signature"] = life.digest(manifest)
        manifest_path = result_root / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError("Existing run has different source/data/plan. Preserved; choose a new run-tag.")
            previous = result_root / "queue_status.json"
            if previous.exists():
                old = json.loads(previous.read_text())
                queue["cases"] = old.get("cases", [])
                queue["started_utc"] = old.get("started_utc", queue["started_utc"])
                queue["resumed_utc"] = life.utc_now()
        else:
            if any(path.name not in ("manifest.json",) for path in result_root.iterdir()):
                raise RuntimeError("Output directory is nonempty without matching manifest; preserved")
            life.atomic_json(manifest_path, manifest)
            for name in signatures["source"]:
                destination = result_root / "source_snapshot" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / name, destination)
        owns_run = True
        refresh_status(queue, result_root)
        print("阶段0：数据协议审计与本机CPU预检；随后单GPU顺序执行，固定一个seed。", flush=True)
        command = [sys.executable, "-u", "scripts/reproduce/audit_que_data_protocol.py", "--data-dir", str(args.data_dir), "--split-config", str(root / "configs/data/paper_split.yaml"), "--output-root", str(result_root / "audit_data_protocol")]
        if supervisor.run(command, log_root / "audit_data_protocol.log"):
            raise RuntimeError("Source-data audit failed; no training started")
        evaluation = json.loads((result_root / "audit_data_protocol/paper_data_statistics.json").read_text())["common_evaluation"]
        preflight = [*life.PREFLIGHT_TESTS, "tests/test_que_comprehensive_runner.py", "tests/test_que_comprehensive_integration.py", "tests/test_que_recurrent_assembly.py"]
        if supervisor.run([sys.executable, "-m", "pytest", "-q", *preflight], log_root / "preflight_tests.log"):
            raise RuntimeError("Server CPU tests failed; no training started")
        by_name = {record["case"]: record for record in queue["cases"]}
        time_limit = started + args.time_budget_hours * 3600 if args.time_budget_hours else float("inf")
        stop_budget = False
        all_cases = list(base_cases)
        for stage in ("A", "B", "C"):
            if stage != "A":
                selection_path = result_root / f"stage_{stage.lower()}_manifest.json"
                if selection_path.exists():
                    selection = json.loads(selection_path.read_text())
                    expected_selection_hash = life.digest({k: v for k, v in selection.items() if k != "selection_sha256"})
                    if selection["parent_manifest_signature"] != manifest["signature"] or selection.get("selection_sha256") != expected_selection_hash:
                        raise RuntimeError("Frozen adaptive manifest changed; outputs preserved")
                    followups = selection["cases"]
                    if any(c["stage"] != stage or c["seed"] != SEED or c["model"] not in MODELS for c in followups):
                        raise RuntimeError("Invalid frozen adaptive cases")
                    if len(all_cases) + len(followups) > MAX_CASES or len({setting_key(c) for c in all_cases + followups}) != len(all_cases) + len(followups):
                        raise RuntimeError("Frozen adaptive cases violate finite or unique-plan constraints")
                else:
                    followups, parents = adaptive_cases([r for r in queue["cases"] if r["stage"] < stage], all_cases, stage)
                    if any(not parents[model] for model in MODELS):
                        raise RuntimeError("A model has no valid Stage A candidate; adaptive selection withheld")
                    selection = {"parent_manifest_signature": manifest["signature"], "cases": followups, "parents": parents,
                                 "selection_mode": PRIMARY_MODE, "test_target_feedback": True, "created_utc": life.utc_now()}
                    selection["selection_sha256"] = life.digest(selection)
                    life.atomic_json(selection_path, selection)
                all_cases.extend(followups)
                queue.update(case_count=len(all_cases), **{f"stage_{stage.lower()}_selected": True})
            current = [case for case in all_cases if case["stage"] == stage]
            for case in current:
                if life.fingerprints(root, args.data_dir) != signatures or life.file_digest(args.paper_config) != manifest["paper_sha256"]:
                    raise RuntimeError("Source/data/reference changed during queue; remaining work stopped and outputs preserved")
                case_root = result_root / "cases" / case["case"]
                run = case_root / case["model"] / f"seed_{SEED}"
                request = {"signature": life.digest({"manifest": manifest["signature"], "settings": setting_key(case)}), "case": case,
                           "model_config": expected_model_config(published, case), "evaluation": evaluation}
                valid, reason = validate_case(run, case, request)
                if not valid and time.monotonic() >= time_limit:
                    queue.update(status="paused_time_budget", active_case=None)
                    stop_budget = True
                    break
                record = by_name.get(case["case"])
                if record is None:
                    record = {"case": case["case"], "model": case["model"], "stage": stage, "settings": case}
                    by_name[case["case"]] = record
                    queue["cases"].append(record)
                record.update(technical_status="PASS(existing)" if valid else "running", validation=reason)
                queue.update(status="running", active_case=case["case"], current_stage=stage)
                refresh_status(queue, result_root)
                print(f"开始任务 {queue['finished_cases']}/{queue['case_count']} (上限{MAX_CASES}): {case['case']} {json.dumps({k: case[k] for k in TRAINING_KEYS}, ensure_ascii=False)}", flush=True)
                if not valid:
                    if args.device != "cpu":
                        life.atomic_json(log_root / f"{case['case']}_gpu.json", life.gpu_preflight(args.gpu_id, args.minimum_free_mib))
                    command = command_for(case, args, case_root)
                    if run.exists():
                        command.append("--overwrite")
                    record["started_utc"] = life.utc_now()
                    tick = time.monotonic()
                    rc = supervisor.run(command, log_root / f"{case['case']}.log")
                    record.update(exit_code=rc, elapsed_seconds=time.monotonic() - tick, finished_utc=life.utc_now())
                    if run.is_dir():
                        life.atomic_json(run / "request_signature.json", request)
                    valid, reason = validate_case(run, case, request, require_receipt=False)
                    if rc == 0 and valid:
                        status = json.loads((run / "status.json").read_text())
                        life.atomic_json(run / "completion_receipt.json", {"request_sha256": life.digest(request), "files": evidence_hashes(run, status)})
                    record.update(technical_status="PASS" if rc == 0 and valid else "FAIL", validation=reason)
                if record["technical_status"].startswith("PASS"):
                    try:
                        record["scores"] = score_case(run, case, paper)
                    except Exception as exc:
                        record.update(technical_status="FAIL", validation=f"scoring_failed:{exc}")
                        record.pop("scores", None)
                refresh_status(queue, result_root)
            if stop_budget:
                break
        if not stop_budget:
            if life.fingerprints(root, args.data_dir) != signatures or life.file_digest(args.paper_config) != manifest["paper_sha256"]:
                raise RuntimeError("Source/data/reference changed before final audit; outputs preserved")
            queue.update(status="recurrent_assembly", active_case=None)
            refresh_status(queue, result_root)
            run_recurrent_assembly(queue, result_root, args.paper_config)
            queue.update(status="final_metric_audit", active_case=None)
            refresh_status(queue, result_root)
            command = [sys.executable, "-u", "scripts/reproduce/audit_que_saved_predictions.py", "--results-root", str(result_root / "cases"), "--paper-config", str(args.paper_config), "--output-root", str(result_root / "audit_all"), "--strict-first-day", "--allow-invalid-evidence"]
            audit_code = supervisor.run(command, log_root / "audit_all.log")
            receipt = json.loads((result_root / "audit_all/audit_summary.json").read_text())
            valid_count = queue["passed_cases"]
            audit_ok = life.strict_audit_passes(receipt, valid_count)
            queue.update(independent_metric_audit=receipt, audit_exit_code=audit_code, technical_success=not queue["failed_cases"] and audit_code == 0 and audit_ok)
            queue["status"] = "completed" if queue["technical_success"] else "completed_with_failures"
            exit_code = 0 if queue["technical_success"] else 1
    except life.InterruptedRun as exc:
        queue.update(status="interrupted", error=str(exc))
        exit_code = 128 + exc.signum
    except Exception as exc:
        queue.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(queue["error"], file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        supervisor.terminate_owned()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        queue.update(exit_code=exit_code, finished_utc=life.utc_now())
        if owns_lock and owns_run:
            refresh_status(queue, result_root)
            try:
                make_bundle(root, result_root, log_root)
            except Exception as exc:
                exit_code = 1
                queue.update(exit_code=1, technical_success=False, bundle_status="failed", bundle_error=str(exc))
                refresh_status(queue, result_root)
                print(f"打包失败，原始结果仍保留：{exc}", file=sys.stderr)
        lock.close()
        print(json.dumps({k: v for k, v in queue.items() if k != "cases"}, ensure_ascii=False, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
