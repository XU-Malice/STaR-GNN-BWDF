#!/usr/bin/env python
"""验证哈希、checkpoint 元数据、协议数量、冻结指标与论文层级。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from paper_release_lib import (
    METRICS,
    PROJECT_ROOT,
    STAR_VARIANTS,
    TASKS,
    better,
    check_test_summary,
    load_protocol,
    read_metrics,
    release_model_dir,
    sha256,
)


def _verify_checksums(root: Path) -> int:
    checksum_file = root / "CHECKSUMS.sha256"
    if not checksum_file.is_file():
        raise FileNotFoundError(checksum_file)
    count = 0
    listed: set[str] = set()
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in listed:
            raise ValueError(f"Duplicate checksum entry: {relative}")
        listed.add(relative)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
        count += 1
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if actual_files != listed:
        missing = sorted(listed - actual_files)
        extra = sorted(actual_files - listed)
        raise ValueError(
            f"Frozen file-set mismatch; missing={missing}, extra={extra}"
        )
    return count


def _close(
    left: float,
    right: float,
    tolerance: float,
    *,
    relative_tolerance: float = 0.0,
) -> bool:
    """比较两个浮点结果。

    冻结文件中的指标应当逐值一致，因此默认只使用很小的绝对容差；
    重新执行 GPU 推理时，可显式传入相对容差，以容纳不同 CUDA/cuDNN
    内核带来的末位浮点差异。checkpoint 哈希、样本数和协议字段不使用
    浮点容差，仍然要求完全一致。
    """
    return math.isclose(
        float(left),
        float(right),
        rel_tol=relative_tolerance,
        abs_tol=tolerance,
    )


def _audit_checkpoint(
    checkpoint_path: Path,
    *,
    family: str,
    model: str,
    task: str,
) -> None:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_horizon = 24 if task == "24h" else 168
    if int(payload.get("horizon", -1)) != expected_horizon:
        raise ValueError(f"Checkpoint horizon mismatch: {checkpoint_path}")
    if int(payload.get("seed", -1)) != 0:
        raise ValueError(f"Checkpoint seed mismatch: {checkpoint_path}")
    if family == "star_gnn":
        if payload.get("model_name") != "star_dcrnn":
            raise ValueError(f"Not a STaR-GNN checkpoint: {checkpoint_path}")
        expected_variant = STAR_VARIANTS[model]
        if payload.get("variant") != expected_variant:
            raise ValueError(f"Variant mismatch: {checkpoint_path}")
        config = payload["resolved_config"]
        training = config["training"]
        paper = load_protocol()["paper_setting"]
        checks = {
            "learning_rate": (training["learning_rate"], paper["learning_rate"]),
            "weight_decay": (training["weight_decay"], paper["weight_decay"]),
            "cl_decay_steps": (
                training["scheduled_sampling"]["cl_decay_steps"],
                paper["cl_decay_steps"],
            ),
            "state_loss_weight": (
                config["innovation"]["dssn_sasr"]["state_loss_weight"],
                paper["state_loss_weight"],
            ),
            "max_epochs": (training["max_epochs"], paper["max_epochs"]),
        }
        for name, (actual, expected) in checks.items():
            if not _close(float(actual), float(expected), 1.0e-12):
                raise ValueError(
                    f"Frozen setting mismatch for {name}: {actual} != {expected}"
                )
    elif payload.get("model_name") != model:
        raise ValueError(f"Baseline model mismatch: {checkpoint_path}")


def _collect_release_metrics(root: Path) -> dict[tuple[str, str], dict[str, float]]:
    values: dict[tuple[str, str], dict[str, float]] = {}
    for label in STAR_VARIANTS:
        for task in TASKS:
            run = release_model_dir(root, "star_gnn", label, task)
            checkpoint = run / "checkpoint_best.pt"
            _audit_checkpoint(checkpoint, family="star_gnn", model=label, task=task)
            check_test_summary(run / "evaluation/test_summary.json", checkpoint)
            values[(task, label)] = read_metrics(
                run / "evaluation/metrics_aggregate_total_common_46.csv"
            )
    # Base/backbone 已经是唯一的 DCRNN；这里只额外核验 STGCN。
    for model in ("stgcn",):
        for task in TASKS:
            run = release_model_dir(root, "baselines", model, task)
            checkpoint = run / "checkpoint_best.pt"
            _audit_checkpoint(checkpoint, family="baselines", model=model, task=task)
            check_test_summary(run / "evaluation/test_summary.json", checkpoint)
    return values


def _verify_expected(
    values: dict[tuple[str, str], dict[str, float]],
    tolerance: float,
) -> None:
    expected = load_protocol()["expected_common46_test"]
    for task in TASKS:
        for label in STAR_VARIANTS:
            for metric in METRICS:
                actual = values[(task, label)][metric]
                target = float(expected[task][label][metric])
                if not _close(actual, target, tolerance):
                    raise ValueError(
                        f"Metric mismatch {task}/{label}/{metric}: "
                        f"{actual} != {target}"
                    )


def _relations(values: dict[tuple[str, str], dict[str, float]]) -> tuple[int, int]:
    relations = (
        ("State", "Base"),
        ("FA-DPR", "Base"),
        ("Full", "State"),
        ("Full", "FA-DPR"),
    )
    passed = 0
    total = 0
    print("\ncommon_46 factorial relations")
    for task in TASKS:
        for left, right in relations:
            relation_pass = 0
            for metric in METRICS:
                ok = better(
                    metric,
                    values[(task, left)][metric],
                    values[(task, right)][metric],
                )
                relation_pass += int(ok)
                passed += int(ok)
                total += 1
            print(f"  {task:4s} {left:7s} > {right:7s}: {relation_pass}/4")
    return passed, total


def _verify_paper_hierarchy(
    values: dict[tuple[str, str], dict[str, float]],
) -> None:
    """验证论文核心叙事，不要求 FA-DPR 的 168 h MAPE 被美化。

    冻结 Test 的硬条件是：State 全面优于 Base，Full 全面优于 State 和
    FA-DPR。FA-DPR 相对 Base 的唯一例外是 168 h MAPE；该例外会在论文
    表中透明保留，而不是通过改指标或筛选 Test 样本消除。
    """
    required = (
        ("State", "Base"),
        ("Full", "State"),
        ("Full", "FA-DPR"),
    )
    failures: list[str] = []
    for task in TASKS:
        for left, right in required:
            for metric in METRICS:
                if not better(
                    metric,
                    values[(task, left)][metric],
                    values[(task, right)][metric],
                ):
                    failures.append(f"{task}/{left}>{right}/{metric}")
    if failures:
        raise ValueError(
            "Frozen Test does not satisfy the registered paper hierarchy: "
            + ", ".join(failures)
        )


def _run_evaluation(
    root: Path,
    *,
    device: str,
    output: Path,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if output.exists():
        raise FileExistsError(f"Verification output is nonempty/refused: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    comparison_rows: list[dict[str, Any]] = []
    try:
        jobs: list[tuple[str, str, str, Path]] = []
        for label, variant in STAR_VARIANTS.items():
            for task in TASKS:
                checkpoint = release_model_dir(
                    root, "star_gnn", label, task
                ) / "checkpoint_best.pt"
                jobs.append(("star", variant, task, checkpoint))
        for model in ("stgcn",):
            for task in TASKS:
                checkpoint = release_model_dir(
                    root, "baselines", model, task
                ) / "checkpoint_best.pt"
                jobs.append(("baseline", model, task, checkpoint))
        for index, (family, model, task, checkpoint) in enumerate(jobs, start=1):
            destination = staging / family / model / task
            destination.mkdir(parents=True, exist_ok=False)
            if family == "star":
                command = [
                    sys.executable,
                    "scripts/innovation/evaluate_star_dcrnn.py",
                    "--variant", model,
                ]
            else:
                command = [
                    sys.executable,
                    "scripts/evaluate/test_model.py",
                    "--model", model,
                ]
            command += [
                "--task", task,
                "--seed", "0",
                "--checkpoint", str(checkpoint),
                "--output-dir", str(destination),
                "--device", device,
                "--batch-size", "16",
            ]
            print(f"[{index:02d}/10] re-evaluate {family}/{model}/{task}", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            frozen_family = "star_gnn" if family == "star" else "baselines"
            frozen_name = (
                next(label for label, value in STAR_VARIANTS.items() if value == model)
                if family == "star" else model
            )
            frozen = read_metrics(
                release_model_dir(root, frozen_family, frozen_name, task)
                / "evaluation/metrics_aggregate_total_common_46.csv"
            )
            repeated = read_metrics(
                destination / "metrics_aggregate_total_common_46.csv"
            )
            for metric in METRICS:
                frozen_value = float(frozen[metric])
                repeated_value = float(repeated[metric])
                absolute_error = abs(repeated_value - frozen_value)
                scale = max(
                    abs(frozen_value),
                    abs(repeated_value),
                    sys.float_info.min,
                )
                relative_error = absolute_error / scale
                allowed_error = max(
                    absolute_tolerance,
                    relative_tolerance * scale,
                )
                passed = _close(
                    frozen_value,
                    repeated_value,
                    absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                comparison_rows.append(
                    {
                        "family": frozen_family,
                        "model": frozen_name,
                        "task": task,
                        "metric": metric,
                        "frozen_value": frozen_value,
                        "repeated_value": repeated_value,
                        "absolute_error": absolute_error,
                        "relative_error": relative_error,
                        "allowed_error": allowed_error,
                        "passed": passed,
                    }
                )

        report_path = staging / "reevaluation_metric_differences.csv"
        fieldnames = list(comparison_rows[0])
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

        failures = [row for row in comparison_rows if not row["passed"]]
        summary = {
            "status": "passed" if not failures else "failed",
            "device": device,
            "jobs": len(jobs),
            "metric_comparisons": len(comparison_rows),
            "passed_comparisons": len(comparison_rows) - len(failures),
            "failed_comparisons": len(failures),
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "maximum_absolute_error": max(
                row["absolute_error"] for row in comparison_rows
            ),
            "maximum_relative_error": max(
                row["relative_error"] for row in comparison_rows
            ),
        }
        (staging / "reevaluation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if failures:
            failed_output = output.with_name(
                f"{output.name}.failed."
                f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            os.replace(staging, failed_output)
            details = "; ".join(
                f"{row['model']}/{row['task']}/{row['metric']}: "
                f"{row['repeated_value']:.10g} != "
                f"{row['frozen_value']:.10g} "
                f"(relative_error={row['relative_error']:.3%})"
                for row in failures
            )
            raise ValueError(
                "Re-evaluation metric audit failed; all inference artifacts "
                f"were preserved at {failed_output}. {details}"
            )
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        "Re-evaluation metrics: "
        f"{len(comparison_rows)}/{len(comparison_rows)} PASS; "
        f"max_abs={summary['maximum_absolute_error']:.6g}; "
        f"max_rel={summary['maximum_relative_error']:.6%}"
    )
    print(f"Re-evaluation artifacts: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=PROJECT_ROOT / "results/paper/frozen_v1",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-5,
        help="冻结指标与注册值的绝对容差；不要用于放宽 checkpoint 校验。",
    )
    parser.add_argument(
        "--reevaluation-absolute-tolerance",
        type=float,
        default=5.0e-4,
        help="跨 CUDA/cuDNN 环境重新推理时的绝对容差。",
    )
    parser.add_argument(
        "--reevaluation-relative-tolerance",
        type=float,
        default=5.0e-4,
        help=(
            "跨 CUDA/cuDNN 环境重新推理时的相对容差；默认0.05%，"
            "不用于哈希、协议或样本数校验。"
        ),
    )
    parser.add_argument("--re-evaluate", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verification-output", type=Path)
    args = parser.parse_args()
    root = args.release.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    file_count = _verify_checksums(root)
    values = _collect_release_metrics(root)
    _verify_expected(values, args.tolerance)
    _verify_paper_hierarchy(values)
    passed, total = _relations(values)
    if (passed, total) != (31, 32):
        raise ValueError(f"Expected 31/32 Test relations, observed {passed}/{total}")
    print(f"\nChecksums: {file_count} files PASS")
    print("Checkpoint metadata: 10/10 PASS")
    print("common_46 count: 10/10 x 46 PASS")
    print("Expected metrics: 32/32 PASS")
    print("Paper hierarchy: State>Base and Full>{State,FA-DPR}: PASS")
    print("Test relations: 31/32 PASS")
    if args.re_evaluate:
        output = args.verification_output or (
            PROJECT_ROOT
            / "results/paper/verification"
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        _run_evaluation(
            root,
            device=args.device,
            output=output.resolve(),
            absolute_tolerance=args.reevaluation_absolute_tolerance,
            relative_tolerance=args.reevaluation_relative_tolerance,
        )


if __name__ == "__main__":
    main()
