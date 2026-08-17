#!/usr/bin/env python
"""审计独立仓库是否真正包含完整的论文复现工件。

该脚本不读取 Test target 做模型选择，也不训练。它只检查文件物理存在性、
数量和目录布局，让后台全面验证在进入 GPU 复推理前就能明确报告缺了什么。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("24h", "168h")
STAR_MODELS = ("Base", "State", "FA-DPR", "Full")
BASELINES = ("stgcn",)
METRICS = ("MAE", "MAPE", "RMSE", "NSE")


def _expected_runs(release: Path) -> list[Path]:
    runs = [
        release / "models/star_gnn" / model / task / "seed_0"
        for model in STAR_MODELS
        for task in TASKS
    ]
    runs.extend(
        release / "models/baselines" / model / task / "seed_0"
        for model in BASELINES
        for task in TASKS
    )
    return runs


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _audit_reevaluation(root: Path) -> None:
    """核对10组复推理和40项跨设备指标差异，而不只统计文件数量。"""
    aggregate = list(root.rglob("metrics_aggregate_total_common_46.csv"))
    predictions = list(root.rglob("predictions.npz"))
    summaries = list(root.rglob("test_summary.json"))
    if (len(aggregate), len(predictions), len(summaries)) != (10, 10, 10):
        raise ValueError(
            "复推理工件数量异常："
            f"aggregate={len(aggregate)}, predictions={len(predictions)}, "
            f"test_summary={len(summaries)}"
        )

    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"复推理未完成：{path}")
        if int(payload.get("protocol_counts", {}).get("common_46", -1)) != 46:
            raise ValueError(f"复推理common-46数量错误：{path}")
        if payload.get("test_targets_used_for_training_or_selection") is not False:
            raise ValueError(f"复推理防泄漏字段错误：{path}")

    summary_path = root / "reevaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_summary = {
        "status": "passed",
        "jobs": 10,
        "metric_comparisons": 40,
        "passed_comparisons": 40,
        "failed_comparisons": 0,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"复推理汇总字段错误：{key}={summary.get(key)!r}，"
                f"期望{expected!r}"
            )

    absolute_tolerance = float(summary["absolute_tolerance"])
    relative_tolerance = float(summary["relative_tolerance"])
    rows = _read_csv(root / "reevaluation_metric_differences.csv")
    if len(rows) != 40:
        raise ValueError(f"复推理指标差异应为40项，实际为{len(rows)}")

    models = (*STAR_MODELS, *BASELINES)
    expected_keys = {
        (model, task, metric)
        for model in models
        for task in TASKS
        for metric in METRICS
    }
    observed_keys: set[tuple[str, str, str]] = set()
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    for row in rows:
        key = (row["model"], row["task"], row["metric"])
        if key in observed_keys:
            raise ValueError(f"复推理指标重复：{key}")
        observed_keys.add(key)
        frozen = float(row["frozen_value"])
        repeated = float(row["repeated_value"])
        absolute_error = abs(repeated - frozen)
        scale = max(abs(frozen), abs(repeated), 1.0e-300)
        relative_error = absolute_error / scale
        allowed_error = max(
            absolute_tolerance,
            relative_tolerance * scale,
        )
        if not math.isclose(
            float(row["absolute_error"]), absolute_error, abs_tol=1.0e-12
        ):
            raise ValueError(f"复推理绝对误差记录不一致：{key}")
        if not math.isclose(
            float(row["relative_error"]), relative_error, abs_tol=1.0e-12
        ):
            raise ValueError(f"复推理相对误差记录不一致：{key}")
        if not math.isclose(
            float(row["allowed_error"]), allowed_error, abs_tol=1.0e-12
        ):
            raise ValueError(f"复推理允许误差记录不一致：{key}")
        if row["passed"].strip().lower() != "true" or absolute_error > allowed_error:
            raise ValueError(f"复推理指标未通过：{key}")
        absolute_errors.append(absolute_error)
        relative_errors.append(relative_error)

    if observed_keys != expected_keys:
        raise ValueError(
            "复推理40项键集合不完整："
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}"
        )
    if not math.isclose(
        float(summary["maximum_absolute_error"]),
        max(absolute_errors),
        abs_tol=1.0e-12,
    ):
        raise ValueError("复推理最大绝对误差汇总不一致。")
    if not math.isclose(
        float(summary["maximum_relative_error"]),
        max(relative_errors),
        abs_tol=1.0e-12,
    ):
        raise ValueError("复推理最大相对误差汇总不一致。")

    print("checkpoint复推理文件：10/10")
    print("checkpoint复推理防泄漏：10/10")
    print("checkpoint复推理指标审计：40/40")
    print(
        "checkpoint复推理最大误差："
        f"abs={max(absolute_errors):.8g}, "
        f"rel={max(relative_errors):.6%}"
    )


def _audit_paper_artifacts() -> None:
    expected_csv_rows = {
        "paper/tables/test_all_models_common46.csv": 10,
        "paper/tables/test_ablation_common46.csv": 8,
        "paper/tables/test_dma_metrics_long.csv": 100,
        "paper/tables/test_day1_day7_metrics.csv": 35,
        "paper/tables/pearson_correlation.csv": 10,
    }
    for relative, expected_rows in expected_csv_rows.items():
        path = PROJECT_ROOT / relative
        rows = _read_csv(path)
        if len(rows) != expected_rows:
            raise ValueError(
                f"论文表格行数错误：{relative}={len(rows)}，"
                f"期望{expected_rows}"
            )
    figures = tuple((PROJECT_ROOT / "paper/figures").glob("*.png")) + tuple(
        (PROJECT_ROOT / "paper/figures").glob("*.pdf")
    )
    if not figures or any(path.stat().st_size == 0 for path in figures):
        raise ValueError("论文图件缺失或为空。")
    print("论文核心表格行数：10/8/100/35/10 PASS")
    print(f"论文PNG/PDF图件：{len(figures)}个，全部非空")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=PROJECT_ROOT / "results/paper/frozen_v1",
    )
    parser.add_argument("--require-paper-artifacts", action="store_true")
    parser.add_argument("--require-reevaluation", type=Path)
    args = parser.parse_args()

    release = args.release.resolve()
    manifest_path = release / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少冻结清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release_id") != "star_gnn_common46_seed0_v1":
        raise ValueError("release_id 不正确。")
    if len(manifest.get("artifacts", {})) != 10:
        raise ValueError("MANIFEST.json 必须登记 10 组模型/任务。")

    required: list[Path] = [
        release / "CHECKSUMS.sha256",
        release / "results_common46.csv",
    ]
    for run in _expected_runs(release):
        required.extend(
            (
                run / "checkpoint_best.pt",
                run / "evaluation/test_summary.json",
                run / "evaluation/predictions.npz",
                run / "evaluation/metrics_aggregate_total_common_46.csv",
                run / "evaluation/metrics_common_46.csv",
            )
        )
    required.extend(
        PROJECT_ROOT / "data/processed/data_build" / name
        for name in (
            "demand_hourly.parquet",
            "weather_hourly.parquet",
            "temporal_hourly.parquet",
            "combined_hourly_features.parquet",
        )
    )
    required.append(
        PROJECT_ROOT / "artifacts/graphs/bwdf_pearson_static_graph.npz"
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "冻结复现工件不完整：\n" + "\n".join(map(str, missing))
        )

    checkpoints = list(release.rglob("checkpoint_best.pt"))
    predictions = list(release.rglob("predictions.npz"))
    summaries = list(release.rglob("test_summary.json"))
    aggregate = list(release.rglob("metrics_aggregate_total_common_46.csv"))
    observed = {
        "checkpoint_best.pt": len(checkpoints),
        "predictions.npz": len(predictions),
        "test_summary.json": len(summaries),
        "aggregate_common46": len(aggregate),
    }
    expected = {name: 10 for name in observed}
    if observed != expected:
        raise ValueError(f"冻结文件数量异常：{observed}，期望：{expected}")

    if args.require_reevaluation is not None:
        _audit_reevaluation(args.require_reevaluation.resolve())

    if args.require_paper_artifacts:
        paper_required = (
            PROJECT_ROOT / "paper/reports/TEST_RESULTS_CN.md",
            PROJECT_ROOT / "paper/tables/test_all_models_common46.csv",
            PROJECT_ROOT / "paper/tables/test_ablation_common46.csv",
            PROJECT_ROOT / "paper/tables/test_dma_metrics_long.csv",
            PROJECT_ROOT / "paper/tables/test_day1_day7_metrics.csv",
            PROJECT_ROOT / "paper/tables/pearson_correlation.csv",
            PROJECT_ROOT / "paper/figures/test_day1_day7_models.png",
            PROJECT_ROOT / "paper/figures/test_day1_day7_ablation.png",
            PROJECT_ROOT / "paper/figures/pearson_correlation_heatmap.png",
        )
        missing_paper = [path for path in paper_required if not path.is_file()]
        if missing_paper:
            raise FileNotFoundError(
                "论文表图生成不完整：\n" + "\n".join(map(str, missing_paper))
            )
        _audit_paper_artifacts()

    print("冻结工件目录：", release)
    print("checkpoint_best.pt：10/10")
    print("predictions.npz：10/10")
    print("test_summary.json：10/10")
    print("common-46 aggregate：10/10")
    print("处理数据：4/4")
    print("Pearson图：1/1")
    if args.require_paper_artifacts:
        print("总体、消融、DMA、Day1-Day7与Pearson表图：PASS")
    print("发布工件物理完整性：PASS")


if __name__ == "__main__":
    main()
