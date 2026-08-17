#!/usr/bin/env python
"""审计从原始数据重新训练产生的10组模型、Test结果和论文表图。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from paper_release_lib import (
    METRICS,
    PROJECT_ROOT,
    STAR_VARIANTS,
    TASKS,
    better,
    check_test_summary,
    read_metrics,
    release_model_dir,
    sha256,
)
from verify_paper_release import _audit_checkpoint


ALL_MODELS = ("Base", "State", "FA-DPR", "Full", "stgcn")


def _run_dir(root: Path, model: str, task: str) -> Path:
    if model in STAR_VARIANTS:
        return root / "star_gnn" / model / task / "seed_0"
    return root / "baselines" / model / task / "seed_0"


def _frozen_dir(root: Path, model: str, task: str) -> Path:
    family = "star_gnn" if model in STAR_VARIANTS else "baselines"
    return release_model_dir(root, family, model, task)


def _audit_training(run: Path, model: str, task: str) -> dict[str, Any]:
    required = (
        run / "checkpoint_best.pt",
        run / "history.csv",
        run / "training_summary.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("从零训练工件缺失：\n" + "\n".join(map(str, missing)))
    summary = json.loads(required[2].read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise ValueError(f"从零训练未完成：{run}")
    if model in STAR_VARIANTS:
        _audit_checkpoint(
            required[0],
            family="star_gnn",
            model=model,
            task=task,
        )
    else:
        _audit_checkpoint(
            required[0],
            family="baselines",
            model=model,
            task=task,
        )
    return summary


def _audit_paper(root: Path) -> None:
    expected = {
        "test_all_models_common46.csv": 10,
        "test_ablation_common46.csv": 8,
        "test_dma_metrics_long.csv": 100,
        "test_day1_day7_metrics.csv": 35,
        "pearson_correlation.csv": 10,
    }
    table_root = root / "paper/tables"
    for name, count in expected.items():
        path = table_root / name
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != count:
            raise ValueError(f"从零论文表行数错误：{name}={len(rows)}，期望{count}")
    figures = list((root / "paper/figures").glob("*.png")) + list(
        (root / "paper/figures").glob("*.pdf")
    )
    if not figures or any(path.stat().st_size == 0 for path in figures):
        raise ValueError("从零论文图件缺失或为空。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reproduction",
        type=Path,
        default=PROJECT_ROOT / "results/paper/reproduction",
    )
    parser.add_argument(
        "--frozen",
        type=Path,
        default=PROJECT_ROOT / "results/paper/frozen_v1",
    )
    parser.add_argument("--error-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--error-absolute-tolerance", type=float, default=0.01)
    parser.add_argument("--nse-absolute-tolerance", type=float, default=0.001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reproduction = args.reproduction.resolve()
    frozen = args.frozen.resolve()
    if not (reproduction / "TRAINING_COMPLETE").is_file():
        raise FileNotFoundError("缺少TRAINING_COMPLETE，禁止审计Test。")
    output = args.output or reproduction / "clean_room_audit"
    output.mkdir(parents=True, exist_ok=False)

    values: dict[tuple[str, str], dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    checkpoint_sha_matches = 0
    graph_hashes: set[str] = set()
    frozen_graph_hashes: set[str] = set()
    for model in ALL_MODELS:
        for task in TASKS:
            run = _run_dir(reproduction, model, task)
            _audit_training(run, model, task)
            evaluation = run / "evaluation"
            checkpoint = run / "checkpoint_best.pt"
            test_summary = check_test_summary(
                evaluation / "test_summary.json",
                checkpoint,
            )
            graph_identity = test_summary.get("graph_identity", {})
            if graph_identity.get("verified") is not True:
                raise ValueError(f"从零图身份未验证：{run}")
            graph_hashes.add(str(graph_identity.get("artifact_sha256")))
            frozen_summary = json.loads(
                (
                    _frozen_dir(frozen, model, task)
                    / "evaluation/test_summary.json"
                ).read_text(encoding="utf-8")
            )
            frozen_graph_hashes.add(
                str(
                    frozen_summary.get("graph_identity", {}).get(
                        "artifact_sha256"
                    )
                )
            )
            if not (evaluation / "predictions.npz").is_file():
                raise FileNotFoundError(evaluation / "predictions.npz")

            current = read_metrics(
                evaluation / "metrics_aggregate_total_common_46.csv"
            )
            reference = read_metrics(
                _frozen_dir(frozen, model, task)
                / "evaluation/metrics_aggregate_total_common_46.csv"
            )
            values[(task, model)] = current
            checkpoint_sha_matches += int(
                sha256(checkpoint)
                == sha256(_frozen_dir(frozen, model, task) / "checkpoint_best.pt")
            )
            for metric in METRICS:
                absolute_error = abs(current[metric] - reference[metric])
                scale = max(abs(current[metric]), abs(reference[metric]), 1.0e-300)
                relative_error = absolute_error / scale
                if metric == "NSE":
                    passed = absolute_error <= args.nse_absolute_tolerance
                    allowed = args.nse_absolute_tolerance
                else:
                    allowed = max(
                        args.error_absolute_tolerance,
                        args.error_relative_tolerance * scale,
                    )
                    passed = absolute_error <= allowed
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "metric": metric,
                        "frozen_value": reference[metric],
                        "from_scratch_value": current[metric],
                        "absolute_error": absolute_error,
                        "relative_error": relative_error,
                        "allowed_error": allowed,
                        "passed": passed,
                    }
                )

    graph_failures: list[str] = []
    if len(graph_hashes) != 1 or "None" in graph_hashes:
        graph_failures.append(
            f"10组从零评估没有共享同一张图：{sorted(graph_hashes)}"
        )
    if graph_hashes != frozen_graph_hashes:
        graph_failures.append(
            "从原始数据重建的Pearson图与冻结论文图SHA不一致："
            f"from_scratch={sorted(graph_hashes)}, "
            f"frozen={sorted(frozen_graph_hashes)}"
        )

    required_relations = (
        ("State", "Base"),
        ("Full", "State"),
        ("Full", "FA-DPR"),
    )
    hierarchy_failures = []
    for task in TASKS:
        for left, right in required_relations:
            for metric in METRICS:
                if not better(
                    metric,
                    values[(task, left)][metric],
                    values[(task, right)][metric],
                ):
                    hierarchy_failures.append(f"{task}/{left}>{right}/{metric}")
    failures = [row for row in rows if not row["passed"]]
    with (output / "from_scratch_metric_differences.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paper_error: str | None = None
    try:
        _audit_paper(reproduction)
    except Exception as exc:  # 保留完整审计报告后再统一失败。
        paper_error = f"{type(exc).__name__}: {exc}"
    all_passed = not (
        failures or hierarchy_failures or graph_failures or paper_error
    )
    summary = {
        "status": "passed" if all_passed else "failed",
        "training_runs": 10,
        "evaluation_runs": 10,
        "common46_runs": 10,
        "metric_comparisons": 40,
        "passed_metric_comparisons": 40 - len(failures),
        "failed_metric_comparisons": len(failures),
        "checkpoint_sha_exact_matches": checkpoint_sha_matches,
        "shared_graph_sha256": next(iter(graph_hashes)),
        "paper_hierarchy": "passed" if not hierarchy_failures else "failed",
        "paper_hierarchy_failures": hierarchy_failures,
        "graph_identity": "passed" if not graph_failures else "failed",
        "graph_failures": graph_failures,
        "paper_artifacts": "passed" if paper_error is None else "failed",
        "paper_artifact_error": paper_error,
        "maximum_absolute_error": max(row["absolute_error"] for row in rows),
        "maximum_relative_error": max(row["relative_error"] for row in rows),
        "error_relative_tolerance": args.error_relative_tolerance,
        "error_absolute_tolerance": args.error_absolute_tolerance,
        "nse_absolute_tolerance": args.nse_absolute_tolerance,
    }
    (output / "from_scratch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not all_passed:
        metric_failed = ", ".join(
            f"{row['model']}/{row['task']}/{row['metric']}"
            for row in failures
        )
        diagnostics = []
        if metric_failed:
            diagnostics.append(f"指标超差={metric_failed}")
        if hierarchy_failures:
            diagnostics.append("层级失败=" + ", ".join(hierarchy_failures))
        diagnostics.extend(graph_failures)
        if paper_error:
            diagnostics.append("论文工件=" + paper_error)
        raise ValueError(
            f"从零复现审计未通过：{'；'.join(diagnostics)}；"
            f"完整40项记录已保存在{output}"
        )

    print("从零训练：10/10 PASS")
    print("从零common-46评估与防泄漏：10/10 PASS")
    print("从零指标对照：40/40 PASS")
    print("从零论文层级：State>Base且Full>{State,FA-DPR} PASS")
    print(f"从零checkpoint SHA完全一致：{checkpoint_sha_matches}/10（记录项）")
    print("从零DMA/逐日/Pearson表图：PASS")
    print(f"从零审计报告：{output}")


if __name__ == "__main__":
    main()
