#!/usr/bin/env python
"""Audit the public GitHub repository boundary before release.

The audit distinguishes frozen scientific artifacts from submission-facing
presentation artifacts. Legacy diagnostic figures may remain tracked, but the
canonical manuscript contract is 2 main tables + 7 main result figures,
Supplementary Tables S1–S3, and Supplementary Figures S1–S3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_DOCS = {
    "docs/README.md",
    "docs/EXPERIMENT_EVIDENCE_AUDIT_FINAL_CN.md",
    "docs/EXPERIMENT_DESIGN_FINAL_CN.md",
    "docs/FULL_PIPELINE_CN.md",
    "docs/MANUSCRIPT_FIGURES_CN.md",
    "docs/MANUSCRIPT_FIGURES_FINAL_CN.md",
    "docs/MSCMNET_BASELINES_CN.md",
    "docs/METHOD_CN.md",
    "docs/PLOTTING_CN.md",
    "docs/RELEASE_CN.md",
    "docs/RESULTS_AND_ARTIFACTS_CN.md",
    "docs/RESULTS_DISCUSSION_JOH_AUDIT_FINAL_CN.md",
    "docs/RESULT_CONSISTENCY_AUDIT_CN.md",
}

DEPRECATED_PUBLIC_FILES = {
    "INSTALL_ON_SERVER_CN.md",
    "configs/paper/dcrnn_24h.yaml",
    "configs/paper/dcrnn_168h.yaml",
    "docs/00_QUICKSTART_CN.md",
    "docs/BASELINES_CN.md",
    "docs/CHECKPOINT_VERIFICATION_CN.md",
    "docs/CLEAN_ROOM_REPRODUCTION_CN.md",
    "docs/CODE_FLOW_CN.md",
    "docs/DATA.md",
    "docs/EXPERIMENTS.md",
    "docs/GITHUB_RELEASE_CHECKLIST_CN.md",
    "docs/PAPER_ARTIFACTS_CN.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RESULTS_PROVENANCE.md",
}

REQUIRED_PUBLIC_FILES = {
    ".github/workflows/ci.yml",
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "README_EN.md",
    "RELEASE_INFO.json",
    "SOURCE_CHECKSUMS.sha256",
    "data/README.md",
    *EXPECTED_DOCS,
    "environment.yml",
    "paper/README.md",
    "paper/captions/SUBMISSION_RESULT_FIGURE_CAPTIONS.md",
    "paper/captions/SUBMISSION_SUPPLEMENTARY_FIGURE_CAPTIONS.md",
    "pyproject.toml",
    "requirements-lock.txt",
    "scripts/reproduce/finalize_public_release.sh",
    "scripts/reproduce/manuscript_plot_style.py",
    "scripts/reproduce/package_frozen_release.py",
    "scripts/reproduce/regenerate_source_checksums.py",
    "scripts/reproduce/render_submission_figures.py",
    "scripts/reproduce/render_supplementary_figures.py",
    "scripts/reproduce/render_submission_tables.py",
    "scripts/reproduce/train_from_scratch.sh",
    "scripts/reproduce/verify_pretrained.sh",
}

FORBIDDEN_SUFFIXES = (
    ".tar", ".tar.gz", ".tgz", ".zip", ".7z", ".log", ".pid", ".pyc"
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checksum_matches(path: Path, expected: str) -> bool:
    """Accept Git-normalized line endings for frozen text inventories."""
    if _sha256(path) == expected:
        return True
    if path.suffix.lower() not in {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}:
        return False
    payload = path.read_bytes()
    lf = payload.replace(b"\r\n", b"\n")
    variants = (lf, lf.replace(b"\n", b"\r\n"))
    return any(hashlib.sha256(item).hexdigest() == expected for item in variants)


def _manifest_files(root: Path) -> tuple[set[str], list[str]]:
    manifest = root / "SOURCE_CHECKSUMS.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    files: set[str] = set()
    errors: list[str] = []
    for number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"源码 SHA 第{number}行格式错误")
            continue
        relative = relative.removeprefix("./")
        candidate = Path(relative)
        if not relative or candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"源码 SHA 含非法路径：{relative}")
            continue
        if relative in files:
            errors.append(f"源码 SHA 重复登记：{relative}")
            continue
        files.add(relative)
        path = root / relative
        if not path.is_file():
            errors.append(f"源码 SHA 登记文件不存在：{relative}")
        elif _sha256(path) != expected:
            errors.append(f"源码 SHA 不一致：{relative}")
    return files, errors


def _git_files(root: Path) -> set[str] | None:
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return None
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return {
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    }


def _audit_frozen(root: Path) -> list[str]:
    release = root / "results/paper/frozen_v1"
    errors: list[str] = []
    manifest_path = release / "MANIFEST.json"
    checksums_path = release / "CHECKSUMS.sha256"
    if not manifest_path.is_file() or not checksums_path.is_file():
        return ["冻结发布缺少 MANIFEST.json 或 CHECKSUMS.sha256"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {
        f"star_gnn/{model}/{task}"
        for model in ("Base", "State", "FA-DPR", "Full")
        for task in ("24h", "168h")
    } | {f"baselines/stgcn/{task}" for task in ("24h", "168h")}
    observed_keys = set(manifest.get("artifacts", {}))
    if observed_keys != expected_keys:
        errors.append(
            "冻结 MANIFEST 模型集合不是唯一10组："
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}"
        )
    if (release / "models/baselines/dcrnn").exists():
        errors.append("仍存在重复 models/baselines/dcrnn")

    counts = {
        "checkpoint": len(list(release.rglob("checkpoint_best.pt"))),
        "prediction": len(list(release.rglob("predictions.npz"))),
        "test_summary": len(list(release.rglob("test_summary.json"))),
    }
    if counts != {"checkpoint": 10, "prediction": 10, "test_summary": 10}:
        errors.append(f"冻结工件数量不是10/10/10：{counts}")

    listed: set[str] = set()
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        listed.add(relative)
        path = release / relative
        if not path.is_file() or not _checksum_matches(path, expected):
            errors.append(f"冻结工件缺失或 SHA 错误：{relative}")
    actual = {
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if actual != listed:
        errors.append("冻结文件集合与 CHECKSUMS.sha256 不一致")
    return errors


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _audit_paper(root: Path) -> list[str]:
    errors: list[str] = []

    source_tables = {
        "paper/tables/literature/table_literature_comparison_common46.csv": 18,
        "paper/tables/literature/table_ablation_common46.csv": 8,
        "paper/tables/literature/table_star_gnn_dma_common46.csv": 20,
        "paper/tables/literature/table_graph_models_dma_common46.csv": 60,
    }
    for relative, expected in source_tables.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"缺少论文源表：{relative}")
        elif _row_count(path) != expected:
            errors.append(
                f"论文源表行数错误：{relative}={_row_count(path)}，期望{expected}"
            )

    ablation_path = root / "paper/tables/literature/table_ablation_common46.csv"
    if ablation_path.is_file():
        with ablation_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_models = (
            "DCRNN",
            "DCRNN + SAS-Norm",
            "DCRNN + FA-DPR",
            "STaR-GNN",
        )
        for task in ("24h", "168h"):
            observed = tuple(
                row["model"] for row in rows if row["task"] == task
            )
            if observed != expected_models:
                errors.append(
                    f"factorial ablation 模型集合错误：{task}={observed}"
                )
            if "STGCN" in observed:
                errors.append(f"STGCN 不应出现在 factorial ablation：{task}")

    required_submission_tables = (
        "paper/tables/submission/table1_overall_performance.md",
        "paper/tables/submission/table2_factorial_ablation.md",
        "paper/tables/submission/tableS1_dma_metrics.md",
        "paper/tables/submission/tableS2_dma_local_margin.md",
        "paper/tables/submission/tableS3_forecast_origin_robustness.csv",
        "paper/tables/submission/tableS3_forecast_origin_robustness.md",
    )
    for relative in required_submission_tables:
        if not (root / relative).is_file():
            errors.append(f"缺少投稿显示表：{relative}")

    required_main = (
        "main_fig1_overall_performance",
        "main_fig2_dma_performance",
        "main_fig3_dma_absolute_24h",
        "main_fig4_dma_absolute_168h",
        "main_fig5_ablation_leadtime",
        "main_fig6_origin_robustness",
        "main_fig7_week_ahead_dynamics",
    )
    for stem in required_main:
        for ext in ("pdf", "svg", "png"):
            path = root / f"paper/figures/submission/{stem}.{ext}"
            if not path.is_file():
                errors.append(f"缺少主图：{path.relative_to(root)}")
            elif path.stat().st_size == 0:
                errors.append(f"主图为空：{path.relative_to(root)}")
    required_supplementary = (
        "supp_figS1_data_cleaning",
        "supp_figS2_weekly_demand_patterns",
        "supp_figS3_origin_mae_ecdf",
    )
    for stem in required_supplementary:
        for ext in ("pdf", "svg", "png"):
            path = root / f"paper/figures/submission/{stem}.{ext}"
            if not path.is_file():
                errors.append(f"缺少补充图：{path.relative_to(root)}")
            elif path.stat().st_size == 0:
                errors.append(f"补充图为空：{path.relative_to(root)}")
    required_audits = (
        "paper/tables/manuscript/submission/main_fig2_dma_ranks.csv",
        "paper/tables/manuscript/submission/main_fig2_dma_pairwise_improvement.csv",
        "paper/tables/manuscript/submission/main_fig3_dma_strongest_competitor.csv",
        "paper/tables/manuscript/submission/main_fig5_daywise_paired_improvement.csv",
        "paper/tables/manuscript/submission/main_fig6_origin_paired_improvement.csv",
        "paper/tables/manuscript/submission/main_fig6_origin_summary.csv",
        "paper/tables/manuscript/submission/main_fig7_diurnal_aggregate_error.csv",
        "paper/tables/manuscript/submission/main_fig7_representative_trajectory.csv",
        "paper/tables/manuscript/submission/main_fig7_representative_selection.json",
        "paper/tables/manuscript/submission/submission_figure_audit.json",
        "paper/tables/manuscript/submission/supp_figS1_data_cleaning_metadata.json",
        "paper/tables/manuscript/submission/supp_figS2_weekly_demand_patterns.csv",
        "paper/tables/manuscript/submission/supp_figS2_weekly_demand_patterns_metadata.json",
        "paper/tables/manuscript/submission/supp_figS3_origin_mae_ecdf.csv",
        "paper/tables/manuscript/submission/supplementary_figure_audit.json",
    )
    for relative in required_audits:
        if not (root / relative).is_file():
            errors.append(f"缺少投稿图件审计：{relative}")

    audit_path = root / (
        "paper/tables/manuscript/submission/submission_figure_audit.json"
    )
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        dma = audit.get("main_fig2_dma", {})
        expected = {
            "24h": {
                "first_place_cells": 36,
                "n_dma_metric_cells": 40,
                "pairwise_wins": 306,
                "n_pairwise_comparisons": 320,
                "graph_baseline_wins": 80,
                "n_graph_baseline_comparisons": 80,
            },
            "168h": {
                "first_place_cells": 27,
                "n_dma_metric_cells": 40,
                "pairwise_wins": 275,
                "n_pairwise_comparisons": 320,
                "graph_baseline_wins": 78,
                "n_graph_baseline_comparisons": 80,
            },
        }
        if dma != expected:
            errors.append(f"Main Fig. 2 DMA audit drift：{dma}")

        local = audit.get("main_fig3_strongest_competitor", {})
        expected_local = {
            "24h": {
                "wins": 36,
                "comparisons": 40,
                "losses_by_dma": {
                    "A": 4, "B": 0, "C": 0, "D": 0, "E": 0,
                    "F": 0, "G": 0, "H": 0, "I": 0, "J": 0,
                },
            },
            "168h": {
                "wins": 27,
                "comparisons": 40,
                "losses_by_dma": {
                    "A": 4, "B": 0, "C": 0, "D": 0, "E": 4,
                    "F": 0, "G": 4, "H": 0, "I": 1, "J": 0,
                },
            },
        }
        if local != expected_local:
            errors.append(f"Main Fig. 3 DMA audit drift：{local}")

        origin_rows = audit.get("main_fig6_origin_summary", [])
        if len(origin_rows) != 16:
            errors.append(
                "Main Fig. 6 origin audit should contain 16 "
                f"horizon-baseline-metric rows, got {len(origin_rows)}"
            )
        elif any(int(row.get("n_origins", -1)) != 46 for row in origin_rows):
            errors.append("Main Fig. 6 origin audit is not common-46")
        elif any(
            int(row.get("n_high_variability", -1)) != 12
            for row in origin_rows
        ):
            errors.append("Main Fig. 6 high-variability stratum is not 12 origins")

    supplementary_audit_path = root / (
        "paper/tables/manuscript/submission/supplementary_figure_audit.json"
    )
    if supplementary_audit_path.is_file():
        supplementary = json.loads(
            supplementary_audit_path.read_text(encoding="utf-8")
        )
        cleaning = supplementary.get("supp_figS1", {})
        if cleaning.get("missing_example", {}).get("dma") != "DMA F":
            errors.append("Supplementary Fig. S1 missing-data DMA drift")
        if cleaning.get("outlier_example", {}).get("dma") != "DMA C":
            errors.append("Supplementary Fig. S1 outlier DMA drift")
        weekly = supplementary.get("supp_figS2", {})
        if weekly.get("selected_dmas") != ["DMA C", "DMA H", "DMA E"]:
            errors.append("Supplementary Fig. S2 representative DMA drift")
        ecdf = supplementary.get("supp_figS3", {})
        if int(ecdf.get("origins_per_model_task", -1)) != 46:
            errors.append("Supplementary Fig. S3 is not common-46")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--require-paper-artifacts", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = PROJECT_ROOT
    source_files, errors = _manifest_files(root)
    git_files = _git_files(root)
    tracked_existing = {
        relative
        for relative in (git_files or set())
        if (root / relative).is_file()
    }
    publish_files = source_files | tracked_existing
    mode = "source-manifest+git" if git_files is not None else "source-manifest"

    missing = REQUIRED_PUBLIC_FILES - publish_files
    if missing:
        errors.append(f"拟发布文件缺失：{sorted(missing)}")
    deprecated = DEPRECATED_PUBLIC_FILES & publish_files
    if deprecated:
        errors.append(f"仍拟发布已合并文档/重复配置：{sorted(deprecated)}")

    docs = {path for path in publish_files if path.startswith("docs/")}
    if docs != EXPECTED_DOCS:
        errors.append(
            "docs/ 文档集合与最终设计不一致："
            f"missing={sorted(EXPECTED_DOCS - docs)}, "
            f"extra={sorted(docs - EXPECTED_DOCS)}"
        )

    for relative in sorted(publish_files):
        lowered = relative.lower()
        path = root / relative
        if lowered.endswith(FORBIDDEN_SUFFIXES):
            errors.append(f"拟发布集合含临时/压缩文件：{relative}")
        if not path.is_file():
            continue
        if path.stat().st_size > 95 * 1024 * 1024:
            errors.append(f"Git 文件超过95 MiB，应转 Release asset：{relative}")
        if path.stat().st_size <= 8 * 1024 * 1024:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                errors.append(f"疑似密钥内容：{relative}")

    if args.require_frozen:
        errors.extend(_audit_frozen(root))
    if args.require_paper_artifacts:
        errors.extend(_audit_paper(root))

    report = {
        "status": "failed" if errors else "passed",
        "audit_mode": mode,
        "public_file_count": len(publish_files),
        "source_manifest_file_count": len(source_files),
        "docs": sorted(docs),
        "canonical_models": [
            "STGCN (independent graph baseline)",
            "DCRNN (Base)",
            "DCRNN + SAS-Norm",
            "DCRNN + FA-DPR",
            "STaR-GNN",
        ],
        "factorial_ablation_models": [
            "DCRNN",
            "DCRNN + SAS-Norm",
            "DCRNN + FA-DPR",
            "STaR-GNN",
        ],
        "submission_contract": {
            "main_tables": 2,
            "main_result_figures": 7,
            "supplementary_figures": 3,
            "supplementary_tables": 3,
        },
        "frozen_checkpoint_count": 10 if args.require_frozen else None,
        "errors": errors,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if errors:
        raise SystemExit("公开仓库审计失败：\n- " + "\n- ".join(errors))

    print("公开仓库结构与发布边界：PASS")
    print(f"审计口径：{mode}，文件数：{len(publish_files)}")
    print(f"docs 文档集合：{len(EXPECTED_DOCS)}/{len(EXPECTED_DOCS)} PASS")
    print("factorial ablation：4 models, no STGCN PASS")
    if args.require_frozen:
        print("唯一 checkpoint：10/10；DCRNN/Base无重复")
    if args.require_paper_artifacts:
        print(
            "Submission Table 1--2 / Main Fig. 1--7 / "
            "Tables S1--S3 / Supplementary Figs. S1--S3：PASS"
        )


if __name__ == "__main__":
    main()
