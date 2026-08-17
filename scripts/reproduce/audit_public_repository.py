#!/usr/bin/env python
"""审计准备上传 GitHub 的公开仓库内容。

本脚本只检查发布边界，不训练模型，也不读取 Test target 做选择。若当前目录
已经是 Git 仓库，则审计 ``git ls-files`` 返回的已跟踪文件；否则审计
``SOURCE_CHECKSUMS.sha256`` 登记的拟发布源码。这样既能用于上传前检查，也能
用于 GitHub 实际克隆后的第二轮检查。
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
    "docs/FULL_PIPELINE_CN.md",
    "docs/METHOD_CN.md",
    "docs/RELEASE_CN.md",
    "docs/RESULTS_AND_ARTIFACTS_CN.md",
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
    "data/README.md",
    *EXPECTED_DOCS,
    "environment.yml",
    "pyproject.toml",
    "requirements-lock.txt",
    "scripts/reproduce/finalize_public_release.sh",
    "scripts/reproduce/package_frozen_release.py",
    "scripts/reproduce/train_from_scratch.sh",
    "scripts/reproduce/verify_pretrained.sh",
}
FORBIDDEN_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".7z",
    ".log",
    ".pid",
    ".pyc",
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
    } | {
        f"baselines/stgcn/{task}" for task in ("24h", "168h")
    }
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
        if not path.is_file() or _sha256(path) != expected:
            errors.append(f"冻结工件缺失或 SHA 错误：{relative}")
    actual = {
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if actual != listed:
        errors.append("冻结文件集合与 CHECKSUMS.sha256 不一致")
    return errors


def _audit_paper(root: Path) -> list[str]:
    expected_rows = {
        "paper/tables/test_all_models_common46.csv": 10,
        "paper/tables/test_ablation_common46.csv": 8,
        "paper/tables/test_dma_metrics_long.csv": 100,
        "paper/tables/test_day1_day7_metrics.csv": 35,
        "paper/tables/pearson_correlation.csv": 10,
    }
    errors: list[str] = []
    for relative, expected in expected_rows.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"缺少论文表格：{relative}")
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        if count != expected:
            errors.append(f"论文表格行数错误：{relative}={count}，期望{expected}")
    figures = list((root / "paper/figures").glob("*.png")) + list(
        (root / "paper/figures").glob("*.pdf")
    )
    if not figures or any(path.stat().st_size == 0 for path in figures):
        errors.append("论文 PNG/PDF 图件缺失或为空")
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
    # SOURCE_CHECKSUMS 是 clean-room 和 GitHub 源码发布的注册边界。Git 已初始化
    # 时额外扫描“仍存在的已跟踪文件”，但不要求新补丁文件在验收前已经 git add。
    tracked_existing = {
        relative
        for relative in (git_files or set())
        if (root / relative).is_file()
    }
    publish_files = source_files | tracked_existing
    mode = "source-manifest+git" if git_files is not None else "source-manifest"

    missing = REQUIRED_PUBLIC_FILES - publish_files
    if not (root / "SOURCE_CHECKSUMS.sha256").is_file():
        errors.append("缺少 SOURCE_CHECKSUMS.sha256")
    if missing:
        errors.append(f"拟发布文件缺失：{sorted(missing)}")
    deprecated = DEPRECATED_PUBLIC_FILES & publish_files
    if deprecated:
        errors.append(f"仍拟发布已合并文档/重复配置：{sorted(deprecated)}")
    docs = {path for path in publish_files if path.startswith("docs/")}
    if docs != EXPECTED_DOCS:
        errors.append(
            "docs/ 应只保留4份中文主文档："
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
            "STGCN",
            "DCRNN (Base)",
            "DCRNN + State",
            "DCRNN + FA-DPR",
            "STaR-GNN",
        ],
        "frozen_checkpoint_count": 10 if args.require_frozen else None,
        "errors": errors,
    }
    output = args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if errors:
        raise SystemExit("公开仓库审计失败：\n- " + "\n- ".join(errors))
    print("公开仓库结构与发布边界：PASS")
    print(f"审计口径：{mode}，文件数：{len(publish_files)}")
    print("中文主文档：4/4")
    if args.require_frozen:
        print("唯一 checkpoint：10/10；DCRNN/Base无重复")
    if args.require_paper_artifacts:
        print("总体/消融/DMA/Day1-Day7/Pearson表图：PASS")


if __name__ == "__main__":
    main()
