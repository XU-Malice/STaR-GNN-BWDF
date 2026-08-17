#!/usr/bin/env python
"""把已冻结的论文 checkpoint 和本机数据复制到独立公开仓库。

该脚本只用于作者在发布前迁移本地大文件。GitHub 源码仓库不应直接提交
处理后数据或 10 个 checkpoint；正式公开时建议把冻结包上传为 GitHub
Release asset。复制采用临时目录加原子替换，不会覆盖非空目标目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_inventory(root: Path) -> dict[str, tuple[int, str]]:
    """返回目录的完整文件清单，用于识别半成品或被修改的目标目录。"""
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, _sha256(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _backup_path(destination: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = destination.with_name(f"{destination.name}.incomplete.{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = destination.with_name(
            f"{destination.name}.incomplete.{stamp}.{suffix}"
        )
        suffix += 1
    return candidate


def _copy_tree_once(source: Path, destination: Path, label: str) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"缺少{label}：{source}")
    if destination.exists() and any(destination.iterdir()):
        # 只有源、目标文件集合和 SHA-256 完全一致时才复用。旧版仅判断目录
        # 非空，断电或中断产生的半成品会被误认为已经导入完成。
        if _tree_inventory(source) == _tree_inventory(destination):
            print(f"REUSE {label}（完整一致）：{destination}")
            return
        backup = _backup_path(destination)
        os.replace(destination, backup)
        print(f"BACKUP 不完整或不一致的{label}：{destination} -> {backup}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        # mkdtemp 已创建目录，因此将源内容复制到其中而不是再次 copytree。
        for child in source.iterdir():
            target = staging / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"COPY {label}：{source} -> {destination}")


def _copy_file_once(source: Path, destination: Path, label: str) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"缺少{label}：{source}")
    if destination.is_file():
        if source.stat().st_size == destination.stat().st_size and (
            _sha256(source) == _sha256(destination)
        ):
            print(f"REUSE {label}（完整一致）：{destination}")
            return
        backup = _backup_path(destination)
        os.replace(destination, backup)
        print(f"BACKUP 不一致的{label}：{destination} -> {backup}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    print(f"COPY {label}：{source} -> {destination}")


def _audit_source(source_root: Path) -> None:
    release = source_root / "results/paper/frozen_v1"
    manifest_path = release / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "旧项目中没有完整冻结发布包。请先在旧项目执行 "
            "python scripts/reproduce/freeze_paper_release.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release_id") != "star_gnn_common46_seed0_v1":
        raise ValueError("冻结发布 release_id 不正确。")
    if len(manifest.get("artifacts", {})) != 10:
        raise ValueError("冻结发布必须包含 10 个模型/任务 checkpoint。")
    required = [
        release / "CHECKSUMS.sha256",
        release / "results_common46.csv",
    ]
    for family, models in {
        "star_gnn": ("Base", "State", "FA-DPR", "Full"),
        "baselines": ("stgcn",),
    }.items():
        for model in models:
            for task in ("24h", "168h"):
                run = release / "models" / family / model / task / "seed_0"
                required.extend(
                    [
                        run / "checkpoint_best.pt",
                        run / "evaluation/test_summary.json",
                        run / "evaluation/predictions.npz",
                    ]
                )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("冻结发布不完整：\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_project",
        type=Path,
        help="保存原始实验和冻结 checkpoint 的旧 DMA-WDF 项目目录。",
    )
    args = parser.parse_args()
    source_root = args.source_project.resolve()
    if source_root == PROJECT_ROOT.resolve():
        raise ValueError("source_project 必须是旧项目，不能是当前独立仓库。")
    _audit_source(source_root)

    _copy_tree_once(
        source_root / "results/paper/frozen_v1",
        PROJECT_ROOT / "results/paper/frozen_v1",
        "10组冻结checkpoint与Test结果",
    )
    _copy_tree_once(
        source_root / "data/processed/data_build",
        PROJECT_ROOT / "data/processed/data_build",
        "checkpoint复评所需处理数据",
    )
    _copy_file_once(
        source_root / "artifacts/graphs/bwdf_pearson_static_graph.npz",
        PROJECT_ROOT / "artifacts/graphs/bwdf_pearson_static_graph.npz",
        "训练期Pearson图",
    )
    graph_diagnostics = source_root / "results/graph/pearson_static"
    if graph_diagnostics.is_dir():
        _copy_tree_once(
            graph_diagnostics,
            PROJECT_ROOT / "results/graph/pearson_static",
            "Pearson图诊断",
        )
    # 对“REUSE”路径也执行物理文件检查，防止把一个非空但不完整的目录误当成
    # 已经成功导入。完整 checkpoint 关系随后由 verify_pretrained 再核验。
    _audit_source(PROJECT_ROOT)
    data_required = (
        PROJECT_ROOT / "data/processed/data_build/demand_hourly.parquet",
        PROJECT_ROOT / "data/processed/data_build/weather_hourly.parquet",
        PROJECT_ROOT / "data/processed/data_build/temporal_hourly.parquet",
        PROJECT_ROOT / "data/processed/data_build/combined_hourly_features.parquet",
    )
    missing_data = [str(path) for path in data_required if not path.is_file()]
    if missing_data:
        raise FileNotFoundError("处理数据导入不完整：\n" + "\n".join(missing_data))
    print("本地论文大文件导入：PASS")


if __name__ == "__main__":
    main()
