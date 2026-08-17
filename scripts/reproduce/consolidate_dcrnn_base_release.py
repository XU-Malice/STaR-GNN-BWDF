#!/usr/bin/env python
"""把冻结发布包中的重复 DCRNN 合并为唯一的 Base/backbone。

该迁移只处理已经存在的冻结工件，不训练模型、不重新选参，也不读取 Test
目标参与任何决策。保留 ``star_gnn/Base``，删除历史 ``baselines/dcrnn``，
随后更新汇总 CSV、写入删除溯源并重建冻结包 SHA-256 清单。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("24h", "168h")
REQUIRED_MODELS = {
    "star_gnn": ("Base", "State", "FA-DPR", "Full"),
    "baselines": ("stgcn",),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(release: Path, family: str, model: str, task: str) -> Path:
    return release / "models" / family / model / task / "seed_0"


def _required_files(run: Path) -> tuple[Path, ...]:
    return (
        run / "checkpoint_best.pt",
        run / "evaluation/test_summary.json",
        run / "evaluation/predictions.npz",
        run / "evaluation/metrics_aggregate_total_common_46.csv",
        run / "evaluation/metrics_common_46.csv",
    )


def _filter_summary(path: Path) -> int:
    """删除独立 DCRNN 行，保留 Base/backbone 行；返回删除行数。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or not fieldnames:
        raise ValueError(f"冻结汇总为空：{path}")

    def legacy_dcrnn(row: dict[str, str]) -> bool:
        values = {str(value).strip().lower() for value in row.values()}
        return "dcrnn" in values and "base" not in values

    kept = [row for row in rows if not legacy_dcrnn(row)]
    removed = len(rows) - len(kept)
    if len(kept) != 10:
        raise ValueError(
            f"合并后 results_common46.csv 应为10行，实际{len(kept)}行；"
            "拒绝覆盖，以免误删。"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    temporary.replace(path)
    return removed


def _rewrite_checksums(release: Path) -> int:
    files = sorted(
        path
        for path in release.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(release).as_posix()}"
        for path in files
    ]
    target = release / "CHECKSUMS.sha256"
    temporary = release / ".CHECKSUMS.sha256.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(target)
    return len(files)


def _embed_protocol_indices(release: Path, duplicate: Path) -> None:
    """把旧 DCRNN 中的协议索引迁入全部 STaR 预测文件。

    旧版 STaR ``predictions.npz`` 只保存 target/prediction，制图器因此隐式
    借用重复 DCRNN 的索引。迁移后每个预测文件自描述，不再依赖待删除目录。
    """
    keys = (
        "operational_indices",
        "strict_within_test_indices",
        "common_46_indices",
    )
    for task in TASKS:
        legacy_path = duplicate / task / "seed_0/evaluation/predictions.npz"
        if legacy_path.is_file():
            with np.load(legacy_path, allow_pickle=False) as payload:
                indices = {key: payload[key] for key in keys}
        else:
            base_path = _run(
                release, "star_gnn", "Base", task
            ) / "evaluation/predictions.npz"
            with np.load(base_path, allow_pickle=False) as payload:
                if any(key not in payload.files for key in keys):
                    raise FileNotFoundError(
                        f"既无历史 DCRNN 索引，也无已迁移 Base 索引：{task}"
                    )
                indices = {key: payload[key] for key in keys}

        if indices["common_46_indices"].shape != (46,):
            raise ValueError(f"{task} common_46_indices数量不是46。")
        for model in REQUIRED_MODELS["star_gnn"]:
            path = _run(
                release, "star_gnn", model, task
            ) / "evaluation/predictions.npz"
            with np.load(path, allow_pickle=False) as payload:
                arrays = {key: payload[key] for key in payload.files}
            arrays.update(indices)
            temporary = path.with_name(f".{path.name}.tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=PROJECT_ROOT / "results/paper/frozen_v1",
    )
    args = parser.parse_args()
    release = args.release.resolve()
    manifest_path = release / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    legacy_keys = ("baselines/dcrnn/24h", "baselines/dcrnn/168h")
    if any(key in artifacts for key in legacy_keys):
        raise ValueError(
            "MANIFEST 仍登记历史 DCRNN；请先安装10组统一补丁，再执行迁移。"
        )
    if len(artifacts) != 10:
        raise ValueError(f"统一 MANIFEST 应登记10组，实际{len(artifacts)}组。")

    missing: list[Path] = []
    for family, models in REQUIRED_MODELS.items():
        for model in models:
            for task in TASKS:
                missing.extend(
                    path
                    for path in _required_files(_run(release, family, model, task))
                    if not path.is_file()
                )
    if missing:
        raise FileNotFoundError("唯一10组冻结工件不完整：\n" + "\n".join(map(str, missing)))

    duplicate = release / "models/baselines/dcrnn"
    _embed_protocol_indices(release, duplicate)
    removed_inventory: dict[str, Any] = {}
    if duplicate.is_dir():
        for path in sorted(duplicate.rglob("*")):
            if path.is_file():
                removed_inventory[path.relative_to(release).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        shutil.rmtree(duplicate)

    removed_rows = _filter_summary(release / "results_common46.csv")
    provenance = {
        "status": "dcrnn_base_unified",
        "canonical_model": "star_gnn/Base (variant=backbone)",
        "canonical_checkpoint_sha256": {
            task: _sha256(_run(release, "star_gnn", "Base", task) / "checkpoint_best.pt")
            for task in TASKS
        },
        "removed_model": "baselines/dcrnn",
        "removed_files": removed_inventory,
        "removed_summary_rows": removed_rows,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "test_used_for_selection": False,
    }
    (release / "DCRNN_BASE_UNIFICATION.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    registered = _rewrite_checksums(release)

    counts = {
        "checkpoint": len(list(release.rglob("checkpoint_best.pt"))),
        "prediction": len(list(release.rglob("predictions.npz"))),
        "summary": len(list(release.rglob("test_summary.json"))),
    }
    if counts != {"checkpoint": 10, "prediction": 10, "summary": 10}:
        raise ValueError(f"统一后冻结工件数量不正确：{counts}")
    print("DCRNN/Base唯一化：PASS")
    print("保留：star_gnn/Base（24h、168h）")
    print("删除：baselines/dcrnn（24h、168h）")
    print(f"冻结文件SHA重新登记：{registered}个")
    print("checkpoint/predictions/test_summary：10/10/10")
    print("训练：未执行")


if __name__ == "__main__":
    main()
