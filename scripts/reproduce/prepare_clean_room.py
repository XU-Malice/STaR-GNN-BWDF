#!/usr/bin/env python
"""从公开源码清单创建全新工作副本，并装入冻结checkpoint发布包。

该脚本只复制 ``SOURCE_CHECKSUMS.sha256`` 登记的公开文件，因此不会把当前
项目的处理数据、训练结果、缓存或本地环境带入 clean-room。冻结发布包作为
独立输入复制，模拟 GitHub 源码仓库与 Release asset 分开下载的真实流程。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_relative(root: Path, value: str, *, label: str) -> tuple[str, Path]:
    """返回严格位于 ``root`` 内的规范相对路径和绝对路径。"""
    relative = value.removeprefix("./")
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label}包含非法路径：{value!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved_root not in resolved.parents:
        raise ValueError(f"{label}包含越界路径：{value!r}")
    return candidate.as_posix(), resolved


def _source_entries(root: Path) -> list[tuple[str, Path]]:
    checksum_path = root / "SOURCE_CHECKSUMS.sha256"
    entries: list[tuple[str, Path]] = []
    listed: set[str] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        relative, source = _checked_relative(
            root,
            relative,
            label="源码SHA清单",
        )
        if relative in listed:
            raise ValueError(f"源码SHA条目重复：{relative}")
        listed.add(relative)
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = _sha256(source)
        if actual != expected:
            raise ValueError(f"源码SHA不一致：{relative}")
        entries.append((relative, source))
    return entries


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as handle:
        root = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"压缩包包含越界路径：{member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"冻结包不允许符号链接：{member.name}")
        handle.extractall(destination, filter="data")


def _resolve_release(source: Path, temporary: Path) -> Path:
    if source.is_dir():
        return source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    extracted = temporary / "release_extracted"
    extracted.mkdir()
    _safe_extract(source.resolve(), extracted)
    candidates = [
        path.parent
        for path in extracted.rglob("MANIFEST.json")
        if (path.parent / "CHECKSUMS.sha256").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "冻结压缩包必须且只能包含一个MANIFEST.json/CHECKSUMS.sha256根目录。"
        )
    return candidates[0]


def _verify_release(root: Path) -> None:
    checksum = root / "CHECKSUMS.sha256"
    if not checksum.is_file() or not (root / "MANIFEST.json").is_file():
        raise FileNotFoundError("冻结发布包缺少MANIFEST或CHECKSUMS。")
    listed: set[str] = set()
    for line in checksum.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        relative, path = _checked_relative(
            root,
            relative,
            label="冻结包SHA清单",
        )
        if relative in listed:
            raise ValueError(f"冻结包SHA条目重复：{relative}")
        listed.add(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"冻结包文件缺失或SHA错误：{relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if actual != listed:
        raise ValueError(
            "冻结包文件集合与SHA清单不一致："
            f"missing={sorted(listed - actual)}, extra={sorted(actual - listed)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--frozen-release", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"clean-room目标已存在，拒绝覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    temporary = Path(tempfile.mkdtemp(prefix="release.", dir=staging))
    try:
        entries = _source_entries(source_root)
        for relative, source in entries:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copy2(
            source_root / "SOURCE_CHECKSUMS.sha256",
            staging / "SOURCE_CHECKSUMS.sha256",
        )

        release = _resolve_release(args.frozen_release, temporary)
        _verify_release(release)
        frozen_target = staging / "results/paper/frozen_v1"
        frozen_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(release, frozen_target)
        shutil.rmtree(temporary)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"clean-room源码：{destination}")
    print(f"公开源码文件：{len(entries)}个，SHA全部通过")
    print("冻结checkpoint发布包：完整复制并通过SHA检查")
    print("当前项目的处理数据、图和训练结果：未复制")


if __name__ == "__main__":
    main()
