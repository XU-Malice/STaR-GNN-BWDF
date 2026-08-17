#!/usr/bin/env python
"""生成可上传到 GitHub Release 的确定性冻结 checkpoint 压缩包。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_release(release: Path) -> list[Path]:
    manifest_path = release / "MANIFEST.json"
    checksum_path = release / "CHECKSUMS.sha256"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError("冻结目录缺少 MANIFEST.json 或 CHECKSUMS.sha256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("artifacts", {})) != 10:
        raise ValueError("只允许打包已经完成 DCRNN/Base 唯一化的10组冻结工件。")
    if (release / "models/baselines/dcrnn").exists():
        raise ValueError("仍存在重复 baselines/dcrnn，拒绝打包。")

    listed: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in listed:
            raise ValueError(f"冻结 SHA 重复登记：{relative}")
        listed[relative] = expected
    actual = {
        path.relative_to(release).as_posix(): path
        for path in release.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if set(actual) != set(listed):
        raise ValueError("冻结目录文件集合与 CHECKSUMS.sha256 不一致。")
    for relative, path in actual.items():
        if _sha256(path) != listed[relative]:
            raise ValueError(f"冻结文件 SHA 不一致：{relative}")
    return sorted(
        (path for path in release.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(release).as_posix(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=PROJECT_ROOT / "results/paper/frozen_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist/STaR-GNN-BWDF-frozen-v1.tar.gz",
    )
    args = parser.parse_args()
    release = args.release.resolve()
    output = args.output.resolve()
    if release == output.parent or release in output.parents:
        raise ValueError("输出压缩包不能放进冻结目录，否则会污染CHECKSUMS文件集合。")
    files = _verify_release(release)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(f"发布资产已存在，拒绝覆盖：{output}")

    # 固定 mtime、uid、gid 和文件顺序，使相同冻结目录生成相同 SHA。
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for source in files:
                    relative = source.relative_to(release)
                    target = Path("results/paper/frozen_v1") / relative
                    info = archive.gettarinfo(str(source), arcname=target.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)

    digest = _sha256(output)
    checksum = output.with_name(output.name + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    size_mib = output.stat().st_size / (1024 * 1024)
    print("GitHub Release checkpoint资产：PASS")
    print(f"文件：{output}")
    print(f"大小：{size_mib:.2f} MiB")
    print(f"SHA-256：{digest}")
    print(f"校验文件：{checksum}")


if __name__ == "__main__":
    main()
