from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_clean_room",
    ROOT / "scripts/reproduce/prepare_clean_room.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_manifest_rejects_tampering_and_traversal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "a.txt"
    payload.write_text("registered\n", encoding="utf-8")
    manifest = source / "SOURCE_CHECKSUMS.sha256"
    manifest.write_text(f"{_sha(payload)}  ./a.txt\n", encoding="utf-8")
    entries = MODULE._source_entries(source)
    assert [(relative, path.name) for relative, path in entries] == [
        ("a.txt", "a.txt")
    ]

    payload.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="源码SHA不一致"):
        MODULE._source_entries(source)

    manifest.write_text(f"{'0' * 64}  ../outside.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非法路径"):
        MODULE._source_entries(source)


def test_release_requires_exact_registered_file_set(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    manifest = release / "MANIFEST.json"
    manifest.write_text("{}\n", encoding="utf-8")
    checksums = release / "CHECKSUMS.sha256"
    checksums.write_text(
        f"{_sha(manifest)}  MANIFEST.json\n",
        encoding="utf-8",
    )
    MODULE._verify_release(release)

    extra = release / "unregistered.bin"
    extra.write_bytes(b"not registered")
    with pytest.raises(ValueError, match="文件集合"):
        MODULE._verify_release(release)


def test_release_manifest_rejects_parent_path(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    (release / "MANIFEST.json").write_text("{}\n", encoding="utf-8")
    (release / "CHECKSUMS.sha256").write_text(
        f"{'0' * 64}  ../outside.bin\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="非法路径"):
        MODULE._verify_release(release)
