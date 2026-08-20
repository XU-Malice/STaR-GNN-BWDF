#!/usr/bin/env python
"""Regenerate SOURCE_CHECKSUMS.sha256 from the tracked public source boundary.

This command is for release preparation.  It deliberately excludes generated
model/data/result artifacts while sealing source code, configs, tests, root
metadata, documentation, and manuscript captions/README files.  A released
checkout should normally run ``verify_source.sh`` rather than regenerate the
manifest.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "SOURCE_CHECKSUMS.sha256"

EXCLUDED_PREFIXES = (
    "artifacts/",
    "data/processed/",
    "dist/",
    "paper/figures/",
    "paper/reports/",
    "paper/tables/",
    "results/",
    "repos/",
)
EXCLUDED_EXACT = {
    "SOURCE_CHECKSUMS.sha256",
}
EXCLUDED_PARTS = {
    ".pytest_cache",
    "__pycache__",
}


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    files = [
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]
    if not files:
        raise RuntimeError("git ls-files returned no tracked files")
    return files


def _include(relative: str) -> bool:
    path = Path(relative)
    if relative in EXCLUDED_EXACT:
        return False
    if any(relative.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        # paper/README.md and paper/captions are manuscript source rather than
        # generated outputs, so retain those explicitly.
        if relative == "paper/README.md" or relative.startswith("paper/captions/"):
            return True
        return False
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    selected: list[str] = []
    for relative in sorted(_tracked_files()):
        if not _include(relative):
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        selected.append(relative)

    lines = [f"{_sha256(ROOT / relative)}  ./{relative}" for relative in selected]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"SOURCE_CHECKSUMS regenerated: {len(selected)} files")
    print(MANIFEST)


if __name__ == "__main__":
    main()
