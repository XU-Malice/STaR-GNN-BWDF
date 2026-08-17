from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_manifest_has_one_dcrnn_base_identity() -> None:
    manifest = json.loads(
        (ROOT / "results/paper/frozen_v1/MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    artifacts = manifest["artifacts"]
    assert len(artifacts) == 10
    assert "star_gnn/Base/24h" in artifacts
    assert "star_gnn/Base/168h" in artifacts
    assert "baselines/dcrnn/24h" not in artifacts
    assert "baselines/dcrnn/168h" not in artifacts


def test_reproduction_does_not_train_a_second_dcrnn() -> None:
    source = (ROOT / "scripts/reproduce/reproduce.py").read_text(encoding="utf-8")
    assert 'for model in ("dcrnn", "stgcn")' not in source
    assert '"baselines" / "stgcn"' in source
    assert '"backbone"' in (
        ROOT / "scripts/reproduce/paper_release_lib.py"
    ).read_text(encoding="utf-8")


def test_public_paper_configs_have_no_second_dcrnn_entry() -> None:
    source_manifest = (ROOT / "SOURCE_CHECKSUMS.sha256").read_text(
        encoding="utf-8"
    )
    assert "configs/paper/dcrnn_24h.yaml" not in source_manifest
    assert "configs/paper/dcrnn_168h.yaml" not in source_manifest
    assert "configs/paper/star_gnn_24h.yaml" in source_manifest
    assert "configs/paper/star_gnn_168h.yaml" in source_manifest
