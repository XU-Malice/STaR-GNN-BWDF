"""Regression guards for shared CLI dispatch and DCRNN isolation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_script(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(
        path.stem + "_test_module",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unified_training_config_dispatch_is_model_specific() -> None:
    module = _load_script("scripts/train/train_model.py")
    assert module._default_config("dcrnn", "24h") == (
        ROOT / "configs" / "train" / "dcrnn_24h.yaml"
    )
    assert module._default_config("stgcn", "168h") == (
        ROOT / "configs" / "train" / "stgcn_168h.yaml"
    )


def test_unified_test_checkpoint_paths_do_not_overlap() -> None:
    module = _load_script("scripts/evaluate/test_model.py")
    dcrnn = module._default_checkpoint("dcrnn", "24h", 0)
    stgcn = module._default_checkpoint("stgcn", "24h", 0)
    assert dcrnn != stgcn
    assert "/dcrnn/" in str(dcrnn)
    assert "/stgcn/" in str(stgcn)


def test_dcrnn_and_stgcn_share_protocol_not_decoder_policy() -> None:
    dcrnn = yaml.safe_load(
        (ROOT / "configs" / "train" / "dcrnn_24h.yaml").read_text(
            encoding="utf-8"
        )
    )
    stgcn = yaml.safe_load(
        (ROOT / "configs" / "train" / "stgcn_24h.yaml").read_text(
            encoding="utf-8"
        )
    )
    for key in ("source", "task", "features", "split", "evaluation"):
        assert dcrnn[key] == stgcn[key]
    assert "scheduled_sampling" in dcrnn["training"]
    assert "scheduled_sampling" not in stgcn["training"]
    assert dcrnn["output"]["base_dir"] != stgcn["output"]["base_dir"]
