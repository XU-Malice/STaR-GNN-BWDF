"""Verified, reversible organization of completed Que experiment directories."""
from __future__ import annotations

import fcntl
import errno
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPPORT = load_module("que_organization_retention_fixture", ROOT / "tests/test_que_paper_finalization.py")
sys.path.insert(0, str(ROOT / "scripts/reproduce"))
try:
    ORG = load_module("que_history_organization_test_target", ROOT / "scripts/reproduce/organize_que_history.py")
finally:
    sys.path.pop(0)
evidence = SUPPORT.evidence


@pytest.fixture
def completed(evidence, monkeypatch):
    """Use a real six-model retention archive, including 24 opaque checkpoints."""
    e = evidence
    SUPPORT.FINAL.finalize(e["project"], e["selection"], e["destination"], execute=True)
    extra = e["project"] / "results/campaign_b"
    (extra / "empty_subdirectory").mkdir(parents=True)
    (extra / "parameters.yaml").write_text("learning_rate: 0.001\n")
    (extra / "metrics.csv").write_text("MAE,RMSE\n1.2,1.5\n")
    monkeypatch.setattr(ORG, "CANONICAL_RELATIVE", "results/paper_final")
    monkeypatch.setattr(ORG, "HISTORY_ROOTS", ("results/campaign", "results/campaign_b"))
    monkeypatch.setattr(ORG.retention, "HISTORY_ROOTS", ("results/campaign",))
    monkeypatch.setattr(ORG.retention, "assert_idle", lambda project: None)
    e["extra"] = extra
    e["archive"] = e["project"] / ORG.ARCHIVE_RELATIVE
    return e


def test_organize_preserves_every_historical_file_and_fixed_models(completed):
    e = completed
    chosen_before = SUPPORT.inventory(e["destination"])
    before = {name: SUPPORT.inventory(e["project"] / "results" / name)
              for name in ("campaign", "campaign_b")}
    original_inodes = {name: (e["project"] / "results" / name).stat().st_ino for name in before}

    ORG.organize(e["project"], execute=True)

    for name, expected in before.items():
        assert not (e["project"] / "results" / name).exists()
        assert SUPPORT.inventory(e["archive"] / name) == expected
        assert (e["archive"] / name).stat().st_ino == original_inodes[name]
    assert (e["archive"] / "campaign_b/empty_subdirectory").is_dir()
    assert SUPPORT.inventory(e["destination"]) == chosen_before
    assert len(list((e["destination"] / "models").rglob("*.pt"))) == 24
    assert e["outside"].read_bytes() == b"not this reproduction project campaign"
    mapping = (e["archive"] / "path_mapping.tsv").read_text()
    assert "results/campaign" in mapping and "results/archive/que_reproduction/campaign" in mapping
    assert (e["archive"] / "README_CN.md").is_file()
    ORG.retention.verify_ready(e["destination"], e["selection"])


def test_preview_keeps_all_histories_in_original_locations(completed):
    e = completed
    before = SUPPORT.inventory(e["history"])
    ORG.organize(e["project"])
    assert SUPPORT.inventory(e["history"]) == before
    assert e["extra"].is_dir()
    assert not (e["archive"] / "campaign").exists()
    assert not (e["archive"] / "campaign_b").exists()
    assert (e["archive"] / "organization_plan.json").is_file()


@pytest.mark.parametrize("status", ["READY", "running", "FAILED"])
def test_noncompleted_cleanup_blocks_every_move(completed, status):
    e = completed
    path = e["destination"] / "cleanup_status.json"
    value = json.loads(path.read_text())
    value["status"] = status
    SUPPORT.write_json(path, value)
    with pytest.raises((RuntimeError, ValueError)):
        ORG.organize(e["project"], execute=True)
    assert e["history"].is_dir() and e["extra"].is_dir()


def test_cleanup_count_mismatch_blocks_every_move(completed):
    e = completed
    path = e["destination"] / "cleanup_status.json"
    value = json.loads(path.read_text())
    value["deleted_files"] -= 1
    SUPPORT.write_json(path, value)
    with pytest.raises((RuntimeError, ValueError)):
        ORG.organize(e["project"], execute=True)
    assert e["history"].is_dir() and e["extra"].is_dir()


def test_corrupted_canonical_model_blocks_every_move(completed):
    e = completed
    checkpoint = next((e["destination"] / "models").rglob("*.pt"))
    checkpoint.write_bytes(b"damaged canonical checkpoint")
    with pytest.raises((RuntimeError, ValueError)):
        ORG.organize(e["project"], execute=True)
    assert e["history"].is_dir() and e["extra"].is_dir()


def test_unlisted_history_and_other_project_results_are_untouched(completed):
    e = completed
    unrelated = ("que_new_unlisted_experiment", "cleanroom_validation", "graph", "paper", "logs")
    for name in unrelated:
        folder = e["project"] / "results" / name
        folder.mkdir()
        (folder / "keep.txt").write_text(name)
    ORG.organize(e["project"], execute=True)
    for name in unrelated:
        assert (e["project"] / "results" / name / "keep.txt").read_text() == name
        assert not (e["archive"] / name).exists()


def test_destination_collision_never_overwrites_existing_directory(completed):
    e = completed
    collision = e["archive"] / "campaign_b"
    collision.mkdir(parents=True)
    (collision / "existing.txt").write_text("already here")
    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        ORG.organize(e["project"], execute=True)
    assert (collision / "existing.txt").read_text() == "already here"
    assert e["history"].is_dir() and e["extra"].is_dir()


@pytest.mark.parametrize("kind", ["root", "inside", "archive_parent"])
def test_symlinks_are_refused_without_modifying_target(completed, tmp_path, kind):
    e = completed
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep.txt").write_text("external data")
    if kind == "root":
        e["extra"].rename(tmp_path / "original_campaign_b")
        e["extra"].symlink_to(external, target_is_directory=True)
    elif kind == "inside":
        (e["extra"] / "linked").symlink_to(external, target_is_directory=True)
    else:
        (e["project"] / "results/archive").symlink_to(external, target_is_directory=True)
    with pytest.raises((RuntimeError, ValueError, OSError)):
        ORG.organize(e["project"], execute=True)
    assert (external / "keep.txt").read_text() == "external data"
    assert e["history"].is_dir()


def test_changed_file_after_preview_refuses_before_any_rename(completed):
    e = completed
    ORG.organize(e["project"])
    (e["extra"] / "parameters.yaml").write_text("learning_rate: 0.2\n")
    with pytest.raises((RuntimeError, ValueError)):
        ORG.organize(e["project"], execute=True)
    assert e["history"].is_dir() and e["extra"].is_dir()


def test_partial_move_after_rename_before_receipt_resumes_safely(completed, monkeypatch):
    e = completed
    before = {name: SUPPORT.inventory(e["project"] / "results" / name)
              for name in ("campaign", "campaign_b")}
    real_rename = ORG.rename_noreplace
    failed = False

    def interrupted(source, destination):
        nonlocal failed
        real_rename(source, destination)
        if not failed:
            failed = True
            raise OSError("simulated interruption after rename")

    monkeypatch.setattr(ORG, "rename_noreplace", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        ORG.organize(e["project"], execute=True)
    assert sum((e["archive"] / name).is_dir() for name in before) == 1
    monkeypatch.setattr(ORG, "rename_noreplace", real_rename)
    ORG.organize(e["project"], execute=True)
    for name, expected in before.items():
        assert SUPPORT.inventory(e["archive"] / name) == expected
        assert not (e["project"] / "results" / name).exists()


def test_modified_partial_destination_is_not_accepted_on_resume(completed, monkeypatch):
    e = completed
    real_rename = ORG.rename_noreplace

    def interrupted(source, destination):
        real_rename(source, destination)
        raise OSError("simulated interruption after rename")

    monkeypatch.setattr(ORG, "rename_noreplace", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        ORG.organize(e["project"], execute=True)
    moved = next(p for p in (e["archive"] / "campaign", e["archive"] / "campaign_b") if p.is_dir())
    next(p for p in moved.rglob("*") if p.is_file()).write_bytes(b"changed after interruption")
    monkeypatch.setattr(ORG, "rename_noreplace", real_rename)
    with pytest.raises((RuntimeError, ValueError)):
        ORG.organize(e["project"], execute=True)
    assert sum((e["archive"] / name).is_dir() for name in ("campaign", "campaign_b")) == 1


def test_successful_repeat_is_idempotent(completed, monkeypatch):
    e = completed
    ORG.organize(e["project"], execute=True)
    before = {name: SUPPORT.inventory(e["archive"] / name) for name in ("campaign", "campaign_b")}

    def no_more_renames(*args):
        raise AssertionError("idempotent call must not rename again")

    monkeypatch.setattr(ORG, "rename_noreplace", no_more_renames)
    ORG.organize(e["project"], execute=True)
    for name, expected in before.items():
        assert SUPPORT.inventory(e["archive"] / name) == expected


def test_active_closeout_lock_blocks_organization(completed):
    e = completed
    lock = e["project"] / "logs/que_paper_closeout.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises((RuntimeError, BlockingIOError)):
            ORG.organize(e["project"], execute=True)
    assert e["history"].is_dir() and e["extra"].is_dir()


def test_atomic_rename_refuses_even_an_empty_existing_destination(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "saved_result.csv").write_text("MAE\n1.2\n")
    identities = source.stat().st_ino, destination.stat().st_ino
    with pytest.raises(OSError) as error:
        ORG.rename_noreplace(source, destination)
    assert error.value.errno in (errno.EEXIST, errno.ENOTEMPTY)
    assert (source.stat().st_ino, destination.stat().st_ino) == identities
    assert (source / "saved_result.csv").read_text() == "MAE\n1.2\n"
    assert not list(destination.iterdir())


def test_missing_libc_renameat2_symbol_uses_working_no_overwrite_syscall(tmp_path, monkeypatch):
    if sys.platform != "linux" or ORG.platform.machine().lower() not in ("x86_64", "amd64", "aarch64", "arm64"):
        pytest.skip("bounded Linux syscall fallback is not applicable on this platform")
    real_libc = ORG.ctypes.CDLL(None, use_errno=True)

    class LibcWithoutRenameat2:
        syscall = real_libc.syscall

    monkeypatch.setattr(ORG.ctypes, "CDLL", lambda *args, **kwargs: LibcWithoutRenameat2())
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    (source / "saved_result.csv").write_text("MAE\n1.2\n")
    original_inode = source.stat().st_ino
    ORG.rename_noreplace(source, destination)
    assert not source.exists()
    assert destination.stat().st_ino == original_inode
    assert (destination / "saved_result.csv").read_text() == "MAE\n1.2\n"
    source.mkdir()
    with pytest.raises(OSError) as error:
        ORG.rename_noreplace(source, destination)
    assert error.value.errno in (errno.EEXIST, errno.ENOTEMPTY)
    assert source.is_dir() and destination.stat().st_ino == original_inode
