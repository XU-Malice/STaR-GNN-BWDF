"""The external observer must preserve verified results before signalling a queue."""
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("total_watch_main_test", ROOT / "scripts/reproduce/watch_que_total_match.py")
watch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = watch
spec.loader.exec_module(watch)


def report(matched=True):
    models = {}
    for i, model in enumerate(watch.MODELS):
        good = matched or i < 5
        rows = [dict(task=task, series="total", metric=metric, value=1.0,
                     paper_value=1.0, difference=0.0 if good else .1,
                     absolute_relative_difference=0.0 if good else .1,
                     tolerance_ratio=0.0 if good else 2.0)
                for task in ("24h", "168h") for metric in ("MAE", "MAPE", "RMSE", "NSE")]
        models[model] = dict(case=f"A_{model}_example", model=model, metrics=rows,
                             matched_count=8 if good else 0, all8_matched=good,
                             settings={}, worst_ratio=0 if good else 2)
    return dict(models=models, all_matched=matched, verification="tentative")


@pytest.fixture
def args(tmp_path):
    project = tmp_path / "project"
    result = project / "results" / watch.DEFAULT_TAG
    result.mkdir(parents=True)
    (result / "queue_status.json").write_text(json.dumps({"status": "running", "cases": []}))
    return watch.parse_args(["--project-root", str(project), "--watch", "--stop-on-success"])


class Handle:
    def __init__(self, events):
        self.events, self.stopped = events, False
    def describe(self):
        return {"pid": 90000, "start_ticks": 123}
    def exited(self):
        return self.stopped
    def request_stop(self):
        self.events.append("stop")
        self.stopped = True
        return True
    def close(self):
        self.events.append("close")


def context(events, values=None, error=None):
    class FakeContext:
        def __init__(self, *args, **kwargs):
            self.values = iter(values or [report()])
        def scan(self, queue):
            return next(self.values)
        def verify_selected(self, value):
            events.append("verify")
            if error == "verify":
                raise ValueError("invalid predictions")
            return {**value, "verified_all_matched": error != "false_verified", "verification": "raw_predictions_and_receipts_verified"}
        def export_to(self, target, value):
            events.append("export")
            if error == "export":
                raise OSError("disk full")
            target.mkdir(parents=True)
            (target / "checkpoint.pt").write_bytes(b"opaque checkpoint bytes, never unpickled")
            (target / "selection.json").write_text(json.dumps(value))
            return target
    return FakeContext


def test_all_six_requires_verified_frozen_archive_before_stop(args, monkeypatch):
    events = []
    real = watch.make_archive
    def archive(path):
        result = real(path)
        events.append("archive")
        return result
    monkeypatch.setattr(watch, "make_archive", archive)
    handle = Handle(events)
    assert watch.run(args, context_factory=context(events), binder=lambda *a: handle) == 0
    assert events == ["verify", "export", "archive", "stop", "close"]
    state = json.loads((args.output_root / "watch_status.json").read_text())
    assert state["status"] == "matched_and_queue_stopped"
    assert state["matched_model_count"] == 6
    assert state["matched_total_cells"] == 48
    with tarfile.open(state["archive"]["path"]) as bundle:
        assert any(x.name.endswith("checkpoint.pt") for x in bundle.getmembers())


@pytest.mark.parametrize("failure", ["verify", "false_verified", "export", "archive"])
def test_failed_validation_or_preservation_never_signals(args, monkeypatch, failure):
    events = []
    if failure == "archive":
        def bad_archive(path):
            raise OSError("no disk space")
        monkeypatch.setattr(watch, "make_archive", bad_archive)
    assert watch.run(args, context_factory=context(events, error=failure), binder=lambda *a: Handle(events)) == 2
    assert "stop" not in events


def test_five_matching_models_do_not_stop_queue(args):
    events = []
    def sleeper(seconds):
        events.append("wait")
        (args.result_root / "queue_status.json").write_text(json.dumps({"status": "completed_with_failures", "cases": []}))
    assert watch.run(args, context_factory=context(events, values=[report(False), report(False)]),
                     binder=lambda *a: Handle(events), sleeper=sleeper) == 0
    assert events == ["wait", "close"]
    assert json.loads((args.output_root / "watch_status.json").read_text())["status"] == "queue_ended_without_all_total_matches"


def test_monitor_only_and_stale_queue_cannot_signal(args):
    events = []
    args.stop_on_success = False
    assert watch.run(args, context_factory=context(events), binder=lambda *a: Handle(events)) == 0
    assert "stop" not in events


def test_monitor_only_never_requires_process_capabilities(args):
    events = []
    args.stop_on_success = False
    def prohibited(*unused):
        pytest.fail("Passive observation must not inspect or bind a process")
    assert watch.run(args, context_factory=context(events), binder=prohibited) == 0
    assert "stop" not in events


def test_missing_pidfd_keeps_polling_and_preserves_accepted_results(args, monkeypatch):
    events = []
    class PidfdUnavailable(RuntimeError):
        pass
    def unsupported(*unused):
        events.append("bind_unavailable")
        raise PidfdUnavailable("Python and libc wrappers absent")
    monkeypatch.setitem(sys.modules, "que_total_watch_process",
                        SimpleNamespace(bind_queue=unsupported, PidfdUnavailable=PidfdUnavailable))
    assert watch.run(args, context_factory=context(events, values=[report(False), report()]),
                     sleeper=lambda s: events.append("wait")) == 0
    assert events == ["bind_unavailable", "wait", "verify", "export"]
    state = json.loads((args.output_root / "watch_status.json").read_text())
    assert state["status"] == "matched_and_preserved"
    assert state["matched_model_count"] == 6 and state["matched_total_cells"] == 48
    assert state["automatic_stop_available"] is False
    assert "PidfdUnavailable" in state["capability_warning"]
    assert Path(state["archive"]["path"]).is_file()


@pytest.mark.parametrize("error", [PermissionError("not owned"), ValueError("identity changed")])
def test_compatibility_handling_never_relaxes_identity_or_permission_refusal(args, monkeypatch, error):
    class PidfdUnavailable(RuntimeError):
        pass
    def refused(*unused):
        raise error
    monkeypatch.setitem(sys.modules, "que_total_watch_process",
                        SimpleNamespace(bind_queue=refused, PidfdUnavailable=PidfdUnavailable))
    events = []
    assert watch.run(args, context_factory=context(events)) == 2
    assert events == []


def test_failed_start_status_does_not_report_results_reset_to_zero(args, capsys):
    def refused(*unused):
        raise ValueError("identity mismatch")
    assert watch.run(args, binder=refused) == 2
    capsys.readouterr()
    assert watch.print_status(args.output_root) == 0
    printed = capsys.readouterr().out
    assert "不代表已有结果归零" in printed
    assert "0/6" not in printed and "0/48" not in printed


def test_wrong_queue_binding_fails_before_evidence_or_signal(args):
    events = []
    def refused(*unused):
        raise ValueError("PID belongs to a different queue")
    assert watch.run(args, context_factory=context(events), binder=refused) == 2
    assert events == []


def test_once_has_no_process_binding(args):
    events = []
    args.watch = args.stop_on_success = False
    def forbidden(*unused):
        raise AssertionError("once must not inspect process")
    assert watch.run(args, context_factory=context(events, values=[report(False)]), binder=forbidden) == 0
    assert events == []


def test_pid_handle_is_bound_only_once_across_polling(args):
    events = []
    def binder(*unused):
        events.append("bind")
        return Handle(events)
    assert watch.run(args, context_factory=context(events, values=[report(False), report()]),
                     binder=binder, sleeper=lambda sec: None) == 0
    assert events.count("bind") == 1
    assert events.count("stop") == 1


def test_no_stop_without_live_bound_queue(args):
    events = []
    assert watch.run(args, context_factory=context(events), binder=lambda *a: None) == 0
    assert "stop" not in events


def test_missing_pid_file_is_not_a_finished_queue(args):
    events = []
    assert watch.run(args, context_factory=context(events, values=[report(False), report()]),
                     binder=lambda *a: None, sleeper=lambda s: events.append("wait")) == 0
    assert events[0] == "wait"
    assert "stop" not in events


def test_cleanup_timeout_never_sends_second_signal_or_escalates(args):
    events = []
    class SlowHandle(Handle):
        def exited(self):
            return False
    assert watch.run(args, context_factory=context(events), binder=lambda *a: SlowHandle(events),
                     sleeper=lambda sec: events.append("cleanup_wait")) == 0
    assert events.count("stop") == 1
    assert events.count("cleanup_wait") == 30
    state = json.loads((args.output_root / "watch_status.json").read_text())
    assert state["status"] == "matched_stop_requested_cleanup_pending"


def test_all_empty_candidates_do_not_retain_stale_comparison(args):
    events = []
    empty = dict(models={m: None for m in watch.MODELS}, all_matched=False, verification="tentative")
    args.watch = args.stop_on_success = False
    args.output_root.mkdir(parents=True)
    (args.output_root / "total_comparison.tsv").write_text("stale candidate\n")
    assert watch.run(args, context_factory=context(events, values=[empty])) == 0
    assert "stale candidate" not in (args.output_root / "total_comparison.tsv").read_text()


def test_duplicate_observer_preserves_active_status(args):
    args.output_root.mkdir(parents=True)
    status = args.output_root / "watch_status.json"
    status.write_text('{"status":"active-other-observer"}')
    with (args.output_root / "watch.lock").open("a+") as lock:
        watch.fcntl.flock(lock, watch.fcntl.LOCK_EX | watch.fcntl.LOCK_NB)
        assert watch.run(args, context_factory=lambda *a, **k: pytest.fail("already locked")) == 2
    assert json.loads(status.read_text())["status"] == "active-other-observer"


@pytest.mark.parametrize("folder", ["src", "scripts", "tests", "configs", "data", "results"])
def test_cli_rejects_output_that_overwrites_active_sources_or_results(tmp_path, folder):
    with pytest.raises(SystemExit):
        watch.parse_args(["--project-root", str(tmp_path), "--output-root", str(tmp_path / folder)])


@pytest.mark.parametrize("options", [["--stop-on-success"], ["--run-tag", "../wrong"],
                                     ["--poll-seconds", "0"], ["--error-relative-tolerance", "nan"],
                                     ["--nse-absolute-tolerance", "-1"]])
def test_invalid_cli_has_no_side_effects(options):
    with pytest.raises(SystemExit):
        watch.parse_args(options)


def test_archive_rejects_symlink(tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "external").symlink_to(__file__)
    with pytest.raises(ValueError):
        watch.make_archive(selected)


def test_archive_detects_source_change_during_pack(tmp_path, monkeypatch):
    selected = tmp_path / "selected"
    selected.mkdir()
    source = selected / "weights.pt"
    source.write_bytes(b"original")
    original = watch.tarfile.TarFile.add
    def changed(self, name, *a, **kw):
        result = original(self, name, *a, **kw)
        if Path(name) == selected:
            source.write_bytes(b"changed")
        return result
    monkeypatch.setattr(watch.tarfile.TarFile, "add", changed)
    with pytest.raises(ValueError):
        watch.make_archive(selected)
    assert not selected.with_suffix(".tar.gz").exists()


def test_status_needs_no_training_or_gpu_environment(tmp_path):
    args = watch.parse_args(["--project-root", str(tmp_path), "--status"])
    assert watch.run(args, context_factory=lambda *a, **kw: pytest.fail("no evidence import"),
                     binder=lambda *a: pytest.fail("no process binding")) == 1
