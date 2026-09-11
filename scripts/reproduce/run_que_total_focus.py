#!/usr/bin/env python3
"""External total-only reconstruction campaign; never edits the live training tree."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

TOOL_ROOT = Path(__file__).resolve().parents[2]
MODELS = ("gru", "lstm", "msnet", "mscmnet_m", "mscmnet_wm", "mscmnet_w")
NAMES = ("GRU", "LSTM", "MSNet", "MSCMNet_M", "MSCMNet_WM", "MSCMNet_W")
KEYS = tuple((task, metric) for task in ("24h", "168h") for metric in ("MAE", "MAPE", "RMSE", "NSE"))
TERMINAL = {"completed", "completed_with_failures", "failed", "interrupted", "paused_time_budget"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def publish(output, state, **updates):
    state.update(updates, updated_utc=datetime.now(timezone.utc).isoformat())
    write(output / "campaign_status.json", state)


def total_rows(model, values, paper):
    rows = []
    for task, metric in KEYS:
        value = float(values[task][metric])
        target = float(paper["tasks"][task][NAMES[MODELS.index(model)]]["total"][metric])
        if not math.isfinite(value) or not math.isfinite(target) or target == 0:
            raise ValueError("Incomplete or nonfinite total metrics")
        delta = value - target
        ratio = abs(delta) / (.01 if metric == "NSE" else abs(target) * .05)
        rows.append(dict(task=task, series="total", metric=metric, value=value, paper_value=target,
                         difference=delta, absolute_relative_difference=abs(delta / target),
                         tolerance_ratio=ratio, within_tolerance=ratio <= 1))
    return rows


def choose_models(candidates):
    selected = {}
    for model in MODELS:
        group = [c for c in candidates if c["model"] == model and len(c["metrics"]) == 8]
        if group:
            selected[model] = min(group, key=lambda c: (max(r["tolerance_ratio"] for r in c["metrics"]),
                                                       sum(r["tolerance_ratio"] for r in c["metrics"]) / 8, c["case"]))
    return selected


def complete_records(ranking, mode, paper):
    result = []
    for candidate in ranking:
        if candidate["mode"] != mode:
            continue
        if len(candidate["metrics"]) != len(KEYS):
            raise ValueError("Audit ranking must contain all eight total metrics")
        values = {task: {} for task in ("24h", "168h")}
        for (task, metric), value in zip(KEYS, candidate["metrics"]):
            values[task][metric] = value
        result.append({**candidate, "metrics": total_rows(candidate["model"], values, paper),
                       "kind": "complete_training", "technical_status": "PASS"})
    return result


def save_table(output, selected, mode="pooled"):
    suffix = "" if mode == "pooled" else "_diagnostic_origin_mean"
    rows = [{"model": m, "case": c["case"], **r} for m, c in selected.items() for r in c["metrics"]]
    path = output / f"total_comparison{suffix}.tsv"
    temporary = path.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["model", "case"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    lines = ["|模型|总体接近项|MAE/MAPE/RMSE最大相对差|NSE最大绝对差|配置|",
             "|---|---:|---:|---:|---|"]
    for model in MODELS:
        c = selected.get(model)
        if c is None:
            lines.append(f"|{model}|无有效候选||||")
            continue
        rs = c["metrics"]
        count = sum(r["within_tolerance"] for r in rs)
        error = max(r["absolute_relative_difference"] for r in rs if r["metric"] != "NSE")
        nse = max(abs(r["difference"]) for r in rs if r["metric"] == "NSE")
        lines.append(f"|{model}|{count}/8|{error:.2%}|{nse:.5f}|{c['case']}|")
    (output / f"current_table{suffix}.md").write_text("\n".join(lines) + "\n")


def collect_validated(context, queue):
    """Validate all completed A/B/C runs in place; no cross-run copy shortcut."""
    plans, candidates, exclusions = context._plans(), [], []
    seen = set()
    for record in queue.get("cases", []):
        if not str(record.get("technical_status", "")).startswith("PASS"):
            continue
        name = record.get("case")
        if name in seen:
            raise ValueError("Duplicate source queue case")
        seen.add(name)
        try:
            case = plans[name]
            if record.get("model") != case["model"] or record.get("settings") != case or record.get("exit_code") != 0:
                raise ValueError("PASS record differs from the frozen source plan")
            run, expected = context._run(case), context._expected(case)
            receipt = read(run / "completion_receipt.json")
            for name_in_receipt in receipt.get("files", {}):
                relative = Path(name_in_receipt)
                if relative.is_absolute() or ".." in relative.parts or str(relative) != name_in_receipt:
                    raise ValueError("Unsafe source receipt path")
                path = run / relative
                if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                    raise ValueError("Missing or non-regular source evidence")
            valid, reason = context.runner.validate_case(run, case, expected)
            if not valid:
                raise ValueError(reason)
            context._verify_reuse(run, case, expected)
            candidates.append(dict(model=case["model"], case=case["case"], run=str(run), settings=case,
                                   evidence_digest=digest(receipt), technical_status="PASS", exit_code=0))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            exclusions.append(dict(case=name, reason=f"{type(exc).__name__}: {exc}"))
    context.check_fingerprints()
    return candidates, exclusions


TRAINING_ENTRY_FILES = frozenset({"scripts/train/train_temporal_baselines.py",
                                "scripts/train/que_shared_gpu_runtime.py"})
INSTALL_METADATA_FILES = frozenset({"PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt",
    "requires.txt", "top_level.txt", "namespace_packages.txt", "not-zip-safe", "zip-safe"})


def _install_metadata(name):
    parts = Path(name).parts
    return (len(parts) == 3 and parts[:2] == ("src", "star_gnn_bwdf.egg-info")
            and parts[2] in INSTALL_METADATA_FILES)


def verify_training_sources(context, tool_root, *, audit_path=None):
    """Compare identical numerical source across two installation layouts.

    Editable-install packaging metadata need not exist in a pure Git export.
    Original full fingerprints, including that metadata, remain unchanged and
    are verified before/after this check. No Python/configuration file is exempt.
    """
    context.check_fingerprints()
    tool_root = Path(tool_root)
    original = context.manifest["signatures"]["source"]
    required = {}
    for name, expected in original.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
            raise ValueError(f"Unsafe frozen source path: {name}")
        if (name.startswith(("src/", "configs/")) or name in TRAINING_ENTRY_FILES) and not _install_metadata(name):
            required[name] = expected
    if (not TRAINING_ENTRY_FILES <= required.keys() or not any(n.startswith("src/") for n in required)
            or not any(n.startswith("configs/") for n in required)):
        raise ValueError("Incomplete shared numerical source inventory")
    exported = {}
    for directory in ("src", "configs"):
        base = tool_root / directory
        if not base.is_dir() or base.is_symlink():
            raise ValueError(f"Missing or non-regular exported source directory: {directory}")
        for path in sorted(base.rglob("*")):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                raise ValueError(f"Non-regular exported source path: {path}")
            if path.is_file():
                exported[str(path.relative_to(tool_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in TRAINING_ENTRY_FILES:
        path = tool_root / name
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"Missing or non-regular exported training entry: {name}")
        exported[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = {name: sha for name, sha in exported.items() if not _install_metadata(name)}
    if set(actual) != set(required):
        raise ValueError("Supplemental training source inventory differs: "
            f"missing={sorted(set(required)-set(actual))}; extra={sorted(set(actual)-set(required))}")
    for name, expected in required.items():
        if actual[name] != expected:
            raise ValueError(f"Supplemental training core differs from existing evidence: {name}")
    metadata = [{"path": name, "original_sha256": original.get(name), "exported_sha256": exported.get(name)}
        for name in sorted(set(original) | set(exported)) if _install_metadata(name)]
    context.check_fingerprints()
    if audit_path is not None:
        write(audit_path, {"status": "PASS", "checked_source_sha256": required,
            "install_metadata": metadata, "original_full_fingerprints_preserved": True,
            "comparison": "Exact numerical source inventory and bytes; packaging metadata recorded separately"})
    if metadata:
        print(f"数值源码和配置核验通过；另记录 {len(metadata)} 个安装元数据文件，不要求其在两种安装方式间一致。", flush=True)
    return sorted(required)


def verify_candidate_evidence(candidates, context):
    for candidate in candidates:
        run = Path(candidate["run"])
        receipt = read(run / "completion_receipt.json")
        if digest(receipt) != candidate["evidence_digest"]:
            raise ValueError(f"Validated completion receipt changed: {candidate['case']}")
        request = read(run / "request_signature.json")
        if request.get("case") != candidate["settings"] or request.get("evaluation") != context.evaluation:
            raise ValueError("Validated request settings or evaluation changed")
        valid, reason = context.runner.validate_case(run, candidate["settings"], request)
        if not valid:
            raise ValueError(f"Candidate changed after collection: {candidate['case']}: {reason}")


def analyse(candidates, context, args, state, audit_api, search_api):
    output = args.output_root
    verify_candidate_evidence(candidates, context)
    identity = digest([(c["case"], c["evidence_digest"]) for c in sorted(candidates, key=lambda c: c["case"])])[:20]
    publish(output, state, status="cpu_metric_audit", candidate_count=len(candidates))
    audit = audit_api.audit_candidates(candidates, context.paper_path, output / "audits" / identity)
    ranking = read(Path(audit["candidates_summary_file"]))
    selected_by_mode = {}
    for mode in ("pooled", "origin_mean"):
        full = complete_records(ranking, mode, context.paper)
        for model in ("gru", "lstm"):
            sources = [c for c in candidates if c["model"] == model]
            if not sources:
                continue
            source_id = digest([(c["case"], c["evidence_digest"]) for c in sorted(sources, key=lambda c: c["case"])])[:20]
            target = output / "cohorts" / model / mode / source_id
            publish(output, state, status="cpu_network_search", searching_model=model, searching_mode=mode)
            report = search_api.search_and_export([Path(c["run"]) for c in sources], context.paper_path, target,
                mode=mode, starts=args.search_starts, sweeps=args.search_sweeps, pair_trials=args.pair_trials, search_seed=0)
            full.append(dict(model=model, case=f"TOTAL_D_{model}_{source_id}", run=str(target), mode=mode,
                kind="independent_network_cohort", metrics=total_rows(model, report["score"]["total_metrics"], context.paper)))
        selected_by_mode[mode] = choose_models(full)
    verify_candidate_evidence(candidates, context)
    for mode in ("pooled", "origin_mean"):
        save_table(output, selected_by_mode[mode], mode)
    selected = selected_by_mode["pooled"]
    total_passed = sum(sum(r["within_tolerance"] for r in c["metrics"]) for c in selected.values())
    model_passed = sum(all(r["within_tolerance"] for r in c["metrics"]) for c in selected.values())
    write(output / "selection.json", dict(primary_mode="pooled", selected_by_mode=selected_by_mode,
        diagnostic_mode_is_not_automatically_promoted=True, test_target_guided=True, original_paper_method_recovered=False))
    publish(output, state, status="analysis_completed", matched_total_cells=total_passed, matched_model_count=model_passed,
            feasibility_blocked=audit["pooled_total48_impossible_even_with_rounding"],
            latest_audit=str(output / "audits" / identity), selection_file=str(output / "selection.json"),
            searching_model=None, searching_mode=None)
    print((output / "current_table.md").read_text(), flush=True)
    return ranking


def terminal_and_lock(queue, project, gpu_id):
    if queue.get("status") not in TERMINAL:
        return None
    stream = (project / "logs" / f"que_gpu_{gpu_id}.lock").open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def prepare_joint_handoff(context, args, state, archive_api, stop_api):
    """Archive verified joint models before requesting the user's queue handoff."""
    output = args.output_root
    saved = output / "joint_closeout_reference.json"
    publish(output, state, status="archiving_joint_models")
    print("正在核验并归档四个联合模型与本轮全部参数记录；原队列此时继续运行。", flush=True)
    if saved.exists():
        reference = read(saved)
        report = archive_api.verify_ready(Path(reference["report_path"]).parent)
        if report["archive_sha256"] != reference["archive_sha256"]:
            raise ValueError("Joint closeout archive differs from the frozen handoff")
    else:
        report = archive_api.archive_joint_closeout(context,
            read(args.source_results / "queue_status.json"), output / "joint_closeout")
        archive_api.verify_ready(Path(report["report_path"]).parent)
        write(saved, report)
    if report.get("status") != "READY":
        raise ValueError("Joint model archive is not complete; original queue left running")
    selected = report["selected_models"]
    if set(selected) != set(MODELS[2:]):
        raise ValueError("Four complete joint models are required before handoff")
    publish(output, state, status="requesting_original_queue_stop",
        joint_archive=report["archive_path"], joint_archive_sha256=report["archive_sha256"],
        joint_selected_cases={m: selected[m]["case"] for m in MODELS[2:]})
    print(f"四个联合模型已完整归档：{report['archive_path']}\nSHA256：{report['archive_sha256']}", flush=True)
    print("正在核验 GRU/LSTM 已结束的计划和原启动器身份，再请求旧队列正常退出。", flush=True)
    result = stop_api.stop_after_closeout(project_root=args.project_root.resolve(),
        source_results=args.source_results.resolve(), archive_root=Path(report["report_path"]).parent,
        validate_archive=archive_api.verify_ready, context=context, execute=True)
    write(output / "original_queue_handoff.json", result)
    print(f"旧队列交接状态：{result['status']}；接下来只搜索和训练 GRU/LSTM。", flush=True)
    return report


def restrict_to_closed_joint_models(candidates, report):
    if report is None:
        return candidates
    selected = report["selected_models"]
    filtered = [c for c in candidates if c["model"] in ("gru", "lstm")
                or c["case"] == selected[c["model"]]["case"]]
    if {c["model"] for c in filtered if c["model"] in MODELS[2:]} != set(MODELS[2:]):
        raise ValueError("A frozen joint-model source is no longer valid")
    return filtered


def print_status(output):
    path = output / "campaign_status.json"
    if not path.exists():
        print("尚未启动总体定向搜索。")
        return 0
    s = read(path)
    print(f"状态：{s.get('status')}；更新时间：{s.get('updated_utc')}")
    print(f"旧队列：{s.get('source_status')}；已核验候选：{s.get('candidate_count', 0)}")
    print(f"总体接近：{s.get('matched_model_count', 0)}/6 模型，{s.get('matched_total_cells', 0)}/48 项")
    print(f"补充训练已结束：{s.get('followups_finished', 0)}/{s.get('followup_count', '待旧队列结束后生成')}；当前：{s.get('active_case')}")
    if s.get("joint_archive"):
        print("四个联合模型完整归档：", s["joint_archive"])
    if s.get("searching_model"):
        print(f"CPU组合搜索：{s['searching_model']} / {s['searching_mode']}")
    if s.get("feasibility_blocked"):
        print("当前pooled口径的部分RMSE/NSE目标不相容；保留原容差并报告最接近值，不声明全部复现。")
    if s.get("error"):
        print("错误：", s["error"])
    table = output / "current_table.md"
    if table.exists():
        print(table.read_text())
    print("明细：", output / "total_comparison.tsv")
    return 0


def bundle_output(output, destination):
    temporary = destination.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(output.rglob("*")):
            if (path.is_file() and not path.is_symlink() and path.suffix not in {".pt", ".lock", ".tmp"}
                    and path.name != "joint_models_complete.tar.gz"):
                archive.add(path, arcname=str(Path(output.name) / path.relative_to(output)), recursive=False)
    temporary.replace(destination)
    sha = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(f"{sha}  {destination.name}\n")
    return str(destination)


def run(args):
    project, output = args.project_root.resolve(), args.output_root.resolve()
    args.output_root = output
    if TOOL_ROOT == project or TOOL_ROOT.is_relative_to(project):
        raise ValueError("Install this campaign outside the live project")
    if output == args.source_results.resolve() or output.is_relative_to(args.source_results.resolve()):
        raise ValueError("Output must be separate from existing training results")
    output.mkdir(parents=True, exist_ok=True)
    (project / "logs").mkdir(exist_ok=True)
    with (output / "campaign.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("已有总体定向搜索在运行。")
            return print_status(output)
        state = dict(status="starting", seed=20240604, primary_mode="pooled", test_target_guided=True,
            training_seeds=[20240604], original_paper_method_recovered=False, followups_finished=0,
            trained_models=["gru", "lstm"], pid=os.getpid())
        publish(output, state)
        print("正在核对原队列的源码、数据和参数；准备读取已完成模型。", flush=True)
        gpu_lock = None
        try:
            sys.path.insert(0, str(TOOL_ROOT / "src"))
            evidence = module("que_focus_evidence", TOOL_ROOT / "scripts/reproduce/que_total_watch_evidence.py")
            context = evidence.Context(project, args.source_results.resolve())
            verify_training_sources(context, TOOL_ROOT, audit_path=output / "training_source_compatibility.json")
            if context._recompute_evaluation() != context.evaluation:
                raise ValueError("Saved evaluation differs from original source data")
            audit_api = module("que_focus_audit", TOOL_ROOT / "scripts/reproduce/audit_que_total_objective.py")
            search_api = module("que_focus_search", TOOL_ROOT / "scripts/reproduce/search_que_total_recurrent.py")
            trainer = module("que_focus_training", TOOL_ROOT / "scripts/train/que_total_followup_training.py")
            runner = module("que_focus_external_runner", TOOL_ROOT / "scripts/train/run_que_comprehensive_reconstruction.py")
            signatures = runner.life.fingerprints(TOOL_ROOT, context.data_dir)
            manifest = dict(version=1, signatures=signatures, paper_sha256=context.manifest["paper_sha256"],
                source_root=str(args.source_results.resolve()), source_manifest_hash=context._manifest_file_hash,
                max_followups_per_model=args.max_followups_per_model, primary_mode="pooled", seed=20240604,
                search_starts=args.search_starts, search_sweeps=args.search_sweeps, pair_trials=args.pair_trials,
                gpu_id=args.gpu_id, memory_limit_gib=6, headroom_gib=2,
                tool_commit=read(TOOL_ROOT / "deployment.json").get("commit"))
            if args.close_joint_first:
                manifest["close_joint_first"] = True
            manifest["signature"] = digest(manifest)
            mp = output / "manifest.json"
            if mp.exists() and read(mp) != manifest:
                raise ValueError("Campaign source/plan changed; choose a separate output directory")
            write(mp, manifest)
            def source_check():
                context.check_fingerprints()
                if runner.life.fingerprints(TOOL_ROOT, context.data_dir) != signatures:
                    raise ValueError("External tool source/data changed")
            closeout = None
            if args.close_joint_first:
                source_check()
                archive_api = module("que_joint_closeout", TOOL_ROOT / "scripts/reproduce/archive_que_joint_closeout.py")
                stop_api = module("que_joint_handoff_stop", TOOL_ROOT / "scripts/reproduce/stop_que_shared_after_closeout.py")
                closeout = prepare_joint_handoff(context, args, state, archive_api, stop_api)
            previous_names, ranking, candidates = set(), [], []
            while True:
                queue = read(args.source_results / "queue_status.json")
                names = {r["case"] for r in queue.get("cases", []) if str(r.get("technical_status", "")).startswith("PASS")}
                terminal = queue.get("status") in TERMINAL
                publish(output, state, source_status=queue.get("status"), source_finished=queue.get("finished_cases"))
                if not ranking or len(names.symmetric_difference(previous_names)) >= args.refresh_cases or (terminal and names != previous_names):
                    source_check()
                    candidates, exclusions = collect_validated(context, queue)
                    candidates = restrict_to_closed_joint_models(candidates, closeout)
                    write(output / "source_exclusions.json", exclusions)
                    if not candidates:
                        raise ValueError("No verified complete source candidates")
                    write(output / "source_candidates.json", candidates)
                    ranking = analyse(candidates, context, args, state, audit_api, search_api)
                    previous_names = names
                if not args.watch:
                    publish(output, state, status="cpu_snapshot_completed")
                    return 0
                gpu_lock = terminal_and_lock(queue, project, args.gpu_id)
                if gpu_lock is not None:
                    # A stale terminal snapshot cannot authorize GPU work after a restart.
                    if read(args.source_results / "queue_status.json").get("status") in TERMINAL:
                        break
                    gpu_lock.close()
                    gpu_lock = None
                publish(output, state, status="waiting_original_queue", active_case=None)
                time.sleep(args.poll_seconds)
            source_check()
            plan_path = output / "followup_plan.json"
            if plan_path.exists():
                plan = read(plan_path)
                if plan.get("campaign_signature") != manifest["signature"] or plan.get("sha256") != digest({k:v for k,v in plan.items() if k != "sha256"}):
                    raise ValueError("Frozen supplemental plan changed")
            else:
                parent_records = complete_records(ranking, "pooled", context.paper)
                # All visited settings prevent duplicates, including invalid or
                # failed artifacts. Only separately audited records select parents.
                parent_records.extend({**r, "technical_status": "HISTORY_ONLY", "metrics": []}
                    for r in queue.get("cases", []) if r.get("model") in ("gru", "lstm"))
                followups = trainer.generate_followups(parent_records, runner, args.max_followups_per_model)
                plan = dict(campaign_signature=manifest["signature"], cases=followups,
                            objective="total8_minimax_then_mean", training_seed=20240604)
                plan["sha256"] = digest(plan)
                write(plan_path, plan)
            preflight = trainer.preflight_followups(plan["cases"], tool_root=TOOL_ROOT,
                data_dir=context.data_dir, output_root=output, runner=runner)
            write(output / "followup_command_preflight.json", preflight)
            records, latest_ranking = [], ranking
            publish(output, state, followup_count=len(plan["cases"]))
            for case in plan["cases"]:
                source_check()
                publish(output, state, status="supplemental_training", active_case=case["case"])
                record = trainer.execute_followup(case, tool_root=TOOL_ROOT, data_dir=context.data_dir,
                    output_root=output, paper_config=context.paper_path, evaluation=context.evaluation,
                    campaign_manifest=manifest, gpu_id=args.gpu_id, source_check=source_check,
                    deadline=float("inf"))
                if record.get("technical_status") == "PAUSED":
                    publish(output, state, status="paused_resource_wait", active_case=case["case"])
                    return 0
                if record.get("exit_code") == 2:
                    raise RuntimeError(f"Supplemental command/configuration failure: {case['case']}; remaining commands withheld")
                records.append(record)
                if str(record.get("technical_status", "")).startswith("PASS"):
                    run = output / "cases" / case["case"] / case["model"] / "seed_20240604"
                    candidates.append(dict(model=case["model"], case=case["case"], run=str(run), settings=case,
                                           evidence_digest=digest(read(run / "completion_receipt.json"))))
                write(output / "followup_status.json", records)
                publish(output, state, followups_finished=len(records), followups_failed=sum(not str(r.get("technical_status", "")).startswith("PASS") for r in records))
                if len(records) % 4 == 0 or len(records) == len(plan["cases"]):
                    latest_ranking = analyse(candidates, context, args, state, audit_api, search_api)
            if not records:
                analyse(candidates, context, args, state, audit_api, search_api)
            source_check()
            publish(output, state, status="completed_search", active_case=None,
                    completion_meaning="finite_search_finished_not_a_claim_of_48_matches", weights_preserved_on_server=True,
                    compact_includes_weights=False)
            archive = bundle_output(output, project.parent / f"{output.name}_compact.tar.gz")
            publish(output, state, archive=archive)
            return 0
        except Exception as exc:
            publish(output, state, status="failed", error=f"{type(exc).__name__}: {exc}")
            print(state["error"], file=sys.stderr, flush=True)
            return 1
        finally:
            if gpu_lock is not None:
                gpu_lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path.home() / "projects/STaR-GNN-BWDF"
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--source-results", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--close-joint-first", action="store_true",
        help="Archive the four joint models, stop the verified old queue, then focus on GRU/LSTM")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--gpu-id", default="7", choices=["7"])
    parser.add_argument("--max-followups-per-model", type=int, default=24, choices=range(25))
    parser.add_argument("--search-starts", type=int, default=12, choices=range(1, 33))
    parser.add_argument("--search-sweeps", type=int, default=8, choices=range(1, 17))
    parser.add_argument("--pair-trials", type=int, default=300, choices=range(2001))
    parser.add_argument("--refresh-cases", type=int, default=16, choices=range(1, 65))
    parser.add_argument("--poll-seconds", type=int, default=60, choices=range(1, 61))
    args = parser.parse_args(argv)
    args.source_results = args.source_results or args.project_root / "results/que_comprehensive_reconstruction_shared_20260908"
    default_tag = "que_recurrent_focus_20260910" if args.close_joint_first else "que_total_focus_20260910"
    args.output_root = args.output_root or args.project_root / "results" / default_tag
    if args.status:
        return print_status(args.output_root)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
