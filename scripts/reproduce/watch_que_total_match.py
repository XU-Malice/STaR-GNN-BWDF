#!/usr/bin/env python3
"""Observe a live Que search externally; optionally stop after verified 48-cell agreement.

Deploy this module and its two helpers outside the active project's source
directories. It does not edit the runner, candidates, training settings or data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time
import uuid

MODELS = ("gru", "lstm", "msnet", "mscmnet_m", "mscmnet_wm", "mscmnet_w")
DEFAULT_TAG = "que_comprehensive_reconstruction_shared_20260908"
TERMINAL = {"completed", "completed_with_failures", "failed", "interrupted", "paused_time_budget"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def make_archive(selected_dir):
    """Finish and seal a self-contained bundle before requesting any stop."""
    files = sorted(p for p in selected_dir.rglob("*") if p.is_file())
    if not files or any(p.is_symlink() for p in selected_dir.rglob("*")):
        raise ValueError("Selected export is empty or contains a symbolic link")
    hashes = {str(p.relative_to(selected_dir)): sha256(p) for p in files}
    write_json(selected_dir / "archive_receipt.json", {"files": hashes, "created_utc": utc_now()})
    archive = selected_dir.with_suffix(".tar.gz")
    temporary = archive.with_name(archive.name + ".partial")
    with temporary.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w:gz", compresslevel=1) as bundle:
            bundle.add(selected_dir, arcname=selected_dir.name, recursive=True)
        raw.flush()
        os.fsync(raw.fileno())
    # Detect changes during archive creation, including selected checkpoint bytes.
    if any(sha256(selected_dir / name) != value for name, value in hashes.items()):
        temporary.unlink()
        raise ValueError("Frozen selected evidence changed while packaging")
    temporary.replace(archive)
    digest = sha256(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
    return {"path": str(archive), "sha256": digest, "bytes": archive.stat().st_size}


def snapshot_summary(report):
    matched, passed = [], 0
    lines = ["|模型|总体接近项|最大误差指标偏差|NSE最大绝对差|配置|",
             "|---|---:|---:|---:|---|"]
    detail = []
    for model in MODELS:
        candidate = report.get("models", {}).get(model)
        if not candidate:
            lines.append(f"|{model}|0/8|暂无有效候选||| ")
            continue
        rows = candidate["metrics"]
        count = candidate["matched_count"]
        passed += count
        if candidate["all8_matched"]:
            matched.append(model)
        error = max(float(r["absolute_relative_difference"]) for r in rows if r["metric"] != "NSE")
        nse = max(abs(float(r["difference"])) for r in rows if r["metric"] == "NSE")
        lines.append(f"|{model}|{count}/8|{error:.2%}|{nse:.5f}|{candidate['case']}|")
        detail.extend({"model": model, "case": candidate["case"], **r} for r in rows)
    return {"matched_models": matched, "matched_model_count": len(matched),
            "matched_total_cells": passed, "table": "\n".join(lines), "comparison": detail}


def publish(output, state, report=None):
    state["updated_utc"] = utc_now()
    if report is not None:
        summary = snapshot_summary(report)
        state.update({k: v for k, v in summary.items() if k not in ("table", "comparison")})
        state["selection"] = report
        (output / "current_table.md").write_text(summary["table"] + "\n", encoding="utf-8")
        destination = output / "total_comparison.tsv"
        temporary = destination.with_suffix(".tsv.tmp")
        fields = list(dict.fromkeys(k for r in summary["comparison"] for k in r)) or ["model", "case", "task", "metric", "value", "paper_value"]
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(summary["comparison"])
        temporary.replace(destination)
    write_json(output / "watch_status.json", state)


def print_status(output):
    path = output / "watch_status.json"
    if not path.is_file():
        print(f"尚无总体判定记录：{path}")
        return 1
    state = json.loads(path.read_text())
    print(f"监测状态：{state.get('status')}；更新时间：{state.get('updated_utc')}")
    if "matched_model_count" in state:
        print(f"总体接近：{state['matched_model_count']}/6 个模型，"
              f"所选配置 {state['matched_total_cells']}/48 项。")
    else:
        print("本次监测尚未完成结果扫描；不代表已有结果归零。")
    print(f"判定标准：误差相对差≤{state['policy']['error_relative_tolerance']:.1%}；"
          f"NSE绝对差≤{state['policy']['nse_absolute_tolerance']:g}；不考核各DMA数值接近。")
    print(f"验证状态：{state.get('selection', {}).get('verification', '尚未验证')}")
    if "automatic_stop_available" in state:
        print("自动停止：" + ("可用；全部达标并完成保存后才请求退出。"
              if state["automatic_stop_available"] else "不可用；继续监测和保存结果，不发送停止信号。"))
    if state.get("capability_warning"):
        print(f"兼容性说明：{state['capability_warning']}")
    if state.get("selection"):
        print(snapshot_summary(state["selection"])["table"])
    if state.get("archive"):
        print(f"已核验结果包：{state['archive']['path']}")
    if state.get("error"):
        print(f"详情：{state['error']}")
    print(f"完整比较：{output / 'total_comparison.tsv'}")
    return 0


def run(args, *, context_factory=None, binder=None, sleeper=time.sleep):
    output, root = args.output_root, args.project_root
    if args.status:
        return print_status(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "watch.lock").open("a+")
    handle = None
    state = {"status": "starting", "started_utc": utc_now(), "observer_pid": os.getpid(),
             "project_root": str(root), "result_root": str(args.result_root),
             "policy": {"error_relative_tolerance": args.error_relative_tolerance,
                        "nse_absolute_tolerance": args.nse_absolute_tolerance,
                        "mode": "pooled", "required_models": list(MODELS),
                        "required_total_cells": 48, "require_dma_closeness": False,
                        "poll_seconds": args.poll_seconds, "stop_on_success": args.stop_on_success},
             "retrospective_paper_target_search": True, "paper_implementation_recovered": False}
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"已有监测程序使用 {output}，本次不重复启动。", flush=True)
            return 2
        publish(output, state)
        # Bind only this queue once. Never follow a replaced PID file to another run.
        # Monitoring alone has no process-control dependency. Only a specific
        # missing capability may degrade; identity/permission refusals stay fatal.
        if args.watch and args.stop_on_success:
            capability_errors = ()
            if binder is None:
                from que_total_watch_process import bind_queue, PidfdUnavailable
                binder = bind_queue
                capability_errors = (PidfdUnavailable,)
            try:
                handle = binder(root, args.result_root, args.pid_file)
            except capability_errors as exc:
                state["capability_warning"] = f"{type(exc).__name__}: {exc}"
                print("当前环境无法安全自动停止队列；继续更新指标并保存达标结果。"
                      + state["capability_warning"], flush=True)
            state["queue_identity"] = handle.describe() if handle else None
            state["automatic_stop_available"] = handle is not None
            if handle is None and not state.get("capability_warning"):
                print("未绑定到运行中的队列；继续只读监测，不会跟随新的PID或停止其他进程。", flush=True)
            elif handle is not None:
                print("自动停止已绑定当前队列；全部48项核验并保存后才请求退出。", flush=True)
        elif args.watch:
            state["automatic_stop_available"] = False
        if context_factory is None:
            from que_total_watch_evidence import Context
            context_factory = Context
        context = context_factory(root, args.result_root,
                                  error_relative_tolerance=args.error_relative_tolerance,
                                  nse_absolute_tolerance=args.nse_absolute_tolerance)
        previous = None
        while True:
            queue = json.loads((args.result_root / "queue_status.json").read_text())
            report = context.scan(queue)
            state.update(status="monitoring", queue_status=queue.get("status"),
                         queue_updated_utc=queue.get("updated_utc"), queue_stage=queue.get("current_stage"))
            publish(output, state, report)
            summary = snapshot_summary(report)
            message = (summary["table"], queue.get("status"), queue.get("current_stage"))
            if message != previous:
                print(f"[{utc_now()}] 总体接近 {summary['matched_model_count']}/6；"
                      f"队列 {queue.get('current_stage')} / {queue.get('status')}", flush=True)
                print(summary["table"], flush=True)
                previous = message
            if report["all_matched"]:
                state["status"] = "verifying_selected_results"
                publish(output, state, report)
                verified = context.verify_selected(report)
                if not verified.get("verified_all_matched"):
                    raise ValueError("Raw prediction verification did not confirm all 48 targets")
                destination = output / ("accepted_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
                selected = context.export_to(destination, verified)
                write_json(selected / "watch_policy.json", state["policy"])
                archive = make_archive(selected)
                state.update(status="matched_and_preserved", archive=archive, selected_root=str(selected))
                publish(output, state, verified)
                print(f"六个模型总体48项已核验并保存：{archive['path']}", flush=True)
                if args.stop_on_success and handle is not None and not handle.exited():
                    # No signals are permitted until verification + copying + hashing + archive all succeed.
                    signalled = handle.request_stop()
                    state.update(status="stop_requested" if signalled else "matched_queue_already_exited",
                                 queue_stop_signal="SIGTERM" if signalled else None)
                    publish(output, state, verified)
                    if signalled:
                        print("已向核验身份的训练队列请求停止；等待其清理子进程并保存日志。", flush=True)
                        for _ in range(30):
                            if handle.exited():
                                state["status"] = "matched_and_queue_stopped"
                                break
                            sleeper(2)
                        if state["status"] == "stop_requested":
                            state["status"] = "matched_stop_requested_cleanup_pending"
                            print("停止信号已发送，队列仍在清理；不会强制结束其他进程。", flush=True)
                        publish(output, state, verified)
                else:
                    state["status"] = "matched_queue_already_exited" if handle is not None and handle.exited() else "matched_and_preserved"
                    publish(output, state, verified)
                return 0
            ended = queue.get("status") in TERMINAL
            # A missing PID record is not evidence that a running queue ended.
            # Continue observing files, but never bind a new/replaced PID later.
            gone = args.watch and handle is not None and handle.exited()
            if ended or gone:
                state["status"] = "queue_ended_without_all_total_matches"
                state["error"] = "已完成的完整候选及可用阶段D组合尚未全部达到总体容差，保留实际差距。"
                publish(output, state, report)
                print(state["error"], flush=True)
                return 0
            if not args.watch:
                state["status"] = "snapshot_complete"
                publish(output, state, report)
                return 0
            sleeper(args.poll_seconds)
    except (KeyboardInterrupt, Exception) as exc:
        state.update(status="observer_failed", error=f"{type(exc).__name__}: {exc}")
        publish(output, state)
        print(f"监测退出：{state['error']}。请查看记录；未通过核验时不会请求停止训练。", file=sys.stderr, flush=True)
        return 2
    finally:
        if handle is not None:
            handle.close()
        lock.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.home() / "projects/STaR-GNN-BWDF")
    parser.add_argument("--run-tag", default=DEFAULT_TAG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--pid-file", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--once", action="store_true", help="Default: one read-only scan, no process signal")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--stop-on-success", action="store_true", help="Opt in: stop the bound queue after verified results are archived")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--error-relative-tolerance", type=float, default=.05)
    parser.add_argument("--nse-absolute-tolerance", type=float, default=.01)
    args = parser.parse_args(argv)
    if not args.run_tag or Path(args.run_tag).name != args.run_tag or args.run_tag in (".", ".."):
        parser.error("run-tag must be one directory name")
    if not math.isfinite(args.poll_seconds) or not 5 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be between 5 and 60")
    for name in ("error_relative_tolerance", "nse_absolute_tolerance"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value < 1:
            parser.error(f"{name} must be finite and between 0 and 1")
    if args.stop_on_success and not args.watch:
        parser.error("stop-on-success requires --watch")
    args.project_root = args.project_root.resolve()
    args.result_root = args.project_root / "results" / args.run_tag
    args.output_root = (args.output_root or args.project_root / "results" / "que_total_match_watch_20260909").resolve()
    args.pid_file = (args.pid_file or args.project_root / "logs/que_shared_gpu7_launcher.pid").resolve()
    protected = [args.project_root / name for name in ("src", "scripts", "configs", "tests", "data")]
    protected.append(args.result_root)
    if (args.output_root == args.project_root or any(args.output_root.is_relative_to(p.resolve()) or p.resolve().is_relative_to(args.output_root) for p in protected)):
        parser.error("output-root must not overwrite or contain source/data/active result directories")
    return args


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(run(parse_args()))
