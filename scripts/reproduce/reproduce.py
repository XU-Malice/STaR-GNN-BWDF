#!/usr/bin/env python
"""原始 BWDF 数据到论文表格的端到端复现编排器。

重要边界：所有模型先完成训练并选择 Validation checkpoint；只有写入
``TRAINING_COMPLETE`` 后才进入 Test 评估。这样既方便一条命令复现，也在代码
流程上避免 Test target 被训练或参数选择提前读取。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from paper_release_lib import PROJECT_ROOT, STAR_VARIANTS, TASKS


CONTROL_DIR: Path | None = None


def _control(current: str) -> None:
    if CONTROL_DIR is None:
        return
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    (CONTROL_DIR / "CURRENT").write_text(current + "\n", encoding="utf-8")


def _run(command: list[str]) -> None:
    """在仓库根目录同步执行一个阶段，失败立即终止全流程。"""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _completed(run: Path) -> bool:
    """只有 checkpoint 与 completed summary 同时存在才允许复用。"""
    checkpoint = run / "checkpoint_best.pt"
    summary = run / "training_summary.json"
    if not checkpoint.is_file() or not summary.is_file():
        return False
    try:
        return json.loads(summary.read_text(encoding="utf-8")).get("status") == "completed"
    except Exception:
        return False


def _train(
    *,
    model: str,
    task: str,
    seed: int,
    device: str,
    output: Path,
    variant: str | None = None,
) -> None:
    """训练一个模型/任务；拒绝覆盖非空但未完成的实验目录。"""
    label = variant if variant is not None else model
    _control(f"TRAIN {label} {task} seed_{seed}")
    if _completed(output):
        print(f"REUSE completed training: {output}")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Incomplete nonempty run is not overwritten: {output}. "
            "Resume it explicitly or move it aside."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if variant is None:
        command = [
            sys.executable,
            "scripts/train/train_model.py",
            "--model", model,
            "--config", f"configs/paper/{model}_{task}.yaml",
        ]
    else:
        command = [
            sys.executable,
            "scripts/innovation/train_star_dcrnn.py",
            "--variant", variant,
            "--config", f"configs/paper/star_gnn_{task}.yaml",
        ]
    command += [
        "--task", task,
        "--seed", str(seed),
        "--device", device,
        "--output-dir", str(output),
    ]
    _run(command)


def _evaluate(
    *,
    model: str,
    task: str,
    seed: int,
    device: str,
    run: Path,
    variant: str | None = None,
) -> None:
    """使用最佳 Validation checkpoint 评估，不再进行任何训练或选参。"""
    label = variant if variant is not None else model
    _control(f"TEST {label} {task} seed_{seed} common_46")
    destination = run / "evaluation"
    summary = destination / "test_summary.json"
    if summary.is_file():
        print(f"REUSE completed evaluation: {destination}")
        return
    # 该函数只会在全局 TRAINING_COMPLETE 标记写入后调用。
    if variant is None:
        command = [
            sys.executable,
            "scripts/evaluate/test_model.py",
            "--model", model,
        ]
    else:
        command = [
            sys.executable,
            "scripts/innovation/evaluate_star_dcrnn.py",
            "--variant", variant,
        ]
    command += [
        "--task", task,
        "--seed", str(seed),
        "--checkpoint", str(run / "checkpoint_best.pt"),
        "--output-dir", str(destination),
        "--device", device,
    ]
    _run(command)


def main() -> None:
    global CONTROL_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluation-device", default="cpu")
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/paper/reproduction",
    )
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument(
        "--control-dir",
        type=Path,
        help="可选状态目录；持续写入CURRENT，便于后台监控。",
    )
    args = parser.parse_args()
    CONTROL_DIR = args.control_dir.resolve() if args.control_dir else None
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds:
        raise ValueError("At least one seed is required.")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_data:
        _control("DATA PREPROCESSING")
        _run(["bash", "scripts/data/run_pipeline.sh"])
        _control("PEARSON GRAPH")
        _run(["bash", "scripts/graph/run_graph_pipeline.sh"])

    runs: list[tuple[str, str, int, Path, str | None]] = []
    for seed in seeds:
        # DCRNN 就是下面 ``backbone`` 变体（论文中的 Base），不能再额外
        # 训练一套独立 DCRNN。外部基线这里只保留 STGCN。
        if not args.skip_baselines:
            for task in TASKS:
                run = root / "baselines" / "stgcn" / task / f"seed_{seed}"
                _train(
                    model="stgcn",
                    task=task,
                    seed=seed,
                    device=args.device,
                    output=run,
                )
                runs.append(("stgcn", task, seed, run, None))
        for label, variant in STAR_VARIANTS.items():
            for task in TASKS:
                run = root / "star_gnn" / label / task / f"seed_{seed}"
                _train(
                    model="star_gnn",
                    task=task,
                    seed=seed,
                    device=args.device,
                    output=run,
                    variant=variant,
                )
                runs.append(("star_gnn", task, seed, run, variant))

    # 硬协议边界：Test target 首次允许被读取的位置就在此标记之后。
    (root / "TRAINING_COMPLETE").write_text("PASS\n", encoding="utf-8")
    _control("TRAINING_COMPLETE; TEST NOW ALLOWED")
    for model, task, seed, run, variant in runs:
        _evaluate(
            model=model,
            task=task,
            seed=seed,
            device=args.evaluation_device,
            run=run,
            variant=variant,
        )
    if seeds == [0] and not args.skip_baselines:
        _control("BUILD PAPER TABLES")
        _run(
            [
                sys.executable,
                "scripts/reproduce/build_paper_tables.py",
                "--input", str(root),
                "--output", str(root / "tables"),
            ]
        )
        _run(
            [
                sys.executable,
                "scripts/reproduce/build_detailed_test_artifacts.py",
                "--release", str(root),
                "--output", str(root / "paper"),
            ]
        )
    _control("REPRODUCTION DONE")
    print(f"Reproduction complete: {root}")


if __name__ == "__main__":
    main()
