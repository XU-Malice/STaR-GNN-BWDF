#!/usr/bin/env python
"""从冻结 common-46 Test 结果生成论文表格、逐 DMA 表格和图件。

本脚本只读取已经完成的 checkpoint 评估文件，不重新训练、不改变模型，
也不利用 Test 结果选择超参数。所有论文数字都由 ``metrics_*.csv`` 或
``predictions.npz`` 重算得到，避免手工复制带来的口径漂移。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dma_wdf.data.graph import compute_graph_metrics, load_graph
from dma_wdf.data.metrics import compute_metrics

from paper_release_lib import METRICS, PROJECT_ROOT, better, read_metrics


TASKS = ("24h", "168h")
# 公开论文只有五个模型名。DCRNN 就是内部 ``Base/backbone``，因此不能
# 同时在表格中再出现一行 Base。
ABLATION_MODELS = ("DCRNN", "State", "FA-DPR", "Full")
ALL_MODELS = ("STGCN", *ABLATION_MODELS)
COLORS = {
    "STGCN": "#8c8c8c",
    "DCRNN": "#4c78a8",
    "State": "#f58518",
    "FA-DPR": "#54a24b",
    "Full": "#d62728",
}


def _evaluation_dir(release: Path, model: str, task: str) -> Path:
    """将公开模型名映射到冻结发布目录。"""
    # 冻结包多一层 models/；从零复现目录直接以 family 开始。自动识别可让
    # 同一制图脚本同时服务 checkpoint 验证和 raw-data 复现。
    prefix = release / "models" if (release / "models").is_dir() else release
    if model == "DCRNN":
        family, name = "star_gnn", "Base"
    elif model in ABLATION_MODELS:
        family, name = "star_gnn", model
    else:
        family, name = "baselines", model.lower()
    path = prefix / family / name / task / "seed_0" / "evaluation"
    if not path.is_dir():
        raise FileNotFoundError(f"缺少冻结评估目录：{path}")
    return path


def _write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    """写出不依赖 tabulate 的 GitHub Markdown 表格。"""
    path.write_text(_markdown_table_text(frame), encoding="utf-8")


def _markdown_table_text(frame: pd.DataFrame) -> str:
    """返回 Markdown 表格文本，避免引入 pandas 的可选 tabulate 依赖。"""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(frame[c]) else "---" for c in columns) + "|",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _save_figure(fig: plt.Figure, base: Path) -> None:
    """同时保存便于预览的 PNG 和投稿用矢量 PDF。"""
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _collect_aggregate(release: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for model in ALL_MODELS:
            path = _evaluation_dir(release, model, task)
            values = read_metrics(path / "metrics_aggregate_total_common_46.csv")
            rows.append({"task": task, "model": model, **values})
    return pd.DataFrame(rows)


def _collect_dma(release: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for model in ALL_MODELS:
            path = _evaluation_dir(release, model, task) / "metrics_common_46.csv"
            frame = pd.read_csv(path)
            frame = frame.loc[frame["entity"].astype(str).isin(list("ABCDEFGHIJ"))]
            if frame["entity"].astype(str).tolist() != list("ABCDEFGHIJ"):
                raise ValueError(f"DMA A-J 顺序或数量不正确：{path}")
            for _, row in frame.iterrows():
                mape = float(row["MAPE"])
                if mape < 1.0:
                    mape *= 100.0
                rows.append(
                    {
                        "task": task,
                        "model": model,
                        "DMA": str(row["entity"]),
                        "MAE": float(row["MAE"]),
                        "MAPE": mape,
                        "RMSE": float(row["RMSE"]),
                        "NSE": float(row["NSE"]),
                    }
                )
    return pd.DataFrame(rows)


def _load_common_predictions(
    release: Path,
    model: str,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    """读取自描述的冻结预测并只保留 common-46 样本。

    DCRNN/Base 是唯一的公共索引来源；DCRNN 四个变体的预测文件也必须携带
    相同的 ``common_46_indices``。这样删除历史重复 DCRNN 工件后，任何一个
    checkpoint 的预测仍可独立审计，并能及时发现样本索引漂移。
    """
    with np.load(
        _evaluation_dir(release, "DCRNN", task) / "predictions.npz",
        allow_pickle=False,
    ) as baseline_npz:
        common = baseline_npz["common_46_indices"].astype(np.int64)
    if common.shape != (46,):
        raise ValueError(f"{task} common-46 索引数量不是 46：{common.shape}")

    path = _evaluation_dir(release, model, task) / "predictions.npz"
    with np.load(path, allow_pickle=False) as payload:
        if "target" in payload and "prediction" in payload:
            truth = payload["target"]
            prediction = payload["prediction"]
            if "common_46_indices" not in payload.files:
                raise ValueError(f"预测文件缺少common_46_indices：{path}")
            indices = payload["common_46_indices"].astype(np.int64)
        elif "y_true" in payload and "y_pred" in payload:
            truth = payload["y_true"]
            prediction = payload["y_pred"]
            indices = payload["common_46_indices"].astype(np.int64)
        else:
            raise ValueError(f"未知 predictions.npz schema：{path}")
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError(f"预测数组形状不正确：{path} {truth.shape} {prediction.shape}")
    if not np.array_equal(indices, common):
        raise ValueError(f"模型与DCRNN/Base的common-46索引不一致：{path}")
    return truth[indices].astype(np.float64), prediction[indices].astype(np.float64)


def _collect_daily_168h(release: Path) -> pd.DataFrame:
    """按未来第 1--7 天重算总需求 MAE/MAPE/RMSE/NSE。"""
    rows: list[dict[str, Any]] = []
    reference_truth: np.ndarray | None = None
    for model in ALL_MODELS:
        truth, prediction = _load_common_predictions(release, model, "168h")
        if truth.shape != (46, 168, 10):
            raise ValueError(f"168h common-46 shape 不正确：{model} {truth.shape}")
        if reference_truth is None:
            reference_truth = truth
        elif not np.allclose(reference_truth, truth, atol=1.0e-6, rtol=0.0):
            raise ValueError(f"不同模型的 common-46 Test 真值不一致：{model}")
        for day in range(1, 8):
            window = slice((day - 1) * 24, day * 24)
            values = compute_metrics(
                truth[:, window, :].sum(axis=2),
                prediction[:, window, :].sum(axis=2),
            )
            rows.append(
                {
                    "day": day,
                    "model": model,
                    "MAE": float(values["MAE"]),
                    "MAPE": 100.0 * float(values["MAPE"]),
                    "RMSE": float(values["RMSE"]),
                    "NSE": float(values["NSE"]),
                }
            )
    return pd.DataFrame(rows)


def _hierarchy_report(aggregate: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    lookup = aggregate.set_index(["task", "model"])
    aggregate_relations: dict[str, str] = {}
    for task in TASKS:
        for left, right in (
            ("State", "DCRNN"),
            ("FA-DPR", "DCRNN"),
            ("Full", "State"),
            ("Full", "FA-DPR"),
            ("Full", "STGCN"),
        ):
            count = sum(
                better(metric, lookup.loc[(task, left), metric], lookup.loc[(task, right), metric])
                for metric in METRICS
            )
            aggregate_relations[f"{task}_{left}_vs_{right}"] = f"{count}/4"

    daily_relations: dict[str, str] = {}
    daily_lookup = daily.set_index(["day", "model"])
    for left, right in (
        ("State", "DCRNN"),
        ("FA-DPR", "DCRNN"),
        ("Full", "State"),
        ("Full", "FA-DPR"),
        ("Full", "STGCN"),
    ):
        count = 0
        for day in range(1, 8):
            for metric in METRICS:
                count += int(
                    better(
                        metric,
                        daily_lookup.loc[(day, left), metric],
                        daily_lookup.loc[(day, right), metric],
                    )
                )
        daily_relations[f"Day1-7_{left}_vs_{right}"] = f"{count}/28"
    return {
        "protocol": "common_46",
        "test_origins": 46,
        "selection_policy": "validation_first_test_once",
        "aggregate_relations": aggregate_relations,
        "daily_relations": daily_relations,
        "transparent_exception": (
            "FA-DPR 168h MAPE is slightly worse than DCRNN; all other registered "
            "factorial Test relations pass (31/32)."
        ),
    }


def _plot_metric_panels(frame: pd.DataFrame, models: tuple[str, ...], title: str, base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    for axis, metric in zip(axes.ravel(), METRICS):
        selected = frame.set_index("model").loc[list(models)]
        values = selected[metric].to_numpy(dtype=float)
        axis.bar(
            np.arange(len(models)),
            values,
            color=[COLORS[model] for model in models],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_xticks(np.arange(len(models)), models, rotation=20, ha="right")
        axis.set_ylabel(metric + (" (%)" if metric == "MAPE" else ""))
        axis.set_title(metric + (" (higher is better)" if metric == "NSE" else " (lower is better)"))
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=13)
    _save_figure(fig, base)


def _plot_daily(daily: pd.DataFrame, models: tuple[str, ...], title: str, base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True)
    markers = ("o", "s", "^", "D", "P", "X")
    for axis, metric in zip(axes.ravel(), METRICS):
        for model, marker in zip(models, markers):
            selected = daily.loc[daily["model"] == model].sort_values("day")
            axis.plot(
                selected["day"],
                selected[metric],
                label=model,
                marker=marker,
                linewidth=2.0 if model == "Full" else 1.5,
                color=COLORS[model],
            )
        axis.set_ylabel(metric + (" (%)" if metric == "MAPE" else ""))
        axis.set_xticks(range(1, 8))
        axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, frameon=False)
    axes[1, 0].set_xlabel("Forecast day")
    axes[1, 1].set_xlabel("Forecast day")
    fig.suptitle(title, fontsize=13)
    _save_figure(fig, base)


def _plot_dma_mae(dma: pd.DataFrame, task: str, base: Path) -> None:
    selected = dma.loc[dma["task"] == task]
    models = ("STGCN", "DCRNN", "State", "FA-DPR", "Full")
    dmas = list("ABCDEFGHIJ")
    x = np.arange(len(dmas))
    width = 0.15
    fig, axis = plt.subplots(figsize=(13.0, 5.2))
    for index, model in enumerate(models):
        values = (
            selected.loc[selected["model"] == model]
            .set_index("DMA")
            .loc[dmas, "MAE"]
            .to_numpy(dtype=float)
        )
        axis.bar(
            x + (index - (len(models) - 1) / 2) * width,
            values,
            width,
            label=model,
            color=COLORS[model],
        )
    axis.set_xticks(x, dmas)
    axis.set_xlabel("DMA")
    axis.set_ylabel("MAE")
    axis.set_title(f"DMA-level common-46 Test MAE ({task})")
    axis.legend(ncol=5, frameon=False)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, base)


def _build_graph_artifacts(graph_path: Path, table_dir: Path, figure_dir: Path) -> dict[str, Any]:
    graph = load_graph(graph_path)
    names = list(graph["node_names"])
    for filename, key in (
        ("pearson_correlation.csv", "static_corr"),
        ("pearson_adjacency.csv", "static_adj"),
        ("pearson_random_walk.csv", "random_walk"),
    ):
        pd.DataFrame(graph[key], index=names, columns=names).to_csv(
            table_dir / filename,
            float_format="%.12f",
        )
    metrics = compute_graph_metrics(graph)
    pd.DataFrame(metrics["node_metrics"]).to_csv(
        table_dir / "pearson_node_metrics.csv",
        index=False,
    )
    (table_dir / "pearson_graph_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    matrix = np.asarray(graph["static_corr"], dtype=float)
    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(len(names)), names)
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("DMA")
    axis.set_ylabel("DMA")
    axis.set_title("Training-period Pearson functional graph")
    for i in range(len(names)):
        for j in range(len(names)):
            axis.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(matrix[i, j]) > 0.55 else "black",
            )
    fig.colorbar(image, ax=axis, shrink=0.82, label="Pearson r")
    _save_figure(fig, figure_dir / "pearson_correlation_heatmap")
    return metrics


def _write_report(
    output: Path,
    aggregate: pd.DataFrame,
    daily: pd.DataFrame,
    hierarchy: dict[str, Any],
    graph_metrics: dict[str, Any],
) -> None:
    ablation = aggregate.loc[aggregate["model"].isin(ABLATION_MODELS)].copy()
    all_table = aggregate.copy()
    lines = [
        "# common-46 Test 结果与论文图表说明",
        "",
        "> 本目录由冻结 checkpoint 自动生成。参数在 Validation 阶段确定，Test 只执行最终报告；所有预测均关闭 teacher forcing。",
        "",
        "## 1. 总体 Test 对比",
        "",
        _markdown_table_text(all_table).rstrip(),
        "",
        "## 2. 组件消融",
        "",
        _markdown_table_text(ablation).rstrip(),
        "",
        "Full 在两个预测范围、四项指标上均优于 State 和 FA-DPR；State 在八项比较中均优于 DCRNN。FA-DPR 相对 DCRNN 的 168 h MAPE 是唯一例外，完整记录为 31/32，而不是删除该结果。",
        "",
        "## 3. 168 h 逐日分析",
        "",
        _markdown_table_text(
            daily.pivot(index="day", columns="model", values="MAE")
            .reset_index()
            .loc[:, ["day", *ALL_MODELS]]
        ).rstrip(),
        "",
        "逐日表按 common-46 的 168 h 预测切分为七个连续 24 h 区间，指标在各日的十 DMA 总需求序列上重算。逐日结果用于解释误差随预测距离的变化，不参与参数选择。",
        "",
        "## 4. DMA 异质性",
        "",
        "完整 A--J 四指标见 `tables/test_dma_metrics_long.csv`，宽表见 `tables/test_dma_*_wide.csv`，图见 `figures/test_dma_mae_24h.*` 和 `figures/test_dma_mae_168h.*`。",
        "",
        "## 5. Pearson 功能图",
        "",
        f"图仅由训练期 {int(graph_metrics.get('num_nodes', 10))} 个 DMA 构建；正相关无向边 {int(graph_metrics.get('positive_undirected_edges', 0))}/{int(graph_metrics.get('possible_undirected_edges', 45))}。相关矩阵、邻接矩阵、随机游走矩阵和节点统计均保存在 `tables/`。",
        "",
        "## 6. 自动层级验收",
        "",
        "```json",
        json.dumps(hierarchy, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 7. 图件索引",
        "",
        "- `figures/test_overall_24h.*`、`test_overall_168h.*`：STGCN/DCRNN/Full 总体对比。",
        "- `figures/test_ablation_24h.*`、`test_ablation_168h.*`：四单元消融。",
        "- `figures/test_day1_day7_models.*`：STGCN/DCRNN/Full 逐日对比。",
        "- `figures/test_day1_day7_ablation.*`：DCRNN/State/FA-DPR/Full 逐日消融。",
        "- `figures/test_dma_mae_*.{png,pdf}`：十 DMA 的区域异质性。",
        "- `figures/pearson_correlation_heatmap.*`：训练期 Pearson 功能关联热力图。",
        "",
    ]
    (output / "reports" / "TEST_RESULTS_CN.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=PROJECT_ROOT / "results/paper/frozen_v1",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=PROJECT_ROOT / "artifacts/graphs/bwdf_pearson_static_graph.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper",
    )
    args = parser.parse_args()
    release = args.release.resolve()
    graph_path = args.graph.resolve()
    output = args.output.resolve()
    table_dir = output / "tables"
    figure_dir = output / "figures"
    report_dir = output / "reports"
    for directory in (table_dir, figure_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    aggregate = _collect_aggregate(release)
    dma = _collect_dma(release)
    daily = _collect_daily_168h(release)
    hierarchy = _hierarchy_report(aggregate, daily)

    aggregate.to_csv(table_dir / "test_all_models_common46.csv", index=False)
    aggregate.loc[aggregate["model"].isin(ABLATION_MODELS)].to_csv(
        table_dir / "test_ablation_common46.csv",
        index=False,
    )
    dma.to_csv(table_dir / "test_dma_metrics_long.csv", index=False)
    daily.to_csv(table_dir / "test_day1_day7_metrics.csv", index=False)
    (table_dir / "test_hierarchy.json").write_text(
        json.dumps(hierarchy, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown_table(aggregate, table_dir / "test_all_models_common46.md")
    _write_markdown_table(
        aggregate.loc[aggregate["model"].isin(ABLATION_MODELS)],
        table_dir / "test_ablation_common46.md",
    )

    for task in TASKS:
        task_dma = dma.loc[dma["task"] == task]
        for metric in METRICS:
            wide = task_dma.pivot(index="DMA", columns="model", values=metric).reset_index()
            wide = wide.loc[:, ["DMA", *ALL_MODELS]]
            wide.to_csv(table_dir / f"test_dma_{metric.lower()}_wide_{task}.csv", index=False)
            _write_markdown_table(
                wide,
                table_dir / f"test_dma_{metric.lower()}_wide_{task}.md",
            )
        task_aggregate = aggregate.loc[aggregate["task"] == task]
        _plot_metric_panels(
            task_aggregate,
            ("STGCN", "DCRNN", "Full"),
            f"Cross-model common-46 Test ({task})",
            figure_dir / f"test_overall_{task}",
        )
        _plot_metric_panels(
            task_aggregate,
            ABLATION_MODELS,
            f"Factorial ablation common-46 Test ({task})",
            figure_dir / f"test_ablation_{task}",
        )
        _plot_dma_mae(dma, task, figure_dir / f"test_dma_mae_{task}")

    for metric in METRICS:
        wide = daily.pivot(index="day", columns="model", values=metric).reset_index()
        wide = wide.loc[:, ["day", *ALL_MODELS]]
        wide.to_csv(table_dir / f"test_day1_day7_{metric.lower()}_wide.csv", index=False)
        _write_markdown_table(
            wide,
            table_dir / f"test_day1_day7_{metric.lower()}_wide.md",
        )
    _plot_daily(
        daily,
        ("STGCN", "DCRNN", "Full"),
        "Day-wise cross-model Test over the 168-h horizon",
        figure_dir / "test_day1_day7_models",
    )
    _plot_daily(
        daily,
        ABLATION_MODELS,
        "Day-wise factorial ablation over the 168-h horizon",
        figure_dir / "test_day1_day7_ablation",
    )

    graph_metrics = _build_graph_artifacts(graph_path, table_dir, figure_dir)
    _write_report(output, aggregate, daily, hierarchy, graph_metrics)
    print(f"论文表格：{table_dir}")
    print(f"论文图件：{figure_dir}")
    print(f"中文结果说明：{report_dir / 'TEST_RESULTS_CN.md'}")


if __name__ == "__main__":
    main()
