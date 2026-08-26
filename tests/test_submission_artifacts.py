from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPRODUCE = ROOT / "scripts" / "reproduce"
if str(REPRODUCE) not in sys.path:
    sys.path.insert(0, str(REPRODUCE))

import manuscript_plot_style as style  # noqa: E402
import render_submission_figures as figures  # noqa: E402
import render_submission_tables as tables  # noqa: E402


def test_submission_visual_hierarchy_is_frozen():
    assert style.MODEL_COLORS["STaR-GNN"] == "#0F4D92"
    assert style.MODEL_COLORS["DCRNN"] == "#5C5C5C"
    assert style.MODEL_COLORS["STGCN"] == "#A6A6A6"
    assert figures.ABLATION_MODELS == (
        "DCRNN",
        "DCRNN + SAS-Norm",
        "DCRNN + FA-DPR",
        "STaR-GNN",
    )
    assert "STGCN" not in figures.ABLATION_MODELS


def test_submission_overall_table_uses_plain_model_names_and_three_decimals():
    frame = pd.read_csv(
        ROOT / "paper/tables/literature/table_literature_comparison_common46.csv"
    )
    text = tables._overall_markdown(frame)
    assert "| 24 h | GRU |" in text
    assert "†" not in text
    assert "GRU (reported)" not in text
    assert "9.424" in text
    assert "9.424199" not in text
    assert "Published reference models" not in text
    assert "Re-evaluated graph models" not in text


def test_submission_ablation_table_is_four_model_factorial():
    frame = pd.read_csv(
        ROOT / "paper/tables/literature/table_ablation_common46.csv"
    )
    text = tables._ablation_markdown(frame)
    assert "STGCN" not in "\n".join(
        line for line in text.splitlines() if line.startswith("| 24 h") or line.startswith("| 168 h")
    )
    assert "**12.208**" in text
    assert "12.234" in text
    assert "**2.014**" in text
    assert "**6.161**" in text
    assert "**0.976**" in text


def test_submission_contract_is_documented():
    paper_readme = (ROOT / "paper/README.md").read_text(encoding="utf-8")
    assert "Main Figure 1 — Overall four-metric performance" in paper_readme
    assert "Main Figure 2 — DMA-level performance breadth" in paper_readme
    assert "Main Figure 3 — DMA-specific local competitive margin" in paper_readme
    assert "Main Figure 4 — Four-metric ablation and lead-time stability" in paper_readme
    assert "Main Figure 5 — Forecast-origin and difficult-window robustness" in paper_readme
    assert "Main Figure 6 — Week-ahead demand dynamics" in paper_readme
    assert "Supplementary Table S2 — DMA-level local margins" in paper_readme
    assert "Supplementary Table S3 — Forecast-origin robustness" in paper_readme


def test_dma_level_comparison_covers_all_models_metrics_and_horizons():
    frame = pd.read_csv(
        ROOT / "paper/tables/literature/table_all_models_dma.csv"
    )
    assert frame.shape == (180, 7)
    assert set(frame["task"]) == {"24h", "168h"}
    assert set(frame["DMA"]) == set("ABCDEFGHIJ")
    assert set(frame["model"]) == set(figures.DMA_MODELS)

    ranks = figures._derive_dma_ranks(frame)
    star = ranks.loc[ranks["model"] == "STaR-GNN"]
    first = star.groupby("task")["rank"].apply(lambda values: int((values == 1).sum()))
    assert first.to_dict() == {"168h": 27, "24h": 36}
    assert star.groupby("task")["rank"].median().to_dict() == {
        "168h": 1.0,
        "24h": 1.0,
    }

    pairwise = figures._derive_dma_pairwise(frame)
    wins = pairwise.groupby("task")["star_better"].sum().astype(int)
    assert wins.to_dict() == {"168h": 275, "24h": 306}
    graph = pairwise.loc[pairwise["baseline_family"] == "graph"]
    graph_wins = graph.groupby("task")["star_better"].sum().astype(int)
    assert graph_wins.to_dict() == {"168h": 78, "24h": 80}


def test_dma_local_margin_supplement_retains_competitors_and_losses():
    frame = pd.read_csv(
        ROOT / "paper/tables/literature/table_all_models_dma.csv"
    )
    text = tables._dma_local_margin_markdown(frame)
    data_rows = [
        line for line in text.splitlines()
        if line.startswith("| 24 h |") or line.startswith("| 168 h |")
    ]
    assert len(data_rows) == 20
    assert "| 24 h | A | GRU | -9.2 | GRU | -5.2 |" in text
    assert "| 168 h | I |" in text
    assert "MSCMNet-WM | -0.046 |" in text
    assert "All values are retained" in text


def test_forecast_origin_robustness_uses_observed_difficulty_and_common46():
    summary = pd.read_csv(
        ROOT
        / "paper/tables/manuscript/submission/main_fig5_origin_summary.csv"
    )
    assert summary.shape[0] == 2 * 2 * 4
    assert set(summary["n_origins"]) == {46}
    assert set(summary["n_high_variability"]) == {12}

    mae_168_dcrnn = summary.loc[
        (summary["task"] == "168h")
        & (summary["baseline"] == "DCRNN")
        & (summary["metric"] == "MAE")
    ].iloc[0]
    assert int(mae_168_dcrnn["wins"]) == 46
    assert int(mae_168_dcrnn["high_variability_wins"]) == 12
    assert float(mae_168_dcrnn["ci95_lower"]) > 0.0

    mape_24_dcrnn = summary.loc[
        (summary["task"] == "24h")
        & (summary["baseline"] == "DCRNN")
        & (summary["metric"] == "MAPE")
    ].iloc[0]
    assert float(mape_24_dcrnn["ci95_lower"]) < 0.0
    assert float(mape_24_dcrnn["ci95_upper"]) > 0.0

    display = tables._origin_robustness_markdown(summary)
    assert "45/46" in display
    assert "12/12" in display
    assert "not treated as independent" in display
