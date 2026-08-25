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
    assert "Main Figure 2 — Four-metric ablation and lead-time stability" in paper_readme
    assert "Main Figure 3 — Four-metric temporal and spatial robustness" in paper_readme
    assert "Main Figure 4 — Week-ahead demand dynamics" in paper_readme
    assert "Supplementary Figure S1" in paper_readme
    assert "Supplementary Figure S2" in paper_readme
