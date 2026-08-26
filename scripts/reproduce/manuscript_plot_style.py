#!/usr/bin/env python
"""Shared publication style for Journal of Hydrology manuscript figures.

This module centralizes the visual grammar used by the canonical submission
renderer.  It is intentionally restrained: one hero method, quiet baselines,
variant-family colors, journal-width canvases, sparse spines, and editable
vector exports.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


HERO_BLUE = "#0F4D92"
BASELINE_DARK = "#5C5C5C"
BASELINE_LIGHT = "#A6A6A6"
SAS_BLUE = "#8FB6D5"
FA_VIOLET = "#9A86B8"
OBSERVED_BLACK = "#1F1F1F"
ZERO_GRAY = "#6E6E6E"
GRID_GRAY = "#D8D8D8"

MODEL_COLORS = {
    "STaR-GNN": HERO_BLUE,
    "DCRNN": BASELINE_DARK,
    "STGCN": BASELINE_LIGHT,
    "DCRNN + SAS-Norm": SAS_BLUE,
    "DCRNN + FA-DPR": FA_VIOLET,
}

MODEL_LINESTYLES = {
    "STaR-GNN": "-",
    "DCRNN": "--",
    "STGCN": ":",
    "DCRNN + SAS-Norm": "-.",
    "DCRNN + FA-DPR": (0, (5, 2)),
}

MODEL_MARKERS = {
    "STaR-GNN": "o",
    "DCRNN": "^",
    "STGCN": "s",
    "DCRNN + SAS-Norm": "D",
    "DCRNN + FA-DPR": "v",
}


def apply_publication_style() -> None:
    """Apply a conservative Nature/Elsevier-compatible Matplotlib style."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 8.2,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "star-gnn-joh-submission",
            "savefig.transparent": False,
        }
    )


def light_to_hero_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "star_blue",
        ["#F4F7FB", "#D9E5F1", "#9EBBD6", "#4C7FAE", HERO_BLUE],
    )


def style_axis(ax: plt.Axes, *, ygrid: bool = False) -> None:
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    if ygrid:
        ax.grid(axis="y", color=GRID_GRAY, linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
    )


def save_publication_figure(fig: plt.Figure, base: Path) -> None:
    """Save vector PDF + editable SVG + 300 dpi PNG preview."""
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    svg_path = base.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    # Matplotlib may emit trailing whitespace in SVG path lines.
    # Normalize generated SVGs so repository checks remain clean.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
