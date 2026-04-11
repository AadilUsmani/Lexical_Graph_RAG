from __future__ import annotations

import logging
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Node:
    """Simple box model for deterministic layout."""

    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> tuple[float, float]:
        return (self.x - self.w / 2, self.y)

    @property
    def right(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y)

    @property
    def top(self) -> tuple[float, float]:
        return (self.x, self.y + self.h / 2)

    @property
    def bottom(self) -> tuple[float, float]:
        return (self.x, self.y - self.h / 2)

    def top_edge(self, t: float) -> tuple[float, float]:
        """Point on top edge with t in [0,1]."""
        return (self.x - self.w / 2 + self.w * t, self.y + self.h / 2)


def init_plotting_style() -> None:
    """IEEE-friendly typography and deterministic style."""
    sns.set_theme(style="white", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.titlesize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def _draw_node(
    ax: plt.Axes,
    node: Node,
    text: str,
    facecolor: str,
    *,
    text_color: str = "#1f2d3a",
    mono: bool = False,
    fontsize: float = 10.0,
) -> None:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (node.x - node.w / 2, node.y - node.h / 2),
            node.w,
            node.h,
            boxstyle="round,pad=0.015",
            facecolor=facecolor,
            edgecolor="#243447",
            linewidth=1.6,
            mutation_aspect=1.0,
            zorder=3,
        )
    )
    ax.text(
        node.x,
        node.y,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight="bold",
        family="monospace" if mono else None,
        zorder=4,
    )


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    lw: float = 1.6,
    style: str = "-|>",
    mutation_scale: float = 14.0,
    shrink_a: float = 6.0,
    shrink_b: float = 8.0,
) -> None:
    ax.add_patch(
        mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=lw,
            color="#2c3e50",
            shrinkA=shrink_a,
            shrinkB=shrink_b,
            zorder=2,
        )
    )


def _draw_context_box(ax: plt.Axes, node: Node, n_blue: int, n_orange: int) -> None:
    """Draw Retrieved Context box and balanced composition bars."""
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (node.x - node.w / 2, node.y - node.h / 2),
            node.w,
            node.h,
            boxstyle="round,pad=0.018",
            facecolor="#f9fbfc",
            edgecolor="#7f8c8d",
            linewidth=1.5,
            zorder=3,
        )
    )
    ax.text(
        node.x,
        node.y + node.h * 0.21,
        "Retrieved Context",
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color="#1f2d3a",
        zorder=4,
    )

    total = max(n_blue + n_orange, 1)
    pad_x = node.w * 0.07
    gap = node.w * 0.03
    bar_h = node.h * 0.31
    bar_y = node.y - node.h * 0.39
    bar_w = (node.w - 2 * pad_x - gap * (total - 1)) / total
    start_x = node.x - node.w / 2 + pad_x

    for i in range(total):
        color = "#4e79a7" if i < n_blue else "#f28e2b"
        ax.add_patch(
            mpatches.Rectangle(
                (start_x + i * (bar_w + gap), bar_y),
                bar_w,
                bar_h,
                facecolor=color,
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )
        )


def _plot_left_panel(ax: plt.Axes) -> None:
    """Latent space bias panel."""
    rng = np.random.default_rng(42)
    blue = "#4e79a7"
    orange = "#f28e2b"
    gray = "#7f8c8d"

    ax.set_title("Latent Space Bias (Unstructured Retrieval)", pad=14, y=1.015)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    query = (0.42, 0.56)
    radius = 0.24

    jpm_points = rng.multivariate_normal(
        mean=[0.36, 0.55],
        cov=[[0.0068, 0.0], [0.0, 0.0085]],
        size=88,
    )
    pypl_points = rng.multivariate_normal(
        mean=[0.73, 0.72],
        cov=[[0.0019, 0.0], [0.0, 0.0023]],
        size=14,
    )

    jpm_points[:, 0] = np.clip(jpm_points[:, 0], 0.08, 0.77)
    jpm_points[:, 1] = np.clip(jpm_points[:, 1], 0.20, 0.88)
    pypl_points[:, 0] = np.clip(pypl_points[:, 0], 0.57, 0.94)
    pypl_points[:, 1] = np.clip(pypl_points[:, 1], 0.48, 0.90)

    ax.scatter(jpm_points[:, 0], jpm_points[:, 1], s=50, color=blue, alpha=0.86, edgecolors="none", zorder=1)
    ax.scatter(pypl_points[:, 0], pypl_points[:, 1], s=58, color=orange, alpha=0.95, edgecolors="none", zorder=1)

    ax.add_patch(
        mpatches.Circle(
            query,
            radius=radius,
            facecolor="none",
            edgecolor="#5d7289",
            linewidth=2.2,
            linestyle="--",
            zorder=2,
        )
    )
    ax.scatter([query[0]], [query[1]], marker="*", s=540, color=gray, edgecolors="#1b1b1b", linewidths=1.4, zorder=5)

    ax.text(
        query[0],
        query[1] - radius - 0.045,
        "Query: Compare JPM and PYPL risks.",
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color="#1f2d3a",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.90),
        zorder=6,
    )
    ax.text(
        0.64,
        0.74,
        r"Search boundary ($k=5$)",
        fontsize=12.8,
        color="#5d7289",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
        zorder=6,
    )

    ax.text(
        0.15,
        0.355,
        "JPM cluster\n(dense mega-nodes)",
        color=blue,
        fontsize=13.5,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.79,
        0.86,
        "PYPL cluster\n(sparser neighborhood)",
        color=orange,
        fontsize=13.5,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=6,
    )

    context_node = Node(x=0.79, y=0.24, w=0.30, h=0.17)
    _draw_context_box(ax, context_node, n_blue=4, n_orange=1)
    _draw_arrow(ax, (query[0] + radius * 0.77, query[1] - radius * 0.49), context_node.top_edge(0.25), lw=2.0)

    left_label = (
        "Dense vector similarity naturally biases toward frequently mentioned "
        "mega-nodes, causing severe context imbalance."
    )
    ax.text(
        0.02,
        0.03,
        textwrap.fill(left_label, width=56),
        ha="left",
        va="bottom",
        fontsize=12.0,
        linespacing=1.25,
        color="#1f2d3a",
        bbox=dict(boxstyle="round,pad=0.50", facecolor="#ffffff", edgecolor="#b0b8bf", alpha=0.98),
        zorder=6,
    )


def _plot_right_panel(ax: plt.Axes) -> None:
    """Entity stratification panel."""
    blue = "#4e79a7"
    orange = "#f28e2b"
    gray = "#7f8c8d"

    ax.set_title("Entity Stratification (Lexical-Structural Retrieval)", pad=14, y=1.015)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    query = (0.50, 0.958)
    ner = Node(0.50, 0.76, 0.30, 0.074)
    anchor_l = Node(0.30, 0.61, 0.38, 0.082)
    anchor_r = Node(0.70, 0.61, 0.36, 0.082)
    collect_l = Node(0.30, 0.47, 0.30, 0.068)
    collect_r = Node(0.70, 0.47, 0.30, 0.068)
    facts_l = [Node(0.20, 0.34, 0.13, 0.058), Node(0.30, 0.34, 0.13, 0.058), Node(0.40, 0.34, 0.13, 0.058)]
    facts_r = [Node(0.60, 0.34, 0.13, 0.058), Node(0.70, 0.34, 0.13, 0.058), Node(0.80, 0.34, 0.13, 0.058)]
    context_node = Node(0.50, 0.17, 0.60, 0.16)

    ax.scatter([query[0]], [query[1]], marker="*", s=540, color=gray, edgecolors="#1b1b1b", linewidths=1.4, zorder=5)
    ax.text(
        query[0],
        query[1] - 0.108,
        "Query: Compare JPM and PYPL risks.",
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color="#1f2d3a",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.90),
        zorder=6,
    )

    _draw_arrow(ax, (query[0], query[1] - 0.03), ner.top, lw=1.9)
    _draw_arrow(ax, ner.bottom, anchor_l.top_edge(0.74), lw=1.8)
    _draw_arrow(ax, ner.bottom, anchor_r.top_edge(0.26), lw=1.8)
    _draw_arrow(ax, anchor_l.bottom, collect_l.top, lw=1.8)
    _draw_arrow(ax, anchor_r.bottom, collect_r.top, lw=1.8)

    for fact in facts_l:
        _draw_arrow(ax, collect_l.bottom, fact.top, lw=1.6)
    for fact in facts_r:
        _draw_arrow(ax, collect_r.bottom, fact.top, lw=1.6)

    left_targets = [0.00, 0.1667, 0.3333]
    right_targets = [0.6667, 0.8333, 1.00]
    for idx, fact in enumerate(facts_l):
        _draw_arrow(ax, fact.bottom, context_node.top_edge(left_targets[idx]), lw=1.5, shrink_b=1.2)
    for idx, fact in enumerate(facts_r):
        _draw_arrow(ax, fact.bottom, context_node.top_edge(right_targets[idx]), lw=1.5, shrink_b=1.2)

    _draw_node(ax, ner, "NER Split", "#bcc2c9", fontsize=12.2)
    _draw_node(ax, anchor_l, "[Anchor: JPMorgan]", blue, text_color="white", fontsize=10.8)
    _draw_node(ax, anchor_r, "[Anchor: PayPal]", orange, text_color="white", fontsize=10.8)
    _draw_node(ax, collect_l, "COLLECT()[0..k]", "#e8ecef", mono=True, fontsize=9.8)
    _draw_node(ax, collect_r, "COLLECT()[0..k]", "#e8ecef", mono=True, fontsize=9.8)

    for fact in facts_l:
        _draw_node(ax, fact, "Fact", "#eaf2fa", fontsize=11.0)
    for fact in facts_r:
        _draw_node(ax, fact, "Fact", "#fef2e8", fontsize=11.0)

    _draw_context_box(ax, context_node, n_blue=3, n_orange=3)

    right_label = (
        "Graph anchoring enforces bounded, 1:1 entity retrieval, guaranteeing "
        "a balanced comparative synthesis."
    )
    ax.text(
        0.02,
        0.03,
        textwrap.fill(right_label, width=60),
        ha="left",
        va="bottom",
        fontsize=11.8,
        linespacing=1.25,
        color="#1f2d3a",
        bbox=dict(boxstyle="round,pad=0.50", facecolor="#ffffff", edgecolor="#b0b8bf", alpha=0.98),
        zorder=6,
    )


def plot_latent_space_bias_vs_entity_stratification(out_dir: Path = Path("figures"), dpi: int = 400) -> tuple[Path, Path]:
    """
    Render publication-quality two-panel figure and save as PNG + PDF.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 9.2), facecolor="white")
    plt.subplots_adjust(left=0.03, right=0.985, top=0.93, bottom=0.17, wspace=0.06)

    _plot_left_panel(axes[0])
    _plot_right_panel(axes[1])

    caption = (
        "Fig X. Latent Space Bias vs. Entity Stratification. Unstructured dense retrieval (left) "
        "is susceptible to semantic density bias, where major institutional hubs monopolize the top-$k$ "
        "context window. The proposed Lexical-Structural pipeline (right) forces balanced, cross-document "
        "synthesis by anchoring retrieval paths to distinct query entities and applying strict collection "
        "bounds. This directly mitigates the precision degradation observed in multi-hop comparative queries "
        "(Section V.D)."
    )
    fig.text(
        0.5,
        0.075,
        textwrap.fill(caption, width=190),
        ha="center",
        va="center",
        fontsize=11.8,
        color="#1f2d3a",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "plot_robust_architecture_final_fixed"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=dpi, format="png", facecolor="white")
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    plt.close(fig)
    log.info("Saved high-quality diagram -> %s", png_path)
    log.info("Saved high-quality diagram -> %s", pdf_path)
    return png_path, pdf_path


def main() -> None:
    init_plotting_style()
    plot_latent_space_bias_vs_entity_stratification(out_dir=Path("figures"))


if __name__ == "__main__":
    main()
