"""
FinTech latency-risk trade-off chart (publication style).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def init_plotting_style() -> None:
    """Initialize paper-style plotting defaults."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_fintech_tradeoff(
    out_dir: Path = Path("figures"),
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot latency vs hallucination risk with an SR 11-7 compliance zone.
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Compliance band: 0% to 2% hallucination.
    ax.axhspan(0, 2, color="#dff5df", alpha=0.9, zorder=0)
    ax.text(
        10.0,
        1.0,
        "Strict Regulatory Compliance Threshold (SR 11-7)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#1f5f1f",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#8bc48b", alpha=0.9),
    )

    # Point A: Baseline Vector RAG / Unconstrained LLM
    point_a_x = 1.2
    point_a_y = 14.0
    a_x_low, a_x_high = 0.5, 2.0
    ax.errorbar(
        point_a_x,
        point_a_y,
        xerr=[[point_a_x - a_x_low], [a_x_high - point_a_x]],
        fmt="o",
        color="#e74c3c",
        ecolor="#e74c3c",
        elinewidth=2,
        capsize=5,
        markersize=10,
        markeredgecolor="black",
        markeredgewidth=0.8,
        zorder=4,
    )
    ax.annotate(
        "Point A: Baseline Vector RAG\nLatency: ~0.5-2.0s | Hallucination: 14%",
        xy=(point_a_x, point_a_y),
        xytext=(3.6, 16.3),
        fontsize=10,
        color="#7a1d13",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color="#7a1d13", lw=1.2),
    )

    # Point B: Stratified GraphRAG
    point_b_x = 18.0
    point_b_y = 0.0
    ax.scatter(
        [point_b_x],
        [point_b_y],
        s=120,
        color="#2e8b57",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )
    ax.annotate(
        "Point B: Stratified GraphRAG\nLatency: ~18s | Hallucination: 0%",
        xy=(point_b_x, point_b_y),
        xytext=(13.2, 4.0),
        fontsize=10,
        color="#1c5a39",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color="#1c5a39", lw=1.2),
    )

    # Narrative dashed arrow: A -> B
    ax.annotate(
        "",
        xy=(point_b_x, point_b_y),
        xytext=(point_a_x, point_a_y),
        arrowprops=dict(
            arrowstyle="->",
            linestyle="--",
            lw=2.2,
            color="#2c3e50",
            mutation_scale=14,
        ),
        zorder=3,
    )
    ax.text(
        9.8,
        8.7,
        "The FinTech Trade-off:\nSacrificing ~17s of latency to eliminate catastrophic generation risk.",
        ha="center",
        va="center",
        fontsize=10,
        color="#2c3e50",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#7f8c8d", alpha=0.95),
    )

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xticks(range(0, 21, 2))
    ax.set_yticks(range(0, 21, 2))
    ax.set_yticklabels([f"{tick}%" for tick in range(0, 21, 2)])
    ax.set_xlabel("Computational Cost: Mean Latency (seconds)")
    ax.set_ylabel("Enterprise Risk: Hallucination Rate")
    ax.set_title("Latency vs Hallucination Risk for Financial RAG Systems", fontweight="bold", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", alpha=0.3, linestyle="--")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf"]:
        out_path = out_dir / f"plot_fintech_latency_hallucination_tradeoff.{fmt}"
        fig.savefig(out_path, dpi=dpi if fmt == "png" else None, bbox_inches="tight", format=fmt)
        log.info("Saved -> %s", out_path)

    return fig, ax


def main() -> None:
    init_plotting_style()
    plot_fintech_tradeoff(out_dir=Path("figures"))
    plt.show()


if __name__ == "__main__":
    main()
