"""
Template 2: Ablation Bar Charts

Used for: Component importance, ablation study results
Style: Grouped bar chart with error bars
Output: PDF/SVG vector graphics at 300 DPI
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_ablation_results(
    ablations,
    accuracies,
    stds,
    colors=None,
    output_dir="results/figures",
    title="SensoryPerceiver: Component Importance (Split-CIFAR-100)",
    ylabel="Accuracy (%)",
    full_model_accuracy=None,
):
    """
    Create publication-quality ablation bar chart.

    Args:
        ablations: List of ablation names (e.g., ['With Replay', 'Without Replay', ...])
        accuracies: List of accuracy values
        stds: List of standard deviations
        colors: Optional list of colors for bars
        output_dir: Directory to save figures
        title: Plot title
        ylabel: Y-axis label
        full_model_accuracy: Accuracy of full model (for reference line)
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    # Default colors if not provided
    if colors is None:
        colors = ["#2E86AB", "#A23B72"] * (len(ablations) // 2)
        if len(ablations) % 2 == 1:
            colors.append("#F18F01")

    x = np.arange(len(ablations))
    bars = ax.bar(
        x,
        accuracies,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        error_kw={"elinewidth": 2},
    )

    # Labels
    ax.set_xlabel("Ablation Study", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ablations, rotation=45, ha="right")

    # Grid and limits
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim([70, 90])

    # Reference line for full model
    if full_model_accuracy is not None:
        ax.axhline(
            y=full_model_accuracy,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            label=f"Full Model ({full_model_accuracy:.1f}%)",
        )
        ax.legend(fontsize=10)

    # Annotations: show drop percentages
    for i, (acc, std) in enumerate(zip(accuracies, stds)):
        if i % 2 == 1 and i > 0:  # Show drop for "without" variants
            drop = accuracies[i - 1] - acc
            ax.text(
                i,
                acc + std + 0.5,
                f"−{drop:.1f}%",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        output_path / "ablation_results.pdf", dpi=300, bbox_inches="tight", format="pdf"
    )
    plt.savefig(
        output_path / "ablation_results.svg", dpi=300, bbox_inches="tight", format="svg"
    )
    plt.savefig(
        output_path / "ablation_results.png", dpi=300, bbox_inches="tight", format="png"
    )
    plt.close()

    print(f"✅ Saved ablation results to {output_path}")


if __name__ == "__main__":
    # Example usage with sample data
    ablations = [
        "With Replay",
        "Without Replay",
        "With MIME",
        "Without MIME",
        "Small (100M)",
        "Medium (234M)",
        "Large (400M)",
    ]
    accuracies = [85.2, 78.1, 85.2, 82.0, 82.1, 85.2, 86.1]
    stds = [0.8, 1.2, 0.8, 1.1, 1.0, 0.8, 0.9]
    colors = [
        "#2E86AB",
        "#A23B72",
        "#2E86AB",
        "#A23B72",
        "#F18F01",
        "#C73E1D",
        "#6A994E",
    ]

    plot_ablation_results(
        ablations, accuracies, stds, colors=colors, full_model_accuracy=85.2
    )
