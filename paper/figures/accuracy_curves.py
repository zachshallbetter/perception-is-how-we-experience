"""
Template 1: Accuracy Comparison Curves

Used for: Training curves, validation accuracy over time, convergence comparison
Style: Line plot with error bands (mean ± confidence interval)
Output: PDF/SVG vector graphics at 300 DPI
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_accuracy_curves(
    epochs,
    sp_mean,
    sp_std,
    perceiver_mean,
    perceiver_std,
    output_dir="results/figures",
    title="Split-CIFAR-100: Continual Learning Performance",
    xlabel="Training Epoch",
    ylabel="Validation Accuracy (%)",
):
    """
    Create publication-quality accuracy curves with error bands.

    Args:
        epochs: Array of epoch numbers (e.g., np.arange(1, 51))
        sp_mean: SensoryPerceiver mean accuracy per epoch
        sp_std: SensoryPerceiver std dev per epoch
        perceiver_mean: Perceiver IO mean accuracy per epoch
        perceiver_std: Perceiver IO std dev per epoch
        output_dir: Directory to save figures
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    # Set publication style
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    plt.rcParams["font.size"] = 11
    plt.rcParams["font.family"] = "sans-serif"

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot with confidence bands
    ax.plot(
        epochs,
        sp_mean,
        label="SensoryPerceiver",
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=5,
        color="#2E86AB",
    )
    ax.fill_between(
        epochs, sp_mean - sp_std, sp_mean + sp_std, alpha=0.2, color="#2E86AB"
    )

    ax.plot(
        epochs,
        perceiver_mean,
        label="Perceiver IO",
        linewidth=2.5,
        marker="s",
        markersize=4,
        markevery=5,
        color="#A23B72",
    )
    ax.fill_between(
        epochs,
        perceiver_mean - perceiver_std,
        perceiver_mean + perceiver_std,
        alpha=0.2,
        color="#A23B72",
    )

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend
    ax.legend(
        fontsize=11,
        loc="lower right",
        frameon=True,
        shadow=False,
        fancybox=False,
        edgecolor="black",
        framealpha=0.95,
    )

    # Clean up
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim([0, 100])

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        output_path / "accuracy_curves.pdf", dpi=300, bbox_inches="tight", format="pdf"
    )
    plt.savefig(
        output_path / "accuracy_curves.svg", dpi=300, bbox_inches="tight", format="svg"
    )
    plt.savefig(
        output_path / "accuracy_curves.png", dpi=300, bbox_inches="tight", format="png"
    )
    plt.close()

    print(f"✅ Saved accuracy curves to {output_path}")


if __name__ == "__main__":
    # Example usage with sample data
    epochs = np.arange(1, 51)

    # Generate sample data (replace with your actual results)
    sp_mean = 0.65 + 0.25 * (1 - np.exp(-epochs / 10)) + np.random.normal(0, 0.02, 50)
    sp_std = 0.03 + 0.01 * np.random.random(50)

    perceiver_mean = (
        0.60 + 0.20 * (1 - np.exp(-epochs / 12)) + np.random.normal(0, 0.025, 50)
    )
    perceiver_std = 0.035 + 0.01 * np.random.random(50)

    plot_accuracy_curves(epochs, sp_mean, sp_std, perceiver_mean, perceiver_std)
