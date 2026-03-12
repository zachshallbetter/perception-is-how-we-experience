"""
Template 3: Efficiency Frontier (Pareto Analysis)

Used for: Parameters vs accuracy tradeoff, model scaling comparisons
Style: Scatter plot with Pareto frontier line
Output: PDF/SVG vector graphics at 300 DPI
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_efficiency_frontier(
    model_names,
    parameters,
    accuracies,
    pareto_models=None,
    colors=None,
    output_dir="results/figures",
    title="SensoryPerceiver: Parameter Efficiency (Split-CIFAR-100)",
    xlabel="Model Parameters (Millions)",
    ylabel="Accuracy (%)",
):
    """
    Create publication-quality efficiency frontier plot.

    Args:
        model_names: List of model names (e.g., ['Small', 'Medium', 'Large'])
        parameters: List of parameter counts in millions
        accuracies: List of accuracy values
        pareto_models: List of model names on the Pareto frontier (optional)
        colors: Optional list of colors for points
        output_dir: Directory to save figures
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Default colors
    if colors is None:
        colors = ["#2E86AB"] * len(model_names)

    # Plot all points
    scatter = ax.scatter(
        parameters,
        accuracies,
        s=300,
        c=colors,
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
        zorder=3,
    )

    # Add labels for each point
    for i, (name, param, acc) in enumerate(zip(model_names, parameters, accuracies)):
        ax.annotate(
            name,
            (param, acc),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    # Plot Pareto frontier if provided
    if pareto_models is not None:
        pareto_indices = [
            model_names.index(m) for m in pareto_models if m in model_names
        ]
        if len(pareto_indices) > 0:
            pareto_x = [parameters[i] for i in pareto_indices]
            pareto_y = [accuracies[i] for i in pareto_indices]

            # Sort by parameters for line plot
            sorted_pairs = sorted(zip(pareto_x, pareto_y), key=lambda x: x[0])
            pareto_x_sorted, pareto_y_sorted = zip(*sorted_pairs)

            ax.plot(
                pareto_x_sorted,
                pareto_y_sorted,
                color="#F18F01",
                linestyle="--",
                linewidth=2.5,
                alpha=0.7,
                label="Pareto Frontier",
                zorder=2,
            )

    # Labels
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Legend
    if pareto_models is not None:
        ax.legend(fontsize=10)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        output_path / "efficiency_frontier.pdf",
        dpi=300,
        bbox_inches="tight",
        format="pdf",
    )
    plt.savefig(
        output_path / "efficiency_frontier.svg",
        dpi=300,
        bbox_inches="tight",
        format="svg",
    )
    plt.savefig(
        output_path / "efficiency_frontier.png",
        dpi=300,
        bbox_inches="tight",
        format="png",
    )
    plt.close()

    print(f"✅ Saved efficiency frontier to {output_path}")


if __name__ == "__main__":
    # Example usage with sample data
    model_names = [
        "Small (100M)",
        "Medium (234M)",
        "Large (400M)",
        "XL (600M)",
        "XXL (800M)",
    ]
    parameters = [100, 234, 400, 600, 800]
    accuracies = [82.1, 85.2, 86.1, 85.8, 85.4]
    pareto_models = ["Small (100M)", "Medium (234M)", "Large (400M)"]

    plot_efficiency_frontier(
        model_names,
        parameters,
        accuracies,
        pareto_models=pareto_models,
        title="SensoryPerceiver: Parameter Efficiency",
    )
