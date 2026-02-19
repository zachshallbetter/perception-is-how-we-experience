"""
Template 7: Radar/Multi-Metric Comparison Chart

Used for: Multi-dimensional capability comparison, overall assessment
Style: Radar chart (polar coordinates)
Output: PDF/SVG vector graphics at 300 DPI
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_radar_comparison(
    categories,
    models_data,
    output_dir="results/figures",
    title="Multi-Metric Model Comparison",
    colors=None,
):
    """
    Create publication-quality radar chart for multi-metric comparison.

    Args:
        categories: List of metric names (e.g., ['Accuracy', 'Speed', 'Parameters', 'Robustness'])
        models_data: Dict of model_name -> list of values (one per category, normalized 0-1)
        output_dir: Directory to save figures
        title: Plot title
        colors: Optional dict of model_name -> color
    """
    num_categories = len(categories)

    # Calculate angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Default colors
    if colors is None:
        colors = {
            'SensoryPerceiver': '#2E86AB',
            'Perceiver IO': '#A23B72',
            'LSTM': '#F18F01',
            'Hand-Coded': '#6A994E',
        }

    # Plot each model
    for model_name, values in models_data.items():
        values = list(values)
        values += values[:1]  # Complete the circle

        color = colors.get(model_name, '#2E86AB')

        ax.plot(
            angles, values,
            'o-',
            linewidth=2.5,
            label=model_name,
            color=color,
            markersize=8,
        )

        # Fill area
        ax.fill(angles, values, alpha=0.15, color=color)

    # Customize axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')

    # Set y-axis limits and ticks
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.set_rlabel_position(0)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Title and legend
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.1),
        fontsize=11,
    )

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path / 'radar_comparison.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path / 'radar_comparison.svg', dpi=300, bbox_inches='tight', format='svg')
    plt.savefig(output_path / 'radar_comparison.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()

    print(f"✅ Saved radar comparison to {output_path}")


if __name__ == '__main__':
    # Example usage: multi-metric comparison across models
    # All values normalized to [0, 1] scale

    categories = [
        'Accuracy',
        'Speed',
        'Robustness',
        'Generalization',
        'Interpretability',
    ]

    models_data = {
        'SensoryPerceiver': [0.87, 0.65, 0.85, 0.82, 0.78],
        'Perceiver IO': [0.82, 0.55, 0.75, 0.70, 0.65],
        'LSTM': [0.80, 0.45, 0.68, 0.65, 0.72],
        'Hand-Coded Fusion': [0.75, 0.90, 0.50, 0.40, 0.95],
    }

    colors = {
        'SensoryPerceiver': '#2E86AB',
        'Perceiver IO': '#A23B72',
        'LSTM': '#F18F01',
        'Hand-Coded Fusion': '#6A994E',
    }

    plot_radar_comparison(
        categories,
        models_data,
        colors=colors,
        title="SensoryPerceiver: Overall Capability Comparison",
    )
