"""
Template 6: Attention Heatmap

Used for: Cross-modal attention visualization, where does model look?
Style: Heatmap with seaborn
Output: PDF/SVG vector graphics at 300 DPI
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_attention_heatmap(
    attention_matrix,
    row_labels=None,
    col_labels=None,
    output_dir="results/figures",
    title="Cross-Modal Attention Weights",
    cmap="YlOrRd",
    vmin=0,
    vmax=1,
):
    """
    Create publication-quality attention heatmap.

    Args:
        attention_matrix: 2D array of attention weights (rows=query, cols=key)
        row_labels: List of row labels (e.g., modality names: ['Vision', 'Proprioception', ...])
        col_labels: List of column labels (e.g., ['Query 1', 'Query 2', ...])
        output_dir: Directory to save figures
        title: Plot title
        cmap: Colormap (default: YlOrRd for attention)
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create heatmap
    sns.heatmap(
        attention_matrix,
        annot=True,  # Show values
        fmt='.2f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.5,
        linecolor='gray',
        xticklabels=col_labels if col_labels else True,
        yticklabels=row_labels if row_labels else True,
        ax=ax,
    )

    # Labels
    ax.set_xlabel('Query Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Key Modality', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Improve readability
    ax.tick_params(axis='both', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path / 'attention_heatmap.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path / 'attention_heatmap.svg', dpi=300, bbox_inches='tight', format='svg')
    plt.savefig(output_path / 'attention_heatmap.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()

    print(f"✅ Saved attention heatmap to {output_path}")


if __name__ == '__main__':
    # Example usage: cross-modal attention for robotic task
    # Rows = source modalities, Columns = query positions in latent space
    row_labels = ['Vision', 'Proprioception', 'Force/Torque', 'Target Position']
    col_labels = [f'Q{i}' for i in range(1, 9)]

    # Simulated attention weights (4 modalities × 8 query positions)
    attention_weights = np.array([
        [0.85, 0.78, 0.72, 0.68, 0.45, 0.38, 0.25, 0.15],  # Vision attends to early queries
        [0.12, 0.18, 0.25, 0.30, 0.62, 0.68, 0.75, 0.65],  # Proprioception spreads across middle/late
        [0.02, 0.03, 0.02, 0.01, 0.35, 0.42, 0.50, 0.58],  # Force attends to late queries
        [0.01, 0.01, 0.01, 0.01, 0.58, 0.52, 0.50, 0.62],  # Target attention high throughout
    ])

    plot_attention_heatmap(
        attention_weights,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Cross-Modal Attention: Franka Panda Reaching Task",
        cmap="YlOrRd",
    )
