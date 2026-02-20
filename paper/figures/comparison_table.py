"""
Template 4: Comparison Table as Publication Image

Used for: Side-by-side method comparisons, results tables
Style: Formatted table with matplotlib table rendering
Output: PDF/SVG vector graphics at 300 DPI
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


def plot_comparison_table(
    column_labels,
    row_labels,
    data,
    output_dir="results/figures",
    title="Performance Comparison: SensoryPerceiver vs Baselines",
    highlight_row=None,
    highlight_color='#FFE5E5',
):
    """
    Create publication-quality comparison table.

    Args:
        column_labels: List of column headers (e.g., ['Model', 'Accuracy', 'Params', 'Speed'])
        row_labels: List of model names (rows)
        data: 2D list/array of cell values (rows × columns)
        output_dir: Directory to save figures
        title: Plot title
        highlight_row: Optional row index to highlight (e.g., SensoryPerceiver row)
        highlight_color: Color for highlighted row
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data (column labels + data)
    table_data = [column_labels]
    table_data.extend(data)

    # Create table
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.15, 0.15, 0.15]
    )

    # Style header row
    for i in range(len(column_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        cell.set_height(0.08)

    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(len(column_labels)):
            cell = table[(i, j)]
            cell.set_height(0.08)

            # Highlight specific row if requested
            if highlight_row is not None and i - 1 == highlight_row:
                cell.set_facecolor(highlight_color)
                cell.set_text_props(weight='bold', fontsize=10)
            else:
                # Alternate row colors for readability
                if i % 2 == 0:
                    cell.set_facecolor('#F5F5F5')
                else:
                    cell.set_facecolor('white')
                cell.set_text_props(fontsize=10)

            # Border
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(1)

    # Title
    fig.text(0.5, 0.95, title, ha='center', fontsize=14, fontweight='bold')

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path / 'comparison_table.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path / 'comparison_table.svg', dpi=300, bbox_inches='tight', format='svg')
    plt.savefig(output_path / 'comparison_table.png', dpi=300, bbox_inches='tight', format='png')
    plt.close()

    print(f"✅ Saved comparison table to {output_path}")


if __name__ == '__main__':
    # Example usage with sample data
    column_labels = ['Method', 'Accuracy (%)', 'Parameters (M)', 'Inference (ms)']

    row_labels = [
        'Hand-Coded Fusion',
        'EKF Fusion',
        'LSTM',
        'Perceiver IO',
        'SensoryPerceiver'
    ]

    data = [
        ['Hand-Coded Fusion', '75.2', '2.1', '12.5'],
        ['EKF Fusion', '78.1', '0.8', '8.3'],
        ['LSTM', '80.3', '5.4', '15.2'],
        ['Perceiver IO', '82.1', '234.0', '45.8'],
        ['SensoryPerceiver', '87.2', '234.0', '48.2'],
    ]

    plot_comparison_table(
        column_labels,
        row_labels,
        data,
        highlight_row=4,  # Highlight SensoryPerceiver
        title="Split-CIFAR-100: Continual Learning Performance Comparison"
    )
