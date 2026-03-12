"""
Publication-quality figure creation templates for SensoryPerceiver research papers.

All figures output to PDF/SVG/PNG at 300 DPI with publication-grade styling,
matching the visual standards of "Attention Is All You Need" (Vaswani et al., 2017).

QUICK START:
    from figures import plot_accuracy_curves
    plot_accuracy_curves(
        epochs=range(1, 51),
        sp_mean=[65, 70, 75, ...],
        sp_std=[2, 1.8, 1.5, ...],
        perceiver_mean=[62, 68, 72, ...],
        perceiver_std=[2.5, 2, 1.8, ...],
    )

FIGURE TEMPLATES (7 types):
    1. Accuracy Curves - Training/validation performance over epochs with error bands
    2. Ablation Results - Component importance with error bars
    3. Efficiency Frontier - Parameter count vs accuracy Pareto analysis
    4. Comparison Table - Method comparison as publication-quality image
    5. Architecture Diagram - System architecture flowchart (requires graphviz)
    6. Attention Heatmap - Cross-modal attention weight visualization
    7. Radar Comparison - Multi-metric capability comparison

STYLING FEATURES:
    ✓ Publication-ready color palette (colorblind-friendly)
    ✓ Vector graphics output (PDF/SVG) at 300 DPI
    ✓ Professional fonts and spacing
    ✓ Error bands and confidence intervals
    ✓ Grid lines and proper axis labeling
    ✓ Legend and figure captions
    ✓ Automatic figure saving to results/figures/

USAGE IN PAPERS:
    - Main Paper (NeurIPS): Figures 1-5 using templates 1, 2, 3
    - Architecture Paper (JMLR): Figures 1-5 using templates 3, 4, 5
    - Application Papers: Figures 1-3 per paper using templates 1, 2, 6, 7

REQUIREMENTS:
    - matplotlib >= 3.5
    - seaborn >= 0.12
    - numpy >= 1.20
    - graphviz (Python package) + graphviz binary (for architecture diagrams)
      Install: pip install graphviz && brew install graphviz (Mac)

OUTPUT:
    All figures saved to results/figures/ with three formats:
    - figure_name.pdf (for LaTeX)
    - figure_name.svg (for web/presentation)
    - figure_name.png (for quick preview)
"""

from .accuracy_curves import plot_accuracy_curves
from .ablation_results import plot_ablation_results
from .efficiency_frontier import plot_efficiency_frontier
from .comparison_table import plot_comparison_table
from .architecture_diagram import plot_architecture_diagram
from .attention_heatmap import plot_attention_heatmap
from .radar_comparison import plot_radar_comparison

__all__ = [
    "plot_accuracy_curves",
    "plot_ablation_results",
    "plot_efficiency_frontier",
    "plot_comparison_table",
    "plot_architecture_diagram",
    "plot_attention_heatmap",
    "plot_radar_comparison",
]

# Convenience dictionary for accessing all templates
TEMPLATES = {
    "accuracy_curves": plot_accuracy_curves,
    "ablation_results": plot_ablation_results,
    "efficiency_frontier": plot_efficiency_frontier,
    "comparison_table": plot_comparison_table,
    "architecture_diagram": plot_architecture_diagram,
    "attention_heatmap": plot_attention_heatmap,
    "radar_comparison": plot_radar_comparison,
}
