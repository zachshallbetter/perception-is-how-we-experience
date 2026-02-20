# Publication-Quality Figure Templates

Complete set of 7 reusable figure templates for SensoryPerceiver research papers, matching the visual standards of "Attention Is All You Need" (Vaswani et al., 2017).

**Status**: ✅ All 7 templates implemented and ready to use

## Templates Overview

### 1. Accuracy Curves (`plot_accuracy_curves`)
**Purpose**: Visualize training/validation performance over time
**Output**: Line plot with error bands
**Use in**: All papers (Weeks 2-10)

```python
from figures import plot_accuracy_curves

plot_accuracy_curves(
    epochs=range(1, 51),
    sp_mean=[65.2, 70.1, 75.3, ...],
    sp_std=[2.1, 1.9, 1.6, ...],
    perceiver_mean=[62.3, 68.5, 72.1, ...],
    perceiver_std=[2.5, 2.2, 1.9, ...],
)
```

### 2. Ablation Results (`plot_ablation_results`)
**Purpose**: Show component importance via ablation studies
**Output**: Bar chart with error bars and drop annotations
**Use in**: Main Paper (Figure 4), Architecture Paper (Figure 5)

```python
from figures import plot_ablation_results

plot_ablation_results(
    ablations=['With Replay', 'Without Replay', 'With MIME', 'Without MIME'],
    accuracies=[85.2, 78.1, 85.2, 82.0],
    stds=[0.8, 1.2, 0.8, 1.1],
    full_model_accuracy=85.2,
)
```

### 3. Efficiency Frontier (`plot_efficiency_frontier`)
**Purpose**: Show parameter-accuracy tradeoff (Pareto analysis)
**Output**: Scatter plot with Pareto frontier line
**Use in**: Architecture Paper (Figure 3), Efficiency Section

```python
from figures import plot_efficiency_frontier

plot_efficiency_frontier(
    model_names=['Small', 'Medium', 'Large', 'XL'],
    parameters=[100, 234, 400, 600],
    accuracies=[82.1, 85.2, 86.1, 85.8],
    pareto_models=['Small', 'Medium', 'Large'],
)
```

### 4. Comparison Table (`plot_comparison_table`)
**Purpose**: Compare methods side-by-side as publication image
**Output**: Formatted table with highlighting
**Use in**: Main Paper (Figure 2), Robotics Paper (Figure 2)

```python
from figures import plot_comparison_table

plot_comparison_table(
    column_labels=['Method', 'Accuracy (%)', 'Parameters (M)', 'Inference (ms)'],
    row_labels=['Hand-Coded', 'EKF', 'LSTM', 'Perceiver IO', 'SensoryPerceiver'],
    data=[
        ['Hand-Coded Fusion', '75.2', '2.1', '12.5'],
        ['EKF Fusion', '78.1', '0.8', '8.3'],
        ['LSTM', '80.3', '5.4', '15.2'],
        ['Perceiver IO', '82.1', '234.0', '45.8'],
        ['SensoryPerceiver', '87.2', '234.0', '48.2'],
    ],
    highlight_row=4,  # Highlight SensoryPerceiver
)
```

### 5. Architecture Diagram (`plot_architecture_diagram`)
**Purpose**: Visualize system architecture as flowchart
**Output**: Graph visualization (PDF/SVG)
**Use in**: Architecture Paper (Figure 1), All papers' Intro
**Requires**: `graphviz` package

```bash
# Install graphviz (binary + Python package)
brew install graphviz  # macOS
pip install graphviz
```

```python
from figures import plot_architecture_diagram

nodes = {
    'input': {'label': 'Input\n(4 modalities)', 'color': '#2E86AB'},
    'tower': {'label': 'PerceiverTower', 'color': '#A23B72'},
    'attention': {'label': 'Cross-Modal\nAttention', 'color': '#6A994E'},
    'output': {'label': 'Output', 'color': '#C73E1D'},
}

edges = [
    ('input', 'tower', {}),
    ('tower', 'attention', {'label': 'modality tokens'}),
    ('attention', 'output', {}),
]

plot_architecture_diagram(nodes, edges)
```

### 6. Attention Heatmap (`plot_attention_heatmap`)
**Purpose**: Visualize cross-modal attention weights
**Output**: Heatmap with annotations
**Use in**: Medical Paper (Figure 3), Robotics Paper (Figure 2)

```python
from figures import plot_attention_heatmap
import numpy as np

# 4 modalities × 8 query positions
attention_weights = np.array([
    [0.85, 0.78, 0.72, 0.68, 0.45, 0.38, 0.25, 0.15],
    [0.12, 0.18, 0.25, 0.30, 0.62, 0.68, 0.75, 0.65],
    [0.02, 0.03, 0.02, 0.01, 0.35, 0.42, 0.50, 0.58],
    [0.01, 0.01, 0.01, 0.01, 0.58, 0.52, 0.50, 0.62],
])

plot_attention_heatmap(
    attention_weights,
    row_labels=['Vision', 'Proprioception', 'Force/Torque', 'Target'],
    col_labels=[f'Q{i}' for i in range(1, 9)],
)
```

### 7. Radar Comparison (`plot_radar_comparison`)
**Purpose**: Compare multiple metrics across models
**Output**: Radar/polar chart
**Use in**: Discussion sections, capability summary

```python
from figures import plot_radar_comparison

categories = ['Accuracy', 'Speed', 'Robustness', 'Generalization', 'Interpretability']

models_data = {
    'SensoryPerceiver': [0.87, 0.65, 0.85, 0.82, 0.78],
    'Perceiver IO': [0.82, 0.55, 0.75, 0.70, 0.65],
    'LSTM': [0.80, 0.45, 0.68, 0.65, 0.72],
    'Hand-Coded': [0.75, 0.90, 0.50, 0.40, 0.95],
}

plot_radar_comparison(categories, models_data)
```

## Integration with Papers

### Week 2: Create All Figures
```bash
cd /path/to/perception-is-how-we-experience
python -c "
from figures import *
import numpy as np

# Load your experimental results from results/summary.csv
# Create accuracy curves
plot_accuracy_curves(...)

# Create ablation results
plot_ablation_results(...)

# Create efficiency frontier
plot_efficiency_frontier(...)

# And so on...
"
```

### LaTeX Integration
All figures save as PDF, ready for LaTeX inclusion:

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/accuracy_curves.pdf}
  \caption{SensoryPerceiver achieves 87.2\% accuracy on Split-CIFAR-100...}
  \label{fig:accuracy}
\end{figure}
```

### Markdown Integration
Reference figures in `FIGURE_TEMPLATES.md`:

```markdown
![Accuracy Curves](figures/accuracy_curves.svg)

**Figure 1**: Training dynamics. SensoryPerceiver (blue) converges faster...
```

## Styling & Customization

All templates use consistent styling:

- **Colors**: Colorblind-friendly palette
  - Primary: #2E86AB (blue)
  - Secondary: #A23B72 (purple)
  - Accent: #F18F01 (orange)
  - Success: #6A994E (green)
  - Alert: #C73E1D (red)

- **Fonts**: Arial 10-12pt for readability
- **Resolution**: 300 DPI for publication
- **Margins**: 0.3 inches on all sides
- **Grid**: Subtle dashed lines for reference

### Customizing Colors

```python
from figures import plot_accuracy_curves

plot_accuracy_curves(
    ...,
    colors=['#E74C3C', '#3498DB'],  # Custom colors
    title="Custom Title"
)
```

## Output Formats

Each template generates 3 files automatically:

- **PDF** (vector, for LaTeX): `figure_name.pdf`
- **SVG** (vector, for web): `figure_name.svg`
- **PNG** (raster, for preview): `figure_name.png`

All saved to `results/figures/` by default.

## Usage Checklist for Week 2

- [ ] Load results from `results/summary.csv`
- [ ] Extract metrics per experiment
- [ ] Create `accuracy_curves.pdf` (training dynamics)
- [ ] Create `ablation_results.pdf` (component importance)
- [ ] Create `efficiency_frontier.pdf` (model scaling)
- [ ] Create `comparison_table.pdf` (baseline comparison)
- [ ] Create `attention_heatmap.pdf` (optional, if visualization available)
- [ ] Create `architecture_diagram.pdf` (system overview)
- [ ] Create `radar_comparison.pdf` (capability summary)
- [ ] Verify all files in `results/figures/`
- [ ] Copy to paper directory
- [ ] Reference in paper text with figure captions

## Troubleshooting

**ImportError: No module named 'graphviz'**
```bash
pip install graphviz
brew install graphviz  # macOS
apt-get install graphviz  # Linux
```

**Figures not saving**
- Check that `results/figures/` directory exists
- Verify write permissions
- Ensure matplotlib backend is set (usually automatic)

**Colors look different in PDF**
- Ensure PDF viewer uses sRGB color space
- Regenerate with `plot_function(..., output_dir="results/figures")`

**Heatmap labels overlapping**
- Reduce label font size or increase figure size
- Use `fig.tight_layout()` after creating figure

## Next Steps

1. **Week 2**: Use these templates to create all figures from experimental results
2. **Weeks 3-10**: Reference figures in paper sections
3. **Final submission**: Include all PDFs in supplementary materials

All templates are production-ready and require only data input to generate publication-quality figures.
