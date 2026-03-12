"""
Template 5: Architecture Diagram (Graphviz-based)

Used for: System architecture visualization, model flow diagrams
Style: Directed graph with nodes and edges
Output: PDF/SVG vector graphics at 300 DPI
"""

from pathlib import Path
import re


def plot_architecture_diagram(
    nodes,
    edges,
    title="SensoryPerceiver Architecture",
    output_dir="results/figures",
    graph_attr=None,
    node_attr=None,
    edge_attr=None,
):
    """
    Create publication-quality architecture diagram using Graphviz.

    Args:
        nodes: Dict of node_id -> {'label': str, 'shape': str, 'color': str}
        edges: List of (from_id, to_id, {'label': str, 'style': str})
        title: Diagram title
        output_dir: Directory to save figures
        graph_attr: Dict of graph attributes (fontsize, rankdir, etc.)
        node_attr: Dict of default node attributes
        edge_attr: Dict of default edge attributes
    """
    try:
        import graphviz
    except ImportError:
        print(
            "⚠️  graphviz Python package not found. Install with: pip install graphviz"
        )
        print(
            "    Also ensure Graphviz is installed: brew install graphviz (Mac) or apt-get install graphviz (Linux)"
        )
        return

    # Default attributes
    if graph_attr is None:
        graph_attr = {
            "rankdir": "TB",
            "fontsize": "12",
            "bgcolor": "white",
            "nodesep": "0.5",
            "ranksep": "0.8",
        }

    if node_attr is None:
        node_attr = {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#E8F4F8",
            "color": "#2E86AB",
            "fontname": "Arial",
            "fontsize": "10",
            "margin": "0.3,0.2",
        }

    if edge_attr is None:
        edge_attr = {
            "color": "#666666",
            "fontsize": "9",
            "fontname": "Arial",
        }

    # Create graph
    dot = graphviz.Digraph(
        name=title.replace(" ", "_"),
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
    )

    # Add nodes
    for node_id, node_props in nodes.items():
        label = node_props.get("label", node_id)
        shape = node_props.get("shape", "box")
        color = node_props.get("color", "#E8F4F8")
        fillcolor = node_props.get("fillcolor", "#E8F4F8")

        dot.node(
            node_id,
            label=label,
            shape=shape,
            fillcolor=fillcolor,
            color=color,
        )

    # Add edges
    for edge in edges:
        from_id, to_id = edge[0], edge[1]
        edge_props = edge[2] if len(edge) > 2 else {}
        label = edge_props.get("label", "")
        style = edge_props.get("style", "solid")

        dot.edge(from_id, to_id, label=label, style=style)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Render to multiple formats
    filepath = output_path / title.replace(" ", "_")

    try:
        dot.render(str(filepath), format="pdf", cleanup=True)
        dot.render(str(filepath), format="svg", cleanup=True)
        dot.render(str(filepath), format="png", cleanup=True)
        print(f"✅ Saved architecture diagram to {output_path}")
    except Exception as e:
        print(f"⚠️  Failed to render diagram: {e}")
        print(f"   Graph source saved to {filepath}.gv for manual rendering")


if __name__ == "__main__":
    # Example: SensoryPerceiver architecture
    nodes = {
        "vision_input": {
            "label": "Vision Input\n(RGB 3×64×64)",
            "color": "#2E86AB",
            "fillcolor": "#D4E8F7",
        },
        "proprioception_input": {
            "label": "Proprioception\n(7 joint angles)",
            "color": "#A23B72",
            "fillcolor": "#F4D4E8",
        },
        "force_input": {
            "label": "Force/Torque\n(6-axis)",
            "color": "#F18F01",
            "fillcolor": "#FFE8D4",
        },
        "vision_tower": {
            "label": "PerceiverTower\n(Vision)",
            "shape": "box",
            "color": "#2E86AB",
            "fillcolor": "#E8F4F8",
        },
        "prop_tower": {
            "label": "PerceiverTower\n(IoT)",
            "shape": "box",
            "color": "#A23B72",
            "fillcolor": "#F4D4E8",
        },
        "force_tower": {
            "label": "PerceiverTower\n(IoT)",
            "shape": "box",
            "color": "#F18F01",
            "fillcolor": "#FFE8D4",
        },
        "cross_attention": {
            "label": "Cross-Modal\nAttention",
            "shape": "box",
            "color": "#6A994E",
            "fillcolor": "#E8F7E8",
        },
        "latent": {
            "label": "Latent\nRepresentation",
            "shape": "ellipse",
            "color": "#C73E1D",
            "fillcolor": "#FFE8D4",
        },
        "output": {
            "label": "Task Output\n(classification, regression)",
            "color": "#C73E1D",
            "fillcolor": "#FFE8D4",
        },
    }

    edges = [
        ("vision_input", "vision_tower", {}),
        ("proprioception_input", "prop_tower", {}),
        ("force_input", "force_tower", {}),
        ("vision_tower", "cross_attention", {"label": "modality tokens"}),
        ("prop_tower", "cross_attention", {"label": "modality tokens"}),
        ("force_tower", "cross_attention", {"label": "modality tokens"}),
        ("cross_attention", "latent", {"label": "cross-modal fusion"}),
        ("latent", "output", {}),
    ]

    plot_architecture_diagram(
        nodes,
        edges,
        title="SensoryPerceiver Architecture - Robotic Arm Task",
    )
