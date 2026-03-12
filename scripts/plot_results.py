"""
Generate visualizations from evaluation results.

Outputs:
- Accuracy curves (train/val over epochs)
- Per-class accuracy bar charts
- Confusion matrices (heatmaps)
- Embedding t-SNE projections
- Latency histograms
"""

import csv
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print(
        "Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_confusion_matrix(confusion_matrix: np.ndarray, output_path: Path):
    """Plot confusion matrix heatmap."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_per_class_accuracy(per_class_acc: dict, output_path: Path):
    """Plot per-class accuracy bar chart."""
    if not MATPLOTLIB_AVAILABLE:
        return

    classes = list(per_class_acc.keys())
    accuracies = [per_class_acc[c] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(classes)), accuracies, color="steelblue", alpha=0.7)

    # Color bars by accuracy
    for bar, acc in zip(bars, accuracies):
        if acc < 0.7:
            bar.set_color("coral")
        elif acc < 0.85:
            bar.set_color("gold")
        else:
            bar.set_color("green")

    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_ylim([0, 1.05])
    ax.axhline(
        y=np.mean(accuracies),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(accuracies):.3f}",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved per-class accuracy to {output_path}")


def plot_training_curves(train_log_path: Path, output_path: Path):
    """Plot training loss and accuracy curves."""
    if not MATPLOTLIB_AVAILABLE:
        return

    epochs = []
    losses = []
    accs = []
    val_losses = []
    val_accs = []

    if train_log_path.exists():
        with open(train_log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                losses.append(float(row["loss"]))
                accs.append(float(row["acc"]))
                val_losses.append(float(row["val_loss"]))
                val_accs.append(float(row["val_acc"]))

        # Loss plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, losses, label="Train Loss", marker="o", markersize=3)
        ax1.plot(epochs, val_losses, label="Val Loss", marker="s", markersize=3)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, accs, label="Train Acc", marker="o", markersize=3)
        ax2.plot(epochs, val_accs, label="Val Acc", marker="s", markersize=3)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved training curves to {output_path}")


def plot_embeddings_tsne(embeddings_path: Path, labels_path: Path, output_path: Path):
    """Plot t-SNE of embeddings (if available)."""
    if not MATPLOTLIB_AVAILABLE:
        return

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("scikit-learn not available, skipping t-SNE")
        return

    if not embeddings_path.exists():
        logger.info("Embeddings file not found, skipping t-SNE visualization")
        return

    embeddings = np.load(embeddings_path)
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=range(len(embeddings_2d)),
        cmap="tab20",
        s=10,
        alpha=0.6,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Embedding Space (t-SNE)")
    plt.colorbar(scatter, ax=ax, label="Sample Index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved t-SNE plot to {output_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing eval_results.json",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save plots"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib/seaborn required for plotting")
        return

    # Find all eval_results.json files
    for eval_file in results_dir.rglob("eval_results.json"):
        exp_name = eval_file.parent.name
        logger.info(f"Processing {exp_name}...")

        with open(eval_file, "r") as f:
            results = json.load(f)

        exp_output_dir = output_dir / exp_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        if "confusion_matrix" in results:
            cm = np.array(results["confusion_matrix"])
            plot_confusion_matrix(cm, exp_output_dir / "confusion_matrix.png")

        # Per-class accuracy
        if "per_class_accuracy" in results:
            plot_per_class_accuracy(
                results["per_class_accuracy"], exp_output_dir / "per_class_accuracy.png"
            )

        # Training curves
        train_log = eval_file.parent / "train_log.csv"
        if train_log.exists():
            plot_training_curves(train_log, exp_output_dir / "training_curves.png")

        # Embeddings t-SNE
        embeddings_file = eval_file.parent / "embeddings.npy"
        if embeddings_file.exists():
            plot_embeddings_tsne(
                embeddings_file, None, exp_output_dir / "embeddings_tsne.png"
            )

    logger.info(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
