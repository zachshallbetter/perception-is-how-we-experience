"""
Visualization utilities for plots and analysis
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(
    train_losses: List[float],
    val_accs: List[float],
    title: str = "Training Progress",
    savepath: Optional[str] = None,
):
    """
    Plot training loss and validation accuracy.

    Args:
        train_losses: Training loss per epoch
        val_accs: Validation accuracy per epoch
        title: Plot title
        savepath: Path to save figure
    """
    if len(train_losses) == 0 or len(val_accs) == 0:
        raise ValueError("train_losses and val_accs must both be non-empty")
    if len(train_losses) != len(val_accs):
        raise ValueError("train_losses and val_accs must have the same length")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = np.arange(len(train_losses))
    ax1.plot(epochs, train_losses, "b-", label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_accs, "g-", label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_attention_weights(
    attention: torch.Tensor,
    title: str = "Cross-Modal Attention",
    savepath: Optional[str] = None,
):
    """
    Visualize attention weight matrix as heatmap.

    Args:
        attention: Attention weights [num_modalities, num_modalities]
        title: Plot title
        savepath: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(attention.cpu().numpy(), cmap="Blues", aspect="auto")
    ax.set_xlabel("Key Modality")
    ax.set_ylabel("Query Modality")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_modality_importance(
    importance_scores: Dict[str, float],
    title: str = "Modality Importance",
    savepath: Optional[str] = None,
):
    """
    Plot modality importance as bar chart.

    Args:
        importance_scores: Dict mapping modality name -> importance score
        title: Plot title
        savepath: Path to save figure
    """
    if not importance_scores:
        raise ValueError("importance_scores must not be empty")

    fig, ax = plt.subplots(figsize=(10, 6))

    modalities = list(importance_scores.keys())
    scores = list(importance_scores.values())

    colors = plt.cm.Spectral(np.linspace(0, 1, len(modalities)))
    ax.bar(modalities, scores, color=colors)
    ax.set_ylabel("Importance Score")
    ax.set_title(title)
    ax.set_ylim([0, max(scores) * 1.1])
    plt.xticks(rotation=45)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    else:
        plt.show()
