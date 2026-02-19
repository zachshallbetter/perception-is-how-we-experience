"""
Evaluation harness for Sensory Perceiver experiments.

Generates:
- Per-class accuracy
- Confusion matrices
- Embedding visualizations (t-SNE)
- Latency measurements
- Cross-modal mutual information
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import json
import csv
import logging
from typing import Dict, Tuple, Optional

from models import SensoryPerceiver
from data_loader import create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluation manager for Sensory Perceiver."""

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        n_classes: int,
        output_dir: Path,
    ):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.n_classes = n_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run full evaluation suite."""
        logger.info("Starting evaluation...")
        self.model.eval()

        all_preds = []
        all_labels = []
        all_embeddings = []
        all_latencies = []

        for batch_idx, batch in enumerate(self.val_loader):
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
            labels = inputs.pop('label')

            # Measure latency
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            t_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

            if torch.cuda.is_available():
                t_start.record()
            outputs, embeddings = self.model(inputs, return_embeddings=True)
            if torch.cuda.is_available():
                t_end.record()
                torch.cuda.synchronize()
                latency = t_start.elapsed_time(t_end)  # ms
                all_latencies.append(latency)

            # Get predictions
            logits = outputs.mean(dim=1)
            _, predicted = logits.max(1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Store embeddings for visualization
            if embeddings is not None:
                pooled_emb = torch.stack([emb.mean(dim=0) for emb in embeddings.values()])
                all_embeddings.append(pooled_emb.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")

        # Aggregate predictions and labels
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        results = self._compute_metrics(all_preds, all_labels)

        # Embeddings analysis
        if all_embeddings:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            results['embeddings_shape'] = all_embeddings.shape
            # Optionally save embeddings for t-SNE visualization
            np.save(self.output_dir / 'embeddings.npy', all_embeddings)

        # Latency analysis
        if all_latencies:
            results['latency_mean'] = float(np.mean(all_latencies))
            results['latency_std'] = float(np.std(all_latencies))
            results['latency_min'] = float(np.min(all_latencies))
            results['latency_max'] = float(np.max(all_latencies))

        return results

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict:
        """Compute classification metrics."""
        # Overall accuracy
        accuracy = np.mean(preds == labels)

        # Per-class accuracy
        per_class_acc = {}
        for c in range(self.n_classes):
            mask = labels == c
            if mask.sum() > 0:
                per_class_acc[f'class_{c}'] = float(np.mean(preds[mask] == labels[mask]))
            else:
                per_class_acc[f'class_{c}'] = None

        # Confusion matrix
        confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for pred, label in zip(preds, labels):
            confusion_matrix[label, pred] += 1

        results = {
            'overall_accuracy': float(accuracy),
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': confusion_matrix.tolist(),
            'n_samples': len(preds),
        }

        return results

    def save_results(self, results: Dict):
        """Save evaluation results."""
        # JSON
        with open(self.output_dir / 'eval_results.json', 'w') as f:
            # Convert numpy arrays for JSON serialization
            json_results = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    json_results[k] = v.tolist()
                else:
                    json_results[k] = v
            json.dump(json_results, f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        if results['latency_mean']:
            logger.info(f"Mean Latency: {results['latency_mean']:.2f} ms")


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct config
    config = checkpoint['config']

    # Recreate model
    model = SensoryPerceiver(
        modality_configs=config['modality_configs'],
        dim=config['d_model'],
        num_heads=config['num_heads'],
        mlp_dim=config['mlp_dim'],
        depth=config['depth'],
        dropout=config['dropout'],
        replay_buffer_size=config['replay_buffer_size'],
        contrastive_temp=config['contrastive_temp'],
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config


def main():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--experiment', type=str, default='continual',
                        choices=['multimodal', 'continual', 'mime_routing', 'iot'])
    parser.add_argument('--dataset', type=str, default='split_cifar')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_file', type=str, default='eval_results.json')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint, device)

    # Create dataloader
    _, val_loader = create_dataloaders(
        experiment=args.experiment,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    n_classes = config['n_classes']

    # Evaluate
    output_dir = Path(args.output_file).parent
    evaluator = Evaluator(
        model=model,
        val_loader=val_loader,
        device=device,
        n_classes=n_classes,
        output_dir=output_dir,
    )

    results = evaluator.evaluate()
    evaluator.save_results(results)


if __name__ == '__main__':
    main()
