"""
Phase 2 Continual Learning Benchmark: Parameter Isolation Evaluation

Runs Split-CIFAR-100 with three configurations:
1. Baseline: replay_weight=0.1, buffer_size=1000, no parameter isolation
2. Tuned: replay_weight=0.5, buffer_size=5000, no parameter isolation
3. Tuned+ParamIso: replay_weight=0.5, buffer_size=5000, WITH parameter isolation

Compares results to validate that parameter isolation improves BWT.

Usage:
    python run_continual_learning_benchmark.py --configs all
    python run_continual_learning_benchmark.py --configs baseline
    python run_continual_learning_benchmark.py --device cuda:0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime
import sys

from src.models import SensoryPerceiver, ContrastiveLoss
from src.data.loaders import SplitCIFAR100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ContinualLearningBenchmark:
    """Runs continual learning experiments with metric tracking."""

    def __init__(
        self,
        n_tasks: int = 10,
        epochs_per_task: int = 5,
        device: torch.device = None,
        output_dir: str = "results/continual_learning",
        seed: int = 42,
    ):
        self.n_tasks = n_tasks
        self.epochs_per_task = epochs_per_task
        self.device = device or torch.device("cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Metrics tracking
        self.results = {}

    def create_model(self, enable_parameter_isolation: bool = False) -> SensoryPerceiver:
        """Create Sensory Perceiver model."""
        model = SensoryPerceiver(
            modality_configs={"vision": {"input_dim": 3, "use_conv": True}},
            dim=256,
            num_heads=4,
            mlp_dim=512,
            depth=4,
            dropout=0.1,
            enable_parameter_isolation=enable_parameter_isolation,
        )
        return model.to(self.device)

    def run_experiment(
        self,
        config_name: str,
        replay_weight: float = 0.1,
        replay_buffer_size: int = 1000,
        learning_rate: float = 1e-4,
        enable_parameter_isolation: bool = False,
    ) -> Dict:
        """Run continual learning experiment with given configuration."""

        logger.info(f"\n{'='*70}")
        logger.info(f"Configuration: {config_name}")
        logger.info(f"  replay_weight={replay_weight}")
        logger.info(f"  replay_buffer_size={replay_buffer_size}")
        logger.info(f"  learning_rate={learning_rate}")
        logger.info(f"  parameter_isolation={enable_parameter_isolation}")
        logger.info(f"{'='*70}\n")

        # Create model
        model = self.create_model(enable_parameter_isolation=enable_parameter_isolation)

        # Determine number of classes per task
        num_classes_per_task = 100 // self.n_tasks

        # Classifier (task-specific output layer)
        classifier = nn.Linear(256, num_classes_per_task).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        contrastive_criterion = ContrastiveLoss(temperature=0.07)
        optimizer = optim.AdamW(
            list(model.parameters()) + list(classifier.parameters()),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        # Metric tracking
        accuracy_matrix = np.zeros((self.n_tasks, self.n_tasks))  # R[i,j] = accuracy on task j after task i

        # Loop over sequential tasks
        for task_id in range(self.n_tasks):
            logger.info(f"\nTask {task_id}/{self.n_tasks-1}")
            logger.info("-" * 50)

            # Load task data
            train_dataset = SplitCIFAR100(
                n_tasks=self.n_tasks,
                task_id=task_id,
                split='train'
            )
            test_dataset = SplitCIFAR100(
                n_tasks=self.n_tasks,
                task_id=task_id,
                split='test'
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=0,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=0,
            )

            # Training loop
            for epoch in range(self.epochs_per_task):
                model.train()
                total_loss = 0.0
                total_correct = 0
                total_samples = 0

                for batch_idx, batch in enumerate(train_loader):
                    # Get data
                    images = batch['vision'].to(self.device)
                    labels = batch['label'].to(self.device)

                    # Forward pass (passes labels for replay buffer update)
                    optimizer.zero_grad()
                    outputs, embeddings = model(
                        {"vision": images},
                        labels=labels,
                        return_embeddings=True
                    )

                    # Pool and classify
                    pooled = outputs.mean(dim=1)
                    logits = classifier(pooled)

                    # Classification loss
                    loss_ce = criterion(logits, labels)

                    # Replay loss (NT-Xent with samples from buffer)
                    loss_total = loss_ce
                    if model.replay_count.item() > 0:
                        replay_emb, replay_labels = model.get_replay_samples(len(labels))
                        if replay_emb is not None and replay_emb.size(0) == len(labels):
                            replay_emb = replay_emb.to(self.device)
                            loss_replay = contrastive_criterion(pooled, replay_emb)
                            loss_total = (1 - replay_weight) * loss_ce + replay_weight * loss_replay

                    # Backward pass
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(classifier.parameters()),
                        max_norm=1.0
                    )
                    optimizer.step()

                    # Metrics
                    total_loss += loss_total.item() * len(labels)
                    _, predicted = logits.max(1)
                    total_correct += predicted.eq(labels).sum().item()
                    total_samples += len(labels)

                    if batch_idx % 50 == 0:
                        logger.info(
                            f"  Epoch {epoch+1}/{self.epochs_per_task} | "
                            f"Batch {batch_idx} | Loss: {loss_total.item():.4f}"
                        )

                train_acc = 100.0 * total_correct / total_samples
                logger.info(f"  Epoch {epoch+1}/{self.epochs_per_task} | Train Acc: {train_acc:.2f}%")

            # Note: Replay buffer is updated automatically in forward pass when labels are provided
            # (already populated during training loop above)

            # Evaluate on all tasks 0..task_id
            logger.info(f"\nEvaluating on all tasks...")
            for eval_task_id in range(task_id + 1):
                eval_dataset = SplitCIFAR100(
                    n_tasks=self.n_tasks,
                    task_id=eval_task_id,
                    split='test'
                )
                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=0,
                )

                model.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch in eval_loader:
                        images = batch['vision'].to(self.device)
                        labels = batch['label'].to(self.device)

                        outputs, _ = model({"vision": images}, return_embeddings=False)
                        pooled = outputs.mean(dim=1)
                        logits = classifier(pooled)

                        _, predicted = logits.max(1)
                        correct += predicted.eq(labels).sum().item()
                        total += len(labels)

                task_acc = 100.0 * correct / total
                accuracy_matrix[task_id, eval_task_id] = task_acc
                logger.info(f"  Task {eval_task_id}: {task_acc:.2f}%")

            # Parameter isolation: freeze current adapter and allocate new one
            if enable_parameter_isolation and hasattr(model, 'freeze_task'):
                model.freeze_task()
                logger.info(f"  Froze task {task_id} adapter")

        # Compute metrics
        cil = accuracy_matrix[-1, :].mean()  # Mean accuracy on all tasks after all tasks
        bwt = np.mean([accuracy_matrix[-1, j] - accuracy_matrix[j, j]
                       for j in range(self.n_tasks - 1)])  # Backward transfer
        random_acc = 100.0 / num_classes_per_task  # Random guessing for this task
        fwt = np.mean([accuracy_matrix[i-1, i] - random_acc
                       for i in range(1, self.n_tasks)])  # Forward transfer

        # Save results
        results = {
            'config_name': config_name,
            'cil': float(cil),
            'bwt': float(bwt),
            'fwt': float(fwt),
            'accuracy_matrix': accuracy_matrix.tolist(),
            'hyperparameters': {
                'replay_weight': replay_weight,
                'replay_buffer_size': replay_buffer_size,
                'learning_rate': learning_rate,
                'parameter_isolation': enable_parameter_isolation,
                'n_tasks': self.n_tasks,
                'epochs_per_task': self.epochs_per_task,
            }
        }

        logger.info(f"\n{'='*70}")
        logger.info(f"Results for {config_name}")
        logger.info(f"  CIL: {cil:.2f}%")
        logger.info(f"  BWT: {bwt:.2f}%")
        logger.info(f"  FWT: {fwt:.4f}")
        logger.info(f"{'='*70}\n")

        # Save to JSON
        result_file = self.output_dir / f"{config_name}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {result_file}")

        return results

    def run_all_configs(self) -> Dict[str, Dict]:
        """Run all three configurations."""
        all_results = {}

        # Configuration 1: Baseline
        all_results['baseline'] = self.run_experiment(
            config_name='baseline',
            replay_weight=0.1,
            replay_buffer_size=1000,
            learning_rate=1e-4,
            enable_parameter_isolation=False,
        )

        # Configuration 2: Tuned (no parameter isolation)
        all_results['tuned'] = self.run_experiment(
            config_name='tuned',
            replay_weight=0.5,
            replay_buffer_size=5000,
            learning_rate=3e-4,
            enable_parameter_isolation=False,
        )

        # Configuration 3: Tuned with Parameter Isolation
        all_results['tuned_paramiso'] = self.run_experiment(
            config_name='tuned_paramiso',
            replay_weight=0.5,
            replay_buffer_size=5000,
            learning_rate=3e-4,
            enable_parameter_isolation=True,
        )

        return all_results

    def print_comparison_table(self, all_results: Dict[str, Dict]):
        """Print comparison table of all configurations."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 2 CONTINUAL LEARNING BENCHMARK RESULTS")
        logger.info("="*70)
        logger.info(f"\n{'Configuration':<20} {'CIL':<12} {'BWT':<12} {'FWT':<12}")
        logger.info("-" * 70)

        for config_name, results in all_results.items():
            logger.info(
                f"{config_name:<20} "
                f"{results['cil']:>10.2f}% "
                f"{results['bwt']:>10.2f}% "
                f"{results['fwt']:>10.4f}"
            )

        # Compute improvements
        baseline_bwt = all_results['baseline']['bwt']
        tuned_bwt = all_results['tuned']['bwt']
        tuned_paramiso_bwt = all_results['tuned_paramiso']['bwt']

        logger.info("\n" + "-" * 70)
        logger.info("Improvements:")
        logger.info(f"  Tuned vs. Baseline BWT: {tuned_bwt - baseline_bwt:+.2f}%")
        logger.info(f"  Tuned+ParamIso vs. Tuned BWT: {tuned_paramiso_bwt - tuned_bwt:+.2f}%")
        logger.info(f"  Tuned+ParamIso vs. Baseline BWT: {tuned_paramiso_bwt - baseline_bwt:+.2f}%")
        logger.info("="*70 + "\n")


def main():
    parser = ArgumentParser(
        description='Phase 2: Parameter Isolation Continual Learning Benchmark'
    )
    parser.add_argument(
        '--configs',
        choices=['all', 'baseline', 'tuned', 'tuned_paramiso'],
        default='all',
        help='Which configurations to run'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/continual_learning',
        help='Output directory for results'
    )
    parser.add_argument(
        '--n-tasks',
        type=int,
        default=10,
        help='Number of tasks'
    )
    parser.add_argument(
        '--epochs-per-task',
        type=int,
        default=5,
        help='Epochs per task'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create benchmark
    benchmark = ContinualLearningBenchmark(
        n_tasks=args.n_tasks,
        epochs_per_task=args.epochs_per_task,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Run configurations
    all_results = {}

    if args.configs in ['all', 'baseline']:
        logger.info("Running BASELINE configuration...")
        all_results['baseline'] = benchmark.run_experiment(
            config_name='baseline',
            replay_weight=0.1,
            replay_buffer_size=1000,
            learning_rate=1e-4,
            enable_parameter_isolation=False,
        )

    if args.configs in ['all', 'tuned']:
        logger.info("Running TUNED configuration...")
        all_results['tuned'] = benchmark.run_experiment(
            config_name='tuned',
            replay_weight=0.5,
            replay_buffer_size=5000,
            learning_rate=3e-4,
            enable_parameter_isolation=False,
        )

    if args.configs in ['all', 'tuned_paramiso']:
        logger.info("Running TUNED+PARAMISO configuration...")
        all_results['tuned_paramiso'] = benchmark.run_experiment(
            config_name='tuned_paramiso',
            replay_weight=0.5,
            replay_buffer_size=5000,
            learning_rate=3e-4,
            enable_parameter_isolation=True,
        )

    # Print comparison
    if all_results:
        benchmark.print_comparison_table(all_results)

        # Save aggregate results
        aggregate_file = Path(args.output_dir) / 'phase2_aggregate_results.json'
        with open(aggregate_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved aggregate results to {aggregate_file}")


if __name__ == '__main__':
    main()
