"""
Training harness for Perceiver IO baseline experiments.

This is the comparison baseline for the Sensory Perceiver paper.
Tests whether Perceiver IO's generic latent bottleneck approach
can match or exceed SensoryPerceiver's modality-specific architecture.

Supports:
- Multimodal classification via latent bottleneck
- Continual learning (without replay - tests if generic architecture helps)
- IoT streaming classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import yaml
from argparse import ArgumentParser
from typing import Dict, Tuple, Optional
import logging

from perceiver_io_baseline import PerceiverIOBaseline
from data_loader import create_dataloaders


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerceiverIOTrainer:
    """Training manager for Perceiver IO baseline."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        device: torch.device,
        output_dir: Path,
        config: Dict,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.train_history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}', leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
            labels = inputs.pop('label')

            # Convert multi-modal inputs to single 2D tensor for Perceiver IO
            # Flatten all modalities into feature vector
            x = self._prepare_input(inputs)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x)

            # Classification loss (Perceiver IO outputs logits directly)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * labels.shape[0]
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.shape[0]

            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*total_correct/total_samples})

        avg_loss = total_loss / total_samples
        avg_acc = 100. * total_correct / total_samples
        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.val_loader:
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
            labels = inputs.pop('label')

            x = self._prepare_input(inputs)
            logits = self.model(x)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.shape[0]
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.shape[0]

        avg_loss = total_loss / total_samples
        avg_acc = 100. * total_correct / total_samples
        return avg_loss, avg_acc

    def _prepare_input(self, inputs: Dict) -> torch.Tensor:
        """
        Convert multi-modal inputs to single 2D tensor for Perceiver IO.

        Perceiver IO expects input of shape (batch, seq_len, input_dim).
        We flatten all modalities into a sequence.
        """
        tensors = []

        for modality_name, data in sorted(inputs.items()):
            if not isinstance(data, torch.Tensor):
                continue

            # Handle 4D image tensors (batch, C, H, W)
            if data.dim() == 4:
                batch_size, C, H, W = data.shape
                # Flatten to (batch, C*H*W)
                data = data.view(batch_size, C*H*W)

            # Handle 3D tensors (batch, seq_len, features) - already 3D
            elif data.dim() == 3:
                pass  # Already in (batch, seq_len, features) format

            # Handle 2D tensors (batch, features)
            elif data.dim() == 2:
                # Add sequence dimension: (batch, 1, features)
                data = data.unsqueeze(1)

            tensors.append(data)

        # Concatenate all modalities into single sequence
        if tensors:
            # Stack along feature dimension if same seq_len, else concatenate
            if all(t.dim() == 3 for t in tensors):
                # All have sequence dimension: concat along feature axis
                x = torch.cat(tensors, dim=2)
            elif all(t.dim() == 2 for t in tensors):
                # All are (batch, features): stack and keep sequence
                x = torch.stack(tensors, dim=1)
            else:
                # Mixed dimensions: try to handle
                x = torch.cat([t if t.dim() == 3 else t.unsqueeze(1)
                              for t in tensors], dim=2)
        else:
            raise ValueError("No valid input tensors found")

        return x

    def fit(self, epochs: int):
        """Train for specified number of epochs."""
        logger.info(f"Starting PerceiverIO training for {epochs} epochs on {self.device}")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)

            logger.info(
                f'Epoch {epoch+1}/{epochs} | '
                f'Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | '
                f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%'
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pt')
                logger.info(f'Saved best model with val_acc={val_acc:.2f}%')

        # Save final model and history
        self.save_checkpoint('latest_model.pt')
        self.save_history()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)

    def save_history(self):
        """Save training history."""
        path = self.output_dir / 'train_log.csv'
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'acc', 'val_loss', 'val_acc'])
            writer.writeheader()
            for i in range(len(self.train_history['loss'])):
                writer.writerow({
                    'epoch': i + 1,
                    'loss': self.train_history['loss'][i],
                    'acc': self.train_history['acc'][i],
                    'val_loss': self.train_history['val_loss'][i],
                    'val_acc': self.train_history['val_acc'][i],
                })


def main():
    parser = ArgumentParser()

    # Experiment
    parser.add_argument('--experiment', type=str, default='continual',
                        choices=['multimodal', 'continual', 'iot'])
    parser.add_argument('--dataset', type=str, default='split_cifar')

    # Model (Perceiver IO specific)
    parser.add_argument('--num_latents', type=int, default=256,
                        help='Number of latent vectors in Perceiver IO bottleneck')
    parser.add_argument('--dim_latent', type=int, default=512,
                        help='Dimension of latent vectors')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--mlp_dim', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)

    # Device
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    # I/O
    parser.add_argument('--output_dir', type=str, default='results/perceiver_io_baseline')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Training PerceiverIOBaseline baseline for comparison with SensoryPerceiver")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        experiment=args.experiment,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Determine input and output dimensions based on dataset
    if args.experiment == 'continual':
        # Image data: CIFAR or MNIST variants
        input_dim = 3 * 32 * 32  # 3-channel 32x32 images
        n_classes = 5  # Split-CIFAR: 5 classes per task
    elif args.experiment == 'iot':
        # Sensor data: smartwatch(7) + room_env(4) + robot_arm(9) = 20 dims
        input_dim = 7 + 4 + 9  # 20
        n_classes = 5
    else:
        # Default multimodal
        input_dim = 3 * 32 * 32 + 128  # Vision + Audio
        n_classes = 10

    logger.info(f"Input dim: {input_dim}, Output dim: {n_classes}")

    # Create Perceiver IO model
    model = PerceiverIOBaseline(
        input_dim=input_dim,
        output_dim=n_classes,
        num_latents=args.num_latents,
        dim_latent=args.dim_latent,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    )

    logger.info(f"Created PerceiverIOBaseline with {sum(p.numel() for p in model.parameters())} parameters")

    # Loss function (no replay mechanism for Perceiver IO - tests generic architecture)
    criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Save config
    config = vars(args)
    config['n_classes'] = n_classes
    config['input_dim'] = input_dim
    config['model_type'] = 'PerceiverIOBaseline'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Train
    trainer = PerceiverIOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        config=config,
    )

    trainer.fit(args.epochs)
    logger.info(f"Training complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
