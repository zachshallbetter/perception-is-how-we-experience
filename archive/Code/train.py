"""
Training harness for Sensory Perceiver experiments.

Supports:
- Multimodal classification
- Continual learning (experience replay, EWC)
- MIME-Type routing evaluation
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

from models import SensoryPerceiver, ContrastiveLoss, PerceiverTower
from data_loader import create_dataloaders, SplitCIFAR100, PermutedMNIST, SyntheticIOTStream


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Training manager for Sensory Perceiver."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        replay_criterion: Optional[nn.Module],
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
        self.replay_criterion = replay_criterion
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

            # Forward pass
            self.optimizer.zero_grad()
            outputs, embeddings = self.model(inputs, return_embeddings=True)

            # Classification loss
            logits = outputs.mean(dim=1)  # Pool over sequence dimension
            loss = self.criterion(logits, labels)

            # Optional: Contrastive replay loss
            if self.replay_criterion is not None and embeddings is not None:
                replay_samples = self.model.get_replay_samples(logits.shape[0])
                pooled_emb = torch.stack([emb.mean(dim=0) for emb in embeddings.values()])
                replay_loss = self.replay_criterion(pooled_emb, replay_samples)
                loss = loss + 0.1 * replay_loss  # Weight replay loss

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

            outputs, _ = self.model(inputs, return_embeddings=False)
            logits = outputs.mean(dim=1)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.shape[0]
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.shape[0]

        avg_loss = total_loss / total_samples
        avg_acc = 100. * total_correct / total_samples
        return avg_loss, avg_acc

    def fit(self, epochs: int):
        """Train for specified number of epochs."""
        logger.info(f"Starting training for {epochs} epochs on {self.device}")

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
                        choices=['multimodal', 'continual', 'mime_routing', 'iot'])
    parser.add_argument('--dataset', type=str, default='split_cifar')

    # Model
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--mlp_dim', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--replay_buffer_size', type=int, default=2000)
    parser.add_argument('--contrastive_temp', type=float, default=0.07)

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
    parser.add_argument('--output_dir', type=str, default='results/exp')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        experiment=args.experiment,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Determine modality configs
    if args.experiment == 'continual':
        modality_configs = {'vision': {'input_dim': 3, 'use_conv': True}}
        n_classes = 5
    elif args.experiment == 'iot':
        modality_configs = {
            'smartwatch': {'input_dim': 7, 'use_conv': False},
            'room_env': {'input_dim': 4, 'use_conv': False},
            'robot_arm': {'input_dim': 9, 'use_conv': False},
        }
        n_classes = 5
    else:
        # Default multimodal
        modality_configs = {
            'vision': {'input_dim': 3, 'use_conv': True},
            'audio': {'input_dim': 128, 'use_conv': False},
        }
        n_classes = 10

    # Create model
    model = SensoryPerceiver(
        modality_configs=modality_configs,
        dim=args.d_model,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        depth=args.depth,
        dropout=args.dropout,
        replay_buffer_size=args.replay_buffer_size,
        contrastive_temp=args.contrastive_temp,
    )

    # Loss functions
    criterion = nn.CrossEntropyLoss()
    replay_criterion = ContrastiveLoss(temperature=args.contrastive_temp)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Save config
    config = vars(args)
    config['modality_configs'] = modality_configs
    config['n_classes'] = n_classes

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        replay_criterion=replay_criterion,
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
