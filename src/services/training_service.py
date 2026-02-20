"""Training service for Sensory Perceiver runtime jobs."""

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.loaders import create_dataloaders
from src.models import ContrastiveLoss, SensoryPerceiver
from src.observability import MetricsCollector, emit_event, generate_trace_id
from src.runtime import (
    ErrorCode,
    TRAIN_CONFIG_SCHEMA_VERSION,
    TrainRuntimeConfig,
    classify_exception,
    resolve_modality_setup,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRunResult:
    """Artifacts produced by a training job."""

    output_dir: Path
    config_file: Path
    best_checkpoint: Path
    latest_checkpoint: Path
    history_file: Path
    metrics_file: Path
    trace_id: str


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
        classifier: Optional[nn.Module] = None,
        replay_mode: str = "er",
    ):
        self.model = model.to(device)
        self.classifier = classifier
        if self.classifier is not None:
            self.classifier = self.classifier.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.replay_mode = replay_mode
        self.replay_criterion = replay_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.train_history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    def _sync_optimizer_params(self):
        """Ensure optimizer includes newly added trainable parameters."""
        known_param_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group["params"]
        }
        new_params = [
            param
            for param in self.model.parameters()
            if param.requires_grad and id(param) not in known_param_ids
        ]
        if self.classifier is not None:
            new_params.extend(
                [
                    param
                    for param in self.classifier.parameters()
                    if param.requires_grad and id(param) not in known_param_ids
                ]
            )
        if new_params:
            self.optimizer.add_param_group({"params": new_params})

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self._sync_optimizer_params()
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch in pbar:
            inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            labels = inputs.pop("label")

            self.optimizer.zero_grad()
            outputs, embeddings = self.model(inputs, labels=labels, return_embeddings=True)

            pooled = outputs.mean(dim=1)
            if self.classifier is not None:
                logits = self.classifier(pooled)
            else:
                logits = pooled
            loss = self.criterion(logits, labels)

            if self.replay_mode == "er":
                replay_inputs, replay_labels = self.model.get_replay_samples(labels.shape[0])
                if replay_inputs is not None:
                    replay_inputs = {key: value.to(self.device) for key, value in replay_inputs.items()}
                    replay_labels = replay_labels.to(self.device)

                    rp_outputs, _ = self.model(replay_inputs, update_buffer=False)
                    rp_pooled = rp_outputs.mean(dim=1)
                    rp_logits = self.classifier(rp_pooled) if self.classifier is not None else rp_pooled
                    replay_loss = self.criterion(rp_logits, replay_labels)
                    loss = loss + replay_loss
            elif (
                self.replay_mode == "contrastive"
                and self.replay_criterion is not None
                and embeddings is not None
            ):
                replay_samples, _ = self.model.get_replay_samples(labels.shape[0])
                if replay_samples is not None:
                    replay_samples = {
                        key: value.to(self.device) for key, value in replay_samples.items()
                    }
                    with torch.no_grad():
                        rp_outputs, _ = self.model(replay_samples, update_buffer=False)
                        rp_features = rp_outputs.mean(dim=1)

                    current_features = pooled
                    n_pairs = min(current_features.size(0), rp_features.size(0))
                    if n_pairs > 0:
                        replay_loss = self.replay_criterion(
                            current_features[:n_pairs], rp_features[:n_pairs]
                        )
                        loss = loss + 0.1 * replay_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.shape[0]
            pbar.set_postfix(
                {"loss": loss.item(), "acc": 100.0 * total_correct / total_samples}
            )

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.val_loader:
            inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            labels = inputs.pop("label")

            outputs, _ = self.model(inputs, labels=None, return_embeddings=False)
            pooled = outputs.mean(dim=1)
            if self.classifier is not None:
                logits = self.classifier(pooled)
            else:
                logits = pooled
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.shape[0]
            _, predicted = logits.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.shape[0]

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        return avg_loss, avg_acc

    def fit(self, epochs: int):
        """Train for specified number of epochs."""
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_history["loss"].append(train_loss)
            self.train_history["acc"].append(train_acc)
            self.train_history["val_loss"].append(val_loss)
            self.train_history["val_acc"].append(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint("best_model.pt")

        self.save_checkpoint("latest_model.pt")
        self.save_history()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "classifier_state_dict": (
                    self.classifier.state_dict() if self.classifier is not None else None
                ),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def save_history(self):
        """Save training history."""
        path = self.output_dir / "train_log.csv"
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=["epoch", "loss", "acc", "val_loss", "val_acc"]
            )
            writer.writeheader()
            for index in range(len(self.train_history["loss"])):
                writer.writerow(
                    {
                        "epoch": index + 1,
                        "loss": self.train_history["loss"][index],
                        "acc": self.train_history["acc"][index],
                        "val_loss": self.train_history["val_loss"][index],
                        "val_acc": self.train_history["val_acc"][index],
                    }
                )


def run_training(runtime_config: TrainRuntimeConfig) -> TrainingRunResult:
    """Execute a training job from validated runtime config."""
    trace_id = generate_trace_id("training")
    metrics = MetricsCollector()
    start_time = time.perf_counter()
    metrics_file = runtime_config.output_dir / "training_metrics.json"
    emit_event(
        logger,
        logging.INFO,
        "training.started",
        trace_id,
        experiment=runtime_config.experiment,
        dataset=runtime_config.dataset,
        output_dir=str(runtime_config.output_dir),
    )

    try:
        torch.manual_seed(runtime_config.seed)
        np.random.seed(runtime_config.seed)

        train_loader, val_loader = create_dataloaders(
            experiment=runtime_config.experiment,
            dataset_name=runtime_config.dataset,
            batch_size=runtime_config.batch_size,
            num_workers=runtime_config.num_workers,
            seed=runtime_config.seed,
        )

        modality_configs, n_classes = resolve_modality_setup(runtime_config.experiment)
        model = SensoryPerceiver(
            modality_configs=modality_configs,
            dim=runtime_config.d_model,
            num_heads=runtime_config.num_heads,
            mlp_dim=runtime_config.mlp_dim,
            depth=runtime_config.depth,
            dropout=runtime_config.dropout,
            replay_buffer_size=runtime_config.replay_buffer_size,
            contrastive_temp=runtime_config.contrastive_temp,
        )

        classifier = nn.Linear(runtime_config.d_model, n_classes).to(runtime_config.device)
        criterion = nn.CrossEntropyLoss()
        replay_criterion = ContrastiveLoss(temperature=runtime_config.contrastive_temp)
        optimizer = optim.AdamW(
            list(model.parameters()) + list(classifier.parameters()),
            lr=runtime_config.learning_rate,
            weight_decay=0.01,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=runtime_config.epochs)

        config = {
            "config_schema_version": TRAIN_CONFIG_SCHEMA_VERSION,
            "experiment": runtime_config.experiment,
            "dataset": runtime_config.dataset,
            "d_model": runtime_config.d_model,
            "num_heads": runtime_config.num_heads,
            "depth": runtime_config.depth,
            "mlp_dim": runtime_config.mlp_dim,
            "dropout": runtime_config.dropout,
            "replay_buffer_size": runtime_config.replay_buffer_size,
            "contrastive_temp": runtime_config.contrastive_temp,
            "replay_mode": runtime_config.replay_mode,
            "batch_size": runtime_config.batch_size,
            "epochs": runtime_config.epochs,
            "learning_rate": runtime_config.learning_rate,
            "warmup_steps": runtime_config.warmup_steps,
            "num_workers": runtime_config.num_workers,
            "seed": runtime_config.seed,
            "device": str(runtime_config.device),
            "output_dir": str(runtime_config.output_dir),
            "modality_configs": modality_configs,
            "n_classes": n_classes,
        }

        output_dir = runtime_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        config_file = output_dir / "config.yaml"
        with open(config_file, "w") as file:
            yaml.dump(config, file)

        trainer = Trainer(
            model=model,
            classifier=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            replay_mode=runtime_config.replay_mode,
            replay_criterion=replay_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=runtime_config.device,
            output_dir=output_dir,
            config=config,
        )
        trainer.fit(runtime_config.epochs)

        duration_sec = time.perf_counter() - start_time
        metrics.increment("training.runs.success")
        metrics.observe("training.duration_sec", duration_sec)
        metrics.write_json(metrics_file)
        emit_event(
            logger,
            logging.INFO,
            "training.completed",
            trace_id,
            duration_sec=duration_sec,
            metrics_file=str(metrics_file),
        )

        return TrainingRunResult(
            output_dir=output_dir,
            config_file=config_file,
            best_checkpoint=output_dir / "best_model.pt",
            latest_checkpoint=output_dir / "latest_model.pt",
            history_file=output_dir / "train_log.csv",
            metrics_file=metrics_file,
            trace_id=trace_id,
        )
    except Exception as exc:
        duration_sec = time.perf_counter() - start_time
        metrics.increment("training.runs.failure")
        metrics.observe("training.duration_sec", duration_sec)
        metrics.write_json(metrics_file)
        app_error = classify_exception(
            exc,
            default_code=ErrorCode.INTERNAL_ERROR,
            context={
                "service": "training",
                "trace_id": trace_id,
                "experiment": runtime_config.experiment,
                "dataset": runtime_config.dataset,
            },
        )
        emit_event(
            logger,
            logging.ERROR,
            "training.failed",
            trace_id,
            code=app_error.code.value,
            retryable=app_error.retryable,
            message=str(app_error),
        )
        raise app_error
