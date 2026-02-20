"""Evaluation service for Sensory Perceiver runtime jobs."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.data.loaders import create_dataloaders
from src.models import SensoryPerceiver
from src.runtime import AppError, ErrorCode, EvalRuntimeConfig


@dataclass(frozen=True)
class EvaluationRunResult:
    """Artifacts produced by an evaluation job."""

    output_file: Path
    results: Dict


class Evaluator:
    """Evaluation manager for Sensory Perceiver."""

    def __init__(
        self,
        model: nn.Module,
        classifier: Optional[nn.Module],
        val_loader,
        device: torch.device,
        n_classes: int,
        output_dir: Path,
    ):
        self.model = model.to(device)
        self.classifier = classifier.to(device) if classifier is not None else None
        self.val_loader = val_loader
        self.device = device
        self.n_classes = n_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run evaluation suite."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_embeddings = []
        all_latencies = []

        for batch in self.val_loader:
            inputs = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            labels = inputs.pop("label")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                t_start = torch.cuda.Event(enable_timing=True)
                t_end = torch.cuda.Event(enable_timing=True)
                t_start.record()
            outputs, embeddings = self.model(inputs, return_embeddings=True)
            if torch.cuda.is_available():
                t_end.record()
                torch.cuda.synchronize()
                all_latencies.append(t_start.elapsed_time(t_end))

            pooled = outputs.mean(dim=1)
            if self.classifier is not None:
                logits = self.classifier(pooled)
            else:
                if pooled.shape[1] < self.n_classes:
                    raise AppError(
                        ErrorCode.CHECKPOINT_LOAD_FAILED,
                        f"Backbone output dim {pooled.shape[1]} < n_classes {self.n_classes}",
                        context={
                            "backbone_dim": pooled.shape[1],
                            "n_classes": self.n_classes,
                        },
                    )
                logits = pooled[:, : self.n_classes]
            _, predicted = logits.max(1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if embeddings is not None:
                pooled_emb = torch.stack([emb.mean(dim=0) for emb in embeddings.values()])
                all_embeddings.append(pooled_emb.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        results = self._compute_metrics(all_preds, all_labels)

        if all_embeddings:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            results["embeddings_shape"] = all_embeddings.shape
            np.save(self.output_dir / "embeddings.npy", all_embeddings)

        if all_latencies:
            results["latency_mean"] = float(np.mean(all_latencies))
            results["latency_std"] = float(np.std(all_latencies))
            results["latency_min"] = float(np.min(all_latencies))
            results["latency_max"] = float(np.max(all_latencies))

        return results

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict:
        """Compute classification metrics."""
        accuracy = np.mean(preds == labels)

        per_class_acc = {}
        for class_id in range(self.n_classes):
            mask = labels == class_id
            if mask.sum() > 0:
                per_class_acc[f"class_{class_id}"] = float(np.mean(preds[mask] == labels[mask]))
            else:
                per_class_acc[f"class_{class_id}"] = None

        confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for pred, label in zip(preds, labels):
            pred_idx = int(pred)
            label_idx = int(label)
            if 0 <= label_idx < self.n_classes and 0 <= pred_idx < self.n_classes:
                confusion_matrix[label_idx, pred_idx] += 1

        return {
            "overall_accuracy": float(accuracy),
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": confusion_matrix.tolist(),
            "n_samples": len(preds),
        }

    def save_results(self, results: Dict):
        """Save evaluation results."""
        output_file = self.output_dir / "eval_results.json"
        with open(output_file, "w") as file:
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            json.dump(json_results, file, indent=2)


def load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, Optional[nn.Module], Dict]:
    """Load model and optional classifier from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise AppError(
            ErrorCode.CHECKPOINT_LOAD_FAILED,
            f"Failed to load checkpoint: {checkpoint_path}",
            context={"checkpoint": checkpoint_path},
        ) from exc

    config = checkpoint["config"]
    model = SensoryPerceiver(
        modality_configs=config["modality_configs"],
        dim=config["d_model"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        depth=config["depth"],
        dropout=config["dropout"],
        replay_buffer_size=config["replay_buffer_size"],
        contrastive_temp=config["contrastive_temp"],
    )

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as exc:
        raise AppError(
            ErrorCode.CHECKPOINT_LOAD_FAILED,
            "Model state dict does not match checkpoint metadata",
            context={"checkpoint": checkpoint_path},
        ) from exc

    classifier = None
    classifier_state_dict = checkpoint.get("classifier_state_dict")
    if classifier_state_dict is not None:
        classifier = nn.Linear(config["d_model"], config["n_classes"])
        try:
            classifier.load_state_dict(classifier_state_dict)
        except Exception as exc:
            raise AppError(
                ErrorCode.CHECKPOINT_LOAD_FAILED,
                "Classifier state dict does not match checkpoint metadata",
                context={"checkpoint": checkpoint_path},
            ) from exc

    return model, classifier, config


def run_evaluation(runtime_config: EvalRuntimeConfig) -> EvaluationRunResult:
    """Execute an evaluation job from validated runtime config."""
    torch.manual_seed(runtime_config.seed)
    np.random.seed(runtime_config.seed)

    model, classifier, checkpoint_config = load_checkpoint(
        str(runtime_config.checkpoint), runtime_config.device
    )
    _, val_loader = create_dataloaders(
        experiment=runtime_config.experiment,
        dataset_name=runtime_config.dataset,
        batch_size=runtime_config.batch_size,
        num_workers=runtime_config.num_workers,
        seed=runtime_config.seed,
    )

    output_dir = runtime_config.output_file.parent
    evaluator = Evaluator(
        model=model,
        classifier=classifier,
        val_loader=val_loader,
        device=runtime_config.device,
        n_classes=checkpoint_config["n_classes"],
        output_dir=output_dir,
    )
    results = evaluator.evaluate()
    evaluator.save_results(results)
    return EvaluationRunResult(
        output_file=output_dir / "eval_results.json",
        results=results,
    )
