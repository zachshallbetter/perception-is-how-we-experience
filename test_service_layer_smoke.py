"""Smoke tests for service-layer entrypoints."""

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models import SensoryPerceiver
from src.runtime import (
    AppError,
    ErrorCode,
    EvalRuntimeConfig,
    OrchestratorRuntimeConfig,
    TRAIN_CONFIG_SCHEMA_VERSION,
    TrainRuntimeConfig,
)
from src.services.evaluation_service import run_evaluation
from src.services.orchestration_service import ExperimentOrchestrator, run_orchestration
from src.services.training_service import run_training


class TinyVisionDataset(Dataset):
    """Small dict-based dataset for service smoke tests."""

    def __init__(self, n_samples: int, n_classes: int = 5):
        self.images = torch.randn(n_samples, 3, 16, 16)
        self.labels = torch.arange(n_samples) % n_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"vision": self.images[idx], "label": self.labels[idx]}


def _mock_create_dataloaders(*_args, **_kwargs):
    train_ds = TinyVisionDataset(n_samples=6, n_classes=5)
    val_ds = TinyVisionDataset(n_samples=4, n_classes=5)
    return (
        DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0),
        DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0),
    )


@contextmanager
def chdir(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class ServiceLayerSmokeTest(unittest.TestCase):
    @patch("src.services.training_service.create_dataloaders", new=_mock_create_dataloaders)
    def test_run_training_writes_core_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = TrainRuntimeConfig(
                experiment="continual",
                dataset="split_cifar",
                d_model=16,
                num_heads=2,
                depth=1,
                mlp_dim=32,
                dropout=0.1,
                replay_buffer_size=16,
                contrastive_temp=0.07,
                distillation_temp=2.0,
                replay_mode="er",
                batch_size=2,
                epochs=1,
                learning_rate=1e-3,
                warmup_steps=0,
                num_workers=0,
                device=torch.device("cpu"),
                seed=1,
                output_dir=Path(tmpdir) / "train_output",
            )
            result = run_training(runtime_config)
            self.assertTrue(result.config_file.exists())
            self.assertTrue(result.latest_checkpoint.exists())
            self.assertTrue(result.history_file.exists())
            self.assertTrue(result.metrics_file.exists())
            self.assertTrue(result.trace_id.startswith("training-"))

    @patch("src.services.training_service.create_dataloaders", new=_mock_create_dataloaders)
    def test_run_training_supports_distill_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = TrainRuntimeConfig(
                experiment="continual",
                dataset="split_cifar",
                d_model=16,
                num_heads=2,
                depth=1,
                mlp_dim=32,
                dropout=0.1,
                replay_buffer_size=16,
                contrastive_temp=0.07,
                distillation_temp=2.0,
                replay_mode="distill",
                batch_size=2,
                epochs=1,
                learning_rate=1e-3,
                warmup_steps=0,
                num_workers=0,
                device=torch.device("cpu"),
                seed=1,
                output_dir=Path(tmpdir) / "train_output",
                continual_task_count=2,
            )
            result = run_training(runtime_config)
            self.assertTrue(result.config_file.exists())
            self.assertTrue(result.latest_checkpoint.exists())
            self.assertTrue(result.history_file.exists())

    @patch("src.services.evaluation_service.create_dataloaders", new=_mock_create_dataloaders)
    def test_run_evaluation_writes_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            output_file = Path(tmpdir) / "eval" / "eval_results.json"

            model = SensoryPerceiver(
                modality_configs={"vision": {"input_dim": 3, "use_conv": True}},
                dim=16,
                num_heads=2,
                mlp_dim=32,
                depth=1,
                dropout=0.1,
                replay_buffer_size=16,
                contrastive_temp=0.07,
            )
            classifier = nn.Linear(16, 5)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "config": {
                    "config_schema_version": TRAIN_CONFIG_SCHEMA_VERSION,
                    "experiment": "continual",
                    "dataset": "split_cifar",
                    "modality_configs": {"vision": {"input_dim": 3, "use_conv": True}},
                    "d_model": 16,
                    "num_heads": 2,
                    "mlp_dim": 32,
                    "depth": 1,
                    "dropout": 0.1,
                    "replay_buffer_size": 16,
                    "contrastive_temp": 0.07,
                    "distillation_temp": 2.0,
                    "replay_mode": "er",
                    "batch_size": 2,
                    "epochs": 1,
                    "learning_rate": 1e-3,
                    "warmup_steps": 0,
                    "num_workers": 0,
                    "seed": 1,
                    "device": "cpu",
                    "output_dir": str(Path(tmpdir) / "train_output"),
                    "n_classes": 5,
                },
            }
            torch.save(checkpoint, checkpoint_path)

            runtime_config = EvalRuntimeConfig(
                checkpoint=checkpoint_path,
                experiment="continual",
                dataset="split_cifar",
                batch_size=2,
                num_workers=0,
                device=torch.device("cpu"),
                output_file=output_file,
                seed=1,
            )
            result = run_evaluation(runtime_config)
            self.assertTrue(result.output_file.exists())
            self.assertIn("overall_accuracy", result.results)
            self.assertTrue(result.metrics_file.exists())
            self.assertTrue(result.trace_id.startswith("evaluation-"))

    def test_orchestrator_rejects_unknown_dataset(self):
        orchestrator = ExperimentOrchestrator(n_gpus=1)
        with self.assertRaises(AppError) as context:
            orchestrator.add_experiment("bad", "unknown_dataset", gpu_id=0, epochs=1)
        self.assertEqual(context.exception.code, ErrorCode.DATASET_MISMATCH)

    def test_run_orchestration_monitor_mode_outputs_dashboard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = OrchestratorRuntimeConfig(
                mode="monitor",
                n_gpus=1,
                parallel=False,
                push_to_github=False,
            )
            with chdir(Path(tmpdir)):
                dashboard_path = run_orchestration(runtime_config)
                self.assertIsNotNone(dashboard_path)
                self.assertTrue(Path(dashboard_path).exists())
                self.assertTrue((Path(tmpdir) / "results" / "orchestration_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
