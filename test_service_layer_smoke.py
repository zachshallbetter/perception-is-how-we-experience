"""Smoke tests for service-layer entrypoints."""

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.models import SensoryPerceiver
from src.runtime.live_schema import ModalityType, SensorWindow, StreamStatus
from src.runtime import (
    AppError,
    ErrorCode,
    EvalRuntimeConfig,
    OrchestratorRuntimeConfig,
    TRAIN_CONFIG_SCHEMA_VERSION,
    TrainRuntimeConfig,
)
from src.services.evaluation_service import run_evaluation
from src.services.inference import LiveTensorAdapter, resolve_inference_plan
from src.services.orchestration_service import ExperimentOrchestrator, run_orchestration
from src.services.training_service import Trainer, run_training


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
    def test_trainer_writes_best_checkpoint_even_when_first_val_acc_is_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = nn.Linear(4, 2)
            optimizer = optim.SGD(model.parameters(), lr=0.1)
            trainer = Trainer(
                model=model,
                train_loader=[],
                val_loader=[],
                criterion=nn.CrossEntropyLoss(),
                replay_criterion=None,
                optimizer=optimizer,
                scheduler=None,
                device=torch.device("cpu"),
                output_dir=Path(tmpdir),
                config={"replay_buffer_size": 0},
                classifier=None,
            )
            trainer.train_epoch = lambda _epoch: (1.0, 0.0)  # type: ignore[method-assign]
            trainer.validate = lambda: (1.0, 0.0)  # type: ignore[method-assign]

            trainer.fit(epochs=1)

            self.assertTrue((Path(tmpdir) / "best_model.pt").exists())
            self.assertTrue((Path(tmpdir) / "latest_model.pt").exists())

    @patch("src.services.training_service.create_dataloaders", new=_mock_create_dataloaders)
    @patch("src.services.training_service.Trainer.fit", autospec=True)
    def test_run_training_fails_if_required_artifacts_are_missing(self, mock_fit):
        def fake_fit(self, epochs: int):
            self.save_checkpoint("latest_model.pt")
            self.save_history()

        mock_fit.side_effect = fake_fit
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
            with self.assertRaises(AppError) as context:
                run_training(runtime_config)
            self.assertEqual(context.exception.code, ErrorCode.IO_ERROR)
            self.assertIn("best_model.pt", context.exception.context["missing_artifacts"])

    def test_live_inference_service_accepts_native_mask_aware_model(self):
        model = SensoryPerceiver(
            modality_configs={"smartwatch": {"input_dim": 7, "use_conv": False}},
            dim=16,
            num_heads=2,
            mlp_dim=32,
            depth=1,
            dropout=0.1,
            replay_buffer_size=8,
            contrastive_temp=0.07,
            enable_parameter_isolation=False,
        )
        classifier = nn.Linear(16, 3)

        class LiveInferenceModel(nn.Module):
            supports_live_valid_mask = True

            def __init__(self, backbone: nn.Module, head: nn.Module):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, inputs):
                outputs, _ = self.backbone(inputs, update_buffer=False, return_embeddings=False)
                pooled = outputs.mean(dim=1)
                return self.head(pooled)

        windows = {
            ModalityType.SMARTWATCH: SensorWindow(
                modality=ModalityType.SMARTWATCH,
                tensor=torch.randn(1, 4, 7),
                valid_mask=torch.tensor([[True, True, False, True]], dtype=torch.bool),
                status=StreamStatus.DEGRADED,
            )
        }

        plan = resolve_inference_plan(
            windows,
            LiveInferenceModel(model, classifier),
            {ModalityType.SMARTWATCH},
        )
        self.assertIsNone(plan.error_code)
        self.assertEqual(tuple(plan.prediction_logits.shape), (1, 3))

    def test_live_inference_service_accepts_legacy_model_via_adapter(self):
        class LegacyTensorModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(4 * 7, 2)

            def forward(self, inputs):
                tensor = inputs["smartwatch"]
                flat = tensor.reshape(tensor.shape[0], -1)
                return self.head(flat)

        wrapped = LiveTensorAdapter(LegacyTensorModel())
        windows = {
            ModalityType.SMARTWATCH: SensorWindow(
                modality=ModalityType.SMARTWATCH,
                tensor=torch.randn(1, 4, 7),
                valid_mask=torch.tensor([[True, False, True, True]], dtype=torch.bool),
                status=StreamStatus.DEGRADED,
            )
        }

        plan = resolve_inference_plan(
            windows,
            wrapped,
            {ModalityType.SMARTWATCH},
        )
        self.assertIsNone(plan.error_code)
        self.assertEqual(tuple(plan.prediction_logits.shape), (1, 2))

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
