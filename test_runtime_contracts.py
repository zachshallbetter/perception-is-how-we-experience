"""Unit tests for runtime contract validation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.runtime import (
    AppError,
    ErrorCode,
    EvalRuntimeConfig,
    SENSOR_STREAMING_CANONICAL_LABEL,
    SENSOR_STREAMING_EXPERIMENT_ID,
    OrchestratorRuntimeConfig,
    TRAIN_CONFIG_SCHEMA_VERSION,
    TrainRuntimeConfig,
    resolve_training_setup,
    validate_sensor_streaming_mime_bindings,
    resolve_device,
    validate_environment,
    validate_experiment_dataset,
    validate_training_config_schema,
)


class RuntimeContractsTest(unittest.TestCase):
    def test_validate_experiment_dataset_rejects_mismatch(self):
        with self.assertRaises(AppError) as context:
            validate_experiment_dataset("continual", "audiovisual")
        self.assertEqual(context.exception.code, ErrorCode.DATASET_MISMATCH)

    def test_resolve_device_rejects_invalid_string(self):
        with self.assertRaises(AppError) as context:
            resolve_device("invalid_device")
        self.assertEqual(context.exception.code, ErrorCode.INVALID_ARGUMENT)

    def test_train_runtime_config_validates_hyperparams(self):
        args = SimpleNamespace(
            experiment="continual",
            dataset="split_cifar",
            d_model=64,
            num_heads=4,
            depth=2,
            mlp_dim=128,
            dropout=1.5,
            replay_buffer_size=10,
            contrastive_temp=0.07,
            distillation_temp=2.0,
            replay_mode="er",
            batch_size=4,
            epochs=1,
            learning_rate=1e-3,
            warmup_steps=0,
            num_workers=0,
            device="cpu",
            seed=1,
            output_dir="/tmp/train",
        )
        with self.assertRaises(AppError) as context:
            TrainRuntimeConfig.from_namespace(args)
        self.assertEqual(context.exception.code, ErrorCode.INVALID_ARGUMENT)

    def test_eval_runtime_config_requires_checkpoint(self):
        args = SimpleNamespace(
            checkpoint="/tmp/does_not_exist_checkpoint.pt",
            experiment="continual",
            dataset="split_cifar",
            batch_size=2,
            num_workers=0,
            device="cpu",
            output_file="/tmp/eval_results.json",
            seed=1,
        )
        with self.assertRaises(AppError) as context:
            EvalRuntimeConfig.from_namespace(args)
        self.assertEqual(context.exception.code, ErrorCode.CHECKPOINT_LOAD_FAILED)

    def test_orchestrator_runtime_config_rejects_invalid_gpu_count(self):
        args = SimpleNamespace(mode="quick", n_gpus=0, parallel=False, push_to_github=False)
        with self.assertRaises(AppError) as context:
            OrchestratorRuntimeConfig.from_namespace(args)
        self.assertEqual(context.exception.code, ErrorCode.INVALID_ARGUMENT)

    def test_train_runtime_config_success(self):
        args = SimpleNamespace(
            experiment="multimodal",
            dataset="audiovisual",
            d_model=64,
            num_heads=4,
            depth=2,
            mlp_dim=128,
            dropout=0.1,
            replay_buffer_size=10,
            contrastive_temp=0.07,
            distillation_temp=2.0,
            replay_mode="contrastive",
            batch_size=4,
            epochs=1,
            learning_rate=1e-3,
            warmup_steps=0,
            num_workers=0,
            device="cpu",
            seed=1,
            output_dir="/tmp/train",
        )
        config = TrainRuntimeConfig.from_namespace(args)
        self.assertEqual(config.experiment, "multimodal")
        self.assertEqual(config.dataset, "audiovisual")
        self.assertEqual(config.device.type, "cpu")

    def test_train_runtime_config_accepts_distill_replay_mode(self):
        args = SimpleNamespace(
            experiment="continual",
            dataset="split_cifar",
            d_model=64,
            num_heads=4,
            depth=2,
            mlp_dim=128,
            dropout=0.1,
            replay_buffer_size=10,
            contrastive_temp=0.07,
            distillation_temp=2.0,
            replay_mode="distill",
            batch_size=4,
            epochs=1,
            learning_rate=1e-3,
            warmup_steps=0,
            num_workers=0,
            device="cpu",
            seed=1,
            output_dir="/tmp/train",
        )
        config = TrainRuntimeConfig.from_namespace(args)
        self.assertEqual(config.replay_mode, "distill")
        self.assertEqual(config.distillation_temp, 2.0)

    def test_training_schema_requires_version(self):
        with self.assertRaises(AppError) as context:
            validate_training_config_schema({"experiment": "continual"})
        self.assertEqual(context.exception.code, ErrorCode.SCHEMA_VALIDATION_FAILED)

    def test_training_schema_rejects_wrong_version(self):
        with self.assertRaises(AppError) as context:
            validate_training_config_schema(
                {
                    "config_schema_version": TRAIN_CONFIG_SCHEMA_VERSION + 1,
                    "experiment": "continual",
                    "dataset": "split_cifar",
                    "d_model": 64,
                    "num_heads": 4,
                    "depth": 2,
                    "mlp_dim": 128,
                    "dropout": 0.1,
                    "replay_buffer_size": 10,
                    "contrastive_temp": 0.07,
                    "replay_mode": "er",
                    "batch_size": 4,
                    "epochs": 1,
                    "learning_rate": 1e-3,
                    "warmup_steps": 0,
                    "num_workers": 0,
                    "seed": 1,
                    "device": "cpu",
                    "output_dir": "/tmp/train",
                    "modality_configs": {"vision": {"input_dim": 3, "use_conv": True}},
                    "n_classes": 5,
                }
        )
        self.assertEqual(context.exception.code, ErrorCode.SCHEMA_VALIDATION_FAILED)

    def test_resolve_training_setup_uses_dataset_default_tasks_for_continual(self):
        modality_configs, n_classes, n_tasks = resolve_training_setup(
            "continual",
            "split_cifar",
            continual_task_count=1,
        )
        self.assertEqual(modality_configs, {"vision": {"input_dim": 3, "use_conv": True}})
        self.assertEqual(n_classes, 5)
        self.assertEqual(n_tasks, 20)

    def test_resolve_training_setup_respects_explicit_continual_task_count(self):
        _, n_classes, n_tasks = resolve_training_setup(
            "continual",
            "split_cifar",
            continual_task_count=5,
        )
        self.assertEqual(n_classes, 20)
        self.assertEqual(n_tasks, 5)

    def test_environment_validation_required_env(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(AppError) as context:
                validate_environment(required_env=["MISSING_VAR"])
        self.assertEqual(context.exception.code, ErrorCode.ENV_VALIDATION_FAILED)

    def test_sensor_streaming_identity_constants(self):
        self.assertEqual(SENSOR_STREAMING_EXPERIMENT_ID, "sensor_streaming")
        self.assertEqual(SENSOR_STREAMING_CANONICAL_LABEL, "sensor_streaming")

    def test_validate_sensor_streaming_mime_bindings_rejects_wrong_type(self):
        with self.assertRaises(AppError) as context:
            validate_sensor_streaming_mime_bindings(
                {
                    "smartwatch": "application/octet-stream",
                }
            )
        self.assertEqual(context.exception.code, ErrorCode.CONFIG_VALIDATION_FAILED)


if __name__ == "__main__":
    unittest.main()
