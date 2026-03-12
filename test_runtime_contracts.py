"""Unit tests for runtime contract validation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.data.loaders import SensorStreamingDataset
from src.data.real_audio import (
    PreparedRealAudioDataset,
    get_real_audio_dataset_ids,
    real_audio_dataset_is_prepared,
)
from src.data.real_sensor import (
    PreparedRealSensorDataset,
    get_real_sensor_dataset_ids,
    real_sensor_dataset_is_prepared,
)
from src.data.real_audio_prepare import prepare_speech_commands_dataset
from src.data.real_sensor_prepare import prepare_wisdm_dataset
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
from src.runtime.live_schema import FrameEnvelope, ModalityType, StreamStatus
from src.runtime.synchronizer import BaseSynchronizer
from src.services.inference import (
    LiveTensorAdapter,
    build_live_model_inputs,
    resolve_inference_plan,
)


class RuntimeContractsTest(unittest.TestCase):
    def test_live_synchronizer_outputs_valid_mask_without_forward_fill(self):
        synchronizer = BaseSynchronizer(window_size_ms=1000, sampling_rate_hz=4)
        envelopes = [
            FrameEnvelope(
                trace_id="trace-1",
                timestamp_ms=0,
                modality_id="smartwatch",
                payload=[1.0] * 7,
            ),
            FrameEnvelope(
                trace_id="trace-1",
                timestamp_ms=500,
                modality_id="smartwatch",
                payload=[2.0] * 7,
            ),
        ]

        windows = synchronizer.build_frozen_window(envelopes, window_start_ms=0, window_end_ms=1000)
        smartwatch = windows[ModalityType.SMARTWATCH]

        self.assertEqual(smartwatch.status, StreamStatus.NOMINAL)
        self.assertEqual(smartwatch.valid_mask.tolist(), [[True, False, True, False]])
        self.assertTrue(torch.allclose(smartwatch.tensor[0, 1, :], torch.zeros(7)))
        self.assertTrue(torch.allclose(smartwatch.tensor[0, 3, :], torch.zeros(7)))

    def test_build_live_model_inputs_preserves_mask_and_status(self):
        window = self._make_sensor_window(
            modality=ModalityType.SMARTWATCH,
            feature_dim=7,
            status=StreamStatus.DEGRADED,
            valid_mask=torch.tensor([[True, False, True]], dtype=torch.bool),
        )

        payload = build_live_model_inputs({ModalityType.SMARTWATCH: window})

        self.assertIn("smartwatch", payload)
        self.assertTrue(torch.equal(payload["smartwatch"]["valid_mask"], window.valid_mask))
        self.assertEqual(payload["smartwatch"]["status"], "degraded")

    def test_resolve_inference_plan_requires_live_capable_model(self):
        class LegacyModel(torch.nn.Module):
            def forward(self, _inputs):
                return torch.randn(1, 3)

        windows = {
            ModalityType.SMARTWATCH: self._make_sensor_window(
                modality=ModalityType.SMARTWATCH,
                feature_dim=7,
            )
        }

        with self.assertRaises(AppError) as context:
            resolve_inference_plan(windows, LegacyModel(), {ModalityType.SMARTWATCH})
        self.assertEqual(context.exception.code, ErrorCode.INVALID_ARGUMENT)

    def test_resolve_inference_plan_passes_structured_live_payload(self):
        class LiveCapableModel(torch.nn.Module):
            supports_live_valid_mask = True

            def forward(self, inputs):
                smartwatch = inputs["smartwatch"]
                assert set(smartwatch.keys()) == {"tensor", "valid_mask", "status"}
                assert smartwatch["status"] == "nominal"
                assert smartwatch["tensor"].shape == (1, 3, 7)
                assert smartwatch["valid_mask"].shape == (1, 3)
                return torch.tensor([[0.1, 0.9]], dtype=torch.float32)

        windows = {
            ModalityType.SMARTWATCH: self._make_sensor_window(
                modality=ModalityType.SMARTWATCH,
                feature_dim=7,
                valid_mask=torch.tensor([[True, True, False]], dtype=torch.bool),
            )
        }

        plan = resolve_inference_plan(windows, LiveCapableModel(), {ModalityType.SMARTWATCH})
        self.assertIsNone(plan.error_code)
        self.assertTrue(plan.is_safe)
        self.assertGreater(plan.confidence_score, 0.65)

    def test_live_tensor_adapter_applies_valid_mask_before_forward(self):
        captured = {}

        class LegacyTensorModel(torch.nn.Module):
            def forward(self, inputs):
                captured.update(inputs)
                return torch.tensor([[0.2, 0.8]], dtype=torch.float32)

        adapter = LiveTensorAdapter(LegacyTensorModel())
        windows = {
            ModalityType.SMARTWATCH: self._make_sensor_window(
                modality=ModalityType.SMARTWATCH,
                feature_dim=7,
                valid_mask=torch.tensor([[True, False, True]], dtype=torch.bool),
            )
        }

        plan = resolve_inference_plan(windows, adapter, {ModalityType.SMARTWATCH})

        self.assertIn("smartwatch", captured)
        self.assertTrue(torch.allclose(captured["smartwatch"][0, 1, :], torch.zeros(7)))
        self.assertIsNone(plan.error_code)

    @staticmethod
    def _make_sensor_window(
        *,
        modality: ModalityType,
        feature_dim: int,
        status: StreamStatus = StreamStatus.NOMINAL,
        valid_mask: torch.Tensor | None = None,
    ):
        from src.runtime.live_schema import SensorWindow

        tensor = torch.ones((1, 3, feature_dim), dtype=torch.float32)
        if valid_mask is None:
            valid_mask = torch.ones((1, 3), dtype=torch.bool)
        return SensorWindow(
            modality=modality,
            tensor=tensor,
            valid_mask=valid_mask,
            status=status,
        )

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

    def test_validate_experiment_dataset_accepts_real_audio_dataset(self):
        validate_experiment_dataset("multimodal", "speech_commands")
        validate_experiment_dataset("multimodal", "esc50")
        validate_experiment_dataset("multimodal", "urbansound8k")

    @patch("src.runtime.contracts.get_real_audio_n_classes", return_value=35)
    def test_resolve_training_setup_real_audio_dataset_uses_manifest_class_count(self, _mock_n_classes):
        modality_configs, n_classes, n_tasks = resolve_training_setup(
            "multimodal",
            "speech_commands",
            continual_task_count=1,
        )
        self.assertEqual(modality_configs, {"audio": {"input_dim": 128, "use_conv": False}})
        self.assertEqual(n_classes, 35)
        self.assertEqual(n_tasks, 1)

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

    def test_resolve_training_setup_sensor_streaming_merged_mode(self):
        modality_configs, n_classes, n_tasks = resolve_training_setup(
            "sensor_streaming",
            "sensor_streaming",
            continual_task_count=1,
            modality_routing_mode="merged",
        )
        self.assertEqual(
            modality_configs, {"sensor_stream": {"input_dim": 20, "use_conv": False}}
        )
        self.assertEqual(n_classes, 5)
        self.assertEqual(n_tasks, 1)

    def test_validate_experiment_dataset_accepts_real_sensor_dataset(self):
        validate_experiment_dataset("sensor_streaming", "wisdm")
        validate_experiment_dataset("sensor_streaming", "pamap2")
        validate_experiment_dataset("sensor_streaming", "opportunity")
        validate_experiment_dataset("sensor_streaming", "uci_har")
        validate_experiment_dataset("sensor_streaming", "hhar")
        validate_experiment_dataset("sensor_streaming", "mhealth")
        validate_experiment_dataset("sensor_streaming", "selfback")
        validate_experiment_dataset("sensor_streaming", "geomagnetic_wlan")

    def test_real_sensor_dataset_registry_includes_integrated_datasets(self):
        self.assertEqual(
            set(get_real_sensor_dataset_ids()),
            {
                "wisdm",
                "pamap2",
                "opportunity",
                "uci_har",
                "hhar",
                "mhealth",
                "selfback",
                "geomagnetic_wlan",
            },
        )

    def test_real_audio_dataset_registry_includes_integrated_audio_datasets(self):
        self.assertEqual(
            set(get_real_audio_dataset_ids()),
            {"speech_commands", "esc50", "urbansound8k", "audioset"},
        )

    @patch("src.runtime.contracts.get_real_sensor_n_classes", return_value=6)
    def test_resolve_training_setup_real_sensor_dataset_uses_manifest_class_count(self, _mock_n_classes):
        modality_configs, n_classes, n_tasks = resolve_training_setup(
            "sensor_streaming",
            "wisdm",
            continual_task_count=1,
            modality_routing_mode="canonical",
        )
        self.assertIn("smartwatch", modality_configs)
        self.assertEqual(n_classes, 6)
        self.assertEqual(n_tasks, 1)

    def test_train_runtime_config_rejects_unknown_sensor_variant(self):
        args = SimpleNamespace(
            experiment="sensor_streaming",
            dataset="sensor_streaming",
            d_model=64,
            num_heads=4,
            depth=2,
            mlp_dim=128,
            dropout=0.1,
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
            modality_routing_mode="canonical",
            sensor_variant="bad_variant",
        )
        with self.assertRaises(AppError) as context:
            TrainRuntimeConfig.from_namespace(args)
        self.assertEqual(context.exception.code, ErrorCode.INVALID_ARGUMENT)

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

    def test_sensor_streaming_dataset_is_class_conditioned(self):
        dataset = SensorStreamingDataset(
            n_sequences=120,
            sequence_length=64,
            n_classes=5,
            seed=7,
        )

        class_signatures = []
        for label in range(5):
            indices = (dataset.labels == label).nonzero(as_tuple=False).flatten()
            self.assertGreater(len(indices), 0)
            smartwatch_mean = dataset.data["smartwatch"][indices, :, 0].mean().item()
            room_mean = dataset.data["room_env"][indices, :, 2].mean().item()
            robot_mean = dataset.data["robot_arm"][indices, :, 0].mean().item()
            class_signatures.append((smartwatch_mean, room_mean, robot_mean))

        rounded = {tuple(round(value, 2) for value in means) for means in class_signatures}
        self.assertEqual(len(rounded), 5)
        smartwatch_codes = {dataset._binding_label(label, "smartwatch") for label in range(5)}
        room_codes = {dataset._binding_label(label, "room_env") for label in range(5)}
        robot_codes = {dataset._binding_label(label, "robot_arm") for label in range(5)}
        self.assertLess(len(smartwatch_codes), 5)
        self.assertLess(len(room_codes), 5)
        self.assertLess(len(robot_codes), 5)

    def test_sensor_streaming_dataset_merged_mode_preserves_label(self):
        dataset = SensorStreamingDataset(
            n_sequences=16,
            sequence_length=32,
            n_classes=5,
            seed=11,
            merge_modalities=True,
        )
        sample = dataset[0]
        self.assertEqual(sample["sensor_stream"].shape, (32, 20))
        self.assertIn("label", sample)

    def test_sensor_streaming_dataset_drop_variant_zeroes_target_modality(self):
        dataset = SensorStreamingDataset(
            n_sequences=8,
            sequence_length=16,
            n_classes=5,
            seed=5,
            sensor_variant="drop_room_env",
        )
        self.assertTrue(torch.allclose(dataset.data["room_env"], torch.zeros_like(dataset.data["room_env"])))
        self.assertFalse(torch.allclose(dataset.data["smartwatch"], torch.zeros_like(dataset.data["smartwatch"])))

    def test_sensor_streaming_dataset_noise_variant_changes_target_modality(self):
        baseline = SensorStreamingDataset(
            n_sequences=8,
            sequence_length=16,
            n_classes=5,
            seed=9,
            sensor_variant="baseline",
        )
        noisy = SensorStreamingDataset(
            n_sequences=8,
            sequence_length=16,
            n_classes=5,
            seed=9,
            sensor_variant="noise_robot_arm",
        )
        self.assertTrue(torch.allclose(baseline.data["smartwatch"], noisy.data["smartwatch"]))
        self.assertFalse(torch.allclose(baseline.data["robot_arm"], noisy.data["robot_arm"]))

    def test_sensor_streaming_validation_split_uses_shifted_generator(self):
        train_ds = SensorStreamingDataset(
            n_sequences=32,
            sequence_length=48,
            n_classes=5,
            seed=13,
            split="train",
        )
        val_ds = SensorStreamingDataset(
            n_sequences=32,
            sequence_length=48,
            n_classes=5,
            seed=13,
            split="val",
        )
        self.assertFalse(torch.allclose(train_ds.data["smartwatch"], val_ds.data["smartwatch"]))
        self.assertFalse(torch.allclose(train_ds.data["room_env"], val_ds.data["room_env"]))
        self.assertFalse(torch.allclose(train_ds.data["robot_arm"], val_ds.data["robot_arm"]))

    def test_prepared_real_sensor_dataset_loads_canonical_split(self):
        with TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "wisdm"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "manifest.json").write_text(
                """{
  "schema_version": 1,
  "dataset": "wisdm",
  "source_name": "WISDM",
  "description": "test",
  "sequence_length": 16,
  "n_classes": 6,
  "class_names": ["a", "b", "c", "d", "e", "f"],
  "splits": {"train": "train.pt", "val": "val.pt"}
}
""",
                encoding="utf-8",
            )
            payload = {
                "labels": torch.tensor([0, 1, 2], dtype=torch.long),
                "smartwatch": torch.randn(3, 16, 7),
                "room_env": torch.randn(3, 16, 4),
                "robot_arm": torch.randn(3, 16, 9),
            }
            torch.save(payload, dataset_dir / "train.pt")
            torch.save(payload, dataset_dir / "val.pt")

            dataset = PreparedRealSensorDataset("wisdm", "train", root=tmpdir)
            merged_dataset = PreparedRealSensorDataset("wisdm", "train", root=tmpdir, merge_modalities=True)

            sample = dataset[0]
            merged = merged_dataset[0]
            self.assertEqual(sample["smartwatch"].shape[-1], 7)
            self.assertEqual(sample["room_env"].shape[-1], 4)
            self.assertEqual(sample["robot_arm"].shape[-1], 9)
            self.assertEqual(merged["sensor_stream"].shape[-1], 20)
            self.assertTrue(real_sensor_dataset_is_prepared("wisdm", tmpdir))

    def test_prepare_wisdm_dataset_builds_prepared_splits(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            for device in ("phone", "watch"):
                for sensor in ("accel", "gyro"):
                    (raw_root / device / sensor).mkdir(parents=True, exist_ok=True)

            def write_subject_file(device: str, sensor: str, subject_id: int, offset: float) -> None:
                path = raw_root / device / sensor / f"data_{device}_{sensor}_{subject_id}.txt"
                lines = []
                timestamp = 0
                for activity_code, activity_offset in (("A", 0.0), ("B", 1.0)):
                    for step in range(12):
                        base = offset + activity_offset + step * 0.1
                        lines.append(
                            f"{subject_id},{activity_code},{timestamp},{base:.3f},{base + 0.5:.3f},{base + 1.0:.3f};"
                        )
                        timestamp += 1
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            write_subject_file("phone", "accel", 1, 0.1)
            write_subject_file("phone", "gyro", 1, 0.2)
            write_subject_file("watch", "accel", 1, 0.3)
            write_subject_file("watch", "gyro", 1, 0.4)
            write_subject_file("phone", "accel", 2, 1.1)
            write_subject_file("phone", "gyro", 2, 1.2)
            write_subject_file("watch", "accel", 2, 1.3)
            write_subject_file("watch", "gyro", 2, 1.4)
            (root / "activity_key.txt").write_text("A=Walking\nB=Jogging\n", encoding="utf-8")

            result = prepare_wisdm_dataset(root, root / "prepared", sequence_length=8)
            self.assertEqual(result["n_classes"], 2)

            prepared_root = root / "prepared" / "wisdm"
            self.assertTrue((prepared_root / "manifest.json").exists())
            self.assertTrue((prepared_root / "train.pt").exists())
            self.assertTrue((prepared_root / "val.pt").exists())

            dataset = PreparedRealSensorDataset("wisdm", "train", root=root / "prepared")
            sample = dataset[0]
            self.assertEqual(tuple(sample["smartwatch"].shape), (8, 7))
            self.assertEqual(tuple(sample["room_env"].shape), (8, 4))
            self.assertEqual(tuple(sample["robot_arm"].shape), (8, 9))

    def test_prepared_real_audio_dataset_loads_canonical_split(self):
        with TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "speech_commands"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "manifest.json").write_text(
                """{
  "schema_version": 1,
  "dataset": "speech_commands",
  "source_name": "Speech Commands",
  "description": "test",
  "sequence_length": 96,
  "feature_dim": 128,
  "feature_type": "log_spectrogram",
  "n_classes": 3,
  "class_names": ["yes", "no", "up"],
  "splits": {"train": "train.pt", "val": "val.pt"}
}
""",
                encoding="utf-8",
            )
            payload = {
                "labels": torch.tensor([0, 1, 2], dtype=torch.long),
                "audio": torch.randn(3, 96, 128),
            }
            torch.save(payload, dataset_dir / "train.pt")
            torch.save(payload, dataset_dir / "val.pt")

            dataset = PreparedRealAudioDataset("speech_commands", "train", root=tmpdir)
            sample = dataset[0]
            self.assertEqual(tuple(sample["audio"].shape), (96, 128))
            self.assertTrue(real_audio_dataset_is_prepared("speech_commands", tmpdir))

    def test_prepare_speech_commands_dataset_builds_prepared_splits(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "speech_commands_hf"
            raw_root.mkdir(parents=True, exist_ok=True)
            (raw_root / "dataset_dict.json").write_text("{}", encoding="utf-8")

            class DummyLabelFeature:
                num_classes = 2

                @staticmethod
                def int2str(index: int) -> str:
                    return ["yes", "no"][index]

            class DummySplit:
                def __init__(self, rows):
                    self._rows = rows
                    self.features = {"label": DummyLabelFeature()}

                def __iter__(self):
                    return iter(self._rows)

            dataset = {
                "train": DummySplit(
                    [
                        {"audio": {"array": [0.1, 0.2, 0.3, 0.4]}, "label": 0},
                        {"audio": {"array": [0.2, 0.1, 0.0, -0.1]}, "label": 1},
                    ]
                ),
                "validation": DummySplit(
                    [
                        {"audio": {"array": [0.0, 0.1, 0.0, -0.1]}, "label": 0},
                    ]
                ),
            }

            with patch("src.data.real_audio_prepare._import_hf_load_from_disk", return_value=lambda path: dataset):
                result = prepare_speech_commands_dataset(root, root / "prepared", sequence_length=128)

            self.assertEqual(result["n_classes"], 2)
            prepared_root = root / "prepared" / "speech_commands"
            self.assertTrue((prepared_root / "manifest.json").exists())
            self.assertTrue((prepared_root / "train.pt").exists())
            self.assertTrue((prepared_root / "val.pt").exists())

            dataset = PreparedRealAudioDataset("speech_commands", "train", root=root / "prepared")
            sample = dataset[0]
            self.assertEqual(tuple(sample["audio"].shape), (96, 128))


if __name__ == "__main__":
    unittest.main()
