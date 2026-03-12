"""
Multimodal dataset loaders for Sensory Perceiver experiments.

- Multimodal audio-visual classification
- Continual learning (Split-CIFAR, Permuted MNIST)
- IoT stream simulation
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .real_audio import PreparedRealAudioDataset, is_real_audio_dataset
from .real_sensor import PreparedRealSensorDataset, is_real_sensor_dataset


class MultimodalDataset(Dataset):
    """Base class for multimodal datasets."""

    def __init__(self, modalities: List[str], split: str = "train"):
        self.modalities = modalities
        self.split = split
        self.data = {}

    def __len__(self):
        return len(self.data.get(self.modalities[0], []))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {}
        for mod in self.modalities:
            sample[mod] = self.data[mod][idx]
        return sample


class SplitCIFAR100(MultimodalDataset):
    """Continual learning: CIFAR-100 split into N sequential tasks.

    Each task: 5 classes (100 classes / 20 tasks = 5 classes per task)
    """

    def __init__(self, n_tasks: int = 20, task_id: int = 0, split: str = "train"):
        import torchvision.transforms as transforms
        from torchvision.datasets import CIFAR100

        super().__init__(["vision"], split=split)
        self.n_tasks = n_tasks
        self.task_id = task_id
        self.classes_per_task = 100 // n_tasks

        # Load CIFAR-100
        cifar = CIFAR100(
            root="/tmp/cifar100",
            train=(split == "train"),
            download=True,
            transform=transforms.ToTensor(),
        )

        # Filter to task classes
        start_class = task_id * self.classes_per_task
        end_class = start_class + self.classes_per_task

        self.data["vision"] = []
        self.labels = []
        for img, label in cifar:
            if start_class <= label < end_class:
                # Normalize to [0, 1]
                self.data["vision"].append(img)
                self.labels.append(label - start_class)

        self.data["vision"] = torch.stack(self.data["vision"])
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        sample["label"] = self.labels[idx]
        return sample


class PermutedMNIST(MultimodalDataset):
    """Continual learning: MNIST with random pixel permutations per task."""

    def __init__(self, n_tasks: int = 20, task_id: int = 0, split: str = "train"):
        import torchvision.transforms as transforms
        from torchvision.datasets import MNIST

        super().__init__(["vision"], split=split)
        self.n_tasks = n_tasks
        self.task_id = task_id

        # Load MNIST
        mnist = MNIST(
            root="/tmp/mnist",
            train=(split == "train"),
            download=True,
            transform=transforms.ToTensor(),
        )

        # Create random permutation for this task
        self.perm = torch.randperm(28 * 28)

        # Apply permutation
        self.data["vision"] = []
        self.labels = []
        for img, label in mnist:
            # Flatten, permute, reshape
            flat = img.view(-1)
            perm_flat = flat[self.perm]
            perm_img = perm_flat.view(1, 28, 28)
            self.data["vision"].append(perm_img)
            self.labels.append(label)

        self.data["vision"] = torch.stack(self.data["vision"])
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        sample["label"] = self.labels[idx]
        return sample


class AudioVisualProxyDataset(Dataset):
    """Proxy for multimodal sequence benchmarking.

    Generates paired vision (image frames) and audio (spectrograms)
    to test the routing capabilities of multimodal models.
    """

    def __init__(
        self,
        n_sequences: int = 1000,
        n_classes: int = 10,
        vision_shape: Tuple[int, ...] = (3, 64, 64),
        audio_shape: Tuple[int, ...] = (1, 64, 64),  # mel spectogram
        seed: int = 42,
    ):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.n_sequences = n_sequences
        self.n_classes = n_classes

        # Pre-generate proxy data to save time during dataloading
        self.vision_data = torch.randn(n_sequences, *vision_shape)
        self.audio_data = torch.randn(n_sequences, *audio_shape)

        self.labels = torch.randint(0, n_classes, (n_sequences,))

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "vision": self.vision_data[idx],
            "audio": self.audio_data[idx],
            "label": self.labels[idx],
        }


class SensorStreamingDataset(Dataset):
    """IoT sensor stream dataset for benchmark validation.

    Modalities:
    - smartwatch: heart rate, accelerometer (3-axis), skin conductance
    - room_env: temperature, humidity, CO2, light
    - robot_arm: 6-DOF joint angles + end-effector force/torque
    """

    def __init__(
        self,
        n_sequences: int = 100,
        sequence_length: int = 500,
        n_classes: int = 5,  # activity types
        seed: int = 42,
        merge_modalities: bool = False,
        sensor_variant: str = "baseline",
        split: str = "train",
    ):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.n_sequences = n_sequences
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.merge_modalities = merge_modalities
        self.sensor_variant = sensor_variant
        self.split = split

        self.labels = torch.randint(0, n_classes, (n_sequences,))
        self.data = self._build_class_conditioned_streams()
        self._apply_sensor_variant()

    def __len__(self):
        return self.n_sequences

    def _build_class_conditioned_streams(self) -> Dict[str, torch.Tensor]:
        smartwatch = torch.empty(self.n_sequences, self.sequence_length, 7)
        room_env = torch.empty(self.n_sequences, self.sequence_length, 4)
        robot_arm = torch.empty(self.n_sequences, self.sequence_length, 9)

        t = torch.linspace(0.0, 1.0, self.sequence_length, dtype=torch.float32)
        for index in range(self.n_sequences):
            label = int(self.labels[index].item())
            context = self._sample_context(index)

            smartwatch[index] = self._make_smartwatch_stream(
                t,
                *self._class_dynamics(
                    self._binding_label(label, "smartwatch"), "smartwatch"
                ),
                context,
            )
            room_env[index] = self._make_room_env_stream(
                t,
                *self._class_dynamics(
                    self._binding_label(label, "room_env"), "room_env"
                ),
                context,
            )
            robot_arm[index] = self._make_robot_arm_stream(
                t,
                *self._class_dynamics(
                    self._binding_label(label, "robot_arm"), "robot_arm"
                ),
                context,
            )

        return {
            "smartwatch": smartwatch,
            "room_env": room_env,
            "robot_arm": robot_arm,
        }

    def _class_dynamics(self, label: int, modality: str) -> Tuple[float, float, float]:
        modality_bias = {"smartwatch": 0.0, "room_env": 0.18, "robot_arm": 0.36}[
            modality
        ]
        split_freq_scale = 1.0 if self.split == "train" else 1.08
        split_phase_shift = 0.0 if self.split == "train" else 0.22
        split_swing_scale = 1.0 if self.split == "train" else 0.92
        phase = (0.35 * label + modality_bias + split_phase_shift) % (2.0 * np.pi)
        freq = float((label + 1) * split_freq_scale)
        swing = (1.0 + 0.15 * label) * split_swing_scale
        return freq, phase, swing

    def _binding_label(self, label: int, modality: str) -> int:
        codebook = {
            0: {"smartwatch": 0, "room_env": 0, "robot_arm": 0},
            1: {"smartwatch": 0, "room_env": 1, "robot_arm": 1},
            2: {"smartwatch": 1, "room_env": 0, "robot_arm": 1},
            3: {"smartwatch": 1, "room_env": 1, "robot_arm": 0},
            4: {"smartwatch": 0, "room_env": 0, "robot_arm": 1},
        }
        base_label = codebook[label][modality]
        if self.sensor_variant != "corrupt_bindings":
            return base_label
        offsets = {"smartwatch": 1, "room_env": 2, "robot_arm": 3}
        return int((base_label + offsets[modality]) % 3)

    def _sample_context(self, index: int) -> Dict[str, float]:
        split_noise_scale = 1.0 if self.split == "train" else 1.35
        split_drift_scale = 1.0 if self.split == "train" else 1.2
        return {
            "tempo_bias": 0.85 + 0.25 * torch.rand(1).item(),
            "amplitude_bias": 0.8 + 0.4 * torch.rand(1).item(),
            "drift_bias": (-0.12 + 0.24 * torch.rand(1).item()) * split_drift_scale,
            "noise_scale": split_noise_scale,
            "context_id": float(index % 4),
        }

    def _apply_sensor_variant(self) -> None:
        if self.sensor_variant.startswith("drop_"):
            dropped_modality = self.sensor_variant.replace("drop_", "", 1)
            self.data[dropped_modality].zero_()
            return

        if self.sensor_variant.startswith("noise_"):
            target = self.sensor_variant.replace("noise_", "", 1)
            self.data[target] = self._inject_noise_and_drift(target, self.data[target])

    def _inject_noise_and_drift(
        self, modality: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        drift_scale = {"smartwatch": 0.45, "room_env": 2.5, "robot_arm": 0.3}[modality]
        noise_scale = {"smartwatch": 0.18, "room_env": 0.35, "robot_arm": 0.14}[
            modality
        ]
        drift = torch.linspace(0.0, 1.0, self.sequence_length).view(
            1, self.sequence_length, 1
        )
        direction = torch.randn(1, 1, tensor.shape[-1])
        return (
            tensor
            + noise_scale * torch.randn_like(tensor)
            + drift_scale * drift * direction
        )

    def _make_smartwatch_stream(
        self,
        t: torch.Tensor,
        freq: float,
        phase: float,
        swing: float,
        context: Dict[str, float],
    ) -> torch.Tensor:
        noise = 0.02 * context["noise_scale"] * torch.randn(self.sequence_length, 7)
        waveform = 2.0 * np.pi * freq * context["tempo_bias"] * t + phase
        shared_drift = context["drift_bias"] * t
        stream = torch.stack(
            [
                68.0 + 5.5 * swing + 4.0 * torch.sin(waveform) + 1.4 * shared_drift,
                0.45 * swing * torch.sin(waveform) + 0.08 * context["context_id"],
                0.35 * swing * torch.cos(waveform * 0.8) + 0.05 * shared_drift,
                0.25 * swing * torch.sin(waveform * 1.3 + 0.4),
                0.32 + 0.05 * swing + 0.04 * torch.cos(waveform * 0.5),
                0.16 * label_band(freq) + 0.08 * torch.sin(waveform * 0.25),
                0.12 * label_band(freq + phase) + 0.06 * torch.cos(waveform * 0.35),
            ],
            dim=1,
        )
        return stream + noise

    def _make_room_env_stream(
        self,
        t: torch.Tensor,
        freq: float,
        phase: float,
        swing: float,
        context: Dict[str, float],
    ) -> torch.Tensor:
        noise = 0.015 * context["noise_scale"] * torch.randn(self.sequence_length, 4)
        waveform = 2.0 * np.pi * freq * context["tempo_bias"] * t + phase
        shared_drift = context["drift_bias"] * t
        stream = torch.stack(
            [
                20.0
                + 0.6 * swing
                + 1.2 * torch.sin(waveform * 0.2)
                + 0.8 * shared_drift,
                45.0 + 3.2 * swing + 4.2 * torch.cos(waveform * 0.17),
                430.0 + 35.0 * swing + 25.0 * torch.sin(waveform * 0.12 + 0.3),
                220.0 + 20.0 * swing + 28.0 * torch.relu(torch.sin(waveform * 0.4)),
            ],
            dim=1,
        )
        return stream + noise

    def _make_robot_arm_stream(
        self,
        t: torch.Tensor,
        freq: float,
        phase: float,
        swing: float,
        context: Dict[str, float],
    ) -> torch.Tensor:
        noise = 0.02 * context["noise_scale"] * torch.randn(self.sequence_length, 9)
        waveform = 2.0 * np.pi * freq * context["tempo_bias"] * t + phase
        shared_drift = context["drift_bias"] * t
        joints = []
        for joint in range(6):
            joints.append(
                0.2 * joint
                + 0.28
                * swing
                * torch.sin(waveform * (0.45 + 0.04 * joint) + joint * 0.3)
                + (0.02 * joint) * shared_drift
            )
        forces = [
            0.55 * swing * torch.cos(waveform * 0.7),
            0.45 * swing * torch.sin(waveform * 0.9 + 0.2),
            0.38 * swing * torch.cos(waveform * 1.1 + 0.4),
        ]
        stream = torch.stack(joints + forces, dim=1)
        return stream + noise

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.merge_modalities:
            merged = torch.cat(
                [
                    self.data["smartwatch"][idx],
                    self.data["room_env"][idx],
                    self.data["robot_arm"][idx],
                ],
                dim=-1,
            )
            return {"sensor_stream": merged, "label": self.labels[idx]}
        return {
            "smartwatch": self.data["smartwatch"][idx],
            "room_env": self.data["room_env"][idx],
            "robot_arm": self.data["robot_arm"][idx],
            "label": self.labels[idx],
        }


def label_band(value: float) -> float:
    return float(np.tanh(value / 3.0))


def create_dataloaders(
    experiment: str,
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    task_id: int = 0,
    n_tasks: int = 20,
    modality_routing_mode: str = "canonical",
    sensor_variant: str = "baseline",
) -> Tuple[DataLoader, DataLoader]:
    """Factory function to create train/val dataloaders.

    Args:
        experiment: 'multimodal', 'continual', 'sensor_streaming'
        dataset_name: 'audiovisual', 'split_cifar', 'sensor_streaming', etc.
        batch_size: Batch size
        num_workers: Data loading workers
        seed: Random seed
        task_id: Task index for continual datasets
        n_tasks: Total task count for continual datasets

    Returns:
        (train_loader, val_loader)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if experiment == "continual":
        if dataset_name == "split_cifar":
            # For continual learning, return single task loaders
            # In practice, iterate over tasks
            train_ds = SplitCIFAR100(n_tasks=n_tasks, task_id=task_id, split="train")
            val_ds = SplitCIFAR100(n_tasks=n_tasks, task_id=task_id, split="test")

        elif dataset_name == "permuted_mnist":
            train_ds = PermutedMNIST(n_tasks=n_tasks, task_id=task_id, split="train")
            val_ds = PermutedMNIST(n_tasks=n_tasks, task_id=task_id, split="test")
        else:
            raise ValueError(f"Unknown continual dataset: {dataset_name}")

    elif experiment == "sensor_streaming":
        if is_real_sensor_dataset(dataset_name):
            train_ds = PreparedRealSensorDataset(
                dataset_name,
                "train",
                merge_modalities=(modality_routing_mode == "merged"),
            )
            val_ds = PreparedRealSensorDataset(
                dataset_name,
                "val",
                merge_modalities=(modality_routing_mode == "merged"),
            )
        elif dataset_name in {"sensor_streaming_smoke", "sensor_streaming_small"}:
            if dataset_name == "sensor_streaming_smoke":
                n_sequences, sequence_length = 80, 100
            else:
                n_sequences, sequence_length = 200, 200
        else:
            n_sequences, sequence_length = 800, 500

            train_ds = SensorStreamingDataset(
                n_sequences=n_sequences,
                sequence_length=sequence_length,
                n_classes=5,
                seed=seed,
                merge_modalities=(modality_routing_mode == "merged"),
                sensor_variant=sensor_variant,
                split="train",
            )
            val_ds = SensorStreamingDataset(
                n_sequences=max(20, n_sequences // 4),
                sequence_length=sequence_length,
                n_classes=5,
                seed=seed + 1,
                merge_modalities=(modality_routing_mode == "merged"),
                sensor_variant=sensor_variant,
                split="val",
            )

    elif experiment == "multimodal":
        if is_real_audio_dataset(dataset_name):
            train_ds = PreparedRealAudioDataset(dataset_name, "train")
            val_ds = PreparedRealAudioDataset(dataset_name, "val")
        elif dataset_name == "audiovisual":
            train_ds = AudioVisualProxyDataset(
                n_sequences=1000, n_classes=10, seed=seed
            )
            val_ds = AudioVisualProxyDataset(
                n_sequences=200, n_classes=10, seed=seed + 1
            )
        else:
            raise ValueError(f"Unknown multimodal dataset: {dataset_name}")

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    use_pin_memory = not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader
