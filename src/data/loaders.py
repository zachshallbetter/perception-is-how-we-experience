"""
Multimodal dataset loaders for Sensory Perceiver experiments.

Supports:
- Multimodal classification (AudioSet, COCO, AV-MNIST, Flickr30K)
- Continual learning (Split-CIFAR, Permuted MNIST)
- IoT stream simulation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, MNIST
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional


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
    """Proxy for AudioSet / Ego4D multimodal benchmarking.

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

        # Pre-generate synthetic data to save time during dataloading
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


class SyntheticIOTStream(Dataset):
    """Synthetic IoT sensor stream for real-world validation.

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
    ):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.n_sequences = n_sequences
        self.sequence_length = sequence_length
        self.n_classes = n_classes

        # Generate synthetic data
        self.data = {
            "smartwatch": torch.randn(
                n_sequences, sequence_length, 7
            ),  # HR + acc(3) + GSR(1) + dummy(2)
            "room_env": torch.randn(
                n_sequences, sequence_length, 4
            ),  # temp, humidity, CO2, light
            "robot_arm": torch.randn(
                n_sequences, sequence_length, 9
            ),  # 6 joints + 3 F/T
        }

        # Normalize to reasonable ranges
        self.data["smartwatch"][:, :, 0] = (
            self.data["smartwatch"][:, :, 0] * 30 + 70
        )  # HR: 70±30
        self.data["room_env"][:, :, 0] = (
            self.data["room_env"][:, :, 0] * 5 + 22
        )  # Temp: 22±5°C
        self.data["room_env"][:, :, 1] = (
            torch.sigmoid(self.data["room_env"][:, :, 1]) * 100
        )  # Humidity: 0-100%

        # Labels: activity classification
        self.labels = torch.randint(0, n_classes, (n_sequences,))

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "smartwatch": self.data["smartwatch"][idx],
            "room_env": self.data["room_env"][idx],
            "robot_arm": self.data["robot_arm"][idx],
            "label": self.labels[idx],
        }


def create_dataloaders(
    experiment: str,
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Factory function to create train/val dataloaders.

    Args:
        experiment: 'multimodal', 'continual', 'mime_routing', 'iot'
        dataset_name: 'coco', 'audioset', 'av_mnist', 'split_cifar', etc.
        batch_size: Batch size
        num_workers: Data loading workers
        seed: Random seed

    Returns:
        (train_loader, val_loader)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if experiment == "continual":
        if dataset_name == "split_cifar":
            # For continual learning, return single task loaders
            # In practice, iterate over tasks
            train_ds = SplitCIFAR100(n_tasks=20, task_id=0, split="train")
            val_ds = SplitCIFAR100(n_tasks=20, task_id=0, split="test")

        elif dataset_name == "permuted_mnist":
            train_ds = PermutedMNIST(n_tasks=20, task_id=0, split="train")
            val_ds = PermutedMNIST(n_tasks=20, task_id=0, split="test")
        else:
            raise ValueError(f"Unknown continual dataset: {dataset_name}")

    elif experiment == "iot":
        train_ds = SyntheticIOTStream(
            n_sequences=800,
            sequence_length=500,
            n_classes=5,
            seed=seed,
        )
        val_ds = SyntheticIOTStream(
            n_sequences=200,
            sequence_length=500,
            n_classes=5,
            seed=seed + 1,
        )

    elif experiment in {"multimodal", "mime_routing"}:
        if dataset_name == "audiovisual":
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
