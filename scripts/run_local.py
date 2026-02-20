#!/usr/bin/env python3
"""
Sensory Perceiver — Local Continual Learning Experiment (M1-ready)

Runs Split-CIFAR-100: N sequential tasks, 5 classes/task.
Measures CIL accuracy, Backward Transfer (BWT), Forward Transfer (FWT).

Purpose:
  1. Full sequential task loop for Split-CIFAR-100 CIL metrics
  2. R[t][j] accuracy matrix → CIL / BWT / FWT
  3. FWT evaluation on next unseen task after each training phase
  4. Auto-detects MPS (Mac M1) > CUDA > CPU
  5. MPS-safe dataloader defaults (pin_memory=False, num_workers=0)
  6. Saves accuracy matrix heatmap + metrics bar chart

Usage:
  python scripts/run_local.py --quick        # 3 tasks, 1 epoch, d_model=128  (~5 min CPU)
  python scripts/run_local.py                # baseline: 10 tasks, 5 epochs, replay_weight=0.1
  python scripts/run_local.py --tuned        # tuned:    10 tasks, 5 epochs, replay_weight=0.5, buffer=5000

Baseline vs Tuned:
  --baseline  replay_weight=0.1, replay_buffer=1000, lr=1e-4  → results/cl_experiment/
  --tuned     replay_weight=0.5, replay_buffer=5000, lr=3e-4  → results/cl_experiment_tuned/

Results: results/cl_experiment/ (baseline) or results/cl_experiment_tuned/ (tuned)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─── Device detection ────────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Device: MPS (Mac M1/M2)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda:0")
    else:
        print("Device: CPU")
        return torch.device("cpu")


# ─── Model ───────────────────────────────────────────────────────────────────


class PerceiverTower(nn.Module):
    """Modality-specific feature extraction.
    Conv2d → flatten spatial → self-attention → MLP → LayerNorm → (seq, B, dim)
    """

    def __init__(
        self,
        in_channels: int,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        use_conv: bool = True,
    ):
        super().__init__()
        self.use_conv = use_conv

        if use_conv:
            self.proj = nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1)
        else:
            self.proj = nn.Linear(in_channels, dim)

        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            x = self.proj(x)  # (B, dim, H', W')
            x = x.flatten(2).permute(2, 0, 1)  # (H'*W', B, dim)
        else:
            x = self.proj(x).permute(1, 0, 2)  # (seq, B, dim)

        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out

        residual = x
        x = residual + self.mlp(self.norm2(x))
        return x  # (seq, B, dim)


class SensoryPerceiver(nn.Module):
    """Multimodal backbone: per-modality towers + cross-modal attention.
    Returns (B, total_seq, dim). Attach a classifier head on top.
    """

    def __init__(
        self,
        modality_configs: dict,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        replay_buffer_size: int = 1000,
    ):
        super().__init__()
        self.towers = nn.ModuleDict(
            {
                name: PerceiverTower(
                    cfg["input_dim"],
                    dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                    cfg.get("use_conv", False),
                )
                for name, cfg in modality_configs.items()
            }
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=False
        )
        self.output_proj = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.norm2 = nn.LayerNorm(dim)

        # Replay buffer (stores RAW inputs and labels)
        self.replay_buffer = {}  # mod_name -> tensor (buffer_size, ...)
        self.register_buffer(
            "replay_labels", torch.zeros(replay_buffer_size, dtype=torch.long)
        )
        self.register_buffer("replay_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("replay_count", torch.zeros(1, dtype=torch.long))
        self.replay_buffer_size = replay_buffer_size
        self.modality_configs = modality_configs

    def forward(
        self, inputs: dict, labels: torch.Tensor = None, update_buffer: bool = True
    ) -> torch.Tensor:
        # ── Initialize buffers on first call if not exists ──
        if update_buffer and labels is not None:
            for name, x in inputs.items():
                if name not in self.replay_buffer:
                    shape = [self.replay_buffer_size] + list(x.shape[1:])
                    self.register_buffer(
                        f"buf_{name}", torch.zeros(*shape, device=x.device)
                    )
                    self.replay_buffer[name] = getattr(self, f"buf_{name}")

        tokens = [self.towers[n](x) for n, x in inputs.items() if n in self.towers]
        combined = torch.cat(tokens, dim=0)  # (total_seq, B, dim)

        residual = combined
        combined = (
            residual
            + self.cross_attn(
                self.norm1(combined), self.norm1(combined), self.norm1(combined)
            )[0]
        )
        combined = combined + self.output_proj(self.norm2(combined))

        out = combined.permute(1, 0, 2)  # (B, total_seq, dim)

        if update_buffer and labels is not None:
            with torch.no_grad():
                B = labels.size(0)

                # ── Reservoir Sampling ──
                for i in range(B):
                    count = self.replay_count.item()
                    if count < self.replay_buffer_size:
                        # Buffer not full, just append
                        idx = count
                    else:
                        # Buffer full, replace with probability buffer_size / total_seen
                        idx = torch.randint(0, count + 1, (1,)).item()
                        if idx >= self.replay_buffer_size:
                            idx = -1  # Do not keep

                    if idx != -1:
                        # Update all modality buffers
                        for name, x_i in inputs.items():
                            if name in self.replay_buffer:
                                self.replay_buffer[name][idx] = x_i[i].detach()
                        # Update labels
                        self.replay_labels[idx] = labels[i].detach()

                    self.replay_count[0] += 1

        return out

    def get_replay_samples(
        self, n: int
    ) -> Tuple[Optional[dict], Optional[torch.Tensor]]:
        count = min(self.replay_count.item(), self.replay_buffer_size)
        if count == 0:
            return None, None
        n = min(n, count)
        idx = torch.randperm(count, device=self.replay_labels.device)[:n]

        samples = {}
        for name in self.replay_buffer:
            samples[name] = self.replay_buffer[name][idx]

        return samples, self.replay_labels[idx]


class SensoryPerceiverCL(nn.Module):
    """Backbone + fixed 100-class CIL head."""

    def __init__(
        self,
        modality_configs: dict,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        n_classes: int = 100,
        dropout: float = 0.1,
        replay_buffer_size: int = 1000,
        contrastive_temp: float = 0.07,
    ):
        super().__init__()
        self.backbone = SensoryPerceiver(
            modality_configs, dim, num_heads, mlp_dim, dropout, replay_buffer_size
        )
        self.classifier = nn.Linear(dim, n_classes)
        self.temp = contrastive_temp

    def forward(
        self, inputs: dict, labels: torch.Tensor = None, update_buffer: bool = True
    ):
        features = self.backbone(
            inputs, labels=labels, update_buffer=update_buffer
        )  # (B, seq, dim)
        pooled = features.mean(dim=1)  # (B, dim)
        logits = self.classifier(pooled)  # (B, n_classes)
        return logits, features

    def contrastive_replay_loss(self, features: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss: current batch embeddings vs. random replay buffer samples."""
        replay_inputs, _ = self.backbone.get_replay_samples(features.size(0))
        if replay_inputs is None:
            return torch.tensor(0.0, device=features.device)

        # We need to run replay inputs through backbone to get features
        with torch.no_grad():
            replay_out = self.backbone(replay_inputs, update_buffer=False)
            replay_features = replay_out.mean(dim=1)

        pooled = nn.functional.normalize(features.mean(dim=1), dim=-1)
        replay = nn.functional.normalize(replay_features, dim=-1)
        sim = torch.mm(pooled, replay.t()) / self.temp
        labels = torch.arange(sim.size(0), device=sim.device)
        return (
            nn.functional.cross_entropy(sim, labels)
            + nn.functional.cross_entropy(sim.t(), labels)
        ) / 2

    def experience_replay_loss(self, batch_size: int) -> torch.Tensor:
        """Standard Experience Replay: CE loss on replayed raw samples."""
        replay_inputs, replay_labels = self.backbone.get_replay_samples(batch_size)
        if replay_inputs is None:
            return torch.tensor(0.0, device=self.classifier.weight.device)

        # Forward pass for replayed samples (updates backbone and head)
        logits, _ = self.forward(replay_inputs, update_buffer=False)
        return nn.functional.cross_entropy(logits, replay_labels)


# ─── Data ────────────────────────────────────────────────────────────────────


class SplitCIFAR100Task(torch.utils.data.Dataset):
    """One task's slice of CIFAR-100 with ABSOLUTE labels (0-99)."""

    _cifar_train = None  # class-level cache — download once
    _cifar_test = None

    def __init__(
        self, task_id: int, classes_per_task: int, train: bool, transform=None
    ):
        start, end = task_id * classes_per_task, (task_id + 1) * classes_per_task

        if train:
            if SplitCIFAR100Task._cifar_train is None:
                SplitCIFAR100Task._cifar_train = CIFAR100(
                    root="/tmp/cifar100", train=True, download=True, transform=transform
                )
            cifar = SplitCIFAR100Task._cifar_train
        else:
            if SplitCIFAR100Task._cifar_test is None:
                SplitCIFAR100Task._cifar_test = CIFAR100(
                    root="/tmp/cifar100",
                    train=False,
                    download=True,
                    transform=transform,
                )
            cifar = SplitCIFAR100Task._cifar_test

        imgs, labels = [], []
        for img, lbl in cifar:
            if start <= lbl < end:
                imgs.append(img)
                labels.append(lbl)  # absolute label

        self.imgs = imgs
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return {"vision": self.imgs[i]}, self.labels[i]


def make_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )


def make_val_loader(task_id, classes_per_task, batch_size, num_workers, pin_memory):
    ds = SplitCIFAR100Task(
        task_id, classes_per_task, train=False, transform=make_transform()
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def make_train_loader(task_id, classes_per_task, batch_size, num_workers, pin_memory):
    ds = SplitCIFAR100Task(
        task_id, classes_per_task, train=True, transform=make_transform()
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ─── Training / Evaluation ───────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, device, replay_weight=0.1):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, features = model(inputs, labels=labels, update_buffer=True)
        loss = ce(logits, labels)

        if replay_weight > 0:
            if getattr(model, "replay_mode", "er") == "er":
                loss = loss + replay_weight * model.experience_replay_loss(
                    labels.size(0)
                )
            else:
                loss = loss + replay_weight * model.contrastive_replay_loss(features)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        logits, _ = model(inputs, labels=None, update_buffer=False)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ─── Metrics ─────────────────────────────────────────────────────────────────


def compute_metrics(R: np.ndarray) -> dict:
    """
    R[t][j] = accuracy on task j's test set after training through task t.
    Diagonal R[j][j] = accuracy on task j right after training on it.
    R[t][t+1] filled during task t for FWT (zero-shot transfer).

    CIL = mean(R[T-1, :])               — final accuracy across all tasks
    BWT = mean(R[T-1,j] - R[j,j])       — forgetting (negative = bad)
          for j in 0..T-2
    FWT = mean(R[j-1, j])               — zero-shot transfer to unseen tasks
          for j in 1..T-1
    """
    T = R.shape[0]
    cil = float(np.nanmean(R[T - 1, :]))
    bwt = (
        float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))
        if T > 1
        else float("nan")
    )
    fwt = (
        float(
            np.nanmean([R[j - 1, j] for j in range(1, T) if not np.isnan(R[j - 1, j])])
        )
        if T > 1
        else float("nan")
    )
    return {"CIL": cil, "BWT": bwt, "FWT": fwt}


# ─── Figures ─────────────────────────────────────────────────────────────────


def save_figures(R: np.ndarray, metrics: dict, out_dir: Path):
    T = R.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Sensory Perceiver — Split-CIFAR-100 Continual Learning",
        fontsize=13,
        fontweight="bold",
    )

    # ── Left: Accuracy matrix heatmap ────────────────────────────────────────
    ax = axes[0]
    masked = np.ma.masked_invalid(R)
    im = ax.imshow(masked, vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")

    ax.set_xlabel("Task evaluated (j)", fontsize=11)
    ax.set_ylabel("Trained through task (t)", fontsize=11)
    ax.set_title("Accuracy matrix R[t][j]", fontsize=11)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"T{j}" for j in range(T)])
    ax.set_yticks(range(T))
    ax.set_yticklabels([f"T{t}" for t in range(T)])

    for t in range(T):
        for j in range(T):
            if not np.isnan(R[t][j]):
                ax.text(
                    j,
                    t,
                    f"{R[t][j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if R[t][j] > 40 else "white",
                )

    # Mark diagonal (R[j][j] — accuracy right after training each task)
    for j in range(T):
        ax.add_patch(
            plt.Rectangle(
                (j - 0.5, j - 0.5), 1, 1, fill=False, edgecolor="blue", linewidth=2
            )
        )

    # ── Right: CIL / BWT / FWT summary bars ──────────────────────────────────
    ax2 = axes[1]
    labels = ["CIL\n(higher=better)", "BWT\n(higher=better)", "FWT\n(higher=better)"]
    values = [
        metrics["CIL"],
        metrics["BWT"],
        metrics["FWT"] if not (metrics["FWT"] != metrics["FWT"]) else 0.0,
    ]
    colors = ["#2196F3", "#4CAF50" if metrics["BWT"] >= 0 else "#F44336", "#FF9800"]

    bars = ax2.bar(labels, values, color=colors, width=0.5, edgecolor="white")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Accuracy / Transfer (%)", fontsize=11)
    ax2.set_title("Summary metrics", fontsize=11)
    ax2.set_ylim(min(-10, min(values) - 5), max(100, max(values) + 10))

    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5 * np.sign(bar.get_height()),
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Config annotation
    cfg = metrics.get("config", {})
    cfg_txt = (
        f"d_model={cfg.get('d_model','?')}  "
        f"n_tasks={cfg.get('n_tasks','?')}  "
        f"epochs/task={cfg.get('epochs_per_task','?')}  "
        f"elapsed={metrics.get('elapsed_seconds',0)/60:.1f} min"
    )
    fig.text(0.5, 0.01, cfg_txt, ha="center", fontsize=9, color="gray")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig_path = out_dir / "accuracy_matrix.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {fig_path}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Sanity check: 3 tasks, 1 epoch, small model (~5 min CPU)",
    )
    parser.add_argument(
        "--tuned",
        action="store_true",
        help="Tuned config: replay_weight=0.5, buffer=5000, lr=3e-4 "
        "→ results/cl_experiment_tuned/",
    )
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--epochs_per_task", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--replay_weight", type=float, default=1.0)
    parser.add_argument("--replay_buffer", type=int, default=5000)
    parser.add_argument(
        "--replay_mode",
        type=str,
        default="er",
        choices=["er", "contrastive"],
        help="ER: Experience Replay (raw samples). Contrastive: NT-Xent on replayed tokens.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/cl_experiment")
    args = parser.parse_args()

    if args.quick:
        args.n_tasks = 3
        args.epochs_per_task = 1
        args.d_model = 128
        args.num_heads = 4
        args.mlp_dim = 256
        args.batch_size = 128
        print("── QUICK MODE: 3 tasks / 1 epoch / d_model=128 ──")

    if args.tuned:
        # Stronger replay signal — designed to reduce catastrophic forgetting
        # replay_weight 0.1→0.5: NT-Xent loss competes meaningfully with CE loss
        # replay_buffer 1000→5000: ~500 exemplars/task at 10 tasks (vs ~100)
        # lr 1e-4→3e-4: helps later tasks overcome accumulated weight inertia
        args.replay_weight = 0.5
        args.replay_buffer = 5000
        args.lr = 3e-4
        if args.output_dir == "results/cl_experiment":  # only override if default
            args.output_dir = "results/cl_experiment_tuned"
        print("── TUNED MODE: replay_weight=0.5 / buffer=5000 / lr=3e-4 ──")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    pin_memory = device.type == "cuda"
    classes_per_task = 5  # 100 CIFAR-100 classes / 20 standard tasks
    n_classes = 100  # full CIL head

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build model ──────────────────────────────────────────────────────────
    model = SensoryPerceiverCL(
        modality_configs={"vision": {"input_dim": 3, "use_conv": True}},
        dim=args.d_model,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        n_classes=n_classes,
        dropout=args.dropout,
        replay_buffer_size=args.replay_buffer,
        contrastive_temp=0.07,
    ).to(device)
    model.replay_mode = args.replay_mode

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}  ({total_params/1e6:.2f}M)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_tasks * args.epochs_per_task)

    # ── Pre-load ALL val loaders (needed for FWT: eval on next unseen task) ──
    print("Loading datasets...", end=" ", flush=True)
    all_val_loaders = [
        make_val_loader(
            t, classes_per_task, args.batch_size, args.num_workers, pin_memory
        )
        for t in range(args.n_tasks)
    ]
    print("done")

    # R[t][j] = accuracy on task j after training through task t
    # Also fills R[t][t+1] for FWT (zero-shot on next unseen task)
    R = np.full((args.n_tasks, args.n_tasks), np.nan)
    start_time = time.time()

    print(f"\n{'='*60}")
    print(
        f"Split-CIFAR-100 CIL  |  {args.n_tasks} tasks × {args.epochs_per_task} epochs  |  "
        f"d_model={args.d_model}"
    )
    print(f"{'='*60}")

    for t in range(args.n_tasks):
        cls_start = t * classes_per_task
        cls_end = cls_start + classes_per_task - 1
        print(f"\n── Task {t+1}/{args.n_tasks} (classes {cls_start}–{cls_end}) ──")

        train_loader = make_train_loader(
            t, classes_per_task, args.batch_size, args.num_workers, pin_memory
        )

        for epoch in range(args.epochs_per_task):
            loss, acc = train_one_epoch(
                model, train_loader, optimizer, device, args.replay_weight
            )
            scheduler.step()
            print(
                f"  epoch {epoch+1}/{args.epochs_per_task}  "
                f"loss={loss:.4f}  train_acc={acc:.1f}%"
            )

        # Evaluate on all seen tasks (0..t)
        print(f"  eval (seen):  ", end="", flush=True)
        for j in range(t + 1):
            R[t][j] = evaluate(model, all_val_loaders[j], device)
            print(f"T{j}={R[t][j]:.1f}%  ", end="", flush=True)

        # FWT: evaluate on next unseen task (t+1) before training it
        if t + 1 < args.n_tasks:
            R[t][t + 1] = evaluate(model, all_val_loaders[t + 1], device)
            print(f"| zero-shot T{t+1}={R[t][t+1]:.1f}%", end="")

        print()

    elapsed = time.time() - start_time
    metrics = compute_metrics(R)
    metrics["elapsed_seconds"] = elapsed
    metrics["config"] = vars(args)
    metrics["R_matrix"] = R.tolist()

    print(f"\n{'='*60}")
    print(f"RESULTS  ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  CIL accuracy:      {metrics['CIL']:.2f}%")
    print(f"  BWT (forgetting):  {metrics['BWT']:+.2f}%")
    print(f"  FWT (transfer):    {metrics['FWT']:.2f}%")

    print()
    print("Accuracy matrix R[t][j]  (blue border = diagonal = R[j][j])")
    header = "       " + "  ".join(f"T{j:<4}" for j in range(args.n_tasks))
    print(header)
    for t in range(args.n_tasks):
        row = "  ".join(
            f"{R[t][j]:5.1f}" if not np.isnan(R[t][j]) else "  —  "
            for j in range(args.n_tasks)
        )
        print(f"  T{t:<3}  {row}")

    # ── Save results ─────────────────────────────────────────────────────────
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved → {results_path}")

    save_figures(R, metrics, out_dir)


if __name__ == "__main__":
    main()
