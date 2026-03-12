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
  python scripts/run_local.py                # baseline: 20 tasks, 5 epochs, replay_weight=0.5
  python scripts/run_local.py --tuned        # tuned:    20 tasks, 5 epochs, replay_weight=0.8, buffer=8000, ewc_lambda=0.6

Baseline vs Tuned:
  --baseline  replay_weight=0.5, replay_buffer=5000, lr=1e-4  → results/cl_experiment/
  --tuned     replay_weight=0.8, replay_buffer=8000, lr=3e-4, ewc_lambda=0.6  → results/cl_experiment_tuned/

Results: results/cl_experiment/ (baseline) or results/cl_experiment_tuned/ (tuned)
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the repository root is on sys.path so `import src...` works even when this
# script is executed from within the `scripts/` directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.models.core import ContrastiveLoss, SensoryPerceiver
from src.runtime import DEFAULT_CONTINUAL_TASKS

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


def pool_features(features: torch.Tensor) -> torch.Tensor:
    return features.mean(dim=1)


def build_optimizer(params, lr):
    return optim.AdamW(params, lr=lr, weight_decay=0.01)


def replay_loss(
    model,
    classifier,
    contrastive_loss_fn,
    device,
    batch_size,
    mode,
    current_pooled,
    teacher_model,
    teacher_classifier,
    distill_temp,
):
    replay_inputs, replay_labels = model.get_replay_samples(batch_size)
    if replay_inputs is None:
        return torch.tensor(0.0, device=device)

    replay_inputs = {k: v.to(device) for k, v in replay_inputs.items()}

    if mode == "er":
        replay_features, _ = model(
            replay_inputs, labels=None, update_buffer=False, return_embeddings=False
        )
        replay_logits = classifier(pool_features(replay_features))
        return nn.functional.cross_entropy(replay_logits, replay_labels.to(device))

    if mode == "distill":
        if teacher_model is None or teacher_classifier is None:
            return torch.tensor(0.0, device=device)

        replay_features, _ = model(
            replay_inputs, labels=None, update_buffer=False, return_embeddings=False
        )
        student_logits = classifier(pool_features(replay_features))
        with torch.no_grad():
            teacher_features, _ = teacher_model(
                replay_inputs,
                labels=None,
                update_buffer=False,
                return_embeddings=False,
            )
            teacher_logits = teacher_classifier(pool_features(teacher_features))
        student_log_probs = nn.functional.log_softmax(
            student_logits / distill_temp, dim=1
        )
        teacher_probs = nn.functional.softmax(teacher_logits / distill_temp, dim=1)
        return nn.functional.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (distill_temp**2)

    replay_features, _ = model(
        replay_inputs, labels=None, update_buffer=False, return_embeddings=False
    )
    replay_pooled = pool_features(replay_features)
    if current_pooled.size(0) != replay_pooled.size(0):
        current_pooled = current_pooled[: replay_pooled.size(0)]
    return contrastive_loss_fn(current_pooled, replay_pooled)


def ewc_penalty(model, ewc_tasks: List[Dict[str, Dict[str, torch.Tensor]]]):
    if not ewc_tasks:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    penalty = 0.0
    for task in ewc_tasks:
        for name, param in model.named_parameters():
            if name in task["fisher"]:
                fisher = task["fisher"][name]
                prev = task["params"][name]
                penalty += (fisher * (param - prev).pow(2)).sum()
    return penalty


def si_penalty(
    model, si_omega: Dict[str, torch.Tensor], si_prev: Dict[str, torch.Tensor]
):
    if not si_omega:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    penalty = 0.0
    for name, param in model.named_parameters():
        if name in si_omega and name in si_prev:
            prev = si_prev[name]
            omega = si_omega[name]
            penalty += (omega * (param - prev).pow(2)).sum()
    return penalty


def ensure_si_state(
    model,
    si_prev: Dict[str, torch.Tensor],
    si_omega: Dict[str, torch.Tensor],
    si_accum: Dict[str, torch.Tensor],
) -> None:
    for name, param in model.named_parameters():
        if name not in si_prev:
            si_prev[name] = param.detach().clone()
        if name not in si_omega:
            si_omega[name] = torch.zeros_like(param)
        if name not in si_accum:
            si_accum[name] = torch.zeros_like(param)


def train_one_epoch(
    model,
    classifier,
    loader,
    optimizer,
    device,
    replay_weight,
    replay_mode,
    contrastive_loss_fn,
    ewc_tasks,
    ewc_lambda,
    si_omega,
    si_prev,
    si_lambda,
    si_accum,
    teacher_model,
    teacher_classifier,
    distill_temp,
):
    model.train()
    classifier.train()
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    if si_lambda > 0:
        ensure_si_state(model, si_prev, si_omega, si_accum)

    for inputs, labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        features, _ = model(inputs, labels=labels, update_buffer=True)
        pooled = pool_features(features)
        logits = classifier(pooled)
        loss = ce(logits, labels)

        if replay_weight > 0:
            loss = loss + replay_weight * replay_loss(
                model,
                classifier,
                contrastive_loss_fn,
                device,
                labels.size(0),
                replay_mode,
                pooled,
                teacher_model,
                teacher_classifier,
                distill_temp,
            )

        if ewc_lambda > 0:
            loss = loss + ewc_lambda * ewc_penalty(model, ewc_tasks)

        if si_lambda > 0:
            loss = loss + si_lambda * si_penalty(model, si_omega, si_prev)

        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()), 1.0
        )

        if si_lambda > 0:
            prev_params = {n: p.detach().clone() for n, p in model.named_parameters()}

        optimizer.step()

        if si_lambda > 0:
            for name, param in model.named_parameters():
                if param.grad is None or name not in si_accum:
                    continue
                delta = param.detach() - prev_params[name]
                si_accum[name] += param.grad.detach() * delta

        total_loss += loss.item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, classifier, loader, device) -> float:
    model.eval()
    classifier.eval()
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        features, _ = model(inputs, labels=None, update_buffer=False)
        logits = classifier(pool_features(features))
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def compute_fisher(
    model,
    classifier,
    loader,
    device,
    max_batches: int,
) -> Dict[str, torch.Tensor]:
    model.eval()
    classifier.eval()
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    ce = nn.CrossEntropyLoss()
    batches = 0

    for inputs, labels in loader:
        if batches >= max_batches:
            break
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        model.zero_grad(set_to_none=True)
        classifier.zero_grad(set_to_none=True)
        features, _ = model(inputs, labels=None, update_buffer=False)
        logits = classifier(pool_features(features))
        loss = ce(logits, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach().pow(2)
        batches += 1

    for name in fisher:
        fisher[name] /= max(1, batches)
    return fisher


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
    parser.add_argument(
        "--n_tasks", type=int, default=DEFAULT_CONTINUAL_TASKS["split_cifar"]
    )
    parser.add_argument("--epochs_per_task", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--replay_weight", type=float, default=0.5)
    parser.add_argument("--replay_buffer", type=int, default=5000)
    parser.add_argument(
        "--replay_mode",
        type=str,
        default="er",
        choices=["er", "contrastive", "distill"],
        help=(
            "ER: label replay on raw samples. "
            "Contrastive: NT-Xent on replayed features. "
            "Distill: KL replay against a frozen previous-task teacher."
        ),
    )
    parser.add_argument("--distill_temp", type=float, default=2.0)
    parser.add_argument("--ewc_lambda", type=float, default=0.4)
    parser.add_argument("--ewc_batches", type=int, default=5)
    parser.add_argument("--si_lambda", type=float, default=0.0)
    parser.add_argument("--si_epsilon", type=float, default=1e-3)
    parser.add_argument("--late_task_boost", type=int, default=2)
    parser.add_argument("--late_task_start", type=int, default=-1)
    parser.add_argument("--no_parameter_isolation", action="store_true")
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
        args.replay_weight = 0.8
        args.replay_buffer = 8000
        args.lr = 3e-4
        args.ewc_lambda = 0.6
        if args.output_dir == "results/cl_experiment":  # only override if default
            args.output_dir = "results/cl_experiment_tuned"
        print(
            "── TUNED MODE: replay_weight=0.8 / buffer=8000 / lr=3e-4 / ewc_lambda=0.6 ──"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    pin_memory = device.type == "cuda"
    classes_per_task = 5  # 100 CIFAR-100 classes / 20 standard tasks
    n_classes = 100  # full CIL head

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.late_task_start < 0:
        args.late_task_start = max(0, args.n_tasks - 3)

    total_epochs = args.n_tasks * args.epochs_per_task
    if args.late_task_boost > 0:
        total_epochs += (
            max(0, args.n_tasks - args.late_task_start) * args.late_task_boost
        )

    # ── Build model ──────────────────────────────────────────────────────────
    model = SensoryPerceiver(
        modality_configs={"vision": {"input_dim": 3, "use_conv": True}},
        dim=args.d_model,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        replay_buffer_size=args.replay_buffer,
        contrastive_temp=0.07,
        enable_parameter_isolation=not args.no_parameter_isolation,
        adapter_hidden_dim=max(128, args.d_model // 2),
    ).to(device)
    classifier = nn.Linear(args.d_model, n_classes).to(device)
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07).to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in classifier.parameters()
    )
    print(f"Model params: {total_params:,}  ({total_params/1e6:.2f}M)")

    optimizer = build_optimizer(
        list(model.parameters()) + list(classifier.parameters()), args.lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

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
    ewc_tasks = []
    si_prev = {name: param.detach().clone() for name, param in model.named_parameters()}
    si_omega = {
        name: torch.zeros_like(param) for name, param in model.named_parameters()
    }
    si_accum = {
        name: torch.zeros_like(param) for name, param in model.named_parameters()
    }
    teacher_model = None
    teacher_classifier = None

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

        epochs_this_task = args.epochs_per_task
        if t >= args.late_task_start:
            epochs_this_task += args.late_task_boost

        for epoch in range(epochs_this_task):
            loss, acc = train_one_epoch(
                model,
                classifier,
                train_loader,
                optimizer,
                device,
                args.replay_weight,
                args.replay_mode,
                contrastive_loss_fn,
                ewc_tasks,
                args.ewc_lambda,
                si_omega,
                si_prev,
                args.si_lambda,
                si_accum,
                teacher_model,
                teacher_classifier,
                args.distill_temp,
            )
            scheduler.step()
            print(
                f"  epoch {epoch+1}/{epochs_this_task}  "
                f"loss={loss:.4f}  train_acc={acc:.1f}%"
            )

        if args.ewc_lambda > 0:
            fisher = compute_fisher(
                model, classifier, train_loader, device, args.ewc_batches
            )
            ewc_tasks.append(
                {
                    "params": {
                        name: param.detach().clone()
                        for name, param in model.named_parameters()
                    },
                    "fisher": fisher,
                }
            )

        if args.si_lambda > 0:
            ensure_si_state(model, si_prev, si_omega, si_accum)
            for name, param in model.named_parameters():
                delta = param.detach() - si_prev[name]
                si_omega[name] += si_accum[name] / (delta.pow(2) + args.si_epsilon)
                si_prev[name] = param.detach().clone()
                si_accum[name].zero_()

        if args.replay_mode == "distill":
            teacher_model = copy.deepcopy(model).to(device)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad_(False)

            teacher_classifier = copy.deepcopy(classifier).to(device)
            teacher_classifier.eval()
            for param in teacher_classifier.parameters():
                param.requires_grad_(False)

        if not args.no_parameter_isolation:
            model.freeze_task()
            if args.si_lambda > 0:
                ensure_si_state(model, si_prev, si_omega, si_accum)
            optimizer = build_optimizer(
                [p for p in model.parameters() if p.requires_grad]
                + list(classifier.parameters()),
                args.lr,
            )

        # Evaluate on all seen tasks (0..t)
        print(f"  eval (seen):  ", end="", flush=True)
        for j in range(t + 1):
            R[t][j] = evaluate(model, classifier, all_val_loaders[j], device)
            print(f"T{j}={R[t][j]:.1f}%  ", end="", flush=True)

        # FWT: evaluate on next unseen task (t+1) before training it
        if t + 1 < args.n_tasks:
            R[t][t + 1] = evaluate(model, classifier, all_val_loaders[t + 1], device)
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
