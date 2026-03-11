import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
import json
import argparse
import time
from tqdm import tqdm
import sys
import torchvision.transforms as T

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.loaders import SplitCIFAR100
from baselines.perceiver import PerceiverBaseline
from baselines.hf_perceiver_io import PerceiverIOBaseline
from baselines.vilt import ViLTBaseline
from baselines.clip import CLIPBaseline
from baselines.flava import FLAVABaseline
from src.runtime import DEFAULT_CONTINUAL_TASKS

# Most vision-language models expect 224x224 inputs
BASELINE_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
    ]
)


def get_baseline_model(model_name: str, num_classes: int = 100):
    if model_name == "perceiver":
        return PerceiverBaseline(num_classes=num_classes, use_pretrained=False)
    elif model_name == "perceiver_io":
        return PerceiverIOBaseline(num_classes=num_classes, use_pretrained=False)
    elif model_name == "vilt":
        return ViLTBaseline(num_classes=num_classes, use_pretrained=False)
    elif model_name == "clip":
        return CLIPBaseline(use_pretrained=False, num_classes=num_classes)
    elif model_name == "flava":
        return FLAVABaseline(num_classes=num_classes, use_pretrained=False)
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Continual Learning Benchmark on Baselines"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["perceiver", "perceiver_io", "vilt", "clip", "flava"],
    )
    parser.add_argument(
        "--n_tasks", type=int, default=DEFAULT_CONTINUAL_TASKS["split_cifar"]
    )
    parser.add_argument("--epochs_per_task", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick", action="store_true", help="Run a fast 2-task, 1-epoch test"
    )
    parser.add_argument("--output_dir", type=str, default="results/baselines_cl")
    args = parser.parse_args()

    if args.quick:
        args.n_tasks = 2
        args.epochs_per_task = 1
        args.batch_size = min(args.batch_size, 4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Model
    # Since we are doing CIL, the network has 100 outputs total.
    print(f"Loading baseline model: {args.model}")
    model = get_baseline_model(args.model, num_classes=100)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    accuracy_matrix = np.zeros((args.n_tasks, args.n_tasks))
    start_time = time.time()

    print(
        f"\\n{'='*60}\\nBaseline CIL Benchmark | Model: {args.model} | {args.n_tasks} tasks\\n{'='*60}\\n"
    )

    for task_id in range(args.n_tasks):
        print(f"\\n── Task {task_id + 1}/{args.n_tasks} ──")

        train_dataset = SplitCIFAR100(
            n_tasks=args.n_tasks, task_id=task_id, split="train"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
        )

        # Training loop (Standard Fine-Tuning, no replay to demonstrate catastrophic forgetting)
        for epoch in range(args.epochs_per_task):
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs_per_task}",
                leave=False,
            )
            for batch in pbar:
                images = batch["vision"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                if args.model in ["vilt", "clip", "flava"]:
                    images = BASELINE_TRANSFORM(images)
                    logits = model(pixel_values=images)
                else:
                    # perceiver, perceiver_io (can handle flexible sequence lengths like 32x32=1024)
                    logits = model(images)

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                total_correct += logits.argmax(1).eq(labels).sum().item()
                total_samples += len(labels)

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100*total_correct/total_samples:.1f}%",
                    }
                )

            print(
                f"  epoch {epoch+1}/{args.epochs_per_task}  loss={total_loss/total_samples:.4f}  train_acc={100*total_correct/total_samples:.1f}%"
            )

        # Evaluation Loop (seen tasks + next zero-shot)
        eval_str = "  eval (seen):  "
        model.eval()

        # Evaluate on all seen tasks
        for eval_task_id in range(task_id + 1):
            test_dataset = SplitCIFAR100(
                n_tasks=args.n_tasks, task_id=eval_task_id, split="test"
            )
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False
            )

            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    images = batch["vision"].to(device)
                    labels = batch["label"].to(device)

                    if args.model in ["vilt", "clip", "flava"]:
                        images = BASELINE_TRANSFORM(images)
                        logits = model(pixel_values=images)
                    else:
                        logits = model(images)

                    correct += logits.argmax(1).eq(labels).sum().item()
                    total += len(labels)

            acc = 100.0 * correct / total
            accuracy_matrix[task_id, eval_task_id] = acc
            eval_str += f"T{eval_task_id}={acc:.1f}%  "

        print(eval_str)

    # Calculate final metrics
    final_cil = accuracy_matrix[-1, :].mean()
    bwt = (
        np.mean(
            [
                accuracy_matrix[-1, j] - accuracy_matrix[j, j]
                for j in range(args.n_tasks - 1)
            ]
        )
        if args.n_tasks > 1
        else 0.0
    )

    print(f"\\n{'='*60}\\nRESULTS ({args.model})\\n{'='*60}")
    print(f"  CIL accuracy:      {final_cil:.2f}%")
    print(f"  BWT (forgetting):  {bwt:.2f}%")

    results = {
        "model": args.model,
        "CIL": float(final_cil),
        "BWT": float(bwt),
        "elapsed_seconds": time.time() - start_time,
        "config": vars(args),
        "R_matrix": accuracy_matrix.tolist(),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\\nResults saved → {output_dir}/results.json")


if __name__ == "__main__":
    main()
