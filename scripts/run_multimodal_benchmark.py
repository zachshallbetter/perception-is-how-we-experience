import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models import SensoryPerceiver
from src.data.loaders import create_dataloaders
from baselines.hf_perceiver_io import PerceiverIOBaseline


def measure_latency(model, batch, device, n_runs=10):
    """Measure forward pass latency."""
    model.eval()

    # Warmup
    for _ in range(2):
        with torch.no_grad():
            if isinstance(model, SensoryPerceiver):
                model(batch, return_embeddings=False)
            else:
                model(batch)

    # Actual measurement
    start_time = time.time()

    with torch.no_grad():
        for _ in range(n_runs):
            if isinstance(model, SensoryPerceiver):
                model(batch, return_embeddings=False)
            else:
                model(batch)

    end_time = time.time()

    avg_latency = ((end_time - start_time) / n_runs) * 1000  # ms
    return avg_latency


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Benchmark: Sensory Perceiver vs Perceiver IO"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Force CPU to bypass MPS deadlock during Perceiver IO attention calculation
    device = torch.device("cpu")
    print(f"Device: {device}\n")

    # Modality configurations matching AudioVisualProxyDataset
    # vision: (3, 224, 224)
    # audio: (1, 128, 128) -> We treat it as 1 channel for conv towers
    configs = {
        "vision": {"input_dim": 3, "use_conv": True},
        "audio": {"input_dim": 1, "use_conv": True},
    }

    print("Initializing Sensory Perceiver...")
    sp_model = SensoryPerceiver(
        modality_configs=configs,
        dim=256,
        num_heads=4,
        mlp_dim=512,
        depth=4,
        dropout=0.1,
    ).to(device)
    sp_classifier = nn.Linear(256, 10).to(device)

    print("Initializing Perceiver IO Baseline...")
    pio_model = PerceiverIOBaseline(num_classes=10, use_pretrained=False).to(device)

    print("Loading AudioVisual Proxy Dataset...")
    train_loader, val_loader = create_dataloaders(
        "multimodal", "audiovisual", batch_size=args.batch_size, num_workers=0
    )

    # Get a sample batch for testing
    sample_batch = next(iter(val_loader))
    inputs = {k: v.to(device) for k, v in sample_batch.items() if k != "label"}
    labels = sample_batch["label"].to(device)

    results = {}

    print(f"\\n{'='*50}\\nParameter Extent\\n{'='*50}")
    sp_params = count_parameters(sp_model) + count_parameters(sp_classifier)
    pio_params = count_parameters(pio_model)
    print(f"Sensory Perceiver: {sp_params:,}")
    print(f"Perceiver IO:      {pio_params:,}")

    results["parameters"] = {"SensoryPerceiver": sp_params, "PerceiverIO": pio_params}

    print(
        f"\\n{'='*50}\\nForward Pass Latency (Batch Size {args.batch_size})\\n{'='*50}"
    )
    sp_latency = measure_latency(sp_model, inputs, device)
    pio_latency = measure_latency(pio_model, inputs, device)

    print(f"Sensory Perceiver: {sp_latency:.2f} ms")
    print(f"Perceiver IO:      {pio_latency:.2f} ms")
    speedup = pio_latency / sp_latency
    print(f"--> SP is {speedup:.2f}x faster")

    results["latency_ms"] = {
        "SensoryPerceiver": sp_latency,
        "PerceiverIO": pio_latency,
        "Speedup": speedup,
    }

    print(f"\\n{'='*50}\\nTraining Validation (Epochs: {args.epochs})\\n{'='*50}")

    def train_model(name, model, classifier, loader, is_sp=False):
        optimizer = optim.AdamW(
            list(model.parameters())
            + (list(classifier.parameters()) if classifier else []),
            lr=args.lr,
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for batch in loader:
                batch_inputs = {
                    k: v.to(device) for k, v in batch.items() if k != "label"
                }
                batch_labels = batch["label"].to(device)

                optimizer.zero_grad()

                if is_sp:
                    outputs, _ = model(batch_inputs, return_embeddings=False)
                    pooled = outputs.mean(dim=1)
                    logits = classifier(pooled)
                else:
                    logits = model(batch_inputs)

                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[{name}] Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    train_model("SensoryPerceiver", sp_model, sp_classifier, train_loader, is_sp=True)
    print("-" * 50)
    train_model("PerceiverIO", pio_model, None, train_loader, is_sp=False)

    out_dir = Path("results/multimodal_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\\nResults saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
