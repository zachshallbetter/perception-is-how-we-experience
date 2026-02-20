"""
Quick unit test for continual learning benchmark using synthetic data.
Verifies the pipeline works before running on real CIFAR-100.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import logging
from src.models import SensoryPerceiver, ContrastiveLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_continual_learning_pipeline():
    """Test continual learning with synthetic data."""

    device = torch.device("cpu")
    n_tasks = 2
    classes_per_task = 10

    logger.info(f"Testing continual learning pipeline with {n_tasks} tasks ({classes_per_task} classes each)")

    # Create model
    model = SensoryPerceiver(
        modality_configs={"vision": {"input_dim": 3, "use_conv": True}},
        dim=256,
        num_heads=4,
        mlp_dim=512,
        depth=4,
        dropout=0.1,
        enable_parameter_isolation=True,
        replay_buffer_size=100,
    ).to(device)

    # Create synthetic data
    batch_size = 16
    img_size = 32

    logger.info(f"Creating synthetic datasets...")

    # Task-specific datasets
    datasets = []
    for task_id in range(n_tasks):
        # Synthetic images: (batch, 3, 32, 32)
        X = torch.randn(64, 3, img_size, img_size)
        # Labels: 0 to classes_per_task-1
        y = torch.randint(0, classes_per_task, (64,))
        datasets.append(TensorDataset(X, y))

    # Setup training
    classifier = nn.Linear(256, classes_per_task).to(device)
    criterion = nn.CrossEntropyLoss()
    contrastive_criterion = ContrastiveLoss(temperature=0.07)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )

    replay_weight = 0.5
    epochs_per_task = 1

    # Accuracy tracking
    accuracy_matrix = np.zeros((n_tasks, n_tasks))

    # Train on tasks sequentially
    for task_id in range(n_tasks):
        logger.info(f"\n--- Task {task_id} ---")

        train_loader = DataLoader(
            datasets[task_id],
            batch_size=batch_size,
            shuffle=True,
        )

        # Train for one epoch
        for epoch in range(epochs_per_task):
            model.train()
            total_loss = 0.0
            total_correct = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs, embeddings = model(
                    {"vision": images},
                    labels=labels,
                    return_embeddings=True
                )

                # Pool and classify
                pooled = outputs.mean(dim=1)
                logits = classifier(pooled)

                # Loss
                loss_ce = criterion(logits, labels)
                loss = loss_ce

                # Add replay loss if buffer has samples
                if model.replay_count.item() > 0:
                    replay_emb, replay_labels_r = model.get_replay_samples(len(labels))
                    if replay_emb is not None and replay_emb.size(0) == len(labels):
                        replay_emb = replay_emb.to(device)
                        loss_replay = contrastive_criterion(pooled, replay_emb)
                        loss = (1 - replay_weight) * loss_ce + replay_weight * loss_replay

                # Backward
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                _, predicted = logits.max(1)
                total_correct += predicted.eq(labels).sum().item()

            train_acc = 100.0 * total_correct / len(datasets[task_id])
            logger.info(f"  Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Loss = {total_loss/len(datasets[task_id]):.4f}")

        # Evaluate on all tasks
        logger.info(f"  Evaluating on tasks 0-{task_id}...")
        for eval_task_id in range(task_id + 1):
            model.eval()
            correct = 0
            total = 0

            test_loader = DataLoader(
                datasets[eval_task_id],
                batch_size=batch_size,
                shuffle=False,
            )

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs, _ = model({"vision": images}, return_embeddings=False)
                    pooled = outputs.mean(dim=1)
                    logits = classifier(pooled)

                    _, predicted = logits.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += len(labels)

            task_acc = 100.0 * correct / total
            accuracy_matrix[task_id, eval_task_id] = task_acc
            logger.info(f"    Task {eval_task_id}: {task_acc:.2f}%")

        # Freeze task and allocate new adapter (parameter isolation)
        if hasattr(model, 'freeze_task'):
            model.freeze_task()
            logger.info(f"  Froze task {task_id} adapter")

        # Print trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Trainable params: {trainable}")

    # Compute metrics
    cil = accuracy_matrix[-1, :].mean()
    bwt = np.mean([accuracy_matrix[-1, j] - accuracy_matrix[j, j]
                   for j in range(n_tasks - 1)])

    logger.info(f"\n{'='*50}")
    logger.info(f"Results Summary:")
    logger.info(f"  CIL: {cil:.2f}%")
    logger.info(f"  BWT: {bwt:.2f}%")
    logger.info(f"  Accuracy Matrix:\n{accuracy_matrix}")
    logger.info(f"{'='*50}")

    return True

if __name__ == '__main__':
    try:
        test_continual_learning_pipeline()
        logger.info("\n✅ Test PASSED")
    except Exception as e:
        logger.error(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
