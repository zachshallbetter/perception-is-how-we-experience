"""Quick continual-learning pipeline test for generated benchmark data."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import unittest

from src.models import SensoryPerceiver, ContrastiveLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContinualLearningQuickTest(unittest.TestCase):
    def test_continual_learning_pipeline(self):
        """Validate continual learning training/eval flow on generated inputs."""
        device = torch.device("cpu")
        n_tasks = 2
        classes_per_task = 10

        logger.info(f"Testing continual learning pipeline with {n_tasks} tasks ({classes_per_task} classes each)")

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

        batch_size = 16
        img_size = 32

        logger.info("Creating generated datasets...")

        datasets = []
        for _task_id in range(n_tasks):
            x_data = torch.randn(64, 3, img_size, img_size)
            y_data = torch.randint(0, classes_per_task, (64,))
            datasets.append(TensorDataset(x_data, y_data))

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
        accuracy_matrix = np.zeros((n_tasks, n_tasks))

        for task_id in range(n_tasks):
            logger.info(f"\n--- Task {task_id} ---")

            train_loader = DataLoader(
                datasets[task_id],
                batch_size=batch_size,
                shuffle=True,
            )

            for epoch in range(epochs_per_task):
                model.train()
                total_loss = 0.0
                total_correct = 0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs, _embeddings = model(
                        {"vision": images},
                        labels=labels,
                        return_embeddings=True
                    )

                    pooled = outputs.mean(dim=1)
                    logits = classifier(pooled)
                    loss_ce = criterion(logits, labels)
                    loss = loss_ce

                    if model.replay_count.item() > 0:
                        replay_inputs, replay_labels = model.get_replay_samples(len(labels))
                        if replay_inputs is not None and replay_labels is not None:
                            replay_inputs = {
                                key: value.to(device) for key, value in replay_inputs.items()
                            }
                            with torch.no_grad():
                                replay_outputs, _ = model(replay_inputs, update_buffer=False)
                                replay_features = replay_outputs.mean(dim=1)
                            n_pairs = min(pooled.size(0), replay_features.size(0))
                            if n_pairs > 0:
                                loss_replay = contrastive_criterion(
                                    pooled[:n_pairs], replay_features[:n_pairs]
                                )
                                loss = (1 - replay_weight) * loss_ce + replay_weight * loss_replay

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * len(labels)
                    _, predicted = logits.max(1)
                    total_correct += predicted.eq(labels).sum().item()

                train_acc = 100.0 * total_correct / len(datasets[task_id])
                logger.info(f"  Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Loss = {total_loss/len(datasets[task_id]):.4f}")

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

            if hasattr(model, 'freeze_task'):
                model.freeze_task()
                logger.info(f"  Froze task {task_id} adapter")

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"  Trainable params: {trainable}")

        cil = accuracy_matrix[-1, :].mean()
        bwt = np.mean([accuracy_matrix[-1, j] - accuracy_matrix[j, j]
                       for j in range(n_tasks - 1)])

        logger.info(f"\n{'='*50}")
        logger.info("Results Summary:")
        logger.info(f"  CIL: {cil:.2f}%")
        logger.info(f"  BWT: {bwt:.2f}%")
        logger.info(f"  Accuracy Matrix:\n{accuracy_matrix}")
        logger.info(f"{'='*50}")

        self.assertTrue(np.isfinite(cil))
        self.assertTrue(np.isfinite(bwt))
        self.assertGreaterEqual(cil, 0.0)

if __name__ == '__main__':
    unittest.main()
