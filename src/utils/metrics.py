"""
Continual Learning and Performance Metrics

Computes standard CL benchmarks:
- Class-Incremental Learning (CIL) accuracy
- Backward Transfer (BWT)
- Forward Transfer (FWT)
- Task accuracy
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_task_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    task_ids: torch.Tensor
) -> Dict[int, float]:
    """
    Compute per-task accuracy.
    
    Args:
        outputs: Model predictions [N, num_classes]
        targets: Ground truth labels [N]
        task_ids: Task identifiers [N]
    
    Returns:
        Dictionary mapping task_id -> accuracy
    """
    accuracies = {}
    for task_id in torch.unique(task_ids):
        mask = task_ids == task_id
        task_correct = (outputs[mask].argmax(1) == targets[mask]).float().mean()
        accuracies[int(task_id)] = float(task_correct)
    return accuracies


def compute_backward_forward_transfer(
    all_task_accuracies: List[Dict[int, float]]
) -> Tuple[float, float]:
    """
    Compute Backward Transfer (BWT) and Forward Transfer (FWT).
    
    BWT = mean(A[i,j] - A[j,j]) for i > j
        Negative = catastrophic forgetting
    
    FWT = mean(A[i,i] - baseline[i]) for i > 0
        How much learning task i helps on future tasks
    
    Args:
        all_task_accuracies: List of accuracy dicts, one per training step
    
    Returns:
        (BWT, FWT) scores
    """
    num_tasks = len(all_task_accuracies)
    bwt_scores = []
    fwt_scores = []
    
    for i in range(1, num_tasks):
        for j in range(i):
            # BWT[i,j] = acc after learning task i - acc after learning task j
            bwt_scores.append(
                all_task_accuracies[i].get(j, 0.0) -
                all_task_accuracies[j].get(j, 0.0)
            )
    
    bwt = np.mean(bwt_scores) if bwt_scores else 0.0
    
    # FWT: improvement on future tasks
    # For task i, FWT = final accuracy - initial accuracy before any training
    for i in range(num_tasks):
        if i < len(all_task_accuracies) - 1:
            # Accuracy on task i after learning all tasks minus initial
            fwt_scores.append(
                all_task_accuracies[-1].get(i, 0.0) - 
                all_task_accuracies[0].get(i, 0.0)
            )
    
    fwt = np.mean(fwt_scores) if fwt_scores else 0.0
    
    return float(bwt), float(fwt)


def compute_continual_learning_metrics(
    all_task_accuracies: List[Dict[int, float]]
) -> Dict[str, float]:
    """
    Compute all standard CL metrics.
    
    Args:
        all_task_accuracies: List of accuracy dicts over time
    
    Returns:
        Dictionary with keys: 'cil', 'bwt', 'fwt', 'final_accuracy'
    """
    num_tasks = len(all_task_accuracies)
    
    # Final task accuracy (Class-Incremental Learning metric)
    final_accs = list(all_task_accuracies[-1].values())
    cil = np.mean(final_accs) if final_accs else 0.0
    
    # Backward and Forward Transfer
    bwt, fwt = compute_backward_forward_transfer(all_task_accuracies)
    
    return {
        'cil': float(cil),
        'bwt': float(bwt),
        'fwt': float(fwt),
        'num_tasks': num_tasks,
    }
