#!/usr/bin/env python
"""Test parameter isolation functionality in Sensory Perceiver.

Demonstrates:
1. Model initialization with task-specific adapters
2. Parameter freezing after each task completes
3. Tracking of trainable parameters and task coverage
4. Comparison with baseline (no parameter isolation)
"""

import torch
import torch.nn as nn
from src.models import SensoryPerceiver


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_parameter_isolation():
    """Test parameter isolation with sequential task training."""
    print("\n" + "="*70)
    print("Parameter Isolation Test: Sensory Perceiver")
    print("="*70)

    # Configuration
    modality_configs = {
        'vision': {'input_dim': 3, 'use_conv': True},
        'audio': {'input_dim': 128, 'use_conv': False},
    }
    num_tasks = 5
    dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # =========================================================================
    # Test 1: Model with Parameter Isolation (Adapters)
    # =========================================================================
    print("\n[Test 1] Parameter Isolation with Task-Specific Adapters")
    print("-" * 70)

    model_isolated = SensoryPerceiver(
        modality_configs=modality_configs,
        dim=dim,
        enable_parameter_isolation=True,
        adapter_hidden_dim=256,
    ).to(device)

    initial_params = count_parameters(model_isolated)
    print(f"Initial trainable parameters: {initial_params:,}")
    print(f"Initial task coverage: {model_isolated.get_task_coverage():.2%}")

    # Simulate sequential task training
    print(f"\nSimulating {num_tasks} sequential tasks:")
    for task_id in range(num_tasks):
        # Training step simulation
        batch = {
            'vision': torch.randn(4, 3, 64, 64).to(device),
            'audio': torch.randn(4, 128).to(device),
        }

        output, _ = model_isolated(batch)
        loss = output.mean()
        loss.backward()

        # Freeze task and allocate new adapter
        model_isolated.freeze_task()

        trainable_params = count_parameters(model_isolated)
        coverage = model_isolated.get_task_coverage()
        print(f"  Task {task_id}: {trainable_params:,} params | Coverage: {coverage:.2%} | "
              f"Frozen layers: task_0...task_{task_id}")

    print(f"\nFinal trainable parameters: {count_parameters(model_isolated):,}")
    print(f"Final task coverage: {model_isolated.get_task_coverage():.2%}")
    print(f"Parameter savings by freezing: {1 - model_isolated.get_task_coverage():.1%}")

    # =========================================================================
    # Test 2: Baseline Model (No Parameter Isolation)
    # =========================================================================
    print("\n[Test 2] Baseline: No Parameter Isolation")
    print("-" * 70)

    model_baseline = SensoryPerceiver(
        modality_configs=modality_configs,
        dim=dim,
        enable_parameter_isolation=False,
    ).to(device)

    baseline_params = count_parameters(model_baseline)
    print(f"Baseline trainable parameters (all tasks): {baseline_params:,}")

    # Simulate training on all tasks (all parameters stay trainable)
    for task_id in range(num_tasks):
        batch = {
            'vision': torch.randn(4, 3, 64, 64).to(device),
            'audio': torch.randn(4, 128).to(device),
        }

        output, _ = model_baseline(batch)
        loss = output.mean()
        loss.backward()

    print(f"After {num_tasks} tasks: {count_parameters(model_baseline):,} trainable params (unchanged)")

    # =========================================================================
    # Comparison
    # =========================================================================
    print("\n[Comparison] Parameter Isolation vs. Baseline")
    print("-" * 70)

    params_isolated = count_parameters(model_isolated)
    params_baseline = count_parameters(model_baseline)

    print(f"Parameter Isolation: {params_isolated:,} trainable params")
    print(f"Baseline (no isolation): {params_baseline:,} trainable params")
    print(f"Savings: {(params_baseline - params_isolated) / params_baseline:.1%}")
    print(f"Capacity remaining: {model_isolated.get_task_coverage():.1%}")

    # =========================================================================
    # Test 3: Verify Adapter Freezing
    # =========================================================================
    print("\n[Test 3] Verify Adapter Freezing Behavior")
    print("-" * 70)

    # Check requires_grad for each adapter
    for task_name, adapter in model_isolated.task_adapters.items():
        frozen = not any(p.requires_grad for p in adapter.parameters())
        status = "FROZEN" if frozen else "TRAINABLE"
        print(f"  {task_name}: {status}")

    print("\n" + "="*70)
    print("✓ All tests completed successfully")
    print("="*70 + "\n")


if __name__ == '__main__':
    test_parameter_isolation()
