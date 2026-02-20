"""
Ablation Study Configurations

Defines all variant configurations needed to validate each component
of SensoryPerceiver and compare against baselines.

Usage:
    python -m src.train --experiment continual --dataset split_cifar --output_dir results/ablation_replay_off
    python -m src.train --experiment continual --dataset split_cifar --output_dir results/ablation_mime_off
    python -m src.train --experiment continual --dataset split_cifar --output_dir results/ablation_efficiency_small
"""

from dataclasses import dataclass, asdict
from typing import Dict, List
import json


@dataclass
class AblationConfig:
    """Single ablation configuration."""

    name: str  # Unique identifier
    experiment: str  # Experiment type (continual, sensor_streaming, multimodal)
    dataset: str  # Dataset name
    description: str  # Human-readable description
    hypothesis: str  # What we're testing

    # Model modifications
    d_model: int = 768
    num_heads: int = 8
    depth: int = 6
    mlp_dim: int = 2048
    dropout: float = 0.1

    # Training modifications
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    seed: int = 0

    # Ablation-specific flags
    use_replay: bool = True
    use_modality_routing: bool = True
    replay_buffer_size: int = 2000
    contrastive_temp: float = 0.07

    # Expected outcomes (for validation)
    expected_accuracy_range: tuple = (0.70, 0.95)
    expected_latency_ms: float = 20.0


class AblationStudy:
    """Manages all ablation configurations."""

    # ============================================================================
    # ABLATION 1: REPLAY MECHANISM
    # ============================================================================

    ABLATION_1_WITH_REPLAY = AblationConfig(
        name="ablation_1_with_replay",
        experiment="continual",
        dataset="split_cifar",
        description="SensoryPerceiver WITH replay mechanism (control)",
        hypothesis="Replay mechanism prevents catastrophic forgetting",
        epochs=50,
        use_replay=True,
        replay_buffer_size=2000,
        expected_accuracy_range=(0.84, 0.86),
    )

    ABLATION_1_WITHOUT_REPLAY = AblationConfig(
        name="ablation_1_without_replay",
        experiment="continual",
        dataset="split_cifar",
        description="SensoryPerceiver WITHOUT replay mechanism",
        hypothesis="Without replay, catastrophic forgetting dominates",
        epochs=50,
        use_replay=False,
        replay_buffer_size=0,
        expected_accuracy_range=(0.76, 0.80),  # Expect lower accuracy
    )

    # ============================================================================
    # ABLATION 2: MIME-TYPE ROUTING
    # ============================================================================

    ABLATION_2_WITH_MIME = AblationConfig(
        name="ablation_2_with_modality_routing",
        experiment="continual",
        dataset="split_cifar",
        description="SensoryPerceiver WITH MIME-Type routing (control)",
        hypothesis="MIME-Type routing outperforms generic latent bottleneck",
        epochs=50,
        use_modality_routing=True,
        expected_accuracy_range=(0.84, 0.86),
        expected_latency_ms=12.0,
    )

    ABLATION_2_WITHOUT_MIME = AblationConfig(
        name="ablation_2_without_modality_routing",
        experiment="continual",
        dataset="split_cifar",
        description="SensoryPerceiver with generic latent bottleneck (like Perceiver IO)",
        hypothesis="Generic bottleneck loses modality-specific information",
        epochs=50,
        use_modality_routing=False,  # Flag to disable modality routing in trainer
        expected_accuracy_range=(0.80, 0.84),  # Expect lower accuracy
        expected_latency_ms=18.0,  # Expect slower (more computation needed)
    )

    # ============================================================================
    # ABLATION 3: EFFICIENCY FRONTIER
    # ============================================================================

    ABLATION_3_EFFICIENCY_SMALL = AblationConfig(
        name="ablation_3_efficiency_small",
        experiment="continual",
        dataset="split_cifar",
        description="SensoryPerceiver - Small (100M params)",
        hypothesis="Small model still beats Perceiver IO on continual learning",
        d_model=512,
        num_heads=6,
        depth=4,
        mlp_dim=1024,
        batch_size=32,
        epochs=50,
        expected_accuracy_range=(0.80, 0.84),  # Slightly lower than full
    )

    ABLATION_3_EFFICIENCY_MEDIUM = AblationConfig(
        name="ablation_3_efficiency_medium",
        experiment="continual",
        dataset="split_cifar",
        description="SensoryPerceiver - Medium (234M params, standard)",
        hypothesis="Medium is sweet spot for efficiency/accuracy tradeoff",
        d_model=768,
        num_heads=8,
        depth=6,
        mlp_dim=2048,
        epochs=50,
        expected_accuracy_range=(0.84, 0.86),  # Standard performance
    )

    ABLATION_3_EFFICIENCY_LARGE = AblationConfig(
        name="ablation_3_efficiency_large",
        experiment="continual",
        dataset="split_cifar",
        description="SensoryPerceiver - Large (400M params)",
        hypothesis="Larger model provides diminishing accuracy gains",
        d_model=1024,
        num_heads=8,
        depth=8,
        mlp_dim=3072,
        batch_size=24,  # Reduce batch size due to memory constraints
        epochs=50,
        expected_accuracy_range=(0.85, 0.87),  # Marginal improvement
    )

    # ============================================================================
    # BONUS: MULTIMODAL ABLATIONS (if time permits)
    # ============================================================================

    ABLATION_4_MULTIMODAL_NO_REPLAY = AblationConfig(
        name="ablation_4_multimodal_no_replay",
        experiment="sensor_streaming",
        dataset="sensor_streaming",
        description="Multimodal IoT WITHOUT replay mechanism",
        hypothesis="Replay helps even in single-pass scenarios",
        epochs=50,
        use_replay=False,
        expected_accuracy_range=(0.88, 0.91),  # Slightly lower
    )

    ABLATION_5_ABLATION_DEPTH_STUDY = AblationConfig(
        name="ablation_5_depth_2",
        experiment="continual",
        dataset="split_cifar",
        description="SensoryPerceiver with depth=2 (minimal)",
        hypothesis="How much does depth matter?",
        d_model=768,
        depth=2,
        epochs=50,
        expected_accuracy_range=(0.80, 0.83),
    )

    @classmethod
    def get_all_configs(cls) -> List[AblationConfig]:
        """Return all ablation configurations."""
        return [
            # Primary ablations
            cls.ABLATION_1_WITH_REPLAY,
            cls.ABLATION_1_WITHOUT_REPLAY,
            cls.ABLATION_2_WITH_MIME,
            cls.ABLATION_2_WITHOUT_MIME,
            cls.ABLATION_3_EFFICIENCY_SMALL,
            cls.ABLATION_3_EFFICIENCY_MEDIUM,
            cls.ABLATION_3_EFFICIENCY_LARGE,
            # Bonus ablations
            cls.ABLATION_4_MULTIMODAL_NO_REPLAY,
            cls.ABLATION_5_ABLATION_DEPTH_STUDY,
        ]

    @classmethod
    def get_config_by_name(cls, name: str) -> AblationConfig:
        """Get config by name."""
        for config in cls.get_all_configs():
            if config.name == name:
                return config
        raise ValueError(f"Config not found: {name}")

    @classmethod
    def print_summary(cls):
        """Print summary of all ablation studies."""
        print("\n" + "=" * 80)
        print("ABLATION STUDY SUMMARY")
        print("=" * 80)

        ablations = {
            "ABLATION 1: Replay Mechanism Impact": [
                cls.ABLATION_1_WITH_REPLAY,
                cls.ABLATION_1_WITHOUT_REPLAY,
            ],
            "ABLATION 2: MIME-Type Routing Impact": [
                cls.ABLATION_2_WITH_MIME,
                cls.ABLATION_2_WITHOUT_MIME,
            ],
            "ABLATION 3: Efficiency Frontier": [
                cls.ABLATION_3_EFFICIENCY_SMALL,
                cls.ABLATION_3_EFFICIENCY_MEDIUM,
                cls.ABLATION_3_EFFICIENCY_LARGE,
            ],
        }

        total_gpu_hours = 0

        for title, configs in ablations.items():
            print(f"\n{title}")
            print("-" * 80)

            for config in configs:
                # Rough GPU time estimate
                gpu_hours = (config.epochs * 50) / 50  # Assuming 50 epochs = 50 min
                total_gpu_hours += gpu_hours

                print(f"  {config.name}")
                print(f"    Description: {config.description}")
                print(f"    Hypothesis:  {config.hypothesis}")
                print(f"    Expected:    {config.expected_accuracy_range[0]*100:.0f}%-{config.expected_accuracy_range[1]*100:.0f}%")
                print(f"    GPU time:    ~{gpu_hours:.1f}h")

        print("\n" + "=" * 80)
        print(f"TOTAL ABLATION GPU TIME: ~{total_gpu_hours:.1f} hours")
        print(f"(Can run in parallel on 3 GPUs in ~{total_gpu_hours/3:.1f} hours)")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Print summary
    AblationStudy.print_summary()

    # Example: Get a specific config
    config = AblationStudy.get_config_by_name("ablation_1_with_replay")
    print(f"\nExample config: {config.name}")
    print(f"  Dataset: {config.dataset}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Use replay: {config.use_replay}")

    # Export as JSON for reference
    all_configs = AblationStudy.get_all_configs()
    config_dicts = [asdict(cfg) for cfg in all_configs]

    with open("ablation_configs.json", "w") as f:
        json.dump(config_dicts, f, indent=2)
    print(f"\nExported {len(config_dicts)} configs to ablation_configs.json")
