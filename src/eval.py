"""CLI entrypoint for evaluating Sensory Perceiver checkpoints."""

import logging
from argparse import ArgumentParser

import torch

from src.runtime import AppError, EvalRuntimeConfig, format_error, validate_environment
from src.services import run_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_parser() -> ArgumentParser:
    """Build the evaluation CLI parser."""
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="continual",
        choices=["multimodal", "continual", "sensor_streaming"],
    )
    parser.add_argument("--dataset", type=str, default="split_cifar")
    parser.add_argument(
        "--modality_routing_mode",
        type=str,
        default="canonical",
        choices=["canonical", "merged"],
        help="Sensor-streaming routing mode: canonical typed modalities or one merged generic stream.",
    )
    parser.add_argument(
        "--sensor_variant",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "drop_smartwatch",
            "drop_room_env",
            "drop_robot_arm",
            "corrupt_bindings",
            "noise_smartwatch",
            "noise_room_env",
            "noise_robot_arm",
        ],
        help="Sensor-streaming ablation variant.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    """Validate runtime config and execute evaluation service."""
    validate_environment()
    runtime_config = EvalRuntimeConfig.from_namespace(build_parser().parse_args())
    logger.info(f"Using device: {runtime_config.device}")
    logger.info(f"Loading checkpoint from {runtime_config.checkpoint}")
    result = run_evaluation(runtime_config)
    logger.info(f"Results saved to {result.output_file.parent}")
    logger.info(f"Overall Accuracy: {result.results['overall_accuracy']:.4f}")
    latency_mean = result.results.get("latency_mean")
    if latency_mean is not None:
        logger.info(f"Mean Latency: {latency_mean:.2f} ms")


if __name__ == "__main__":
    try:
        main()
    except AppError as error:
        logger.error(format_error(error))
        raise SystemExit(2)
