"""CLI entrypoint for training Sensory Perceiver models."""

import logging
from argparse import ArgumentParser

import torch

from src.runtime import AppError, TrainRuntimeConfig, format_error, validate_environment
from src.services import run_training

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_parser() -> ArgumentParser:
    """Build the training CLI parser."""
    parser = ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        default="continual",
        choices=["multimodal", "continual", "sensor_streaming"],
    )
    parser.add_argument("--dataset", type=str, default="split_cifar")

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--mlp_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--replay_buffer_size", type=int, default=2000)
    parser.add_argument("--contrastive_temp", type=float, default=0.07)
    parser.add_argument(
        "--replay_mode", type=str, default="er", choices=["er", "contrastive", "feature"]
    )
    parser.add_argument(
        "--adapter_rollover",
        action="store_true",
        help="Enable task-adapter freeze/rollover between continual tasks.",
    )
    parser.add_argument(
        "--continual_task_count",
        type=int,
        default=1,
        help="Number of continual tasks to run sequentially when adapter rollover is enabled.",
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="results/exp")
    return parser


def main():
    """Validate runtime config and execute training service."""
    validate_environment()
    runtime_config = TrainRuntimeConfig.from_namespace(build_parser().parse_args())
    logger.info(f"Using device: {runtime_config.device}")
    result = run_training(runtime_config)
    logger.info(f"Training complete. Results saved to {result.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except AppError as error:
        logger.error(format_error(error))
        raise SystemExit(2)
