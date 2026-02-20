"""CLI entrypoint for experiment orchestration."""

import logging
import sys
from argparse import ArgumentParser

from src.runtime import AppError, OrchestratorRuntimeConfig, format_error
from src.services import run_orchestration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("experiment_orchestrator.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def build_parser() -> ArgumentParser:
    """Build the orchestrator CLI parser."""
    parser = ArgumentParser(description="Sensory Perceiver Experiment Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "monitor"],
        default="quick",
        help="Experiment mode",
    )
    parser.add_argument("--n-gpus", type=int, default=3, help="Number of available GPUs")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (one per GPU)",
    )
    parser.add_argument(
        "--push-to-github",
        action="store_true",
        help="Auto-commit results to GitHub",
    )
    return parser


def main():
    """Validate runtime config and execute orchestration service."""
    runtime_config = OrchestratorRuntimeConfig.from_namespace(build_parser().parse_args())
    logger.info("Sensory Perceiver Experiment Orchestrator")
    logger.info(
        f"Mode: {runtime_config.mode} | GPUs: {runtime_config.n_gpus} | "
        f"Parallel: {runtime_config.parallel}"
    )
    dashboard_path = run_orchestration(runtime_config)
    if dashboard_path is not None:
        logger.info(f"Open monitor: file://{dashboard_path}")


if __name__ == "__main__":
    try:
        main()
    except AppError as error:
        logger.error(format_error(error))
        raise SystemExit(2)
