"""Experiment orchestration service."""

import json
import logging
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.runtime import (
    AppError,
    ErrorCode,
    OrchestratorRuntimeConfig,
    validate_experiment_dataset,
)

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration for an experiment run."""

    def __init__(self, name: str, dataset: str, gpu_id: int, epochs: int = 50):
        self.name = name
        self.dataset = dataset
        self.gpu_id = gpu_id
        self.epochs = epochs
        self.output_dir = Path(f"results/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.status = "pending"
        self.start_time = None
        self.end_time = None
        self.result = None


class ExperimentOrchestrator:
    """Manages experiment execution, GPU allocation, and monitoring."""

    def __init__(self, n_gpus: int = 3, monitor_interval: int = 30):
        self.n_gpus = n_gpus
        self.monitor_interval = monitor_interval
        self.experiments: List[ExperimentConfig] = []
        self.active_experiments: Dict[int, ExperimentConfig] = {}
        self.results_file = Path("results/experiment_log.json")
        self._results_lock = threading.Lock()

    def _resolve_experiment_for_dataset(self, dataset: str) -> str:
        """Resolve canonical experiment id for a dataset."""
        if dataset == "iot":
            experiment = "iot"
        elif dataset in {"split_cifar", "permuted_mnist"}:
            experiment = "continual"
        elif dataset == "audiovisual":
            experiment = "mime_routing"
        else:
            raise AppError(
                ErrorCode.DATASET_MISMATCH,
                f"Unsupported dataset for orchestrator: {dataset}",
                context={"dataset": dataset},
            )
        validate_experiment_dataset(experiment, dataset)
        return experiment

    def add_experiment(
        self, name: str, dataset: str, gpu_id: Optional[int] = None, epochs: int = 50
    ):
        """Add experiment to queue."""
        self._resolve_experiment_for_dataset(dataset)
        if gpu_id is None:
            gpu_id = len(self.experiments) % self.n_gpus
        exp = ExperimentConfig(name, dataset, gpu_id, epochs)
        self.experiments.append(exp)
        logger.info(f"Added experiment: {name} (GPU {gpu_id}, dataset {dataset})")

    def queue_full_benchmark(self):
        """Queue full benchmark: 3 datasets × 3 seeds = 9 runs."""
        datasets = [
            ("synthetic_iot", "iot", 50),
            ("split_cifar", "split_cifar", 50),
            ("permuted_mnist", "permuted_mnist", 50),
        ]

        for dataset_name, dataset_code, epochs in datasets:
            for seed in range(3):
                exp_name = f"{dataset_name}_seed{seed}"
                gpu_id = len(self.experiments) % self.n_gpus
                exp = ExperimentConfig(exp_name, dataset_code, gpu_id, epochs)
                exp.seed = seed
                self.experiments.append(exp)
                logger.info(f"Queued {exp_name} on GPU {gpu_id}")

    def queue_quick_test(self):
        """Queue quick test: 1 synthetic experiment."""
        exp = ExperimentConfig("quick_test_iot", "iot", gpu_id=0, epochs=10)
        self.experiments.append(exp)
        logger.info("Queued quick test on GPU 0")

    def run_experiment(self, exp: ExperimentConfig) -> bool:
        """Execute a single experiment."""
        exp.status = "running"
        exp.start_time = datetime.now()
        experiment_name = self._resolve_experiment_for_dataset(exp.dataset)

        cmd = [
            "python",
            "-m",
            "src.train",
            "--experiment",
            experiment_name,
            "--dataset",
            exp.dataset,
            "--epochs",
            str(exp.epochs),
            "--batch_size",
            "32",
            "--device",
            f"cuda:{exp.gpu_id}",
            "--output_dir",
            str(exp.output_dir),
        ]
        if hasattr(exp, "seed"):
            cmd.extend(["--seed", str(exp.seed)])

        logger.info(f"Starting: {exp.name} (GPU {exp.gpu_id})")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=24 * 3600,
            )
            if result.returncode == 0:
                logger.info(f"✓ Completed: {exp.name}")
                exp.status = "completed"
                exp.result = self._run_evaluation(exp)
                return True
            logger.error(f"✗ Failed: {exp.name}")
            logger.error(f"STDOUT: {result.stdout[-500:]}")
            logger.error(f"STDERR: {result.stderr[-500:]}")
            exp.status = "failed"
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout: {exp.name}")
            exp.status = "failed"
            return False
        except Exception as exc:
            logger.error(f"✗ Exception in {exp.name}: {exc}")
            exp.status = "failed"
            return False
        finally:
            exp.end_time = datetime.now()
            self._save_results()

    def _run_evaluation(self, exp: ExperimentConfig) -> Dict:
        """Run evaluation on completed experiment."""
        checkpoint = exp.output_dir / "best_model.pt"
        if not checkpoint.exists():
            logger.warning(f"Checkpoint not found: {checkpoint}")
            return {}
        experiment_name = self._resolve_experiment_for_dataset(exp.dataset)

        cmd = [
            "python",
            "-m",
            "src.eval",
            "--checkpoint",
            str(checkpoint),
            "--experiment",
            experiment_name,
            "--dataset",
            exp.dataset,
            "--device",
            f"cuda:{exp.gpu_id}",
            "--output_file",
            str(exp.output_dir / "eval_results.json"),
        ]
        logger.info(f"Evaluating: {exp.name}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                results_file = exp.output_dir / "eval_results.json"
                if results_file.exists():
                    with open(results_file) as file:
                        return json.load(file)
            else:
                logger.error(f"Evaluation failed for {exp.name}")
                return {}
        except Exception as exc:
            logger.error(f"Evaluation error for {exp.name}: {exc}")
            return {}
        return {}

    def run_all(self, parallel: bool = False):
        """Execute all queued experiments."""
        if not self.experiments:
            logger.warning("No experiments queued")
            return
        logger.info(f"Starting experiment run: {len(self.experiments)} experiments")
        if parallel:
            self._run_parallel()
        else:
            self._run_sequential()
        logger.info("All experiments completed!")
        self._print_summary()

    def _run_sequential(self):
        """Run experiments one at a time."""
        for index, exp in enumerate(self.experiments):
            logger.info(f"\n[{index + 1}/{len(self.experiments)}] Running {exp.name}")
            self.run_experiment(exp)
            time.sleep(5)

    def _run_parallel(self):
        """Run experiments in parallel (one queue per GPU)."""
        gpu_queues = {gpu_id: [] for gpu_id in range(self.n_gpus)}
        for exp in self.experiments:
            gpu_id = exp.gpu_id % self.n_gpus
            exp.gpu_id = gpu_id
            gpu_queues[gpu_id].append(exp)

        def run_gpu_queue(gpu_id: int, queue: List[ExperimentConfig]):
            for exp in queue:
                logger.info(f"\n[GPU {gpu_id}] Running {exp.name}")
                self.run_experiment(exp)
                time.sleep(5)

        with ThreadPoolExecutor(max_workers=self.n_gpus) as executor:
            futures = [
                executor.submit(run_gpu_queue, gpu_id, queue)
                for gpu_id, queue in gpu_queues.items()
                if queue
            ]
            for future in as_completed(futures):
                future.result()

    def _save_results(self):
        """Save experiment results to JSON."""
        with self._results_lock:
            results = []
            for exp in self.experiments:
                results.append(
                    {
                        "name": exp.name,
                        "dataset": exp.dataset,
                        "gpu_id": exp.gpu_id,
                        "status": exp.status,
                        "start_time": (
                            exp.start_time.isoformat() if exp.start_time else None
                        ),
                        "end_time": exp.end_time.isoformat() if exp.end_time else None,
                        "duration_hours": (
                            (exp.end_time - exp.start_time).total_seconds() / 3600
                            if exp.start_time and exp.end_time
                            else None
                        ),
                        "output_dir": str(exp.output_dir),
                        "result": exp.result,
                    }
                )
            with open(self.results_file, "w") as file:
                json.dump(results, file, indent=2)

    def _print_summary(self):
        """Print summary of all experiments."""
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 70)

        completed = [exp for exp in self.experiments if exp.status == "completed"]
        failed = [exp for exp in self.experiments if exp.status == "failed"]

        logger.info(f"Total: {len(self.experiments)}")
        logger.info(f"✓ Completed: {len(completed)}")
        logger.info(f"✗ Failed: {len(failed)}")

        if completed:
            total_hours = sum(
                (exp.end_time - exp.start_time).total_seconds() / 3600
                for exp in completed
            )
            avg_acc = (
                np.mean(
                    [exp.result.get("overall_accuracy", 0) for exp in completed if exp.result]
                )
                if completed
                else 0
            )
            logger.info(f"\nCompute time: {total_hours:.1f} GPU-hours")
            logger.info(f"Avg accuracy: {avg_acc * 100:.2f}%")

        if failed:
            logger.warning("\nFailed experiments:")
            for exp in failed:
                logger.warning(f"  - {exp.name}")
        logger.info("=" * 70)

    def generate_dashboard(self):
        """Generate HTML monitoring dashboard."""
        dashboard_file = Path("monitor_dashboard.html")
        results = []
        if self.results_file.exists():
            with open(self.results_file) as file:
                results = json.load(file)

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Sensory Perceiver Experiment Monitor</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; }}
        .summary {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .stat {{ background: #e8f5e9; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat.failed {{ background: #ffebee; }}
        .stat.running {{ background: #fff3e0; }}
        .stat h3 {{ margin: 0; color: #666; font-size: 12px; }}
        .stat .number {{ font-size: 32px; font-weight: bold; color: #2e7d32; margin: 10px 0; }}
        .stat.failed .number {{ color: #c62828; }}
        .stat.running .number {{ color: #f57c00; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #f5f5f5; padding: 10px; text-align: left; border-bottom: 2px solid #ddd; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f9f9f9; }}
        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-pending {{ background: #e0e0e0; color: #666; }}
        .status-running {{ background: #fff3e0; color: #f57c00; }}
        .status-completed {{ background: #c8e6c9; color: #2e7d32; }}
        .status-failed {{ background: #ffcdd2; color: #c62828; }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 3px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: #4caf50;
            width: 0%;
            transition: width 0.3s;
        }}
        .timestamp {{ color: #999; font-size: 12px; }}
        .footer {{ text-align: center; color: #999; margin-top: 40px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sensory Perceiver Experiment Monitor</h1>
        <p class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div class="summary">
            <div class="stat">
                <h3>Total Experiments</h3>
                <div class="number">{len(results)}</div>
            </div>
            <div class="stat">
                <h3>Completed</h3>
                <div class="number">{len([r for r in results if r['status'] == 'completed'])}</div>
            </div>
            <div class="stat failed">
                <h3>Failed</h3>
                <div class="number">{len([r for r in results if r['status'] == 'failed'])}</div>
            </div>
        </div>
        <h2>Experiment Status</h2>
        <table>
            <tr>
                <th>Experiment</th>
                <th>Dataset</th>
                <th>Status</th>
                <th>Start Time</th>
                <th>Duration (h)</th>
                <th>Accuracy</th>
            </tr>
"""
        for result in results:
            status_class = f"status-{result['status']}"
            duration = (
                f"{result['duration_hours']:.2f}" if result["duration_hours"] else "—"
            )
            accuracy = (
                f"{result['result'].get('overall_accuracy', 0)*100:.2f}%"
                if result.get("result")
                else "—"
            )
            html_content += f"""
            <tr>
                <td><strong>{result['name']}</strong></td>
                <td>{result['dataset']}</td>
                <td><span class="status-badge {status_class}">{result['status'].upper()}</span></td>
                <td>{result.get('start_time', '—')[:10]}</td>
                <td>{duration}</td>
                <td>{accuracy}</td>
            </tr>
"""
        html_content += """
        </table>
        <h2>Progress</h2>
"""
        if results:
            total = len(results)
            completed = len([r for r in results if r["status"] == "completed"])
            progress_pct = (completed / total) * 100
            html_content += f"""
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_pct}%"></div>
            </div>
            <p>{completed}/{total} experiments completed ({progress_pct:.0f}%)</p>
"""
        html_content += """
        <div class="footer">
            <p>Auto-refreshes every 30 seconds | View results: <code>results/</code></p>
        </div>
    </div>
</body>
</html>
"""
        with open(dashboard_file, "w") as file:
            file.write(html_content)
        logger.info(f"Dashboard saved to {dashboard_file}")
        return str(dashboard_file.absolute())


def run_orchestration(runtime_config: OrchestratorRuntimeConfig) -> Optional[str]:
    """Execute orchestrator flow from validated runtime config."""
    orchestrator = ExperimentOrchestrator(n_gpus=runtime_config.n_gpus)
    if runtime_config.mode == "full":
        orchestrator.queue_full_benchmark()
    elif runtime_config.mode == "quick":
        orchestrator.queue_quick_test()
    elif runtime_config.mode == "monitor":
        return orchestrator.generate_dashboard()

    dashboard_path = orchestrator.generate_dashboard()
    logger.info(f"Open monitor: file://{dashboard_path}")
    logger.info(f"Running {len(orchestrator.experiments)} experiments...")
    orchestrator.run_all(parallel=runtime_config.parallel)

    if runtime_config.push_to_github:
        logger.info("Pushing results to GitHub...")
        subprocess.run(["python", "scripts/push_results.py"], check=False)
    return dashboard_path
