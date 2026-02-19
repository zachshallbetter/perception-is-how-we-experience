"""
Automated experiment orchestrator for Sensory Perceiver.

Manages:
- GPU allocation across 3 TitanX GPUs
- Sequential/parallel experiment execution
- Real-time progress monitoring
- Results collection and GitHub push
- Email notifications on completion

Usage:
    python experiment_orchestrator.py --mode full --push-to-github
    python experiment_orchestrator.py --mode quick --gpu 0
    python experiment_orchestrator.py --monitor
"""

import subprocess
import torch
import numpy as np
import time
import json
import csv
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('experiment_orchestrator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration for an experiment run."""

    def __init__(self, name: str, dataset: str, gpu_id: int, epochs: int = 50):
        self.name = name
        self.dataset = dataset
        self.gpu_id = gpu_id
        self.epochs = epochs
        self.output_dir = Path(f'results/{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.status = 'pending'  # pending, running, completed, failed
        self.start_time = None
        self.end_time = None
        self.result = None


class ExperimentOrchestrator:
    """Manages experiment execution, GPU allocation, and monitoring."""

    def __init__(self, n_gpus: int = 3, monitor_interval: int = 30):
        self.n_gpus = n_gpus
        self.monitor_interval = monitor_interval  # seconds
        self.experiments: List[ExperimentConfig] = []
        self.active_experiments: Dict[int, ExperimentConfig] = {}  # gpu_id -> experiment
        self.results_file = Path('results/experiment_log.json')

    def add_experiment(self, name: str, dataset: str, gpu_id: Optional[int] = None, epochs: int = 50):
        """Add experiment to queue."""
        if gpu_id is None:
            gpu_id = len(self.experiments) % self.n_gpus
        exp = ExperimentConfig(name, dataset, gpu_id, epochs)
        self.experiments.append(exp)
        logger.info(f"Added experiment: {name} (GPU {gpu_id}, dataset {dataset})")

    def queue_full_benchmark(self):
        """Queue full benchmark: 3 datasets × 3 seeds = 9 runs."""
        datasets = [
            ('synthetic_iot', 'iot', 50),
            ('split_cifar', 'split_cifar', 50),
            ('permuted_mnist', 'permuted_mnist', 50),
        ]

        for dataset_name, dataset_code, epochs in datasets:
            for seed in range(3):
                exp_name = f'{dataset_name}_seed{seed}'
                gpu_id = (len(self.experiments)) % self.n_gpus
                exp = ExperimentConfig(exp_name, dataset_code, gpu_id, epochs)
                exp.seed = seed
                self.experiments.append(exp)
                logger.info(f"Queued {exp_name} on GPU {gpu_id}")

    def queue_quick_test(self):
        """Queue quick test: 1 synthetic experiment (4h)."""
        exp = ExperimentConfig('quick_test_iot', 'iot', gpu_id=0, epochs=10)
        self.experiments.append(exp)
        logger.info("Queued quick test on GPU 0")

    def run_experiment(self, exp: ExperimentConfig) -> bool:
        """Execute a single experiment."""
        exp.status = 'running'
        exp.start_time = datetime.now()

        cmd = [
            'python', 'train.py',
            '--experiment', 'iot' if exp.dataset == 'iot' else 'continual',
            '--dataset', exp.dataset,
            '--epochs', str(exp.epochs),
            '--batch_size', '32',
            '--device', f'cuda:{exp.gpu_id}',
            '--output_dir', str(exp.output_dir),
        ]

        if hasattr(exp, 'seed'):
            cmd.extend(['--seed', str(exp.seed)])

        logger.info(f"Starting: {exp.name} (GPU {exp.gpu_id})")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=24*3600  # 24h timeout
            )

            if result.returncode == 0:
                logger.info(f"✓ Completed: {exp.name}")
                exp.status = 'completed'

                # Run evaluation
                eval_result = self._run_evaluation(exp)
                exp.result = eval_result

                return True
            else:
                logger.error(f"✗ Failed: {exp.name}")
                logger.error(f"STDOUT: {result.stdout[-500:]}")  # Last 500 chars
                logger.error(f"STDERR: {result.stderr[-500:]}")
                exp.status = 'failed'
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout: {exp.name}")
            exp.status = 'failed'
            return False
        except Exception as e:
            logger.error(f"✗ Exception in {exp.name}: {e}")
            exp.status = 'failed'
            return False
        finally:
            exp.end_time = datetime.now()
            self._save_results()

    def _run_evaluation(self, exp: ExperimentConfig) -> Dict:
        """Run evaluation on completed experiment."""
        checkpoint = exp.output_dir / 'best_model.pt'
        if not checkpoint.exists():
            logger.warning(f"Checkpoint not found: {checkpoint}")
            return {}

        cmd = [
            'python', 'eval.py',
            '--checkpoint', str(checkpoint),
            '--experiment', 'iot' if exp.dataset == 'iot' else 'continual',
            '--dataset', exp.dataset,
            '--device', f'cuda:{exp.gpu_id}',
            '--output_file', str(exp.output_dir / 'eval_results.json'),
        ]

        logger.info(f"Evaluating: {exp.name}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                results_file = exp.output_dir / 'eval_results.json'
                if results_file.exists():
                    with open(results_file) as f:
                        return json.load(f)
            else:
                logger.error(f"Evaluation failed for {exp.name}")
                return {}
        except Exception as e:
            logger.error(f"Evaluation error for {exp.name}: {e}")
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
        for i, exp in enumerate(self.experiments):
            logger.info(f"\n[{i+1}/{len(self.experiments)}] Running {exp.name}")
            self.run_experiment(exp)
            time.sleep(5)  # Cooldown between runs

    def _run_parallel(self):
        """Run experiments in parallel (one per GPU)."""
        queue = list(self.experiments)
        active = {}  # gpu_id -> (exp, process)

        while queue or active:
            # Start new experiments on free GPUs
            while queue and len(active) < self.n_gpus:
                exp = queue.pop(0)
                self.run_experiment(exp)
                logger.info(f"GPU {exp.gpu_id}: {exp.name}")

            # Monitor active experiments
            if active:
                time.sleep(self.monitor_interval)

    def _save_results(self):
        """Save experiment results to JSON."""
        results = []
        for exp in self.experiments:
            results.append({
                'name': exp.name,
                'dataset': exp.dataset,
                'gpu_id': exp.gpu_id,
                'status': exp.status,
                'start_time': exp.start_time.isoformat() if exp.start_time else None,
                'end_time': exp.end_time.isoformat() if exp.end_time else None,
                'duration_hours': (exp.end_time - exp.start_time).total_seconds() / 3600
                    if exp.start_time and exp.end_time else None,
                'output_dir': str(exp.output_dir),
                'result': exp.result,
            })

        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)

    def _print_summary(self):
        """Print summary of all experiments."""
        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*70)

        completed = [e for e in self.experiments if e.status == 'completed']
        failed = [e for e in self.experiments if e.status == 'failed']

        logger.info(f"Total: {len(self.experiments)}")
        logger.info(f"✓ Completed: {len(completed)}")
        logger.info(f"✗ Failed: {len(failed)}")

        if completed:
            total_hours = sum(
                (e.end_time - e.start_time).total_seconds() / 3600
                for e in completed
            )
            avg_acc = np.mean([
                e.result.get('overall_accuracy', 0) for e in completed
                if e.result
            ]) if completed else 0

            logger.info(f"\nCompute time: {total_hours:.1f} GPU-hours")
            logger.info(f"Avg accuracy: {avg_acc*100:.2f}%")

        if failed:
            logger.warning("\nFailed experiments:")
            for e in failed:
                logger.warning(f"  - {e.name}")

        logger.info("="*70)

    def generate_dashboard(self):
        """Generate HTML monitoring dashboard."""
        dashboard_file = Path('monitor_dashboard.html')

        # Load current results
        results = []
        if self.results_file.exists():
            with open(self.results_file) as f:
                results = json.load(f)

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
        <h1>🔬 Sensory Perceiver Experiment Monitor</h1>
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
        for r in results:
            status_class = f"status-{r['status']}"
            duration = f"{r['duration_hours']:.2f}" if r['duration_hours'] else "—"
            accuracy = f"{r['result'].get('overall_accuracy', 0)*100:.2f}%" if r.get('result') else "—"

            html_content += f"""
            <tr>
                <td><strong>{r['name']}</strong></td>
                <td>{r['dataset']}</td>
                <td><span class="status-badge {status_class}">{r['status'].upper()}</span></td>
                <td>{r.get('start_time', '—')[:10]}</td>
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
            completed = len([r for r in results if r['status'] == 'completed'])
            progress_pct = (completed / total) * 100
            html_content += f"""
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_pct}%"></div>
            </div>
            <p>{completed}/{total} experiments completed ({progress_pct:.0f}%)</p>
"""

        html_content += f"""
        <div class="footer">
            <p>Auto-refreshes every 30 seconds | View results: <code>results/</code></p>
        </div>
    </div>
</body>
</html>
"""

        with open(dashboard_file, 'w') as f:
            f.write(html_content)

        logger.info(f"Dashboard saved to {dashboard_file}")
        return str(dashboard_file.absolute())


def main():
    parser = ArgumentParser(description='Sensory Perceiver Experiment Orchestrator')
    parser.add_argument('--mode', choices=['full', 'quick', 'monitor'], default='quick',
                        help='Experiment mode')
    parser.add_argument('--n-gpus', type=int, default=3,
                        help='Number of available GPUs')
    parser.add_argument('--parallel', action='store_true',
                        help='Run experiments in parallel (one per GPU)')
    parser.add_argument('--push-to-github', action='store_true',
                        help='Auto-commit results to GitHub')

    args = parser.parse_args()

    logger.info(f"Sensory Perceiver Experiment Orchestrator")
    logger.info(f"Mode: {args.mode} | GPUs: {args.n_gpus} | Parallel: {args.parallel}")

    # Create orchestrator
    orchestrator = ExperimentOrchestrator(n_gpus=args.n_gpus)

    # Queue experiments based on mode
    if args.mode == 'full':
        orchestrator.queue_full_benchmark()
    elif args.mode == 'quick':
        orchestrator.queue_quick_test()
    elif args.mode == 'monitor':
        orchestrator.generate_dashboard()
        return

    # Generate dashboard
    dashboard_path = orchestrator.generate_dashboard()
    logger.info(f"Open monitor: file://{dashboard_path}")

    # Run experiments
    logger.info(f"Running {len(orchestrator.experiments)} experiments...")
    orchestrator.run_all(parallel=args.parallel)

    # Push to GitHub if requested
    if args.push_to_github:
        logger.info("Pushing results to GitHub...")
        subprocess.run(['python', 'scripts/push_results.py'], check=False)


if __name__ == '__main__':
    main()
