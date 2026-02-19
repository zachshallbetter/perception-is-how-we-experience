"""
Aggregate evaluation results from multiple experiment runs.

Outputs:
- summary.csv: Mean ± std across seeds
- aggregate_stats.json: Full statistics
"""

import json
import csv
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> Dict[str, List[Dict]]:
    """Load all eval_results.json files from results directory."""
    results_by_experiment = {}

    for result_file in results_dir.rglob('eval_results.json'):
        # Extract experiment name and seed from path
        parts = result_file.parent.name.split('_seed_')
        if len(parts) != 2:
            logger.warning(f"Skipping {result_file}: unexpected naming convention")
            continue

        exp_name = parts[0]
        seed = int(parts[1])

        with open(result_file, 'r') as f:
            results = json.load(f)

        if exp_name not in results_by_experiment:
            results_by_experiment[exp_name] = []

        results['seed'] = seed
        results_by_experiment[exp_name].append(results)

    return results_by_experiment


def compute_statistics(results_list: List[Dict]) -> Dict:
    """Compute mean and std for a list of results."""
    if not results_list:
        return {}

    accuracies = [r['overall_accuracy'] for r in results_list]
    seeds = [r.get('seed', -1) for r in results_list]

    stats = {
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'min_accuracy': float(np.min(accuracies)),
        'max_accuracy': float(np.max(accuracies)),
        'n_runs': len(results_list),
        'seeds': seeds,
    }

    # Latency stats
    latencies = [r.get('latency_mean') for r in results_list if r.get('latency_mean') is not None]
    if latencies:
        stats['mean_latency_ms'] = float(np.mean(latencies))
        stats['std_latency_ms'] = float(np.std(latencies))

    return stats


def main():
    parser = ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='summary.csv')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    logger.info(f"Loading results from {results_dir}")
    results_by_exp = load_results(results_dir)

    if not results_by_exp:
        logger.warning("No results found!")
        return

    # Compute statistics for each experiment
    aggregate_stats = {}
    for exp_name, results_list in results_by_exp.items():
        stats = compute_statistics(results_list)
        aggregate_stats[exp_name] = stats
        logger.info(f"{exp_name}: Acc={stats['mean_accuracy']:.4f}±{stats['std_accuracy']:.4f}")

    # Save summary CSV
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['experiment', 'mean_accuracy', 'std_accuracy', 'n_runs', 'seeds']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp_name, stats in sorted(aggregate_stats.items()):
            row = {
                'experiment': exp_name,
                'mean_accuracy': f"{stats['mean_accuracy']:.4f}",
                'std_accuracy': f"{stats['std_accuracy']:.4f}",
                'n_runs': stats['n_runs'],
                'seeds': ','.join(map(str, stats['seeds'])),
            }
            writer.writerow(row)

    logger.info(f"Summary saved to {output_file}")

    # Save full statistics JSON
    json_file = output_file.parent / 'aggregate_stats.json'
    with open(json_file, 'w') as f:
        json.dump(aggregate_stats, f, indent=2)

    logger.info(f"Full statistics saved to {json_file}")


if __name__ == '__main__':
    main()
