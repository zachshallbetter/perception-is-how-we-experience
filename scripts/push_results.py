"""
Auto-commit experiment results to GitHub.

Usage:
    python scripts/push_results.py --message "Add results from full benchmark"
    python scripts/push_results.py --branch results/20260220
"""

import json
import logging
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_git_command(cmd: list, check: bool = True):
    """Run a git command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        logger.error(f"Git error: {e}")
        return False, str(e)


def push_results(message: str = None, branch: str = "main"):
    """Commit and push experiment results."""
    results_dir = Path("results")

    if not results_dir.exists():
        logger.warning("No results directory found")
        return False

    # Check if git repo
    success, _ = run_git_command(["git", "rev-parse", "--git-dir"], check=False)
    if not success:
        logger.error("Not a git repository")
        return False

    # Add results
    logger.info("Staging results...")
    run_git_command(["git", "add", "results/"])
    run_git_command(["git", "add", "monitor_dashboard.html"])

    # Create commit message
    if message is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Count experiments
        try:
            log_file = Path("results/experiment_log.json")
            if log_file.exists():
                with open(log_file) as f:
                    results = json.load(f)
                    completed = len([r for r in results if r["status"] == "completed"])
                    total = len(results)
                    message = f"Add experimental results: {completed}/{total} completed ({timestamp})"
            else:
                message = f"Add experiment results ({timestamp})"
        except:
            message = f"Add experiment results ({timestamp})"

    # Commit
    logger.info(f"Committing: {message}")
    success, output = run_git_command(["git", "commit", "-m", message], check=False)

    if not success:
        if "nothing to commit" in output:
            logger.info("No changes to commit")
            return True
        else:
            logger.error(f"Commit failed: {output}")
            return False

    # Push
    logger.info(f"Pushing to {branch}...")
    success, output = run_git_command(["git", "push", "origin", branch], check=False)

    if success:
        logger.info("✓ Results pushed to GitHub")
        return True
    else:
        logger.error(f"Push failed: {output}")
        return False


def main():
    parser = ArgumentParser(description="Push experiment results to GitHub")
    parser.add_argument(
        "--message", type=str, default=None, help="Custom commit message"
    )
    parser.add_argument(
        "--branch", type=str, default="main", help="Target branch (default: main)"
    )

    args = parser.parse_args()

    success = push_results(message=args.message, branch=args.branch)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
