"""
Pre-flight validation for Sensory Perceiver experiments.

Checks all critical dependencies and configurations before running.
Must pass before attempting full benchmark.

Usage:
    python validate_setup.py
    python validate_setup.py --verbose
"""

import sys
import subprocess
from pathlib import Path
from argparse import ArgumentParser
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    pass


class Validator:
    """Comprehensive setup validator."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.checks = []
        self.passed = 0
        self.failed = 0

    def _log(self, message: str, level: str = 'info'):
        if level == 'info':
            logger.info(f"  ✓ {message}")
        elif level == 'warn':
            logger.warning(f"  ⚠ {message}")
        elif level == 'error':
            logger.error(f"  ✗ {message}")

    def check_python(self) -> bool:
        """Verify Python version."""
        logger.info("\n[1/8] Checking Python...")
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if sys.version_info.major < 3 or sys.version_info.minor < 10:
            self._log(f"Python {version} (need 3.10+)", 'error')
            return False
        self._log(f"Python {version}")
        return True

    def check_pytorch(self) -> bool:
        """Verify PyTorch and CUDA support."""
        logger.info("\n[2/8] Checking PyTorch...")
        try:
            import torch
            self._log(f"PyTorch {torch.__version__}")

            if not torch.cuda.is_available():
                self._log("CUDA not available", 'error')
                return False

            n_gpus = torch.cuda.device_count()
            self._log(f"GPU devices: {n_gpus}")

            if n_gpus < 1:
                self._log("No GPUs detected", 'error')
                return False

            for i in range(n_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                self._log(f"  GPU {i}: {gpu_name}")

            return True
        except ImportError:
            self._log("PyTorch not installed", 'error')
            return False
        except Exception as e:
            self._log(f"PyTorch error: {e}", 'error')
            return False

    def check_dependencies(self) -> bool:
        """Verify core dependencies."""
        logger.info("\n[3/8] Checking dependencies...")
        required = [
            ('numpy', 'numpy'),
            ('torch', 'torch'),
            ('torchvision', 'torchvision'),
            ('torchaudio', 'torchaudio'),
            ('timm', 'timm'),
            ('tensorboard', 'tensorboard'),
            ('tqdm', 'tqdm'),
            ('pyyaml', 'yaml'),
            ('omegaconf', 'omegaconf'),
            ('scikit-learn', 'sklearn'),
        ]

        all_ok = True
        for label, import_name in required:
            try:
                __import__(import_name)
                self._log(f"{label}")
            except ImportError:
                self._log(f"{label} (missing)", 'error')
                all_ok = False

        return all_ok

    def check_optional_dependencies(self) -> bool:
        """Verify optional dependencies."""
        logger.info("\n[4/8] Checking optional dependencies...")
        optional = [
            ('matplotlib', 'Visualization'),
            ('seaborn', 'Enhanced plots'),
            ('librosa', 'Audio processing'),
            ('cv2', 'Image processing'),
        ]

        all_ok = True
        for pkg, desc in optional:
            try:
                __import__(pkg)
                self._log(f"{pkg} - {desc}")
            except ImportError:
                self._log(f"{pkg} - {desc} (optional, can install later)", 'warn')

        return True  # Optional failures are warnings, not hard fails

    def check_disk_space(self) -> bool:
        """Verify sufficient disk space."""
        logger.info("\n[5/8] Checking disk space...")
        try:
            import shutil
            usage = shutil.disk_usage('results' if Path('results').exists() else '.')
            free_gb = usage.free / (1024 ** 3)

            self._log(f"Free space: {free_gb:.1f} GB")

            if free_gb < 50:
                self._log(f"Low disk space (need ~50GB for full benchmark)", 'warn')
                return False

            return True
        except Exception as e:
            self._log(f"Disk check error: {e}", 'warn')
            return True

    def check_files(self) -> bool:
        """Verify critical files exist."""
        logger.info("\n[6/8] Checking critical files...")
        required_files = [
            'src/train.py',
            'src/eval.py',
            'src/models/core.py',
            'src/data/loaders.py',
            'src/experiments/experiment_orchestrator.py',
            'requirements.txt',
        ]

        all_ok = True
        for fname in required_files:
            if Path(fname).exists():
                self._log(fname)
            else:
                self._log(f"{fname} (missing)", 'error')
                all_ok = False

        return all_ok

    def check_data_directories(self) -> bool:
        """Create/verify data directories."""
        logger.info("\n[7/8] Checking data directories...")
        dirs = [
            'results',
            'data',
        ]

        for dirname in dirs:
            path = Path(dirname)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            self._log(f"{dirname}/")

        return True

    def run_mini_test(self) -> bool:
        """Run tiny training test (1 batch, 1 epoch)."""
        logger.info("\n[8/8] Running mini training test...")

        try:
            result = subprocess.run(
                [
                    'python', '-m', 'src.train',
                    '--experiment', 'sensor_streaming',
                    '--dataset', 'sensor_streaming',
                    '--epochs', '1',
                    '--batch_size', '4',
                    '--device', 'cuda:0',
                    '--output_dir', 'results/validation_test',
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
            )

            if result.returncode == 0:
                self._log("Training test passed")
                return True
            else:
                error = result.stderr.split('\n')[-10:]  # Last 10 lines
                self._log(f"Training test failed", 'error')
                if self.verbose:
                    for line in error:
                        if line.strip():
                            logger.error(f"    {line}")
                return False

        except subprocess.TimeoutExpired:
            self._log("Training test timeout (GPU issue?)", 'error')
            return False
        except Exception as e:
            self._log(f"Training test error: {e}", 'error')
            return False

    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("="*60)
        logger.info("Sensory Perceiver Setup Validation")
        logger.info("="*60)

        checks = [
            ("Python version", self.check_python),
            ("PyTorch & CUDA", self.check_pytorch),
            ("Core dependencies", self.check_dependencies),
            ("Optional dependencies", self.check_optional_dependencies),
            ("Disk space", self.check_disk_space),
            ("Critical files", self.check_files),
            ("Data directories", self.check_data_directories),
            ("Mini training test", self.run_mini_test),
        ]

        results = []
        for name, check_fn in checks:
            try:
                result = check_fn()
                results.append((name, result))
                if not result:
                    self.failed += 1
                else:
                    self.passed += 1
            except Exception as e:
                logger.error(f"\nException in {name}: {e}")
                results.append((name, False))
                self.failed += 1

        # Summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)

        for name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{status:8} | {name}")

        logger.info("="*60)
        logger.info(f"Total: {self.passed} passed, {self.failed} failed")
        logger.info("="*60)

        if self.failed == 0:
            logger.info("\n✓ All checks passed! Ready to run experiments.\n")
            logger.info("Next steps:")
            logger.info("  1. Run quick test: .\\run_experiments.ps1 -Mode quick")
            logger.info("  2. Run full benchmark: .\\run_experiments.ps1 -Mode full -Parallel")
            logger.info("  3. Monitor: Open monitor_dashboard.html in browser\n")
            return True
        else:
            logger.error("\n✗ Validation failed. Fix errors above before proceeding.\n")
            return False


def main():
    parser = ArgumentParser(description='Validate Sensory Perceiver setup')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose error output')
    args = parser.parse_args()

    validator = Validator(verbose=args.verbose)
    success = validator.validate_all()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
