"""Regression guards preventing legacy command/reference reintroduction."""

import unittest
from pathlib import Path


class LegacyGuardrailsTest(unittest.TestCase):
    def test_no_legacy_command_forms_in_docs_and_scripts(self):
        files = [
            Path("README.md"),
            Path("docs/training-guide.md"),
            Path("docs/baselines.md"),
            Path("scripts/README.md"),
            Path("scripts/validate_setup.py"),
            Path("src/experiments/ablation_study_config.py"),
        ]
        forbidden_patterns = [
            "python train.py",
            "python eval.py",
            "python src/train.py",
            "python src/eval.py",
            "python experiment_orchestrator.py",
            "run_m1.py",
            "src/models.py",
            "mime_routing",
            "generate_synthetic_iot.py",
            "--experiment iot",
            "--dataset iot",
        ]

        for file_path in files:
            content = file_path.read_text(encoding="utf-8")
            for forbidden in forbidden_patterns:
                self.assertNotIn(
                    forbidden,
                    content,
                    msg=f"Forbidden legacy pattern {forbidden!r} found in {file_path}",
                )

    def test_models_package_uses_static_imports_only(self):
        content = Path("src/models/__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("importlib.util.spec_from_file_location", content)
        self.assertIn("from .core import", content)

    def test_orchestrator_does_not_use_legacy_iot_name(self):
        content = Path("src/services/orchestration_service.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("synthetic_iot", content)


if __name__ == "__main__":
    unittest.main()
