"""Models for Sensory Perceiver architecture with parameter isolation support."""

# Import from parent directory's models.py module
import importlib.util
import os

# Load models.py from parent directory
models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models.py')
spec = importlib.util.spec_from_file_location("_models", models_path)
_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_models)

# Re-export classes
PerceiverTower = _models.PerceiverTower
TaskAdapter = _models.TaskAdapter
SensoryPerceiver = _models.SensoryPerceiver
ContrastiveLoss = _models.ContrastiveLoss

__all__ = [
    'PerceiverTower',
    'TaskAdapter',
    'SensoryPerceiver',
    'ContrastiveLoss',
]
