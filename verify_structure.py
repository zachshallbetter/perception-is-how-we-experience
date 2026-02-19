import os
from pathlib import Path

expected_structure = {
    "main-paper": ["Perception Is How We Experience.md"],
    "main-paper/sections": [
        "01-abstract.md", "02-introduction.md", "03-related-work.md",
        "04-methods.md", "05-experiments.md", "06-results.md",
        "07-discussion.md", "08-conclusion.md"
    ],
    "main-paper/figures": ["__init__.py", "README.md"],
    "supplementary-papers": [],
    "supplementary-papers/architecture-paper": [],
    "supplementary-papers/orchestrator-paper": [],
    "supplementary-papers/applications/robotics": [],
    "supplementary-papers/applications/medical": [],
    "supplementary-papers/applications/smart-home": [],
    "src": ["models.py", "train.py", "eval.py"],
    "src/data": ["loaders.py"],
    "src/experiments": ["orchestrator.py"],
    "src/experiments/configs": ["experiment_config.yaml"],
    "src/utils": ["metrics.py", "visualization.py"],
    "baselines/perceiver_io": ["baseline.py", "ablations.py"],
    "scripts": ["run_experiments.ps1", "run_experiments.sh", "validate_setup.py"],
    "planning": ["START_HERE.md", "MASTER_EXECUTION_PLAN.md", "EXECUTION_READY.md", "MULTI_PAPER_STRATEGY.md"],
    "research": ["PAPER_OUTLINES.md", "FIGURE_TEMPLATES.md", "APPLICATION_FRAMEWORKS.md", "VALIDATION_AUDIT.md"],
    "config": ["CLAUDE.md", "requirements.txt", "SETUP_GUIDE.md", "experiment_config.yaml"],
    "docs": ["ARCHITECTURE_OVERVIEW.md", "CONTRIBUTION_GUIDE.md", "TROUBLESHOOTING.md", "FAQ.md"],
    "reference/literature/papers": [],
    "reference/examples": [],
    "reference/sensor-types": [],
    "reference/reviews": [],
    "reference/code-snippets": ["dynamic-weighting.md"],
    ".": ["README.md", "QUICK_START.md", "PROGRESS.md"],
}

base_dir = Path("/Users/zachshallbetter/Projects/perception-is-how-we-experience")

missing_dirs = []
missing_files = []

for d, files in expected_structure.items():
    dir_path = base_dir / d if d != "." else base_dir
    if not dir_path.exists():
        missing_dirs.append(str(dir_path.relative_to(base_dir)))
    else:
        for f in files:
            file_path = dir_path / f
            if not file_path.exists():
                missing_files.append(str(file_path.relative_to(base_dir)))

if not missing_dirs and not missing_files:
    print("All expected directories and files are present!")
else:
    if missing_dirs:
        print("Missing directories:")
        for md in missing_dirs:
            print(f"  - {md}")
    if missing_files:
        print("Missing files:")
        for mf in missing_files:
            print(f"  - {mf}")
