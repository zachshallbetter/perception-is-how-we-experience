# Contributing to perception-is-how-we-experience

Thanks for your interest. This is a solo-maintained research repository, so review is best-effort (see
[SUPPORT.md](SUPPORT.md)). Open an issue before a large change so the direction can be agreed first.

## Setup

```sh
pip install -r requirements.txt
```

Baseline wrappers need `transformers` 4.x in a separate environment (`requirements-baselines.txt`); the core
training environment does not. The full environment steps are in [config/SETUP_GUIDE.md](config/SETUP_GUIDE.md) and
the canonical workflow is in [docs/training-guide.md](docs/training-guide.md).

## Checks

Run the runtime and service checks before longer jobs, and the setup preflight:

```sh
python3 -m unittest test_runtime_contracts.py test_service_layer_smoke.py
python scripts/validate_setup.py --verbose
```

`test_legacy_guardrails.py` scans the README and the docs for legacy command forms; run it with
`python3 -m unittest test_legacy_guardrails.py` after editing either. A short training job to confirm a change end to end:

```sh
python -m src.train --experiment continual --dataset split_cifar --epochs 1 --output_dir results/smoke_train
```

## Conventions

- Use module entrypoints (`python -m src.*`) as the canonical execution path.
- The paper source of truth is `paper/sections/`; figures and tables come from `paper/figures/generated/results_package/`.
- Claims carry one of the README's labels (Implemented, Evidenced, Conceptual, Patent candidate, Blocked / open
  problem). Do not state a result the packaged evidence does not support, and never add an invented citation or number.
- **Commits** use a conventional prefix with a sentence (`feat:`, `docs:`, `chore:`), for example
  `docs: Clarify the core distinction hypothesis`.

## Pull requests

Keep one change per pull request and fill in the pull request template.
