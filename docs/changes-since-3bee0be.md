# Changes Since `3bee0be`

This document summarizes the material changes made since commit `3bee0be` (`chore(private): untrack ignored artifacts and legacy review outputs`).

It is intended to be the single retrospective record for code, experiment workflow, Titan operations, and paper assets added during the current implementation pass.

## 1. Training and Runtime Corrections

The continual-learning runtime was tightened so the documented training paths are valid again and the benchmark contract is explicit.

- Fixed dataset/task resolution so continual runs default to benchmark task counts instead of silently collapsing into invalid label/head combinations.
- Restored the documented `src.train` continual path by resolving effective task count and class count from the dataset contract.
- Replaced interpreter assumptions in `scripts/validate_setup.py` with `sys.executable` so environment validation works correctly inside Titan virtual environments.
- Added/updated runtime and smoke tests to cover the corrected contract behavior.

Key files:

- `src/runtime/contracts.py`
- `src/runtime/__init__.py`
- `src/services/training_service.py`
- `src/train.py`
- `scripts/validate_setup.py`
- `test_runtime_contracts.py`
- `test_service_layer_smoke.py`

## 2. Canonical Paper Benchmark Workflow

The repo now has one explicit paper-authoritative workflow for Split-CIFAR-100 on Titan.

- Canonical core path: `scripts/run_local.py --tuned`
- Canonical baselines: `scripts/run_baselines_cl.py --model perceiver` and `--model perceiver_io`
- Canonical task schedule: `n_tasks=20`
- Canonical Titan split:
  - GPU 0: core model and follow-up ablations
  - GPU 1: Perceiver baseline
  - GPU 2: Perceiver IO baseline

Supporting changes:

- Baseline runner now accepts deterministic seeds.
- Baseline defaults were made Titan-safe for 12 GB cards.
- Baseline dependencies were separated so the core environment does not inherit unnecessary transformer constraints.

Key files:

- `scripts/run_local.py`
- `scripts/run_baselines_cl.py`
- `scripts/run_all_baselines.sh`
- `requirements-baselines.txt`
- `docs/training-guide.md`
- `docs/baselines.md`
- `README.md`

## 3. Titan Automation and Campaign Control

Titan is now set up to run set-and-forget experiment queues with detached workers, live monitoring, and post-run paper package regeneration.

Added campaign launchers:

- `scripts/launch_paper_benchmarks_titan.sh`
- `scripts/launch_titan_continuous_campaign.sh`
- `scripts/launch_bwt_improvement_campaign.sh`
- `scripts/launch_distill_bwt_campaign.sh`
- `scripts/launch_distill_confirmation_campaign.sh`
- `scripts/launch_tuned_core_expansion_campaign.sh`

Added monitoring and recovery helpers:

- `scripts/campaign_status.sh`
- `scripts/distill_campaign_status.sh`
- `scripts/monitor_titan_campaign.sh`
- `scripts/restart_campaign_gpu0.sh`
- `scripts/restart_campaign_gpu_baseline.sh`
- `scripts/summarize_bwt_campaign.py`

Pipeline improvement:

- The main campaign launchers now ensure the `results-package-watcher` tmux session exists automatically.
- This removes the need to manually start paper package regeneration for the main Titan workflows.

## 4. New Mechanism Class: Distillation Replay

The next-mechanism step is no longer another nearby EWC/SI/buffer tweak. A distinct replay mechanism class was added: task-boundary teacher distillation replay on buffered raw samples.

Implemented behavior:

- New replay mode: `distill`
- Distillation operates from a frozen previous-task teacher snapshot
- Replay uses buffered raw samples and KL alignment rather than only label replay or feature contrast
- Distillation temperature is configurable per campaign lane

Supporting changes were made across:

- the local benchmark runner
- the service-layer training path
- the runtime contracts
- the CLI training entrypoint

Key files:

- `scripts/run_local.py`
- `src/services/training_service.py`
- `src/runtime/contracts.py`
- `src/train.py`
- `scripts/launch_distill_bwt_campaign.sh`

## 5. BWT Exploration Outcomes

The exploratory BWT sweep completed and produced a clear outcome: none of the local EWC/SI/buffer variants beat the tuned replay-centric core baseline.

Observed pattern:

- `EWC 1.0` was the strongest exploratory result in that sweep
- larger ER buffer helped relative to weaker ER settings
- mild SI did not improve BWT enough
- stronger SI and contrastive replay underperformed clearly
- the tuned replay-centric core remained the best current result

This changed the experimental direction:

- stop spending the next cycle on nearby regularization tweaks
- test a genuinely different mechanism class instead
- use the distillation replay campaign as the next decision point

## 6. Paper Realignment

The paper was rewritten so its claims match the implementation and measured evidence.

Conceptual corrections:

- The project is framed as a machine-perception architecture with typed modality semantics, modality-specific towers, shared integration, and continual adaptation.
- The strongest current evidence is still in replay-centric continual learning, not a fully validated broad multimodal benchmark suite.

Results framing now follows a three-step evidence ladder:

1. feature-only replay failure
2. raw-input replay recovery
3. tuned-core current best result

The paper also now states clearly that:

- parameter isolation is implemented but not yet the empirically dominant result
- the current BWT sweep is exploratory and does not replace the tuned-core headline

Updated paper files include:

- `paper/sections/00-implementation-status-matrix.md`
- `paper/sections/01-abstract.md`
- `paper/sections/02-introduction.md`
- `paper/sections/05-experiments.md`
- `paper/sections/06-results.md`
- `paper/sections/07-discussion.md`
- `paper/sections/08-conclusion.md`
- `paper/sections/10-concept-implementation-audit.md`
- `paper/Perception Is How We Experience.md`

## 7. Results Package and Publication Assets

The repo now builds a publication-oriented results package instead of requiring manual figure assembly.

Added generator:

- `paper/figures/build_results_package.py`

Generated package:

- `paper/figures/generated/results_package/`

Outputs include:

- overview table images
- phase progression figure
- tuned-core multiseed figure
- BWT sweep ranking figure
- copied source accuracy matrices
- CSV tables
- captions
- LaTeX include fragment
- provenance manifest
- distillation campaign package status file

The package generator can now fold in distillation vs tuned-ER control results as they complete.

## 8. Automatic Package Rebuilds

Paper asset regeneration is now tied to experiment completion.

Added watcher:

- `scripts/watch_results_package.sh`

Behavior:

- fingerprints completed `distill_*` and `tuned_er_*` `results.json` files
- rebuilds the paper results package only when that completed artifact set changes
- writes logs and state under `results/titan_logs/`

This watcher is intended to stay up as a long-running tmux session on Titan and is now auto-started by the main campaign launchers.

## 9. Dashboard and Operator Tooling

Monitoring is no longer limited to raw `tail -f` and `nvidia-smi`.

Added renderer and remote wrapper:

- `scripts/render_titan_dashboard.py`
- `scripts/titan_remote_dashboard.sh`

These provide:

- GPU utilization, memory, and temperature summaries
- active run stage detection
- ETA approximations
- BWT leaderboard summaries
- current paper-core aggregate metric display

## 10. Operational State On Titan

At the time of writing, Titan has:

- detached tmux-driven campaign execution
- automatic results package rebuild support
- one-command distillation campaign status
- queued decision branches for the next experiment:
  - confirm distillation if it wins
  - expand tuned-core evidence if it does not

The intended operating loop is now:

1. launch a campaign
2. let the watcher update paper assets automatically
3. inspect status via dashboard or campaign status script
4. make the next experiment decision from completed evidence

## 11. Remaining Follow-Through

The next high-value actions after this change set are:

- let the current distillation campaign finish
- compare distillation replay against tuned ER directly
- either promote distillation into the paper headline or strengthen tuned-core multiseed confidence instead of introducing another mechanism class immediately
