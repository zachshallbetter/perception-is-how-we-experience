# Training Guide: Canonical Paper Workflow

This guide defines the single paper-authoritative workflow for Split-CIFAR-100 continual learning on Titan Node.

## 1. Canonical Contract

Use this exact run contract for paper metrics:

- Core model: `scripts/run_local.py --tuned`
- Baselines: `scripts/run_baselines_cl.py --model perceiver` and `--model perceiver_io`
- Task schedule: `n_tasks=20`
- Baseline batch size on 12 GB cards: `batch_size=4`
- Core and baseline dependencies in separate Python environments

The intended Titan GPU split is:

- GPU 0: core model runs and replay-depth ablation
- GPU 1: Perceiver baseline runs
- GPU 2: Perceiver IO baseline runs

## 2. Environment Setup

Core environment:

```bash
pip install -r requirements.txt
```

Baseline environment:

```bash
pip install -r requirements.txt
pip install -r requirements-baselines.txt
```

`requirements-baselines.txt` is intentionally separate so `transformers` remains out of the core training environment.

## 3. Launch Modes

Single 3-run paper launch:

```bash
CORE_PYTHON=/ai/envs/perception/bin/python \
BASELINE_PYTHON=/ai/envs/torch-cu118-smoke/bin/python \
bash scripts/launch_paper_benchmarks_titan.sh
```

Continuous campaign (set-and-forget multi-seed queue):

```bash
CORE_PYTHON=/ai/envs/perception/bin/python \
BASELINE_PYTHON=/ai/envs/torch-cu118-smoke/bin/python \
CORE_SEEDS=42,43,44 \
BASELINE_SEEDS=42,43,44 \
bash scripts/launch_titan_continuous_campaign.sh
```

This launcher keeps each GPU busy with a sequential queue and writes per-run logs under `results/titan_logs/`.

Distillation replay campaign (new mechanism class vs tuned ER control):

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_distill_bwt_campaign.sh
```

This campaign compares task-boundary teacher distillation replay (`--replay_mode distill`) against additional tuned ER control seeds.

## 4. Monitoring

The canonical status command from your Mac is:

```bash
bash scripts/titan_remote_dashboard.sh
```

Watch mode:

```bash
bash scripts/titan_remote_dashboard.sh --watch 60
```

This renders a live dashboard from Titan with:

- GPU utilization and memory
- active run stages and ETA estimates
- BWT leaderboard from completed runs
- paper-core aggregate metrics

To rebuild the paper package automatically whenever completed `distill_*` or `tuned_er_*` results land:

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/watch_results_package.sh 60
```

Recommended detached Titan session:

```bash
tmux new-session -d -s results-package-watcher \
  "cd /ai/projects/Perception && PYTHON_BIN=/ai/envs/perception/bin/python bash scripts/watch_results_package.sh 60"
```

The campaign launchers now ensure this watcher session exists automatically, so you should not need to start it manually for:

- `launch_paper_benchmarks_titan.sh`
- `launch_titan_continuous_campaign.sh`
- `launch_distill_bwt_campaign.sh`
- `launch_distill_confirmation_campaign.sh`
- `launch_tuned_core_expansion_campaign.sh`

Distillation campaign status:

```bash
bash scripts/distill_campaign_status.sh distill-campaign-YYYYMMDD_HHMMSS
```

Decision triggers after the current campaign:

If distillation replay wins on BWT:

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
DISTILL_TEMP=4.0 \
bash scripts/launch_distill_confirmation_campaign.sh
```

If distillation replay does not win:

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_tuned_core_expansion_campaign.sh
```

Automatic decision helper:

```bash
bash scripts/decide_distill_followup.sh
```

Trigger the chosen next campaign automatically:

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/decide_distill_followup.sh --apply
```

The helper waits for the current package to contain the full initial distillation comparison set, then chooses between the confirmation and tuned-core expansion launchers using mean BWT.

Low-level commands remain useful during debugging:

```bash
tmux ls
nvidia-smi
tail -f results/titan_logs/<campaign-log>.log
```

Start periodic health monitoring for a running campaign:

```bash
bash scripts/monitor_titan_campaign.sh <campaign-id> 60 3
```

Attach a specific worker session:

```bash
tmux attach -t <campaign-id>-gpu0
tmux attach -t <campaign-id>-gpu1
tmux attach -t <campaign-id>-gpu2
```

Direct renderer on Titan:

```bash
/ai/envs/perception/bin/python scripts/render_titan_dashboard.py \
  --results-root results \
  --log-dir results/titan_logs \
  --md-out results/titan_dashboard.md \
  --json-out results/titan_dashboard.json
```

## 5. Outputs

Core run artifacts:

- `results/paper_runlocal_tuned_20t_*`
- `results.json`
- `accuracy_matrix.png`

Baseline artifacts:

- `results/paper_baseline_perceiver_20t_*`
- `results/paper_baseline_perceiver_io_20t_*`

Replay-depth ablation artifacts:

- `results/replay_depth_ablation/run_*`
- `ablation_runs.json`
- `ablation_summary.csv`
- `ablation_report.md` (after running the report script)

Distillation replay artifacts:

- `results/distill_t2_seed*`
- `results/distill_t4_seed*`
- `results/tuned_er_seed*`

Sensor/MIME ablation artifacts:

- `results/sensor_streaming_canonical_seed*`
- `results/sensor_streaming_merged_seed*`

## 6. Result Reporting

Generate a replay-depth markdown report:

```bash
python scripts/report_replay_depth_ablation.py \
  --ablation_dir results/replay_depth_ablation/run_YYYYMMDD_HHMMSS
```

Aggregate/plot once campaigns complete:

```bash
python scripts/aggregate_results.py --results_dir results
python scripts/plot_results.py --results_dir results --output_dir results/plots
```

## 7. Pre-Run Validation

Run checks before long jobs:

```bash
python3 -m unittest test_runtime_contracts.py test_service_layer_smoke.py
python scripts/validate_setup.py --verbose
```

## 8. Notes and Constraints

- `src.train` remains valid for service-layer and ablation studies, and now defaults continual datasets to benchmark task counts unless overridden.
- Paper-primary continual metrics should be generated from `scripts/run_local.py --tuned` and the two baseline scripts above.
- If GPU memory pressure occurs on Titan X cards, reduce baseline batch size before reducing task count.
- The canonical perception-side follow-up is `scripts/launch_sensor_mime_ablation_titan.sh`, which compares typed sensor modalities against a merged generic stream on the same `sensor_streaming` benchmark using class-conditioned synthetic sensor patterns and seed-matched evaluation. On 12 GB Titan X cards, keep the launcher at its conservative default `BATCH_SIZE=2` unless you have verified larger batches fit both canonical and merged lanes.
- The next sensor experiment sequence is:
  - aggregate canonical vs merged routing results with `python3 scripts/summarize_sensor_routing_results.py`
  - leave-one-modality-out with `bash scripts/launch_sensor_leave_one_out_titan.sh`
  - MIME-binding corruption with `bash scripts/launch_sensor_mime_corruption_titan.sh`
  - single-modality noise robustness with `bash scripts/launch_sensor_noise_robustness_titan.sh`
  - capacity-matched canonical vs merged control with `bash scripts/launch_sensor_capacity_matched_control_titan.sh`
