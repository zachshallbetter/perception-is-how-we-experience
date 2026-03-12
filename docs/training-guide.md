# Training Guide: Canonical Titan Workflow

This guide defines the paper-authoritative workflow for Titan Node and separates three experiment lanes clearly:

1. continual-learning paper lane,
2. exploratory mechanism lane,
3. sensor/MIME perception lane.

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

Sensor/MIME baseline routing campaign:

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_sensor_mime_ablation_titan.sh
```

Bounded autonomous sensor follow-on:

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/run_sensor_sequence_autonomous.sh <baseline-family-id>
```

This supervisor waits for the baseline family to finish, summarizes the canonical-vs-merged result, and then launches the next sensor evidence stages in order:

- capacity-matched control
- leave-one-modality-out
- MIME binding corruption
- single-modality noise robustness

It is intentionally bounded. It does not run forever or recursively chase "perfect" scores, and it treats the capacity-matched comparison as a required gate before promoting a positive routing claim.
Each launched family now uses a canonical family id for tmux sessions, result directories, and eval artifact discovery, so the supervisor can resume cleanly without drifting onto timestamp-only output names.
Sensor branch workers are also balanced per GPU now: each GPU carries one canonical seed and one merged seed, which keeps utilization steadier than isolating the lighter merged queue on a single card.

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

Sensor sequence supervisor status:

```bash
tail -f results/titan_logs/sensor-sequence-supervisor.log
```

One-command sensor ladder status:

```bash
bash scripts/sensor_sequence_status.sh <family>
```

One-command real-audio sequence status:

```bash
bash scripts/real_audio_sequence_status.sh <sequence-id>
```

Autonomy report output:

```bash
cat results/sensor_sequence_summaries/<family>-autonomy-report.json
```

The autonomy report now updates while the ladder is running, not only at the end. That means you can see:

- the currently active branch family
- artifact progress for that branch
- final failure state and error message if the supervisor stops early

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
- `results/sensor_streaming_<variant>_<routing>_seed*`
- `results/sensor_sequence_summaries/*.json`

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
- The 6-seed tuned-core family is now the canonical current continual-learning evidence, not the earlier 3-seed snapshot.
- Sensor/MIME results should only be promoted into the main paper once the canonical-vs-merged aggregation completes and the follow-on stress tests are interpreted together.
- The current `sensor_streaming` benchmark is hardened beyond the earlier random-label scaffold: compositional class codes, shared nuisance variation, and a shifted validation regime are now part of the generator. It is still synthetic and should be described that way.
- Paper-primary continual metrics should be generated from `scripts/run_local.py --tuned` and the two baseline scripts above.
- If GPU memory pressure occurs on Titan X cards, reduce baseline batch size before reducing task count.
- The canonical perception-side follow-up is `scripts/launch_sensor_mime_ablation_titan.sh`, which compares typed sensor modalities against a merged generic stream on the same `sensor_streaming` benchmark using a hardened synthetic generator with compositional class codes, shared nuisance variation, and a shifted validation regime. On 12 GB Titan X cards, keep the launcher at its conservative default `BATCH_SIZE=2` unless you have verified larger batches fit both canonical and merged lanes.
- The next sensor experiment sequence is:
  - aggregate canonical vs merged routing results with `python3 scripts/summarize_sensor_routing_results.py`
  - leave-one-modality-out with `bash scripts/launch_sensor_leave_one_out_titan.sh`
  - MIME-binding corruption with `bash scripts/launch_sensor_mime_corruption_titan.sh`
  - single-modality noise robustness with `bash scripts/launch_sensor_noise_robustness_titan.sh`
  - capacity-matched canonical vs merged control with `bash scripts/launch_sensor_capacity_matched_control_titan.sh`
- To queue that whole sequence behind a currently running baseline family on Titan, use:
  - `bash scripts/run_sensor_sequence_autonomous.sh <family>`
- The autonomous supervisor is bounded. It does not loop forever chasing “perfect” results; it stops in one of four decision states:
  - `canonical-dominant`
  - `capacity-confounded`
  - `competitive`
  - `merged-favored`
- Real-data progression uses the same `sensor_streaming` experiment family, but support is explicitly staged:
  - `wisdm`: runnable once raw files exist under `data/raw/wisdm`
  - `pamap2`, `opportunity`: scaffold only
  - sensor/body candidate queue only: `uci_har`, `hhar`, `mhealth`, `ubiqlog`, `selfback`, `geomagnetic_wlan`
  - audio candidate queue only: `speech_commands`, `esc50`, `urbansound8k`, `audioset`
- Prepare WISDM with:
  - `python3 scripts/prepare_real_sensor_dataset.py --dataset wisdm --raw-dir data/raw/wisdm`
- Launch one real-data routing comparison with:
  - `PYTHON_BIN=/ai/envs/perception/bin/python DATASET=wisdm bash scripts/launch_real_sensor_routing_titan.sh`
- Run the bounded real-data supervisor with:
  - `PYTHON_BIN=/ai/envs/perception/bin/python bash scripts/run_real_sensor_sequence_autonomous.sh real-seq-001`
- Check the real-data supervisor with:
  - `bash scripts/real_sensor_sequence_status.sh real-seq-001`
- The real-data supervisor only launches datasets that are actually prepared. It auto-prepares `wisdm` if raw files are present and reports scaffold-only / candidate datasets explicitly instead of pretending they are runnable.
- Audio now has its first runnable Titan path:
  - `speech_commands`: integrated into the `multimodal` runtime family and bounded Titan supervisor
  - `esc50`, `urbansound8k`, `audioset`: still blocked/candidate until prepared parsers land
- Run the bounded real-audio supervisor with:
  - `PYTHON_BIN=/ai/envs/perception/bin/python bash scripts/run_real_audio_sequence_autonomous.sh real-audio-seq-001`
- Check the real-audio supervisor with:
  - `bash scripts/real_audio_sequence_status.sh real-audio-seq-001`
