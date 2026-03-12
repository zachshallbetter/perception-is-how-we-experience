# Utility Scripts for Sensory Perceiver Experiments

## Dataset Preparation

### `generate_sensor_streaming_data.py`
Generate IoT sensor-stream benchmark data for validation (no external dependencies).

```bash
python scripts/generate_sensor_streaming_data.py \
  --n_sequences 100 \
  --sequence_length 500 \
  --output_dir data/sensor-streaming
```

**Output**: Multimodal sensor data (smartwatch, environmental, robot arm)

### `prepare_real_sensor_dataset.py`
Prepare or bootstrap a prepared real-sensor dataset for:
- `wisdm`
- `pamap2`
- `opportunity`
- `uci_har`
- `hhar`
- `mhealth`
- `selfback`
- `geomagnetic_wlan`

```bash
python3 scripts/prepare_real_sensor_dataset.py \
  --dataset wisdm \
  --raw-dir data/raw/wisdm
```

Behavior:
- integrated real-sensor datasets parse raw files and write prepared canonical `train.pt` / `val.pt`
- `ubiqlog` remains blocked because a canonical supervised task definition does not exist yet

Prepared outputs:
- `data/real-sensor/<dataset>/manifest.json`
- `data/real-sensor/<dataset>/README.md`
- `data/real-sensor/wisdm/train.pt`
- `data/real-sensor/wisdm/val.pt`

This is the canonical entrypoint for beginning real-data ingestion without introducing a parallel experiment family.

### `launch_real_sensor_routing_titan.sh`
Launch a prepared real-data canonical-vs-merged routing comparison.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
DATASET=wisdm \
bash scripts/launch_real_sensor_routing_titan.sh
```

This currently assumes the dataset has already been prepared under `data/real-sensor/<dataset>/`.

### `summarize_real_sensor_results.py`
Aggregate completed real-data routing runs for one dataset/family.

```bash
python3 scripts/summarize_real_sensor_results.py \
  --results-root results \
  --dataset wisdm \
  --family real-seq-001-wisdm
```

### `run_real_sensor_sequence_autonomous.sh`
Run the bounded Titan-local real-data progression supervisor.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/run_real_sensor_sequence_autonomous.sh real-seq-001
```

### `chain_real_sensor_sequence.sh`
Wait for one bounded real-sensor sequence to finish, then launch the next one automatically.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/chain_real_sensor_sequence.sh real-seq-001 real-seq-002
```

### `real_sensor_ai_supervisor.py`
Real-data supervisor for prepared datasets.

Current policy:
- auto-prepares integrated sensor datasets if raw files exist
- runs canonical vs merged once a dataset is prepared
- leaves `ubiqlog` explicitly blocked until a canonical supervised task is defined

### `real_sensor_sequence_status.sh`
Print the live/final status of a real-data supervision sequence.

```bash
bash scripts/real_sensor_sequence_status.sh real-seq-001
```

### `prepare_real_audio_dataset.py`
Prepare or bootstrap a prepared real-audio dataset for:
- `speech_commands`
- `esc50`
- `urbansound8k`
- `audioset`

```bash
python3 scripts/prepare_real_audio_dataset.py \
  --dataset speech_commands \
  --raw-dir data/raw/audio
```

Behavior:
- `speech_commands`: parses the saved Hugging Face dataset and writes prepared canonical `train.pt` / `val.pt`
- `esc50`: prepares canonical `train.pt` / `val.pt` from the extracted ESC-50 archive
- `urbansound8k`: prepares canonical `train.pt` / `val.pt` from the extracted UrbanSound8K archive, using `ffmpeg` fallback for non-PCM WAVs
- `audioset`: remains blocked because only metadata is staged locally

Prepared outputs:
- `data/real-audio/<dataset>/manifest.json`
- `data/real-audio/<dataset>/README.md`
- `data/real-audio/speech_commands/train.pt`
- `data/real-audio/speech_commands/val.pt`

### `launch_real_audio_routing_titan.sh`
Launch a prepared real-audio benchmark family on Titan.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
DATASET=speech_commands \
bash scripts/launch_real_audio_routing_titan.sh
```

This currently assumes the dataset has already been prepared under `data/real-audio/<dataset>/`.

### `summarize_real_audio_results.py`
Aggregate completed real-audio runs for one dataset/family.

```bash
python3 scripts/summarize_real_audio_results.py \
  --results-root results \
  --dataset speech_commands \
  --family real-audio-seq-001-speech_commands
```

### `run_real_audio_sequence_autonomous.sh`
Run the bounded Titan-local real-audio supervisor.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/run_real_audio_sequence_autonomous.sh real-audio-seq-001
```

### `chain_real_audio_sequence.sh`
Wait for one bounded real-audio sequence to finish, then launch the next one automatically.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/chain_real_audio_sequence.sh real-audio-seq-001 real-audio-seq-002
```

### `real_audio_ai_supervisor.py`
Real-audio supervisor for prepared datasets.

Current policy:
- auto-prepares `speech_commands`, `esc50`, and `urbansound8k` if raw files exist
- runs the prepared `multimodal` audio benchmark once a dataset is prepared
- leaves `audioset` explicitly blocked until local audio media exists

### `real_audio_sequence_status.sh`
Print the live/final status of a real-audio supervision sequence.

```bash
bash scripts/real_audio_sequence_status.sh real-audio-seq-001
```

### `download_audioset.py` (user implementation)
Download and preprocess AudioSet (22K balanced subset).
- Requires: `yt-dlp`, `librosa`
- Converts to 16kHz mono WAV files
- User should implement using AudioSet's official download instructions

### `preprocess_coco.py` (user implementation)
Preprocess MSCOCO 2017 validation split (5K images).
- Download from: https://cocodataset.org/
- Expected structure: `data/coco/images/val2017/` + annotations JSON

### `generate_av_mnist.py` (user implementation)
Generate audio-visual MNIST pairs (5K samples).
- Combines visual digits with generated speech

### `preprocess_flickr.py` (user implementation)
Preprocess Flickr30K (31K image-text pairs).
- Download from: https://www.flickr30k.org/
- Extract images and captions to parallel files

### `generate_split_cifar.py` (user implementation)
Generate Split-CIFAR-100 (20 sequential tasks, 5 classes each).
- Automatically downloads CIFAR-100
- Splits into task-specific subsets

### `generate_permuted_mnist.py` (user implementation)
Generate Permuted MNIST (20 tasks with random pixel permutations).
- Automatically downloads MNIST
- Applies task-specific permutations

## Training & Evaluation

### `src.train` (main training loop)
Train Sensory Perceiver on any experiment type.

```bash
python -m src.train \
  --experiment continual \
  --dataset split_cifar \
  --batch_size 32 \
  --epochs 50 \
  --output_dir results/my_experiment
```

**Outputs**:
- `best_model.pt`: Best validation checkpoint
- `latest_model.pt`: Final model
- `train_log.csv`: Training history
- `config.yaml`: Experiment configuration

### `src.eval`
Evaluate trained model on validation set.

```bash
python -m src.eval \
  --checkpoint results/my_experiment/best_model.pt \
  --experiment continual \
  --dataset split_cifar \
  --output_file results/my_experiment/eval_results.json
```

**Outputs**:
- `eval_results.json`: Per-class accuracy, confusion matrix, latency
- `embeddings.npy`: Per-modality embeddings (for visualization)

## Benchmarking

### `launch_paper_benchmarks_titan.sh`
Launch one canonical 3-GPU paper run in detached `tmux` sessions.

```bash
CORE_PYTHON=/ai/envs/perception/bin/python \
BASELINE_PYTHON=/ai/envs/torch-cu118-smoke/bin/python \
bash scripts/launch_paper_benchmarks_titan.sh
```

### `launch_titan_continuous_campaign.sh`
Launch a set-and-forget multi-seed queue:

- GPU 0: `run_local.py --tuned` seeds + replay-depth ablation
- GPU 1: `run_baselines_cl.py --model perceiver` seeds
- GPU 2: `run_baselines_cl.py --model perceiver_io` seeds

```bash
CORE_PYTHON=/ai/envs/perception/bin/python \
BASELINE_PYTHON=/ai/envs/torch-cu118-smoke/bin/python \
CORE_SEEDS=42,43,44 \
BASELINE_SEEDS=42,43,44 \
bash scripts/launch_titan_continuous_campaign.sh
```

This is the canonical paper-run path for Titan Node.

### `launch_distill_bwt_campaign.sh`
Launch a genuinely different replay mechanism class: task-boundary teacher distillation on replayed raw samples.

- GPU 0: `distill` replay, temperature `2.0`, seeds `42,43`
- GPU 1: `distill` replay, temperature `4.0`, seeds `42,43`
- GPU 2: tuned ER control seeds `45,46`

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_distill_bwt_campaign.sh
```

This is the right next campaign when nearby EWC / SI / buffer sweeps fail to beat the tuned replay baseline.

### `distill_campaign_status.sh`
One-command live status for a running distillation replay campaign.

```bash
bash scripts/distill_campaign_status.sh distill-campaign-YYYYMMDD_HHMMSS
```

Shows:
- worker session status
- GPU utilization
- per-lane task / epoch stage
- completed output counts
- current package watcher fold-in status

### `launch_distill_confirmation_campaign.sh`
Confirmation campaign to run only if distillation replay beats tuned ER on BWT.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
DISTILL_TEMP=4.0 \
bash scripts/launch_distill_confirmation_campaign.sh
```

This launches additional distill seeds and matched tuned-ER controls.

### `launch_tuned_core_expansion_campaign.sh`
Paper-strengthening follow-up to run only if distillation replay does not beat tuned ER.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_tuned_core_expansion_campaign.sh
```

This expands the tuned-core multiseed evidence instead of introducing yet another mechanism class immediately.

### `launch_sensor_mime_ablation_titan.sh`
Launch the perception-side sensor/MIME ablation on Titan.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_sensor_mime_ablation_titan.sh
```

Titan X note:
- the launcher defaults to `BATCH_SIZE=2` and `EPOCHS=20` because the canonical 3-tower route OOMs at larger batch sizes on 12 GB cards

This compares:

- `canonical`: three typed sensor modalities derived from the MIME registry
- `merged`: one generic 20-feature stream without typed modality separation
- each seed evaluates on the matching deterministic validation split used for that training run

This is the highest-value next experiment for supporting the broader perception claim beyond Split-CIFAR continual learning.
The launcher now uses a canonical family id for tmux sessions, log files, and result directories so the autonomous supervisor and summarizer resolve the same artifacts.
It also balances sensor branch work per GPU by pairing one canonical seed with one merged seed on each card, which reduces long idle periods caused by the lighter merged route finishing early.

Variant control:
- `SENSOR_VARIANT=baseline` compares canonical vs merged routing on the standard sensor benchmark
- `SENSOR_VARIANT=drop_smartwatch|drop_room_env|drop_robot_arm` runs leave-one-modality-out stress tests
- `SENSOR_VARIANT=corrupt_bindings` keeps tensors shaped the same but scrambles modality/class semantic alignment
- `SENSOR_VARIANT=noise_smartwatch|noise_room_env|noise_robot_arm` injects heavy drift/noise into one modality
- `CANONICAL_D_MODEL`, `MERGED_D_MODEL`, `CANONICAL_MLP_DIM`, and `MERGED_MLP_DIM` allow capacity-matched controls

### `summarize_sensor_routing_results.py`
Aggregate completed sensor-streaming eval runs by variant and routing mode.

```bash
python3 scripts/summarize_sensor_routing_results.py --results-root results
```

### `launch_sensor_leave_one_out_titan.sh`
Launch the three leave-one-modality-out routing comparisons.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_sensor_leave_one_out_titan.sh
```

### `launch_sensor_mime_corruption_titan.sh`
Launch the MIME-binding corruption comparison.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_sensor_mime_corruption_titan.sh
```

### `launch_sensor_noise_robustness_titan.sh`
Launch the three single-modality noise robustness comparisons.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_sensor_noise_robustness_titan.sh
```

### `find_sensor_capacity_match.py`
Estimate a merged-route `d_model` that roughly matches the canonical route parameter count.

```bash
python3 scripts/find_sensor_capacity_match.py
```

### `launch_sensor_capacity_matched_control_titan.sh`
Launch a capacity-matched canonical vs merged comparison using the helper above.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_sensor_capacity_matched_control_titan.sh
```

### `run_sensor_sequence_autonomous.sh`
Wait for the current baseline sensor family to finish, aggregate it, then run the full follow-on sequence automatically via the Titan-local strategy supervisor:
- capacity-matched control
- leave-one-out
- MIME-binding corruption
- single-modality noise robustness

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/run_sensor_sequence_autonomous.sh 20260312_001338
```

### `sensor_ai_supervisor.py`
Bounded Titan-local supervisor for the sensor evidence ladder.

Decision states:
- `canonical-dominant`
- `capacity-confounded`
- `competitive`
- `merged-favored`

Stop conditions:
- baseline family aggregated
- capacity-matched control completed before any positive routing claim is promoted
- all stress-test families completed

Outputs:
- progress log: `results/titan_logs/sensor-sequence-supervisor.log`
- autonomy report: `results/sensor_sequence_summaries/<family>-autonomy-report.json`
- resume-safe behavior: completed branch families are skipped if their expected eval artifacts already exist under the canonical family id
- live report updates while branches are running, including active family, progress counts, and explicit failure state if the supervisor stops early

### `sensor_sequence_status.sh`
One-command status view for the live sensor ladder.

```bash
bash scripts/sensor_sequence_status.sh 20260312_011301
```

Outputs:
- baseline result progress
- any active branch-family progress
- current GPU snapshot
- final autonomy state and gap summary when available
- recent supervisor log lines for that family

### `decide_distill_followup.sh`
Choose the next launcher automatically once the current distillation replay comparison has enough completed runs.

Dry run:

```bash
bash scripts/decide_distill_followup.sh
```

Apply the decision immediately:

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/decide_distill_followup.sh --apply
```

Decision rule:

- wait until the package contains at least 4 completed distillation runs and 2 tuned-ER control runs
- compare mean `BWT`
- launch `launch_distill_confirmation_campaign.sh` if distillation wins
- otherwise launch `launch_tuned_core_expansion_campaign.sh`

### `monitor_titan_campaign.sh`
Periodic monitor for campaign health and GPU saturation.

```bash
bash scripts/monitor_titan_campaign.sh paper-campaign-YYYYMMDD_HHMMSS 60 3
```

Arguments:
- campaign id
- polling interval seconds (default `60`)
- idle alert threshold in consecutive polls (default `3`)

### `watch_results_package.sh`
Poll for newly completed `distill_*` and `tuned_er_*` `results.json` files and rebuild the paper package only when the completed artifact set changes.

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/watch_results_package.sh 60
```

Outputs:
- rebuild log: `results/titan_logs/results-package-watcher.log`
- state file: `results/titan_logs/results-package-watcher.state`
- refreshed paper package: `paper/figures/generated/results_package/`

Recommended Titan usage:

```bash
tmux new-session -d -s results-package-watcher \
  "cd /ai/projects/Perception && PYTHON_BIN=/ai/envs/perception/bin/python bash scripts/watch_results_package.sh 60"
```

The main Titan launchers now auto-start this watcher if it is missing, so manual startup is only needed for ad hoc experiments or recovery.

### `render_titan_dashboard.py`
Render the current Titan state as Markdown and JSON using:
- `nvidia-smi`
- active `run_local.py` / `run_baselines_cl.py` processes
- live `results/titan_logs/*.log`
- completed `results/*/results.json`

Directly on Titan:

```bash
/ai/envs/perception/bin/python scripts/render_titan_dashboard.py \
  --results-root results \
  --log-dir results/titan_logs \
  --md-out results/titan_dashboard.md \
  --json-out results/titan_dashboard.json
```

### `titan_remote_dashboard.sh`
Mac-side wrapper that SSHes to Titan and renders the dashboard remotely.

One-shot:

```bash
bash scripts/titan_remote_dashboard.sh
```

Watch mode:

```bash
bash scripts/titan_remote_dashboard.sh --watch 60
```

### `run_replay_depth_ablation.py`
Run replay-depth ablations across three conditions:
- `feature_only` (`--replay_mode feature`)
- `raw_er` (`--replay_mode er`)
- `raw_contrastive` (`--replay_mode contrastive`)

```bash
python scripts/run_replay_depth_ablation.py \
  --experiment continual \
  --dataset split_cifar \
  --seeds 42,43,44 \
  --epochs 5 \
  --device cpu
```

**Outputs**:
- `ablation_runs.json`: per-run records + aggregate stats
- `ablation_runs.csv`: flat run table
- `ablation_summary.csv`: condition-level mean/std metrics

### `report_replay_depth_ablation.py`
Generate a Markdown report from a completed replay-depth ablation run.

```bash
python scripts/report_replay_depth_ablation.py \
  --ablation_dir results/replay_depth_ablation/run_YYYYMMDD_HHMMSS
```

**Output**:
- `ablation_report.md` in the selected ablation directory

### `run_full_benchmark.sh` (legacy bulk benchmark path)
Execute all 3 experiments × 3 seeds with evaluation.

```bash
bash scripts/run_full_benchmark.sh
# Outputs: results/benchmark_YYYYMMDD_HHMMSS/
```

**Runs** (parallel-friendly):
1. Continual Learning: Split-CIFAR-100 (3 seeds)
2. Continual Learning: Permuted MNIST (3 seeds)
3. IoT Streaming: Generated sensor data (3 seeds)
4. Aggregation: Compile statistics

**Duration**: ~6-12 hours on A100 (2h per seed × 3 seeds × 2 datasets)

## Analysis & Visualization

### `aggregate_results.py`
Aggregate results across multiple experiment runs.

```bash
python scripts/aggregate_results.py \
  --results_dir results/benchmark_20260219_120000 \
  --output_file results/benchmark_20260219_120000/summary.csv
```

**Outputs**:
- `summary.csv`: Mean ± std accuracy per experiment
- `aggregate_stats.json`: Full statistics (latency, per-seed breakdowns)

### `plot_results.py`
Generate visualization plots from evaluation results.

```bash
python scripts/plot_results.py \
  --results_dir results/benchmark_20260219_120000 \
  --output_dir results/benchmark_20260219_120000/plots
```

**Plots Generated**:
- `confusion_matrix.png`: Heatmap of prediction errors
- `per_class_accuracy.png`: Bar chart by class
- `training_curves.png`: Loss & accuracy over epochs
- `embeddings_tsne.png`: t-SNE projection of learned representations

## Workflow Example

```bash
# 1. Generate IoT benchmark data
python scripts/generate_sensor_streaming_data.py --output_dir data/sensor-streaming

# 2. Train on IoT experiment
python -m src.train \
  --experiment sensor_streaming \
  --batch_size 32 \
  --epochs 50 \
  --output_dir results/sensor_streaming_seed0

# 3. Evaluate
python -m src.eval \
  --checkpoint results/sensor_streaming_seed0/best_model.pt \
  --experiment sensor_streaming \
  --output_file results/sensor_streaming_seed0/eval_results.json

# 4. Visualize
python scripts/plot_results.py \
  --results_dir results/sensor_streaming_seed0 \
  --output_dir results/sensor_streaming_seed0/plots
```

## Notes

- **External datasets** (AudioSet, COCO, Flickr30K): Users must implement download scripts or use standard dataset repositories
- **Generated data**: Use `generate_sensor_streaming_data.py` for quick validation without external dependencies
- **GPU memory**: Reduce `--batch_size` or `--d_model` if OOM errors occur
- **Reproducibility**: All scripts accept `--seed` parameter for deterministic runs
